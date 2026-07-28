#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.family.evaluate_stage2_candidate_fusion import load_source  # noqa: E402
from training.family.train_stage2_listwise_ranker import precursor_formula_features  # noqa: E402


SetKey = Tuple[int, ...]


def parse_named_source(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"source must be NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    return name.strip(), path.strip()


def target_keys(y: np.ndarray) -> List[SetKey]:
    return [tuple(np.flatnonzero(row > 0.5).tolist()) for row in y]


def formula_fold(formula: str, folds: int, seed: int) -> int:
    digest = hashlib.sha1(f"{seed}|{formula}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % int(folds)


class StackingFeatureBuilder:
    def __init__(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        train_meta: pd.DataFrame,
        precursor_names: Sequence[str],
        query_components: int,
        chemistry_components: int,
        seed: int,
    ) -> None:
        self.query_svd = TruncatedSVD(n_components=query_components, random_state=seed).fit(train_x)
        self.label_chemistry = precursor_formula_features([str(value) for value in precursor_names])
        self.chemistry_svd = TruncatedSVD(n_components=chemistry_components, random_state=seed)
        self.label_chemistry_embedding = self.chemistry_svd.fit_transform(self.label_chemistry).astype(np.float32)
        self.label_frequency = np.asarray(train_y.sum(axis=0), dtype=np.float32) + 1.0
        self.set_frequency = Counter(target_keys(train_y))
        train_families = train_meta["family_signature_primary"].fillna("UNK").astype(str).to_numpy()
        self.family_vocab = {value: index for index, value in enumerate(sorted(set(train_families)))}
        self.family_support = Counter(train_families.tolist())
        self.family_label_frequency: Dict[str, np.ndarray] = {}
        for family in sorted(set(train_families)):
            self.family_label_frequency[family] = (
                train_y[train_families == family].sum(axis=0).astype(np.float32) + 1.0
            )

    def encode_queries(self, x: np.ndarray, formulas: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        query_embedding = self.query_svd.transform(x).astype(np.float32)
        query_formula_raw = precursor_formula_features([str(value) for value in formulas])
        query_formula_embedding = self.chemistry_svd.transform(query_formula_raw).astype(np.float32)
        return query_embedding, query_formula_raw, query_formula_embedding

    def candidate_features(
        self,
        candidate: SetKey,
        query_embedding: np.ndarray,
        query_formula_raw: np.ndarray,
        query_formula_embedding: np.ndarray,
        family: str,
        ranks: Sequence[int],
        source_limit: int,
    ) -> np.ndarray:
        labels = np.asarray(candidate, dtype=np.int64)
        rank_array = np.asarray(ranks, dtype=np.float32)
        present = rank_array > 0
        reciprocal = np.zeros_like(rank_array, dtype=np.float32)
        reciprocal[present] = 1.0 / rank_array[present]
        log_rank = np.where(
            present, -np.log1p(rank_array), -math.log1p(source_limit * 2)
        ).astype(np.float32)
        top10 = ((rank_array > 0) & (rank_array <= 10)).astype(np.float32)
        rank_features = np.concatenate(
            [
                reciprocal,
                log_rank,
                top10,
                np.asarray(
                    [
                        float(present.sum()) / max(1, len(ranks)),
                        float(reciprocal.max()),
                        float(reciprocal.mean()),
                        float(rank_array[present].min()) / source_limit if present.any() else 2.0,
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        label_frequency = self.label_frequency[labels]
        family_frequency = self.family_label_frequency.get(family, self.label_frequency)[labels]
        prior_features = np.asarray(
            [
                float(len(candidate)) / 6.0,
                float(np.log(label_frequency).mean()),
                float(np.log(label_frequency).max()),
                float(np.log(label_frequency).min()),
                float(np.log(family_frequency).mean()),
                math.log1p(self.set_frequency.get(candidate, 0)),
                math.log1p(self.family_support.get(family, 0)),
                float(self.family_vocab.get(family, -1)),
            ],
            dtype=np.float32,
        )
        label_chemistry_embedding = self.label_chemistry_embedding[labels]
        candidate_mean = label_chemistry_embedding.mean(axis=0)
        candidate_max = label_chemistry_embedding.max(axis=0)
        candidate_raw = self.label_chemistry[labels].max(axis=0)
        query_element = query_formula_raw[:118]
        candidate_element = candidate_raw[:118]
        query_group = query_formula_raw[118:136]
        candidate_group = candidate_raw[118:136]
        query_element_present = query_element > 1e-6
        candidate_element_present = candidate_element > 1e-6
        query_group_present = query_group > 1e-6
        candidate_group_present = candidate_group > 1e-6
        exact_elements = float(np.logical_and(query_element_present, candidate_element_present).sum())
        exact_groups = float(np.logical_and(query_group_present, candidate_group_present).sum())
        chemistry_scalars = np.asarray(
            [
                exact_elements / max(1, int(query_element_present.sum())),
                exact_elements / max(1, int(candidate_element_present.sum())),
                exact_groups / max(1, int(query_group_present.sum())),
                exact_groups / max(1, int(candidate_group_present.sum())),
                float(np.minimum(query_element, candidate_element).sum()),
                float(np.minimum(query_group, candidate_group).sum()),
                float(np.abs(query_element - candidate_element).mean()),
                float(np.abs(query_group - candidate_group).mean()),
            ],
            dtype=np.float32,
        )
        chemistry_features = np.concatenate(
            [
                candidate_mean,
                candidate_max,
                query_formula_embedding * candidate_max,
                np.abs(query_formula_embedding - candidate_mean),
                chemistry_scalars,
            ]
        )
        return np.concatenate(
            [query_embedding, rank_features, prior_features, chemistry_features]
        ).astype(np.float32)


def build_candidate_matrix(
    builder: StackingFeatureBuilder,
    x: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    sources: Sequence[Sequence[Sequence[SetKey]]],
    source_limit: int,
) -> tuple[np.ndarray, np.ndarray, List[int], List[List[SetKey]]]:
    targets = target_keys(y)
    formulas = meta["formula"].fillna("").astype(str).tolist()
    families = meta["family_signature_primary"].fillna("UNK").astype(str).tolist()
    query_embedding, query_formula_raw, query_formula_embedding = builder.encode_queries(x, formulas)
    matrices = []
    labels = []
    groups: List[int] = []
    union_rows: List[List[SetKey]] = []
    for row_index, target in enumerate(targets):
        union: List[SetKey] = []
        seen = set()
        rank_maps = []
        for source in sources:
            current = source[row_index][:source_limit]
            rank_maps.append({candidate: rank + 1 for rank, candidate in enumerate(current)})
            for candidate in current:
                if candidate not in seen:
                    seen.add(candidate)
                    union.append(candidate)
        current_matrix = np.vstack(
            [
                builder.candidate_features(
                    candidate,
                    query_embedding[row_index],
                    query_formula_raw[row_index],
                    query_formula_embedding[row_index],
                    families[row_index],
                    [rank_map.get(candidate, 0) for rank_map in rank_maps],
                    source_limit,
                )
                for candidate in union
            ]
        )
        matrices.append(current_matrix)
        labels.append(np.asarray([int(candidate == target) for candidate in union], dtype=np.int8))
        groups.append(len(union))
        union_rows.append(union)
    return np.vstack(matrices), np.concatenate(labels), groups, union_rows


def group_offsets(groups: Sequence[int]) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(np.asarray(groups, dtype=np.int64))])


def select_query_rows(offsets: np.ndarray, query_indices: np.ndarray) -> np.ndarray:
    return np.concatenate([np.arange(offsets[index], offsets[index + 1]) for index in query_indices])


def rank_queries(
    scores: np.ndarray,
    query_indices: np.ndarray,
    offsets: np.ndarray,
    union_rows: Sequence[Sequence[SetKey]],
    output: List[List[SetKey]],
) -> None:
    score_offset = 0
    for query_index in query_indices:
        size = int(offsets[query_index + 1] - offsets[query_index])
        current = scores[score_offset : score_offset + size]
        order = np.argsort(-current, kind="stable")
        output[int(query_index)] = [union_rows[int(query_index)][int(index)] for index in order]
        score_offset += size


def exact_metrics(targets: Sequence[SetKey], rows: Sequence[Sequence[SetKey]]) -> Dict[str, float]:
    return {
        f"exact_hit@{k}": float(np.mean([target in set(row[:k]) for target, row in zip(targets, rows)]))
        for k in (1, 3, 5, 10, 20, 50, 100, 500)
    }


def make_ranker(args: argparse.Namespace, seed: int, n_estimators: int | None = None) -> lgb.LGBMRanker:
    return lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[1, 3, 5, 10],
        n_estimators=int(n_estimators or args.n_estimators),
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Five-fold OOF stacker for heterogeneous Stage2 candidate rankers.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--source", action="append", required=True, help="Repeat NAME=validation_candidates.jsonl")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--source_limit", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--query_components", type=int, default=32)
    parser.add_argument("--chemistry_components", type=int, default=16)
    parser.add_argument("--n_estimators", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=0.03)
    parser.add_argument("--num_leaves", type=int, default=63)
    parser.add_argument("--min_child_samples", type=int, default=50)
    parser.add_argument("--early_stopping_rounds", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    train_pack = np.load(input_dir / "train.npz", allow_pickle=True)
    val_pack = np.load(input_dir / "val.npz", allow_pickle=True)
    train_x = np.asarray(train_pack["x"], dtype=np.float32)
    train_y = np.asarray(train_pack["y_multi_hot"], dtype=np.float32)
    val_x = np.asarray(val_pack["x"], dtype=np.float32)
    val_y = np.asarray(val_pack["y_multi_hot"], dtype=np.float32)
    train_meta = pd.read_csv(input_dir / "train_meta.csv", low_memory=False)
    val_meta = pd.read_csv(input_dir / "val_meta.csv", low_memory=False)
    precursor_names = json.loads((input_dir / "precursor_names.json").read_text(encoding="utf-8"))
    named_paths = dict(parse_named_source(value) for value in args.source)
    source_rows = [load_source(path, len(val_x), args.source_limit) for path in named_paths.values()]
    builder = StackingFeatureBuilder(
        train_x, train_y, train_meta, precursor_names,
        args.query_components, args.chemistry_components, args.seed,
    )
    matrix, labels, groups, union_rows = build_candidate_matrix(
        builder, val_x, val_y, val_meta, source_rows, args.source_limit
    )
    targets = target_keys(val_y)
    offsets = group_offsets(groups)
    formulas = val_meta["canonical_formula"].fillna(val_meta["formula"]).astype(str).tolist()
    fold_ids = np.asarray([formula_fold(value, args.folds, args.seed) for value in formulas], dtype=np.int64)
    oof_rows: List[List[SetKey]] = [[] for _ in range(len(val_x))]
    fold_reports = []
    best_iterations = []
    for fold in range(args.folds):
        train_queries = np.flatnonzero(fold_ids != fold)
        holdout_queries = np.flatnonzero(fold_ids == fold)
        train_rows = select_query_rows(offsets, train_queries)
        holdout_rows = select_query_rows(offsets, holdout_queries)
        train_groups = [groups[int(index)] for index in train_queries]
        holdout_groups = [groups[int(index)] for index in holdout_queries]
        model = make_ranker(args, args.seed + fold)
        model.fit(
            matrix[train_rows], labels[train_rows], group=train_groups,
            eval_set=[(matrix[holdout_rows], labels[holdout_rows])],
            eval_group=[holdout_groups],
            callbacks=[
                lgb.early_stopping(args.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(50),
            ],
        )
        scores = model.predict(matrix[holdout_rows], num_iteration=model.best_iteration_)
        rank_queries(scores, holdout_queries, offsets, union_rows, oof_rows)
        best_iterations.append(int(model.best_iteration_))
        fold_targets = [targets[int(index)] for index in holdout_queries]
        fold_rankings = [oof_rows[int(index)] for index in holdout_queries]
        fold_reports.append({
            "fold": fold,
            "n_queries": int(len(holdout_queries)),
            "best_iteration": int(model.best_iteration_),
            **exact_metrics(fold_targets, fold_rankings),
        })
        model.booster_.save_model(str(run_dir / f"oof_fold_{fold}.txt"))

    oof_metrics = exact_metrics(targets, oof_rows)
    final_iterations = max(20, int(np.median(best_iterations)))
    final_model = make_ranker(args, args.seed + 1000, n_estimators=final_iterations)
    final_model.fit(matrix, labels, group=groups)
    final_model.booster_.save_model(str(run_dir / "final_stacked_ranker.txt"))
    joblib.dump(
        {
            "builder": builder,
            "source_names": list(named_paths),
            "source_limit": int(args.source_limit),
            "feature_dim": int(matrix.shape[1]),
            "final_iterations": final_iterations,
        },
        run_dir / "stacking_state.joblib",
    )
    with (run_dir / "oof_val_candidates.jsonl").open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(oof_rows):
            handle.write(json.dumps({"row_index": row_index, "candidate_label_ids": [list(key) for key in row]}) + "\n")
    report: Dict[str, Any] = {
        "protocol": "five_fold_oof_validation_formula_disjoint_exact_precursor_set_stacking",
        "config": vars(args),
        "sources": named_paths,
        "data": {
            "n_queries": len(val_x),
            "n_candidate_rows": int(len(matrix)),
            "feature_dim": int(matrix.shape[1]),
            "mean_union_candidates": float(np.mean(groups)),
            "oracle_union_recall": float(np.mean([target in set(row) for target, row in zip(targets, union_rows)])),
        },
        "folds": fold_reports,
        "oof_validation": oof_metrics,
        "final_iterations": final_iterations,
    }
    (run_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
