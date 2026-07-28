#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.family.train_stage2_listwise_ranker import precursor_formula_features


SetKey = Tuple[int, ...]


def load_candidates(path: Path, n_rows: int, limit: int) -> tuple[List[List[SetKey]], List[List[float]]]:
    rows: List[List[SetKey]] = [[] for _ in range(n_rows)]
    scores: List[List[float]] = [[] for _ in range(n_rows)]
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            row = int(record["row_index"])
            values = record["candidate_label_ids"][:limit]
            raw_scores = record.get("scores", [])[:limit]
            rows[row] = [tuple(sorted({int(value) for value in item})) for item in values]
            scores[row] = [
                float(raw_scores[index]) if index < len(raw_scores) else -math.log1p(index)
                for index in range(len(values))
            ]
    return rows, scores


def target_keys(y: np.ndarray) -> List[SetKey]:
    return [tuple(np.flatnonzero(row > 0.5).tolist()) for row in y]


def zscore(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if not len(array):
        return array
    return (array - array.mean()) / max(float(array.std()), 1e-6)


class FeatureBuilder:
    def __init__(
        self,
        y_train: np.ndarray,
        x_train: np.ndarray,
        train_meta: pd.DataFrame,
        components: int,
        ridge_alpha: float,
        seed: int,
        precursor_names: Sequence[str] | None = None,
        chemistry_components: int = 0,
    ) -> None:
        y_sparse = sparse.csr_matrix(y_train)
        svd = TruncatedSVD(n_components=components, random_state=seed)
        svd.fit(y_sparse)
        self.label_embedding = svd.components_.T.astype(np.float32)
        set_embedding = np.asarray(y_sparse @ self.label_embedding, dtype=np.float32)
        norms = np.linalg.norm(set_embedding, axis=1, keepdims=True)
        set_embedding /= np.maximum(norms, 1e-6)
        self.ridge = Ridge(alpha=ridge_alpha).fit(x_train, set_embedding)
        self.label_frequency = np.asarray(y_train.sum(axis=0), dtype=np.float32) + 1.0
        self.set_frequency = Counter(target_keys(y_train))
        self.family_label_frequency: Dict[str, np.ndarray] = {}
        self.source_label_frequency: Dict[str, np.ndarray] = {}
        self.chemistry_components = int(chemistry_components)
        self.label_chemistry = None
        self.label_chemistry_embedding = None
        self.chemistry_svd = None
        if self.chemistry_components > 0:
            if precursor_names is None:
                raise ValueError("precursor names are required for chemistry features")
            self.label_chemistry = precursor_formula_features([str(value) for value in precursor_names])
            self.chemistry_svd = TruncatedSVD(n_components=self.chemistry_components, random_state=seed)
            self.label_chemistry_embedding = self.chemistry_svd.fit_transform(self.label_chemistry).astype(np.float32)
        families = train_meta["family_signature_primary"].fillna("UNK").astype(str).to_numpy()
        sources = train_meta["source_dataset"].fillna("").astype(str).to_numpy()
        for family in np.unique(families):
            self.family_label_frequency[family] = y_train[families == family].sum(axis=0).astype(np.float32) + 1.0
        for source in np.unique(sources):
            self.source_label_frequency[source] = y_train[sources == source].sum(axis=0).astype(np.float32) + 1.0

    def predict_embedding(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(self.ridge.predict(x), dtype=np.float32)
        return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-6)

    def query_chemistry(self, formulas: Sequence[str]) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self.chemistry_svd is None:
            return None, None
        raw = precursor_formula_features([str(value) for value in formulas])
        encoded = self.chemistry_svd.transform(raw).astype(np.float32)
        return raw, encoded

    def row_features(
        self,
        predicted: np.ndarray,
        candidates: Sequence[SetKey],
        raw_scores: Sequence[float],
        family: str,
        source: str,
        query_chemistry_raw: np.ndarray | None = None,
        query_chemistry_embedding: np.ndarray | None = None,
    ) -> np.ndarray:
        if not candidates:
            chemistry_dim = self.chemistry_components * 4 + (8 if self.chemistry_components else 0)
            return np.zeros((0, self.label_embedding.shape[1] * 3 + 10 + chemistry_dim), dtype=np.float32)
        analytical = zscore(raw_scores)
        family_frequency = self.family_label_frequency.get(family, self.label_frequency)
        source_frequency = self.source_label_frequency.get(source, self.label_frequency)
        rows = []
        for rank, candidate in enumerate(candidates):
            labels = np.asarray(candidate, dtype=np.int64)
            candidate_embedding = self.label_embedding[labels].sum(axis=0)
            candidate_embedding /= max(float(np.linalg.norm(candidate_embedding)), 1e-6)
            product = predicted * candidate_embedding
            difference = np.abs(predicted - candidate_embedding)
            global_prior = float(np.log(self.label_frequency[labels]).mean())
            family_prior = float(np.log(family_frequency[labels]).mean())
            source_prior = float(np.log(source_frequency[labels]).mean())
            chemistry_features = np.zeros(0, dtype=np.float32)
            if self.label_chemistry_embedding is not None:
                if query_chemistry_raw is None or query_chemistry_embedding is None:
                    raise ValueError("query chemistry descriptors are required")
                candidate_chemistry_embedding = self.label_chemistry_embedding[labels]
                candidate_mean = candidate_chemistry_embedding.mean(axis=0)
                candidate_max = candidate_chemistry_embedding.max(axis=0)
                candidate_raw = self.label_chemistry[labels].max(axis=0)
                query_element = query_chemistry_raw[:118]
                candidate_element = candidate_raw[:118]
                query_group = query_chemistry_raw[118:136]
                candidate_group = candidate_raw[118:136]
                query_element_present = query_element > 1e-6
                candidate_element_present = candidate_element > 1e-6
                query_group_present = query_group > 1e-6
                candidate_group_present = candidate_group > 1e-6
                exact_elements = float(np.logical_and(query_element_present, candidate_element_present).sum())
                exact_groups = float(np.logical_and(query_group_present, candidate_group_present).sum())
                scalars = np.asarray(
                    [
                        exact_elements / max(1, int(query_element_present.sum())),
                        exact_elements / max(1, int(candidate_element_present.sum())),
                        exact_groups / max(1, int(query_group_present.sum())),
                        exact_groups / max(1, int(candidate_group_present.sum())),
                        float(np.minimum(query_element, candidate_element).sum()),
                        float(np.minimum(query_group, candidate_group).sum()),
                        float(np.abs(query_element - candidate_raw[:118]).mean()),
                        float(np.abs(query_group - candidate_raw[118:136]).mean()),
                    ],
                    dtype=np.float32,
                )
                chemistry_features = np.concatenate(
                    [
                        candidate_mean,
                        candidate_max,
                        query_chemistry_embedding * candidate_max,
                        np.abs(query_chemistry_embedding - candidate_mean),
                        scalars,
                    ]
                ).astype(np.float32)
            rows.append(
                np.concatenate(
                    [
                        candidate_embedding,
                        product,
                        difference,
                        chemistry_features,
                        np.asarray(
                            [
                                float(np.dot(predicted, candidate_embedding)),
                                float(analytical[rank]) if rank < len(analytical) else 0.0,
                                -math.log1p(rank),
                                float(len(candidate)),
                                global_prior,
                                family_prior,
                                source_prior,
                                math.log1p(self.set_frequency.get(candidate, 0)),
                                float(max(self.label_frequency[labels])),
                                float(min(self.label_frequency[labels])),
                            ],
                            dtype=np.float32,
                        ),
                    ]
                )
            )
        return np.asarray(rows, dtype=np.float32)


def build_matrix(
    builder: FeatureBuilder,
    predicted: np.ndarray,
    candidates: Sequence[Sequence[SetKey]],
    scores: Sequence[Sequence[float]],
    targets: Sequence[SetKey],
    meta: pd.DataFrame,
    require_positive: bool,
    query_chemistry_raw: np.ndarray | None = None,
    query_chemistry_embedding: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    features = []
    labels = []
    groups = []
    row_indices = []
    for row, row_candidates in enumerate(candidates):
        if not row_candidates:
            continue
        target = targets[row]
        if require_positive and target not in set(row_candidates):
            continue
        current = builder.row_features(
            predicted[row],
            row_candidates,
            scores[row],
            str(meta.iloc[row].get("family_signature_primary", "UNK")),
            str(meta.iloc[row].get("source_dataset", "")),
            None if query_chemistry_raw is None else query_chemistry_raw[row],
            None if query_chemistry_embedding is None else query_chemistry_embedding[row],
        )
        features.append(current)
        labels.append(np.asarray([int(candidate == target) for candidate in row_candidates], dtype=np.int8))
        groups.append(len(row_candidates))
        row_indices.append(row)
    return np.vstack(features), np.concatenate(labels), groups, row_indices


def rank_rows(
    model: lgb.LGBMRanker,
    matrix: np.ndarray,
    groups: Sequence[int],
    row_indices: Sequence[int],
    candidates: Sequence[Sequence[SetKey]],
) -> List[List[SetKey]]:
    prediction = model.predict(matrix)
    output: List[List[SetKey]] = [[] for _ in range(len(candidates))]
    offset = 0
    for row, size in zip(row_indices, groups):
        scores = prediction[offset : offset + size]
        order = np.argsort(-scores, kind="stable")
        output[row] = [candidates[row][int(index)] for index in order]
        offset += size
    return output


def exact_metrics(targets: Sequence[SetKey], rows: Sequence[Sequence[SetKey]]) -> Dict[str, float]:
    return {
        f"exact_hit@{k}": float(np.mean([target in set(row[:k]) for target, row in zip(targets, rows)]))
        for k in (1, 3, 5, 10, 20, 50, 100, 500)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LambdaMART ranker for exact precursor-set candidates.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--train_candidates", required=True)
    parser.add_argument("--val_candidates", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--train_limit", type=int, default=200)
    parser.add_argument("--val_limit", type=int, default=500)
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--ridge_alpha", type=float, default=10.0)
    parser.add_argument("--chemistry_components", type=int, default=0)
    parser.add_argument("--n_estimators", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=0.03)
    parser.add_argument("--num_leaves", type=int, default=63)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--device", choices=("cpu", "gpu", "cuda"), default="cpu")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    packs = {split: np.load(input_dir / f"{split}.npz", allow_pickle=True) for split in ("train", "val")}
    meta = {split: pd.read_csv(input_dir / f"{split}_meta.csv", low_memory=False) for split in ("train", "val")}
    x_train = np.asarray(packs["train"]["x"], dtype=np.float32)
    x_val = np.asarray(packs["val"]["x"], dtype=np.float32)
    y_train = np.asarray(packs["train"]["y_multi_hot"], dtype=np.float32)
    y_val = np.asarray(packs["val"]["y_multi_hot"], dtype=np.float32)
    targets = {"train": target_keys(y_train), "val": target_keys(y_val)}
    candidates = {}
    scores = {}
    candidates["train"], scores["train"] = load_candidates(Path(args.train_candidates), len(x_train), args.train_limit)
    candidates["val"], scores["val"] = load_candidates(Path(args.val_candidates), len(x_val), args.val_limit)

    precursor_names = json.loads((input_dir / "precursor_names.json").read_text(encoding="utf-8"))
    builder = FeatureBuilder(
        y_train, x_train, meta["train"], args.components, args.ridge_alpha, args.seed,
        precursor_names=precursor_names,
        chemistry_components=args.chemistry_components,
    )
    embedding = {"train": builder.predict_embedding(x_train), "val": builder.predict_embedding(x_val)}
    query_chemistry = {
        split: builder.query_chemistry(meta[split]["formula"].fillna("").astype(str).tolist())
        for split in ("train", "val")
    }
    train_matrix, train_labels, train_groups, train_rows = build_matrix(
        builder, embedding["train"], candidates["train"], scores["train"], targets["train"], meta["train"], True,
        *query_chemistry["train"],
    )
    val_matrix, val_labels, val_groups, val_rows = build_matrix(
        builder, embedding["val"], candidates["val"], scores["val"], targets["val"], meta["val"], False,
        *query_chemistry["val"],
    )
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[1, 3, 5, 10],
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=args.seed,
        n_jobs=-1,
        device_type=args.device,
        verbosity=-1,
    )
    model.fit(
        train_matrix,
        train_labels,
        group=train_groups,
        eval_set=[(val_matrix, val_labels)],
        eval_group=[val_groups],
        callbacks=[lgb.early_stopping(100, verbose=True), lgb.log_evaluation(25)],
    )
    ranked = rank_rows(model, val_matrix, val_groups, val_rows, candidates["val"])
    report: Dict[str, Any] = {
        "protocol": "val_formula_disjoint_exact_precursor_set_lambdarank",
        "config": vars(args),
        "data": {
            "train_queries": len(train_groups),
            "train_candidates": int(len(train_labels)),
            "val_queries": len(val_groups),
            "val_candidates": int(len(val_labels)),
            "feature_dim": int(train_matrix.shape[1]),
        },
        "best_iteration": int(model.best_iteration_),
        "validation": exact_metrics(targets["val"], ranked),
    }
    model.booster_.save_model(str(run_dir / "lambdarank.txt"))
    (run_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "val_candidates.jsonl").open("w", encoding="utf-8") as handle:
        for row, values in enumerate(ranked):
            handle.write(json.dumps({"row_index": row, "candidate_label_ids": [list(value) for value in values]}) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
