#!/usr/bin/env python3
"""Cross-fit a query gate for a validation-trained miss-only reranker."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from training.family.evaluate_stage2_candidate_fusion import load_source
from training.family.train_stage2_listwise_ranker import balanced_row_weights
from training.family.train_stage2_oof_candidate_stacker import formula_group_folds
from training.family.train_stage2_structured_energy_ranker import (
    load_matsci_pca_views,
    seed_everything,
    standardize_from_train,
    targets_from_matrix,
)
from training.family.train_stage2_validation_meta_lambdarank import (
    parse_expert_source,
    protected_expert_union,
    score_sorted_rows,
)
from training.family.train_stage2_within_family_variant_ranker import exact_metrics


SetKey = Tuple[int, ...]


def jaccard(left: set[SetKey], right: set[SetKey]) -> float:
    union = left | right
    return len(left & right) / max(1, len(union))


def agreement_features(
    base_row: Sequence[SetKey], expert_rows: Sequence[Sequence[SetKey]], limit: int
) -> np.ndarray:
    base = list(dict.fromkeys(base_row))[: int(limit)]
    base_set = set(base)
    sources = [list(dict.fromkeys(row))[: int(limit)] for row in expert_rows]
    votes = Counter(candidate for row in sources for candidate in row)
    outsider_votes = [count for candidate, count in votes.items() if candidate not in base_set]
    pairwise = [
        jaccard(set(sources[left]), set(sources[right]))
        for left in range(len(sources))
        for right in range(left + 1, len(sources))
    ]
    features = [
        len(votes) / max(1.0, float(len(sources) * int(limit))),
        max(votes.values(), default=0) / max(1.0, float(len(sources))),
        float(np.mean(list(votes.values()))) / max(1.0, float(len(sources))) if votes else 0.0,
        max(outsider_votes, default=0) / max(1.0, float(len(sources))),
        sum(value >= 2 for value in outsider_votes) / max(1.0, float(len(outsider_votes))),
        sum(value >= 3 for value in outsider_votes) / max(1.0, float(len(outsider_votes))),
        float(np.mean(pairwise)) if pairwise else 0.0,
        float(np.max(pairwise)) if pairwise else 0.0,
        float(np.mean([votes.get(candidate, 0) for candidate in base]))
        / max(1.0, float(len(sources))),
        max((votes.get(candidate, 0) for candidate in base), default=0)
        / max(1.0, float(len(sources))),
    ]
    for row in sources:
        current = set(row)
        features.extend(
            [
                jaccard(base_set, current),
                float(bool(row) and row[0] in base_set),
                len(set(row[:3]) & base_set) / 3.0,
                len(current & base_set) / max(1.0, float(int(limit))),
                float(np.mean([votes.get(candidate, 0) for candidate in row]))
                / max(1.0, float(len(sources))),
                votes.get(row[0], 0) / max(1.0, float(len(sources))) if row else 0.0,
            ]
        )
    return np.asarray(features, dtype=np.float32)


def gated_merge(
    base_row: Sequence[SetKey], ranked_row: Sequence[SetKey], protected_prefix: int
) -> List[SetKey]:
    base = list(dict.fromkeys(base_row))
    ranked = list(dict.fromkeys(ranked_row))
    prefix = min(int(protected_prefix), 10, len(base))
    selected = list(base[:prefix])
    selected_set = set(selected)
    for candidate in ranked:
        if candidate not in selected_set:
            selected.append(candidate)
            selected_set.add(candidate)
        if len(selected) >= 10:
            break
    for candidate in [*base, *ranked]:
        if candidate not in selected_set:
            selected.append(candidate)
            selected_set.add(candidate)
    return selected


def apply_gate_policy(
    base_rows: Sequence[Sequence[SetKey]],
    ranked_rows: Sequence[Sequence[SetKey]],
    probabilities: np.ndarray,
    threshold: float,
    protected_prefix: int,
) -> List[List[SetKey]]:
    return [
        gated_merge(base, ranked, int(protected_prefix))
        if float(probability) >= float(threshold)
        else list(base)
        for base, ranked, probability in zip(base_rows, ranked_rows, probabilities)
    ]


def select_gate_policy(
    targets: Sequence[SetKey],
    base_rows: Sequence[Sequence[SetKey]],
    ranked_rows: Sequence[Sequence[SetKey]],
    probabilities: np.ndarray,
) -> tuple[dict[str, object], List[List[SetKey]], List[dict[str, object]]]:
    base_hit = np.asarray(
        [target in set(row[:10]) for target, row in zip(targets, base_rows)], dtype=bool
    )
    thresholds = sorted(
        set(
            [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]
            + [float(value) for value in np.quantile(probabilities, np.linspace(0, 1, 41))]
        )
    )
    best_key = None
    best: dict[str, object] = {}
    best_rows: List[List[SetKey]] = []
    trials: List[dict[str, object]] = []
    for prefix in (0, 1, 3, 5, 7, 9):
        for threshold in thresholds:
            rows = apply_gate_policy(
                base_rows, ranked_rows, probabilities, threshold, prefix
            )
            hit = np.asarray(
                [target in set(row[:10]) for target, row in zip(targets, rows)], dtype=bool
            )
            metrics = exact_metrics(targets, rows)
            trial = {
                "strategy": "validation_miss_gate",
                "threshold": float(threshold),
                "protected_prefix": int(prefix),
                "gated_rows": int((probabilities >= float(threshold)).sum()),
                "new_hits_over_base": int((hit & ~base_hit).sum()),
                "lost_hits_vs_base": int((base_hit & ~hit).sum()),
                **metrics,
            }
            trials.append(trial)
            key = (
                float(metrics["exact_hit@10"]),
                float(metrics["exact_hit@5"]),
                float(metrics["exact_hit@1"]),
                -float(trial["lost_hits_vs_base"]),
                -float(trial["gated_rows"]),
            )
            if best_key is None or key > best_key:
                best_key = key
                best = trial
                best_rows = rows
    return best, best_rows, trials


def fixed_gate_policy(path: str) -> dict[str, object]:
    report = json.loads(Path(path).resolve().read_text(encoding="utf-8"))
    policy = report.get("validation", {}).get("best")
    if not isinstance(policy, dict):
        policy = report.get("best")
    if not isinstance(policy, dict):
        raise ValueError("fixed gate report must contain validation.best or best")
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--matsci_embeddings", required=True)
    parser.add_argument("--matsci_components", type=int, default=64)
    parser.add_argument("--base_candidates", required=True)
    parser.add_argument("--expert_source", action="append", default=[])
    parser.add_argument("--expert_manifest", default="")
    parser.add_argument("--reranker_scores_npz", required=True)
    parser.add_argument("--base_limit", type=int, default=100)
    parser.add_argument("--expert_limit", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--num_boost_round", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=0.03)
    parser.add_argument("--num_leaves", type=int, default=15)
    parser.add_argument("--min_data_in_leaf", type=int, default=20)
    parser.add_argument("--feature_fraction", type=float, default=0.9)
    parser.add_argument("--bagging_fraction", type=float, default=0.9)
    parser.add_argument("--miss_weight", type=float, default=1.0)
    parser.add_argument("--num_threads", type=int, default=64)
    parser.add_argument("--seed", type=int, default=8521)
    parser.add_argument("--evaluation_split", choices=("val", "test"), default="val")
    parser.add_argument("--resume_model", default="")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--fixed_policy_json", default="")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", required=True)
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--output_probabilities_npz", default="")
    args = parser.parse_args()

    if bool(args.eval_only) and not str(args.resume_model).strip():
        parser.error("--eval_only requires --resume_model")
    if str(args.evaluation_split) == "test" and not bool(args.eval_only):
        parser.error("test is allowed only with --eval_only")
    if str(args.evaluation_split) == "test" and not str(args.fixed_policy_json).strip():
        parser.error("test requires --fixed_policy_json")
    if not bool(args.eval_only) and str(args.evaluation_split) != "val":
        parser.error("gate training is restricted to validation")
    if float(args.miss_weight) <= 0:
        parser.error("--miss_weight must be positive")

    try:
        import lightgbm as lgb
    except ImportError as error:
        raise RuntimeError("LightGBM is required for the miss gate") from error

    seed_everything(int(args.seed))
    split = str(args.evaluation_split)
    input_dir = Path(args.input_dir).resolve()
    train_pack = np.load(input_dir / "train.npz", allow_pickle=True)
    evaluation_pack = np.load(input_dir / f"{split}.npz", allow_pickle=True)
    train_x = np.asarray(train_pack["x"], dtype=np.float32)
    evaluation_x = np.asarray(evaluation_pack["x"], dtype=np.float32)
    targets = targets_from_matrix(
        np.asarray(evaluation_pack["y_multi_hot"], dtype=np.float32)
    )
    evaluation_meta = pd.read_csv(input_dir / f"{split}_meta.csv", low_memory=False)
    _, train_matsci, evaluation_matsci, matsci_metadata = load_matsci_pca_views(
        Path(args.matsci_embeddings).resolve(),
        int(args.matsci_components),
        int(args.seed),
        split,
    )
    _, evaluation_query, _, _ = standardize_from_train(
        np.hstack([train_x, train_matsci]).astype(np.float32),
        np.hstack([evaluation_x, evaluation_matsci]).astype(np.float32),
    )

    expert_specs = [parse_expert_source(value) for value in args.expert_source]
    if str(args.expert_manifest).strip():
        manifest = json.loads(
            Path(args.expert_manifest).resolve().read_text(encoding="utf-8")
        )
        expert_specs.extend(
            (str(name), str(Path(path).resolve()))
            for name, path in manifest.get("expert_paths", {}).items()
            if str(name) != "base"
        )
    if not expert_specs:
        parser.error("at least one expert source or manifest is required")
    base_rows = load_source(args.base_candidates, len(targets), int(args.base_limit))
    expert_rows = [
        load_source(path, len(targets), int(args.expert_limit)) for _, path in expert_specs
    ]
    pool_rows = protected_expert_union(
        base_rows, expert_rows, int(args.base_limit), int(args.expert_limit)
    )
    score_pack = np.load(Path(args.reranker_scores_npz).resolve(), allow_pickle=False)
    raw_scores = np.asarray(score_pack["raw_scores"], dtype=np.float32)
    spans = [tuple(map(int, value)) for value in score_pack["spans"]]
    if len(spans) != len(targets) or (spans and spans[-1][1] != len(raw_scores)):
        raise ValueError("reranker score spans do not align with evaluation rows")
    ranked_rows = score_sorted_rows(pool_rows, raw_scores, spans)
    agreement = np.asarray(
        [
            agreement_features(
                base_rows[row_index],
                [source[row_index] for source in expert_rows],
                int(args.expert_limit),
            )
            for row_index in range(len(targets))
        ],
        dtype=np.float32,
    )
    features = np.hstack([agreement, evaluation_query]).astype(np.float32)
    base_hit = np.asarray(
        [target in set(row[:10]) for target, row in zip(targets, base_rows)], dtype=bool
    )
    labels = (~base_hit).astype(np.float32)
    formula_groups = evaluation_meta["family_group_key"].fillna("UNK").astype(str).to_numpy()
    families = (
        evaluation_meta["family_signature_primary"].fillna("UNK").astype(str).to_numpy()
    )

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": float(args.learning_rate),
        "num_leaves": int(args.num_leaves),
        "min_data_in_leaf": int(args.min_data_in_leaf),
        "feature_fraction": float(args.feature_fraction),
        "bagging_fraction": float(args.bagging_fraction),
        "bagging_freq": 1,
        "lambda_l2": 0.2,
        "num_threads": int(args.num_threads),
        "seed": int(args.seed),
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    fold_reports: List[dict[str, object]] = []
    if bool(args.eval_only):
        booster = lgb.Booster(model_file=str(Path(args.resume_model).resolve()))
        probabilities = np.asarray(booster.predict(features), dtype=np.float32)
    else:
        probabilities = np.zeros(len(targets), dtype=np.float32)
        for fold_index, (train_rows, holdout_rows) in enumerate(
            formula_group_folds(
                formula_groups, n_splits=int(args.folds), seed=int(args.seed)
            )
        ):
            weights = balanced_row_weights(
                train_rows, formula_groups, families, 0.5, 0.0
            ).astype(np.float32)
            weights *= np.where(
                labels[train_rows] > 0.5, float(args.miss_weight), 1.0
            ).astype(np.float32)
            dataset = lgb.Dataset(
                features[train_rows], label=labels[train_rows], weight=weights
            )
            booster = lgb.train(
                {**params, "seed": int(args.seed) + fold_index},
                dataset,
                num_boost_round=int(args.num_boost_round),
                callbacks=[lgb.log_evaluation(period=0)],
            )
            probabilities[holdout_rows] = np.asarray(
                booster.predict(features[holdout_rows]), dtype=np.float32
            )
            fold_reports.append(
                {
                    "fold": int(fold_index),
                    "train_rows": int(len(train_rows)),
                    "holdout_rows": int(len(holdout_rows)),
                }
            )
        all_rows = np.arange(len(targets), dtype=np.int64)
        final_weights = balanced_row_weights(
            all_rows, formula_groups, families, 0.5, 0.0
        ).astype(np.float32)
        final_weights *= np.where(
            labels > 0.5, float(args.miss_weight), 1.0
        ).astype(np.float32)
        booster = lgb.train(
            params,
            lgb.Dataset(features, label=labels, weight=final_weights),
            num_boost_round=int(args.num_boost_round),
            callbacks=[lgb.log_evaluation(period=0)],
        )

    if str(args.fixed_policy_json).strip():
        policy = fixed_gate_policy(args.fixed_policy_json)
        output_rows = apply_gate_policy(
            base_rows,
            ranked_rows,
            probabilities,
            float(policy["threshold"]),
            int(policy["protected_prefix"]),
        )
        output_hit = np.asarray(
            [target in set(row[:10]) for target, row in zip(targets, output_rows)], dtype=bool
        )
        best = {
            **policy,
            "gated_rows": int((probabilities >= float(policy["threshold"])).sum()),
            "new_hits_over_base": int((output_hit & ~base_hit).sum()),
            "lost_hits_vs_base": int((base_hit & ~output_hit).sum()),
            **exact_metrics(targets, output_rows),
        }
        trials = [best]
    else:
        best, output_rows, trials = select_gate_policy(
            targets, base_rows, ranked_rows, probabilities
        )

    report = {
        "protocol": {
            "name": "validation_formula_group_crossfit_base_miss_gate",
            "evaluation_split": split,
            "test_policy_frozen": bool(str(args.fixed_policy_json).strip()),
            "rows": int(len(targets)),
            "formula_groups": int(len(set(formula_groups.tolist()))),
            "features": int(features.shape[1]),
            "misses": int(labels.sum()),
            "matsci": matsci_metadata,
            "expert_sources": [
                {"name": name, "path": path} for name, path in expert_specs
            ],
        },
        "model": {
            "num_boost_round": int(args.num_boost_round),
            "num_leaves": int(args.num_leaves),
            "learning_rate": float(args.learning_rate),
            "miss_weight": float(args.miss_weight),
        },
        "folds": fold_reports,
        "base": exact_metrics(targets, base_rows),
        "validation": {"best": best, "trial_count": int(len(trials))},
    }
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_candidates = Path(args.output_candidates_jsonl).resolve()
    output_candidates.parent.mkdir(parents=True, exist_ok=True)
    with output_candidates.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(output_rows):
            handle.write(
                json.dumps(
                    {
                        "row_index": int(row_index),
                        "candidate_label_ids": [list(candidate) for candidate in row],
                    }
                )
                + "\n"
            )
    output_model = Path(args.output_model).resolve()
    output_model.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(output_model))
    output_probabilities = (
        Path(args.output_probabilities_npz).resolve()
        if str(args.output_probabilities_npz).strip()
        else output_json.with_name("val_gate_probabilities.npz")
    )
    np.savez_compressed(output_probabilities, probabilities=probabilities)
    print(json.dumps({"base": report["base"], "best": best}, indent=2))


if __name__ == "__main__":
    main()
