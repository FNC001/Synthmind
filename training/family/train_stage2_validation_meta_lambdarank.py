#!/usr/bin/env python3
"""Cross-fit a validation meta-ranker over aligned precursor experts.

Base experts are trained on the train split.  Their validation predictions are
cross-fitted by formula group to select a deployment meta architecture.  The
final meta model is then fit on all validation predictions and may be applied
to test exactly once with the selected slate policy frozen.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from training.family.evaluate_stage2_candidate_fusion import load_source
from training.family.evaluate_stage2_oof_chemistry_rescore import (
    family_length_modes,
    label_chemistry,
)
from training.family.evaluate_stage2_precursor_family_slate import precursor_family
from training.family.evaluate_stage2_score_ensemble import candidate_fingerprint
from training.family.train_stage2_lightgbm_ranker import fixed_trial_from_report
from training.family.train_stage2_listwise_ranker import balanced_row_weights
from training.family.train_stage2_matscibert_cross_encoder import (
    candidate_rank_feature_maps,
    candidate_rank_features,
)
from training.family.train_stage2_oof_candidate_stacker import (
    CandidatePriorBuilder,
    TemplatePriorBuilder,
    formula_group_folds,
)
from training.family.train_stage2_structured_energy_ranker import (
    apply_fixed_trial,
    best_trials_by_strategy,
    build_pair_features,
    evaluate_grid,
    load_matsci_pca_views,
    seed_everything,
    standardize_from_train,
    targets_from_matrix,
)
from training.family.train_stage2_within_family_variant_ranker import exact_metrics


SetKey = Tuple[int, ...]


def parse_expert_source(raw: str) -> tuple[str, str]:
    name, separator, path = str(raw).partition("=")
    if not separator or not name.strip() or not path.strip():
        raise ValueError("expert source must use name=/path/to/candidates.jsonl")
    return name.strip(), str(Path(path.strip()).resolve())


def protected_expert_union(
    base_rows: Sequence[Sequence[SetKey]],
    expert_rows: Sequence[Sequence[Sequence[SetKey]]],
    base_limit: int,
    expert_limit: int,
) -> List[List[SetKey]]:
    output: List[List[SetKey]] = []
    for row_index, base in enumerate(base_rows):
        row = list(dict.fromkeys(base))[: int(base_limit)]
        seen = set(row)
        for source in expert_rows:
            for candidate in source[int(row_index)][: int(expert_limit)]:
                if candidate not in seen:
                    seen.add(candidate)
                    row.append(candidate)
        output.append(row)
    return output


def row_subset(
    matrix: np.ndarray,
    labels: np.ndarray,
    spans: Sequence[tuple[int, int]],
    row_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.asarray(row_indices, dtype=np.int64)
    groups = np.asarray([spans[int(row)][1] - spans[int(row)][0] for row in rows], dtype=np.int32)
    positions = np.concatenate(
        [np.arange(spans[int(row)][0], spans[int(row)][1], dtype=np.int64) for row in rows]
    )
    return matrix[positions], labels[positions], groups


def row_positions(
    spans: Sequence[tuple[int, int]], row_indices: Sequence[int]
) -> np.ndarray:
    return np.concatenate(
        [
            np.arange(spans[int(row)][0], spans[int(row)][1], dtype=np.int64)
            for row in np.asarray(row_indices, dtype=np.int64)
        ]
    )


def lgb_parameters(args: argparse.Namespace, seed: int) -> dict[str, object]:
    return {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3, 5, 10],
        "lambdarank_truncation_level": 10,
        "learning_rate": float(args.learning_rate),
        "num_leaves": int(args.num_leaves),
        "min_data_in_leaf": int(args.min_data_in_leaf),
        "feature_fraction": float(args.feature_fraction),
        "bagging_fraction": float(args.bagging_fraction),
        "bagging_freq": 1,
        "lambda_l1": float(args.lambda_l1),
        "lambda_l2": float(args.lambda_l2),
        "max_bin": 255,
        "num_threads": int(args.num_threads),
        "seed": int(seed),
        "feature_fraction_seed": int(seed),
        "bagging_seed": int(seed),
        "data_random_seed": int(seed),
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }


def score_sorted_rows(
    rows: Sequence[Sequence[SetKey]],
    scores: np.ndarray,
    spans: Sequence[tuple[int, int]],
) -> List[List[SetKey]]:
    output: List[List[SetKey]] = []
    for row, (start, end) in zip(rows, spans):
        values = list(dict.fromkeys(row))
        order = sorted(
            range(len(values)),
            key=lambda index: (-float(scores[int(start) + int(index)]), int(index)),
        )
        output.append([values[index] for index in order])
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--matsci_embeddings", required=True)
    parser.add_argument("--matsci_components", type=int, default=64)
    parser.add_argument("--base_candidates", required=True)
    parser.add_argument("--expert_source", action="append", default=[])
    parser.add_argument("--base_limit", type=int, default=100)
    parser.add_argument("--expert_limit", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--num_boost_round", type=int, default=450)
    parser.add_argument("--learning_rate", type=float, default=0.03)
    parser.add_argument("--num_leaves", type=int, default=63)
    parser.add_argument("--min_data_in_leaf", type=int, default=80)
    parser.add_argument("--feature_fraction", type=float, default=0.9)
    parser.add_argument("--bagging_fraction", type=float, default=0.9)
    parser.add_argument("--lambda_l1", type=float, default=0.01)
    parser.add_argument("--lambda_l2", type=float, default=0.2)
    parser.add_argument("--group_balance_power", type=float, default=1.0)
    parser.add_argument("--family_balance_power", type=float, default=0.25)
    parser.add_argument("--base_miss_weight", type=float, default=1.0)
    parser.add_argument("--exclude_base_rank_features", action="store_true")
    parser.add_argument(
        "--train_accessible_base_misses_only",
        action="store_true",
        help="Fit fold and final meta models only on base misses whose truth is in the pool.",
    )
    parser.add_argument("--num_threads", type=int, default=64)
    parser.add_argument("--seed", type=int, default=8518)
    parser.add_argument("--evaluation_split", choices=("val", "test"), default="val")
    parser.add_argument("--resume_model", default="")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--fixed_trial_json", default="")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", required=True)
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--output_scores_npz", default="")
    args = parser.parse_args()

    if not args.expert_source:
        parser.error("at least one --expert_source is required")
    if bool(args.eval_only) and not str(args.resume_model).strip():
        parser.error("--eval_only requires --resume_model")
    if str(args.evaluation_split) == "test" and not bool(args.eval_only):
        parser.error("test is allowed only with --eval_only")
    if str(args.evaluation_split) == "test" and not str(args.fixed_trial_json).strip():
        parser.error("test requires --fixed_trial_json")
    if not bool(args.eval_only) and str(args.evaluation_split) != "val":
        parser.error("meta training is restricted to the validation split")
    if float(args.base_miss_weight) <= 0:
        parser.error("--base_miss_weight must be positive")

    try:
        import lightgbm as lgb
    except ImportError as error:
        raise RuntimeError("LightGBM is required for validation meta-ranking") from error

    seed_everything(int(args.seed))
    started = time.time()
    input_dir = Path(args.input_dir).resolve()
    split = str(args.evaluation_split)
    train_pack = np.load(input_dir / "train.npz", allow_pickle=True)
    evaluation_pack = np.load(input_dir / f"{split}.npz", allow_pickle=True)
    train_y = np.asarray(train_pack["y_multi_hot"], dtype=np.float32)
    evaluation_y = np.asarray(evaluation_pack["y_multi_hot"], dtype=np.float32)
    train_x = np.asarray(train_pack["x"], dtype=np.float32)
    evaluation_x = np.asarray(evaluation_pack["x"], dtype=np.float32)
    targets = targets_from_matrix(evaluation_y)
    train_meta = pd.read_csv(input_dir / "train_meta.csv", low_memory=False)
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
    names = [
        str(value)
        for value in json.loads((input_dir / "precursor_names.json").read_text(encoding="utf-8"))
    ]
    label_families = [precursor_family(name) for name in names]
    label_elements, label_groups, label_metals = label_chemistry(names)
    train_seen = np.asarray(train_y.sum(axis=0) > 0)
    length_modes = family_length_modes(train_meta, train_y)
    prior_builder = CandidatePriorBuilder(train_y, train_meta)
    template_builder = TemplatePriorBuilder(train_y, train_meta, names)
    evaluation_families = (
        evaluation_meta["family_signature_primary"].fillna("UNK").astype(str).to_numpy()
    )
    formula_groups = evaluation_meta["family_group_key"].fillna("UNK").astype(str).to_numpy()

    expert_specs = [parse_expert_source(value) for value in args.expert_source]
    base_rows = load_source(args.base_candidates, len(targets), int(args.base_limit))
    expert_rows = [
        load_source(path, len(targets), int(args.expert_limit)) for _, path in expert_specs
    ]
    pool_rows = protected_expert_union(
        base_rows,
        expert_rows,
        int(args.base_limit),
        int(args.expert_limit),
    )
    rank_maps = [
        *(
            []
            if bool(args.exclude_base_rank_features)
            else [candidate_rank_feature_maps(base_rows, int(args.base_limit))]
        ),
        *[
            candidate_rank_feature_maps(rows, int(args.expert_limit))
            for rows in expert_rows
        ],
    ]
    base_top10_hit = np.asarray(
        [target in set(row[:10]) for target, row in zip(targets, base_rows)], dtype=bool
    )
    hard_query_weight = np.where(
        base_top10_hit, 1.0, float(args.base_miss_weight)
    ).astype(np.float32)

    pair_rows: List[np.ndarray] = []
    query_rows: List[np.ndarray] = []
    labels: List[float] = []
    hashes: List[np.uint64] = []
    spans: List[tuple[int, int]] = []
    offset = 0
    for row_index, candidates in enumerate(pool_rows):
        family = str(evaluation_families[int(row_index)])
        row = evaluation_meta.iloc[int(row_index)]
        target = targets[int(row_index)]
        for candidate in candidates:
            pair = build_pair_features(
                candidate,
                row,
                family,
                int(length_modes.get(family, length_modes["__GLOBAL__"])),
                label_elements,
                label_groups,
                label_metals,
                train_seen,
                prior_builder,
                template_builder,
            )
            source_ranks = [
                candidate_rank_features(source_maps, row_index, candidate)
                for source_maps in rank_maps
            ]
            pair_rows.append(np.concatenate([pair, *source_ranks]).astype(np.float32))
            query_rows.append(evaluation_query[int(row_index)])
            labels.append(float(candidate == target))
            hashes.append(np.uint64(candidate_fingerprint(candidate)))
        spans.append((offset, offset + len(candidates)))
        offset += len(candidates)
    matrix = np.hstack(
        [np.asarray(pair_rows, dtype=np.float32), np.asarray(query_rows, dtype=np.float32)]
    ).astype(np.float32)
    label_array = np.asarray(labels, dtype=np.float32)
    positive_groups = int(sum(label_array[start:end].sum() > 0 for start, end in spans))
    pool_positive = np.asarray(
        [label_array[start:end].sum() > 0 for start, end in spans], dtype=bool
    )
    print(
        json.dumps(
            {
                "pairs": int(len(label_array)),
                "features": int(matrix.shape[1]),
                "positive_groups": positive_groups,
                "candidate_oracle": positive_groups / max(1, len(targets)),
                "preprocessing_minutes": (time.time() - started) / 60.0,
            }
        ),
        flush=True,
    )

    if bool(args.eval_only):
        booster = lgb.Booster(model_file=str(Path(args.resume_model).resolve()))
        raw_scores = np.asarray(booster.predict(matrix), dtype=np.float32)
        fold_reports: List[dict[str, object]] = []
    else:
        raw_scores = np.zeros(len(label_array), dtype=np.float32)
        fold_reports = []
        folds = formula_group_folds(
            formula_groups, n_splits=int(args.folds), seed=int(args.seed)
        )
        for fold_index, (train_rows, holdout_rows) in enumerate(folds):
            effective_train_rows = (
                train_rows[(~base_top10_hit[train_rows]) & pool_positive[train_rows]]
                if bool(args.train_accessible_base_misses_only)
                else train_rows
            )
            x_fold, y_fold, fold_groups = row_subset(
                matrix, label_array, spans, effective_train_rows
            )
            query_weights = balanced_row_weights(
                effective_train_rows,
                formula_groups,
                evaluation_families,
                float(args.group_balance_power),
                float(args.family_balance_power),
            ).astype(np.float32)
            query_weights *= hard_query_weight[effective_train_rows]
            sample_weights = np.repeat(query_weights, fold_groups).astype(np.float32)
            dataset = lgb.Dataset(
                x_fold,
                label=y_fold,
                group=fold_groups,
                weight=sample_weights,
            )
            booster = lgb.train(
                lgb_parameters(args, int(args.seed) + fold_index),
                dataset,
                num_boost_round=int(args.num_boost_round),
                callbacks=[lgb.log_evaluation(period=0)],
            )
            positions = row_positions(spans, holdout_rows)
            raw_scores[positions] = np.asarray(
                booster.predict(matrix[positions]), dtype=np.float32
            )
            fold_reports.append(
                {
                    "fold": int(fold_index),
                    "train_rows": int(len(effective_train_rows)),
                    "holdout_rows": int(len(holdout_rows)),
                    "train_pairs": int(len(y_fold)),
                }
            )
            print(json.dumps(fold_reports[-1]), flush=True)

        all_rows = np.arange(len(targets), dtype=np.int64)
        final_train_rows = (
            all_rows[(~base_top10_hit) & pool_positive]
            if bool(args.train_accessible_base_misses_only)
            else all_rows
        )
        final_matrix, final_labels, all_groups = row_subset(
            matrix, label_array, spans, final_train_rows
        )
        all_query_weights = balanced_row_weights(
            final_train_rows,
            formula_groups,
            evaluation_families,
            float(args.group_balance_power),
            float(args.family_balance_power),
        ).astype(np.float32)
        all_query_weights *= hard_query_weight[final_train_rows]
        final_dataset = lgb.Dataset(
            final_matrix,
            label=final_labels,
            group=all_groups,
            weight=np.repeat(all_query_weights, all_groups).astype(np.float32),
        )
        booster = lgb.train(
            lgb_parameters(args, int(args.seed)),
            final_dataset,
            num_boost_round=int(args.num_boost_round),
            callbacks=[lgb.log_evaluation(period=0)],
        )

    if str(args.fixed_trial_json).strip():
        selected, output_rows = apply_fixed_trial(
            targets,
            pool_rows,
            raw_scores,
            spans,
            label_families,
            fixed_trial_from_report(args.fixed_trial_json),
        )
        trials = [selected]
    else:
        selected, output_rows, trials = evaluate_grid(
            targets,
            pool_rows,
            raw_scores,
            spans,
            label_families,
            (0.0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8),
            (0, 1, 3, 5, 7, 9, 10),
            (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
            (20, 50, 100, 150, 200),
        )

    report = {
        "protocol": {
            "name": "validation_meta_lambdarank_formula_group_crossfit",
            "evaluation_split": split,
            "test_policy_frozen": bool(str(args.fixed_trial_json).strip()),
            "base_experts_fit_on_train": True,
            "meta_oof_folds": 0 if bool(args.eval_only) else int(args.folds),
            "rows": int(len(targets)),
            "formula_groups": int(len(set(formula_groups.tolist()))),
            "pairs": int(len(label_array)),
            "feature_dimension": int(matrix.shape[1]),
            "candidate_oracle": positive_groups / max(1, len(targets)),
            "base_top10_hits": int(base_top10_hit.sum()),
            "base_top10_misses": int((~base_top10_hit).sum()),
            "accessible_base_misses": int(
                sum(
                    (not bool(base_top10_hit[row_index]))
                    and label_array[start:end].sum() > 0
                    for row_index, (start, end) in enumerate(spans)
                )
            ),
            "matsci": matsci_metadata,
            "base_candidates": str(Path(args.base_candidates).resolve()),
            "expert_sources": [
                {"name": name, "path": path} for name, path in expert_specs
            ],
        },
        "model": {
            "num_boost_round": int(args.num_boost_round),
            "num_leaves": int(args.num_leaves),
            "learning_rate": float(args.learning_rate),
            "group_balance_power": float(args.group_balance_power),
            "family_balance_power": float(args.family_balance_power),
            "base_miss_weight": float(args.base_miss_weight),
            "exclude_base_rank_features": bool(args.exclude_base_rank_features),
            "train_accessible_base_misses_only": bool(
                args.train_accessible_base_misses_only
            ),
        },
        "folds": fold_reports,
        "base": exact_metrics(targets, pool_rows),
        "raw_score_ranking": exact_metrics(
            targets, score_sorted_rows(pool_rows, raw_scores, spans)
        ),
        "validation": {
            "best": selected,
            "best_by_strategy": best_trials_by_strategy(trials),
        },
        "elapsed_minutes": (time.time() - started) / 60.0,
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
    output_scores = (
        Path(args.output_scores_npz).resolve()
        if str(args.output_scores_npz).strip()
        else output_json.with_name("val_scores.npz")
    )
    np.savez_compressed(
        output_scores,
        raw_scores=raw_scores,
        spans=np.asarray(spans, dtype=np.int64),
        candidate_hashes=np.asarray(hashes, dtype=np.uint64),
    )
    print(json.dumps({"base": report["base"], "best": selected}, indent=2), flush=True)


if __name__ == "__main__":
    main()
