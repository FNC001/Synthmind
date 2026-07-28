#!/usr/bin/env python3
"""Train a formula-group-disjoint LambdaRank precursor-set selector.

The ranker learns from OOF candidate slates only.  Validation chooses the
boosting iteration and a global slate policy; a final test run is permitted
only in evaluation-only mode with that validation policy frozen.
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
from training.family.train_stage2_matscibert_cross_encoder import (
    candidate_rank_feature_maps,
    candidate_rank_features,
)
from training.family.train_stage2_listwise_ranker import balanced_row_weights
from training.family.train_stage2_oof_candidate_stacker import (
    CandidatePriorBuilder,
    TemplatePriorBuilder,
)
from training.family.train_stage2_structured_energy_ranker import (
    CandidateRegistry,
    apply_fixed_trial,
    best_trials_by_strategy,
    build_pair_features,
    build_training_pools,
    build_validation_pairs,
    evaluate_grid,
    load_matsci_pca_views,
    merge_candidate_sources,
    seed_everything,
    standardize_from_train,
    targets_from_matrix,
)
from training.family.train_stage2_within_family_variant_ranker import exact_metrics


SetKey = Tuple[int, ...]


def flatten_training_pool(
    pair_features: np.ndarray,
    mask: np.ndarray,
    query_indices: np.ndarray,
    query_features: np.ndarray,
    include_query_features: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten padded query pools while preserving LightGBM group order."""

    groups = np.asarray(mask.sum(axis=1), dtype=np.int32)
    if np.any(groups < 2):
        raise ValueError("every ranking group must contain a positive and a negative")
    pair_flat = np.asarray(pair_features[mask], dtype=np.float32)
    labels = np.concatenate(
        [np.r_[np.float32(1.0), np.zeros(int(size) - 1, dtype=np.float32)] for size in groups]
    )
    if not include_query_features:
        return pair_flat, labels, groups
    query_flat = np.repeat(
        np.asarray(query_features[query_indices], dtype=np.float32), groups, axis=0
    )
    return np.hstack([pair_flat, query_flat]).astype(np.float32), labels, groups


def flatten_evaluation_pairs(
    pair_features: np.ndarray,
    query_indices: np.ndarray,
    query_features: np.ndarray,
    include_query_features: bool,
) -> np.ndarray:
    pair = np.asarray(pair_features, dtype=np.float32)
    if not include_query_features:
        return pair
    return np.hstack(
        [pair, np.asarray(query_features[query_indices], dtype=np.float32)]
    ).astype(np.float32)


def fixed_trial_from_report(path: str) -> dict[str, object]:
    report = json.loads(Path(path).resolve().read_text(encoding="utf-8"))
    trial = report.get("validation", {}).get("best")
    if not isinstance(trial, dict):
        trial = report.get("best")
    if not isinstance(trial, dict):
        raise ValueError("fixed trial report must contain validation.best or best")
    return trial


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--matsci_embeddings", required=True)
    parser.add_argument("--matsci_components", type=int, default=64)
    parser.add_argument("--base_val_candidates", required=True)
    parser.add_argument("--train_candidate_source", action="append", default=[])
    parser.add_argument("--train_aux_rank_source", action="append", default=[])
    parser.add_argument("--val_aux_rank_source", action="append", default=[])
    parser.add_argument("--candidate_limit", type=int, default=800)
    parser.add_argument("--source_union_limit", type=int, default=128)
    parser.add_argument("--train_pool_limit", type=int, default=128)
    parser.add_argument("--cross_family_negatives", type=int, default=32)
    parser.add_argument("--aux_rank_limit", type=int, default=100)
    parser.add_argument("--exclude_query_features", action="store_true")
    parser.add_argument("--num_boost_round", type=int, default=1600)
    parser.add_argument("--early_stopping_rounds", type=int, default=120)
    parser.add_argument("--learning_rate", type=float, default=0.025)
    parser.add_argument("--num_leaves", type=int, default=127)
    parser.add_argument("--min_data_in_leaf", type=int, default=100)
    parser.add_argument("--feature_fraction", type=float, default=0.85)
    parser.add_argument("--bagging_fraction", type=float, default=0.85)
    parser.add_argument("--lambda_l1", type=float, default=0.01)
    parser.add_argument("--lambda_l2", type=float, default=0.1)
    parser.add_argument("--group_balance_power", type=float, default=0.0)
    parser.add_argument("--family_balance_power", type=float, default=0.0)
    parser.add_argument("--num_threads", type=int, default=64)
    parser.add_argument("--seed", type=int, default=8516)
    parser.add_argument("--evaluation_split", choices=("val", "test"), default="val")
    parser.add_argument("--resume_model", default="")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--fixed_trial_json", default="")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", required=True)
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--output_scores_npz", default="")
    args = parser.parse_args()

    if not args.train_candidate_source:
        parser.error("at least one --train_candidate_source is required")
    if len(args.train_aux_rank_source) != len(args.val_aux_rank_source):
        parser.error("train and evaluation auxiliary rank sources must have equal counts")
    if bool(args.eval_only) and not str(args.resume_model).strip():
        parser.error("--eval_only requires --resume_model")
    if str(args.evaluation_split) == "test" and not bool(args.eval_only):
        parser.error("test evaluation is allowed only with --eval_only")
    if str(args.evaluation_split) == "test" and not str(args.fixed_trial_json).strip():
        parser.error("test evaluation requires --fixed_trial_json")

    try:
        import lightgbm as lgb
    except ImportError as error:
        raise RuntimeError("LightGBM is required for this ranker") from error

    seed_everything(int(args.seed))
    started = time.time()
    input_dir = Path(args.input_dir).resolve()
    split = str(args.evaluation_split)
    train_pack = np.load(input_dir / "train.npz", allow_pickle=True)
    eval_pack = np.load(input_dir / f"{split}.npz", allow_pickle=True)
    train_y = np.asarray(train_pack["y_multi_hot"], dtype=np.float32)
    eval_y = np.asarray(eval_pack["y_multi_hot"], dtype=np.float32)
    train_x = np.asarray(train_pack["x"], dtype=np.float32)
    eval_x = np.asarray(eval_pack["x"], dtype=np.float32)
    train_targets = targets_from_matrix(train_y)
    eval_targets = targets_from_matrix(eval_y)
    train_meta = pd.read_csv(input_dir / "train_meta.csv", low_memory=False)
    eval_meta = pd.read_csv(input_dir / f"{split}_meta.csv", low_memory=False)
    _, train_matsci, eval_matsci, matsci_metadata = load_matsci_pca_views(
        Path(args.matsci_embeddings).resolve(),
        int(args.matsci_components),
        int(args.seed),
        split,
    )
    train_query, eval_query, _, _ = standardize_from_train(
        np.hstack([train_x, train_matsci]).astype(np.float32),
        np.hstack([eval_x, eval_matsci]).astype(np.float32),
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
    train_families = train_meta["family_signature_primary"].fillna("UNK").astype(str).to_numpy()
    eval_families = eval_meta["family_signature_primary"].fillna("UNK").astype(str).to_numpy()

    train_source_paths = [str(Path(path).resolve()) for path in args.train_candidate_source]
    train_sources = [
        load_source(path, len(train_targets), int(args.source_union_limit))
        for path in train_source_paths
    ]
    train_source_cache = dict(zip(train_source_paths, train_sources))
    base_rows = load_source(args.base_val_candidates, len(eval_targets), int(args.candidate_limit))
    train_merged_rows = [
        merge_candidate_sources(train_sources, row_index, int(args.source_union_limit))
        for row_index in range(len(train_targets))
    ]
    train_rank_maps = candidate_rank_feature_maps(
        train_merged_rows, int(args.source_union_limit)
    )
    eval_rank_maps = candidate_rank_feature_maps(base_rows, int(args.candidate_limit))
    train_aux_rank_maps: List[List[Dict[SetKey, np.ndarray]]] = []
    eval_aux_rank_maps: List[List[Dict[SetKey, np.ndarray]]] = []
    for train_path, eval_path in zip(args.train_aux_rank_source, args.val_aux_rank_source):
        resolved_train_path = str(Path(train_path).resolve())
        cached = train_source_cache.get(resolved_train_path)
        train_rows = (
            [row[: int(args.aux_rank_limit)] for row in cached]
            if cached is not None
            else load_source(train_path, len(train_targets), int(args.aux_rank_limit))
        )
        eval_rows = load_source(eval_path, len(eval_targets), int(args.aux_rank_limit))
        train_aux_rank_maps.append(
            candidate_rank_feature_maps(train_rows, int(args.aux_rank_limit))
        )
        eval_aux_rank_maps.append(
            candidate_rank_feature_maps(eval_rows, int(args.aux_rank_limit))
        )

    def pair_features(
        row_index: int,
        candidate: SetKey,
        meta: pd.DataFrame,
        families: np.ndarray,
        rank_maps: Sequence[Dict[SetKey, np.ndarray]],
        aux_maps: Sequence[Sequence[Dict[SetKey, np.ndarray]]],
    ) -> np.ndarray:
        family = str(families[int(row_index)])
        base = build_pair_features(
            candidate,
            meta.iloc[int(row_index)],
            family,
            int(length_modes.get(family, length_modes["__GLOBAL__"])),
            label_elements,
            label_groups,
            label_metals,
            train_seen,
            prior_builder,
            template_builder,
        )
        return np.concatenate(
            [
                base,
                candidate_rank_features(rank_maps, row_index, candidate),
                *[
                    candidate_rank_features(source_maps, row_index, candidate)
                    for source_maps in aux_maps
                ],
            ]
        ).astype(np.float32)

    registry = CandidateRegistry()
    training_pool = build_training_pools(
        train_targets,
        train_meta,
        train_sources,
        label_families,
        int(args.train_pool_limit),
        int(args.source_union_limit),
        registry,
        lambda row_index, candidate: pair_features(
            row_index,
            candidate,
            train_meta,
            train_families,
            train_rank_maps,
            train_aux_rank_maps,
        ),
        int(args.seed),
        int(args.cross_family_negatives),
    )
    eval_query_indices, eval_candidate_ids, eval_pair, spans = build_validation_pairs(
        base_rows,
        eval_meta,
        registry,
        lambda row_index, candidate: pair_features(
            row_index,
            candidate,
            eval_meta,
            eval_families,
            eval_rank_maps,
            eval_aux_rank_maps,
        ),
    )
    include_query = not bool(args.exclude_query_features)
    x_train, y_train, train_groups = flatten_training_pool(
        training_pool.pair_features,
        training_pool.mask,
        training_pool.query_indices,
        train_query,
        include_query,
    )
    query_weights = balanced_row_weights(
        training_pool.query_indices,
        train_meta["family_group_key"].fillna("UNK").astype(str).to_numpy(),
        train_families,
        float(args.group_balance_power),
        float(args.family_balance_power),
    ).astype(np.float32)
    train_weights = np.repeat(query_weights, train_groups).astype(np.float32)
    x_eval = flatten_evaluation_pairs(
        eval_pair, eval_query_indices, eval_query, include_query
    )
    y_eval = np.asarray(
        [
            float(registry.keys[int(candidate_id)] == eval_targets[int(row_index)])
            for row_index, candidate_id in zip(eval_query_indices, eval_candidate_ids)
        ],
        dtype=np.float32,
    )
    eval_groups = np.asarray([end - start for start, end in spans], dtype=np.int32)
    print(
        json.dumps(
            {
                "train_pairs": int(len(y_train)),
                "evaluation_pairs": int(len(y_eval)),
                "features": int(x_train.shape[1]),
                "positive_evaluation_groups": int(
                    sum(y_eval[start:end].sum() > 0 for start, end in spans)
                ),
                "preprocessing_minutes": (time.time() - started) / 60.0,
            }
        ),
        flush=True,
    )

    if bool(args.eval_only):
        booster = lgb.Booster(model_file=str(Path(args.resume_model).resolve()))
        best_iteration = int(booster.current_iteration())
    else:
        params = {
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
            "seed": int(args.seed),
            "feature_fraction_seed": int(args.seed),
            "bagging_seed": int(args.seed),
            "data_random_seed": int(args.seed),
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
        }
        train_data = lgb.Dataset(
            x_train, label=y_train, group=train_groups, weight=train_weights
        )
        eval_data = lgb.Dataset(x_eval, label=y_eval, group=eval_groups, reference=train_data)
        callbacks = [lgb.log_evaluation(period=25)]
        if int(args.early_stopping_rounds) > 0:
            callbacks.append(
                lgb.early_stopping(int(args.early_stopping_rounds), first_metric_only=False)
            )
        booster = lgb.train(
            params,
            train_data,
            num_boost_round=int(args.num_boost_round),
            valid_sets=[eval_data],
            valid_names=[split],
            callbacks=callbacks,
        )
        best_iteration = int(booster.best_iteration or booster.current_iteration())

    raw_scores = np.asarray(
        booster.predict(x_eval, num_iteration=best_iteration), dtype=np.float32
    )
    if str(args.fixed_trial_json).strip():
        selected, output_rows = apply_fixed_trial(
            eval_targets,
            base_rows,
            raw_scores,
            spans,
            label_families,
            fixed_trial_from_report(args.fixed_trial_json),
        )
        trials = [selected]
    else:
        selected, output_rows, trials = evaluate_grid(
            eval_targets,
            base_rows,
            raw_scores,
            spans,
            label_families,
            (0.0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8),
            (0, 1, 3, 5, 7, 9, 10),
            (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
            (20, 50, 100, 200, 400),
        )

    report = {
        "protocol": {
            "name": "stage2_lightgbm_lambdarank_formula_group_disjoint",
            "evaluation_split": split,
            "test_policy_frozen": bool(str(args.fixed_trial_json).strip()),
            "train_rows": int(len(train_targets)),
            "evaluation_rows": int(len(eval_targets)),
            "train_pairs": int(len(y_train)),
            "evaluation_pairs": int(len(y_eval)),
            "feature_dimension": int(x_train.shape[1]),
            "include_query_features": bool(include_query),
            "group_balance_power": float(args.group_balance_power),
            "family_balance_power": float(args.family_balance_power),
            "matsci": matsci_metadata,
            "candidate_sources": train_source_paths,
            "auxiliary_rank_sources": [
                {"train": str(a), "evaluation": str(b)}
                for a, b in zip(args.train_aux_rank_source, args.val_aux_rank_source)
            ],
        },
        "model": {
            "best_iteration": int(best_iteration),
            "num_leaves": int(args.num_leaves),
            "learning_rate": float(args.learning_rate),
        },
        "base": exact_metrics(eval_targets, base_rows),
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
    booster.save_model(str(output_model), num_iteration=best_iteration)
    output_scores = (
        Path(args.output_scores_npz).resolve()
        if str(args.output_scores_npz).strip()
        else output_json.with_name("val_scores.npz")
    )
    np.savez_compressed(
        output_scores,
        raw_scores=raw_scores,
        spans=np.asarray(spans, dtype=np.int64),
        row_indices=eval_query_indices.astype(np.int64),
        candidate_hashes=np.asarray(
            [candidate_fingerprint(registry.keys[int(value)]) for value in eval_candidate_ids],
            dtype=np.uint64,
        ),
    )
    print(json.dumps({"base": report["base"], "best": selected}, indent=2), flush=True)


if __name__ == "__main__":
    main()
