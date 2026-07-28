#!/usr/bin/env python3
"""Exploratory formula-group OOF calibration of frozen Stage2 candidate scores.

This script intentionally reports cross-fitted validation diagnostics rather
than a fixed-heldout score.  It is used to test whether a train-OOF calibrator
is worth the substantially higher cost of generating matching transformer OOF
scores on the full training split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from training.family.evaluate_stage2_candidate_fusion import load_source
from training.family.evaluate_stage2_oof_chemistry_rescore import (
    family_length_modes,
    label_chemistry,
)
from training.family.evaluate_stage2_precursor_family_slate import precursor_family
from training.family.train_stage2_oof_candidate_stacker import (
    CandidatePriorBuilder,
    TemplatePriorBuilder,
    formula_group_folds,
)
from training.family.train_stage2_structured_energy_ranker import (
    best_trials_by_strategy,
    build_pair_features,
    evaluate_grid,
    targets_from_matrix,
)
from training.family.train_stage2_within_family_variant_ranker import exact_metrics


SetKey = Tuple[int, ...]


def row_score_features(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mean = float(values.mean(dtype=np.float64))
    std = max(float(values.std(dtype=np.float64)), 1.0e-6)
    z = (values - mean) / std
    ranks = np.arange(1, len(values) + 1, dtype=np.float32)
    rank10 = float(values[min(9, len(values) - 1)])
    return np.column_stack(
        [
            values,
            z,
            values - float(values[0]),
            values - rank10,
            1.0 / np.log2(ranks + 1.0),
            np.log1p(ranks) / np.log1p(max(2, len(values))),
        ]
    ).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--base_val_candidates", required=True)
    parser.add_argument("--score_npz", required=True)
    parser.add_argument("--candidate_limit", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n_estimators", type=int, default=800)
    parser.add_argument("--num_leaves", type=int, default=31)
    parser.add_argument("--learning_rate", type=float, default=0.03)
    parser.add_argument("--min_child_samples", type=int, default=30)
    parser.add_argument("--n_jobs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", required=True)
    parser.add_argument("--output_scores_npz", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    train_pack = np.load(input_dir / "train.npz", allow_pickle=True)
    val_pack = np.load(input_dir / "val.npz", allow_pickle=True)
    train_y = np.asarray(train_pack["y_multi_hot"], dtype=np.float32)
    val_y = np.asarray(val_pack["y_multi_hot"], dtype=np.float32)
    val_x = np.asarray(val_pack["x"], dtype=np.float32)
    targets = targets_from_matrix(val_y)
    train_meta = pd.read_csv(input_dir / "train_meta.csv", low_memory=False)
    val_meta = pd.read_csv(input_dir / "val_meta.csv", low_memory=False)
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
    families = val_meta["family_signature_primary"].fillna("UNK").astype(str).to_numpy()

    base_rows = load_source(
        args.base_val_candidates, len(targets), int(args.candidate_limit)
    )
    with np.load(Path(args.score_npz).resolve(), allow_pickle=False) as pack:
        raw_scores = np.asarray(pack["raw_scores"], dtype=np.float32)
        spans = np.asarray(pack["spans"], dtype=np.int64)
    if len(spans) != len(targets):
        raise ValueError("score spans must contain one row per validation query")

    feature_rows: List[np.ndarray] = []
    label_rows: List[np.ndarray] = []
    output_spans: List[Tuple[int, int]] = []
    offset = 0
    for row_index, (target, candidates, span) in enumerate(zip(targets, base_rows, spans)):
        start, end = (int(span[0]), int(span[1]))
        values = raw_scores[start:end][: len(candidates)]
        candidates = list(candidates[: len(values)])
        family = str(families[row_index])
        score_features = row_score_features(values)
        pair = np.asarray(
            [
                build_pair_features(
                    candidate,
                    val_meta.iloc[row_index],
                    family,
                    int(length_modes.get(family, length_modes["__GLOBAL__"])),
                    label_elements,
                    label_groups,
                    label_metals,
                    train_seen,
                    prior_builder,
                    template_builder,
                )
                for candidate in candidates
            ],
            dtype=np.float32,
        )
        query = np.repeat(val_x[row_index][None, :], len(candidates), axis=0)
        features = np.concatenate([score_features, pair, query], axis=1).astype(np.float32)
        labels = np.asarray([candidate == target for candidate in candidates], dtype=np.int8)
        feature_rows.append(features)
        label_rows.append(labels)
        output_spans.append((offset, offset + len(candidates)))
        offset += len(candidates)

    groups = val_meta["family_group_key"].fillna("UNK").astype(str).to_numpy()
    splits = formula_group_folds(groups, int(args.folds), int(args.seed))
    oof_scores = np.zeros(offset, dtype=np.float32)
    fold_reports = []
    for fold, (train_indices, query_indices) in enumerate(splits):
        train_indices = [int(index) for index in train_indices if label_rows[int(index)].any()]
        matrix = np.vstack([feature_rows[index] for index in train_indices])
        labels = np.concatenate([label_rows[index] for index in train_indices])
        train_groups = [len(feature_rows[index]) for index in train_indices]
        model = lgb.LGBMRanker(
            objective="lambdarank",
            label_gain=[0, 1],
            lambdarank_truncation_level=20,
            n_estimators=int(args.n_estimators),
            learning_rate=float(args.learning_rate),
            num_leaves=int(args.num_leaves),
            min_child_samples=int(args.min_child_samples),
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=5.0,
            random_state=int(args.seed) + fold * 1009,
            n_jobs=int(args.n_jobs),
            verbosity=-1,
        )
        model.fit(matrix, labels, group=train_groups)
        query_indices = [int(row_index) for row_index in query_indices]
        query_matrix = np.vstack([feature_rows[row_index] for row_index in query_indices])
        query_scores = np.asarray(model.predict(query_matrix), dtype=np.float32)
        query_offset = 0
        for row_index in query_indices:
            start, end = output_spans[row_index]
            size = int(end - start)
            oof_scores[start:end] = query_scores[query_offset : query_offset + size]
            query_offset += size
        fold_reports.append(
            {
                "fold": int(fold),
                "train_rows": int(len(train_indices)),
                "query_rows": int(len(query_indices)),
            }
        )

    alpha_grid = (0.0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4)
    protected_grid = (0, 1, 3, 5, 7, 9, 10)
    minimum_gain_grid = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    best, ranked, trials = evaluate_grid(
        targets,
        base_rows,
        oof_scores,
        output_spans,
        label_families,
        alpha_grid,
        protected_grid,
        minimum_gain_grid,
    )
    report = {
        "protocol": "exploratory_val_formula_group_oof_score_calibration_not_fixed_heldout",
        "config": vars(args),
        "rows": int(len(targets)),
        "feature_dim": int(feature_rows[0].shape[1]),
        "base": exact_metrics(targets, base_rows),
        "candidate_oracle": float(
            np.mean([target in set(row) for target, row in zip(targets, base_rows)])
        ),
        "best": best,
        "best_by_strategy": best_trials_by_strategy(trials),
        "folds": fold_reports,
    }
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_candidates = Path(args.output_candidates_jsonl).resolve()
    output_candidates.parent.mkdir(parents=True, exist_ok=True)
    with output_candidates.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(ranked):
            handle.write(
                json.dumps(
                    {"row_index": row_index, "candidate_label_ids": [list(value) for value in row]}
                )
                + "\n"
            )
    output_scores = Path(args.output_scores_npz).resolve()
    output_scores.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_scores,
        raw_scores=oof_scores,
        spans=np.asarray(output_spans, dtype=np.int64),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
