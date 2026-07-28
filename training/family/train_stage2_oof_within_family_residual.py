#!/usr/bin/env python3
"""Formula-group OOF residual ranker for exact variants within base families."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from training.family.evaluate_stage2_candidate_fusion import load_source
from training.family.evaluate_stage2_oof_chemistry_rescore import (
    family_length_modes,
    json_set,
    label_chemistry,
)
from training.family.evaluate_stage2_precursor_family_slate import family_key, precursor_family
from training.family.train_stage2_oof_candidate_stacker import (
    CandidatePriorBuilder,
    MatSciFeatureBuilder,
    TemplatePriorBuilder,
    formula_group_folds,
    load_matsci_views,
)
from training.family.train_stage2_within_family_variant_ranker import (
    build_features,
    exact_metrics,
    family_slot_rerank,
    targets_from_matrix,
)


SetKey = Tuple[int, ...]


def rank_features(candidates: Sequence[SetKey], label_families: Sequence[str]) -> np.ndarray:
    keys = [family_key(candidate, label_families) for candidate in candidates]
    first_rank: Dict[Tuple[str, ...], int] = {}
    local_rank: Dict[Tuple[str, ...], int] = {}
    top10_counts: Dict[Tuple[str, ...], int] = {}
    for rank, key in enumerate(keys, start=1):
        first_rank.setdefault(key, rank)
        if rank <= 10:
            top10_counts[key] = top10_counts.get(key, 0) + 1
    output = []
    for rank, key in enumerate(keys, start=1):
        within = local_rank.get(key, 0) + 1
        local_rank[key] = within
        output.append(
            [
                1.0 / math.log2(rank + 2.0),
                float(rank <= 1),
                float(rank <= 3),
                float(rank <= 5),
                float(rank <= 10),
                float(rank <= 20),
                float(rank <= 50),
                1.0 / math.log2(first_rank[key] + 2.0),
                1.0 / math.log2(within + 2.0),
                min(top10_counts.get(key, 0), 10) / 10.0,
                min(len(key), 10) / 10.0,
            ]
        )
    return np.asarray(output, dtype=np.float32)


def matrix_for_rows(
    feature_rows: Sequence[np.ndarray],
    label_rows: Sequence[np.ndarray],
    indices: Sequence[int],
    require_positive: bool,
) -> tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    matrices = []
    labels = []
    groups = []
    kept = []
    for row_index in indices:
        row_labels = label_rows[int(row_index)]
        if require_positive and not bool(np.any(row_labels > 0)):
            continue
        matrices.append(feature_rows[int(row_index)])
        labels.append(row_labels)
        groups.append(len(row_labels))
        kept.append(int(row_index))
    return np.vstack(matrices), np.concatenate(labels), groups, kept


def fit_model(
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: Sequence[int],
    seed: int,
    args: argparse.Namespace,
) -> lgb.LGBMClassifier:
    weights = np.ones(len(labels), dtype=np.float32)
    offset = 0
    for size in groups:
        local = labels[offset : offset + int(size)]
        positive = np.flatnonzero(local > 0) + offset
        weights[positive] *= max(1, int(size) - 1) ** float(args.positive_weight_power)
        offset += int(size)
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        num_leaves=int(args.num_leaves),
        min_child_samples=int(args.min_child_samples),
        colsample_bytree=0.9,
        reg_lambda=2.0,
        random_state=int(seed),
        n_jobs=int(args.n_jobs),
        verbosity=-1,
    )
    model.fit(matrix, labels, sample_weight=weights)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--base_candidates", required=True)
    parser.add_argument("--matsci_embeddings", default="")
    parser.add_argument("--candidate_limit", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n_estimators", type=int, default=1600)
    parser.add_argument("--num_leaves", type=int, default=63)
    parser.add_argument("--learning_rate", type=float, default=0.02)
    parser.add_argument("--min_child_samples", type=int, default=30)
    parser.add_argument("--positive_weight_power", type=float, default=0.75)
    parser.add_argument("--matsci_components", type=int, default=32)
    parser.add_argument("--matsci_ridge_alpha", type=float, default=10.0)
    parser.add_argument("--n_jobs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", required=True)
    parser.add_argument("--output_model", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    train_y = np.asarray(
        np.load(input_dir / "train.npz", allow_pickle=True)["y_multi_hot"], dtype=np.float32
    )
    val_y = np.asarray(
        np.load(input_dir / "val.npz", allow_pickle=True)["y_multi_hot"], dtype=np.float32
    )
    targets = targets_from_matrix(val_y)
    train_meta = pd.read_csv(input_dir / "train_meta.csv", low_memory=False)
    meta = pd.read_csv(input_dir / "val_meta.csv", low_memory=False)
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

    matsci_builder = None
    val_direct = val_projected = None
    if str(args.matsci_embeddings).strip():
        label_views, train_query_views, val_query_views = load_matsci_views(
            Path(args.matsci_embeddings).resolve(), input_dir, "val", names
        )
        matsci_builder = MatSciFeatureBuilder(
            label_views,
            train_query_views,
            train_y,
            int(args.matsci_components),
            float(args.matsci_ridge_alpha),
            int(args.seed),
        )
        val_direct, val_projected = matsci_builder.transform_queries(val_query_views)

    candidates = load_source(args.base_candidates, len(targets), int(args.candidate_limit))
    families = meta["family_signature_primary"].fillna("UNK").astype(str).to_numpy()
    feature_rows = []
    label_rows = []
    for row_index, (row, target) in enumerate(zip(candidates, targets)):
        family = str(families[row_index])
        matrix = build_features(
            row,
            json_set(meta.iloc[row_index]["target_cation_elements"]),
            json_set(meta.iloc[row_index]["target_anion_elements"]),
            family,
            int(length_modes.get(family, length_modes["__GLOBAL__"])),
            label_elements,
            label_groups,
            label_metals,
            train_seen,
            prior_builder,
            template_builder,
            matsci_builder,
            None if val_direct is None else val_direct[row_index],
            None if val_projected is None else val_projected[row_index],
        )
        feature_rows.append(np.concatenate([matrix, rank_features(row, label_families)], axis=1))
        label_rows.append(np.asarray([candidate == target for candidate in row], dtype=np.int8))

    groups = meta["family_group_key"].fillna("UNK").astype(str).to_numpy()
    splits = formula_group_folds(groups, int(args.folds), int(args.seed))
    score_rows = [np.zeros(len(row), dtype=np.float32) for row in candidates]
    fold_reports = []
    for fold, (train_indices, query_indices) in enumerate(splits):
        train_matrix, train_labels, train_groups, kept = matrix_for_rows(
            feature_rows, label_rows, train_indices, require_positive=True
        )
        model = fit_model(
            train_matrix,
            train_labels,
            train_groups,
            int(args.seed) + fold * 1009,
            args,
        )
        query_matrix, _, query_groups, query_kept = matrix_for_rows(
            feature_rows, label_rows, query_indices, require_positive=False
        )
        query_scores = model.predict_proba(query_matrix)[:, 1]
        offset = 0
        for row_index, size in zip(query_kept, query_groups):
            score_rows[int(row_index)] = query_scores[offset : offset + int(size)]
            offset += int(size)
        raw_rows = [
            [candidate for _, candidate in sorted(zip(score_rows[int(row)], candidates[int(row)]), reverse=True)]
            for row in query_indices
        ]
        fold_reports.append(
            {
                "fold": int(fold),
                "train_rows_with_positive": int(len(kept)),
                "query_rows": int(len(query_indices)),
                **exact_metrics([targets[int(row)] for row in query_indices], raw_rows),
            }
        )

    trials = []
    best = None
    best_rows = []
    for protected_prefix in range(0, 11):
        ranked = [
            family_slot_rerank(row, scores, label_families, 10, protected_prefix)
            for row, scores in zip(candidates, score_rows)
        ]
        current = {"protected_prefix": protected_prefix, **exact_metrics(targets, ranked)}
        trials.append(current)
        key = (current["exact_hit@10"], current["exact_hit@5"], current["exact_hit@1"])
        if best is None or key > best[0]:
            best = (key, current)
            best_rows = ranked
    assert best is not None

    full_matrix, full_labels, full_groups, kept = matrix_for_rows(
        feature_rows, label_rows, np.arange(len(targets)), require_positive=True
    )
    full_model = fit_model(full_matrix, full_labels, full_groups, int(args.seed) + 100000, args)
    report = {
        "protocol": "val_formula_group_disjoint_oof_within_family_residual_ranking",
        "config": vars(args),
        "validation": {
            "rows": len(targets),
            "rows_with_target_in_candidate_pool": int(len(kept)),
            "base": exact_metrics(targets, candidates),
            "best": best[1],
            "trials": trials,
            "folds": fold_reports,
        },
        "feature_dim": int(feature_rows[0].shape[1]),
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    candidate_output = Path(args.output_candidates_jsonl).resolve()
    candidate_output.parent.mkdir(parents=True, exist_ok=True)
    with candidate_output.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(best_rows):
            handle.write(
                json.dumps(
                    {"row_index": row_index, "candidate_label_ids": [list(value) for value in row]}
                )
                + "\n"
            )
    model_output = Path(args.output_model).resolve()
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": full_model,
            "prior_builder": prior_builder,
            "template_builder": template_builder,
            "matsci_builder": matsci_builder,
            "protected_prefix": int(best[1]["protected_prefix"]),
            "label_families": label_families,
            "length_modes": length_modes,
        },
        model_output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
