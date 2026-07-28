#!/usr/bin/env python3
"""Formula-group OOF selector for a protected Top-9 plus one learned safe slot.

The first nine candidates of a trusted base ranking are immutable.  The model
chooses only the tenth candidate from the union of the base tenth candidate
and complementary frozen experts.  This turns broad candidate fusion into a
low-risk residual decision and makes every possible loss/gain at Top-10
explicit in the validation report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
from training.family.train_stage2_oof_candidate_stacker import (
    CandidatePriorBuilder,
    MatSciFeatureBuilder,
    TemplatePriorBuilder,
    append_matsci_features,
    build_row_candidates_and_features,
    formula_group_folds,
    load_matsci_views,
    parse_named_source,
)
from training.family.train_stage2_within_family_variant_ranker import (
    exact_metrics,
    targets_from_matrix,
)


SetKey = Tuple[int, ...]


def safe_slot_merge(
    base: Sequence[SetKey], slot_candidate: SetKey, union: Sequence[SetKey], protected: int = 9
) -> List[SetKey]:
    """Preserve a base prefix and deterministically append de-duplicated fallbacks."""
    output: List[SetKey] = []
    seen: set[SetKey] = set()
    for candidate in [*base[: int(protected)], slot_candidate, *base[int(protected) + 1 :], *union]:
        if candidate and candidate not in seen:
            output.append(candidate)
            seen.add(candidate)
    return output


def matrix_for_rows(
    feature_rows: Sequence[np.ndarray],
    label_rows: Sequence[np.ndarray],
    indices: Sequence[int],
    require_positive: bool,
) -> tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    matrices: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    groups: List[int] = []
    kept: List[int] = []
    for row_index in indices:
        current_labels = label_rows[int(row_index)]
        if require_positive and not bool(np.any(current_labels > 0)):
            continue
        matrices.append(feature_rows[int(row_index)])
        labels.append(current_labels)
        groups.append(int(len(current_labels)))
        kept.append(int(row_index))
    if not matrices:
        feature_dim = int(feature_rows[0].shape[1]) if feature_rows else 0
        return np.zeros((0, feature_dim), dtype=np.float32), np.zeros(0), [], []
    return np.vstack(matrices), np.concatenate(labels), groups, kept


def fit_model(
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: Sequence[int],
    seed: int,
    args: argparse.Namespace,
    row_weights: Sequence[float] | None = None,
) -> Any:
    weights = np.ones(len(labels), dtype=np.float32)
    offset = 0
    if row_weights is None:
        row_weights = np.ones(len(groups), dtype=np.float32)
    if len(row_weights) != len(groups):
        raise ValueError("row_weights must align with query groups")
    for size, row_weight in zip(groups, row_weights):
        local = labels[offset : offset + int(size)]
        positive = np.flatnonzero(local > 0) + offset
        weights[offset : offset + int(size)] *= float(row_weight)
        weights[positive] *= max(1, int(size) - 1) ** float(args.positive_weight_power)
        offset += int(size)
    common = dict(
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        num_leaves=int(args.num_leaves),
        min_child_samples=int(args.min_child_samples),
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=3.0,
        random_state=int(seed),
        n_jobs=int(args.n_jobs),
        verbosity=-1,
    )
    if str(args.model_objective) == "lambdarank":
        model = lgb.LGBMRanker(
            objective="lambdarank",
            label_gain=[0, 1],
            lambdarank_truncation_level=20,
            **common,
        )
        model.fit(matrix, labels, group=list(groups), sample_weight=weights)
    else:
        model = lgb.LGBMClassifier(objective="binary", **common)
        model.fit(matrix, labels, sample_weight=weights)
    return model


def predict_scores(model: Any, matrix: np.ndarray, objective: str) -> np.ndarray:
    if str(objective) == "lambdarank":
        return np.asarray(model.predict(matrix), dtype=np.float32)
    return np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float32)


def slot_rows(
    base_rows: Sequence[Sequence[SetKey]],
    candidates: Sequence[Sequence[SetKey]],
    scores: Sequence[np.ndarray],
    margin: float,
    protected: int,
) -> tuple[List[List[SetKey]], np.ndarray]:
    output: List[List[SetKey]] = []
    switched = np.zeros(len(base_rows), dtype=bool)
    for row_index, (base, union, row_scores) in enumerate(zip(base_rows, candidates, scores)):
        prefix = set(base[: int(protected)])
        base_slot = next((value for value in base[int(protected) :] if value not in prefix), ())
        score_map = {candidate: float(score) for candidate, score in zip(union, row_scores)}
        available = [candidate for candidate in union if candidate not in prefix]
        best = max(available, key=lambda value: (score_map[value], value)) if available else base_slot
        base_score = score_map.get(base_slot, 0.0)
        if not best or score_map.get(best, 0.0) < base_score + float(margin):
            best = base_slot
        switched[row_index] = bool(best and best != base_slot)
        output.append(safe_slot_merge(base, best, union, int(protected)))
    return output, switched


def multi_slot_rows(
    base_rows: Sequence[Sequence[SetKey]],
    candidates: Sequence[Sequence[SetKey]],
    scores: Sequence[np.ndarray],
    margin: float,
    protected: int,
) -> tuple[List[List[SetKey]], np.ndarray]:
    """Replace weak unprotected Top-10 entries with confident union outsiders."""
    output: List[List[SetKey]] = []
    switched = np.zeros(len(base_rows), dtype=bool)
    for row_index, (base, union, row_scores) in enumerate(zip(base_rows, candidates, scores)):
        unique_base = list(dict.fromkeys(base))
        slate_size = min(10, len(unique_base))
        selected = list(unique_base[:slate_size])
        score_map = {candidate: float(score) for candidate, score in zip(union, row_scores)}
        replaceable = list(range(min(int(protected), slate_size), slate_size))
        replaceable.sort(
            key=lambda index: (score_map.get(selected[index], -np.inf), -index)
        )
        selected_set = set(selected)
        outsiders = [candidate for candidate in union if candidate not in selected_set]
        outsiders.sort(key=lambda candidate: (-score_map[candidate], candidate))
        for outsider in outsiders:
            if not replaceable:
                break
            base_index = replaceable[0]
            base_score = score_map.get(selected[base_index], -np.inf)
            if score_map[outsider] < base_score + float(margin):
                break
            selected_set.discard(selected[base_index])
            selected[base_index] = outsider
            selected_set.add(outsider)
            replaceable.pop(0)
        switched[row_index] = selected != unique_base[:slate_size]
        seen = set(selected)
        merged = list(selected)
        for candidate in [*unique_base, *union]:
            if candidate and candidate not in seen:
                merged.append(candidate)
                seen.add(candidate)
        output.append(merged)
    return output, switched


def slice_metrics(
    targets: Sequence[SetKey], rows: Sequence[Sequence[SetKey]], mask: np.ndarray
) -> Dict[str, float]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return {"rows": 0, "exact_hit@10": float("nan")}
    return {
        "rows": int(len(indices)),
        "exact_hit@10": float(
            np.mean([targets[int(index)] in set(rows[int(index)][:10]) for index in indices])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--base_candidates", required=True)
    parser.add_argument("--expert", action="append", default=[], help="Repeat NAME=PATH")
    parser.add_argument("--aux_base_candidates", default="")
    parser.add_argument(
        "--aux_expert",
        action="append",
        default=[],
        help="Repeat NAME=train-OOF-PATH; requires --source_agnostic",
    )
    parser.add_argument("--aux_weight", type=float, default=0.35)
    parser.add_argument("--source_agnostic", action="store_true")
    parser.add_argument(
        "--training_mode",
        choices=("validation_oof", "aux_only"),
        default="validation_oof",
        help="aux_only fits exclusively on frozen train-OOF candidate sources.",
    )
    parser.add_argument(
        "--selection_strategy",
        choices=("single_slot", "multi_slot"),
        default="single_slot",
    )
    parser.add_argument(
        "--model_objective", choices=("binary", "lambdarank"), default="binary"
    )
    parser.add_argument("--expert_limit", type=int, default=10)
    parser.add_argument("--base_output_limit", type=int, default=100)
    parser.add_argument("--union_limit", type=int, default=160)
    parser.add_argument("--protected_prefix", type=int, default=9)
    parser.add_argument("--matsci_embeddings", default="")
    parser.add_argument("--matsci_components", type=int, default=32)
    parser.add_argument("--matsci_ridge_alpha", type=float, default=10.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n_estimators", type=int, default=1800)
    parser.add_argument("--num_leaves", type=int, default=63)
    parser.add_argument("--learning_rate", type=float, default=0.02)
    parser.add_argument("--min_child_samples", type=int, default=25)
    parser.add_argument("--positive_weight_power", type=float, default=0.9)
    parser.add_argument("--n_jobs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", required=True)
    parser.add_argument("--output_model", required=True)
    args = parser.parse_args()
    if not args.expert:
        parser.error("at least one complementary --expert is required")
    if bool(str(args.aux_base_candidates).strip()) != bool(args.aux_expert):
        parser.error("--aux_base_candidates and at least one --aux_expert must be used together")
    if args.aux_expert and not bool(args.source_agnostic):
        parser.error("auxiliary sources require --source_agnostic so train/val sources may differ")
    if str(args.training_mode) == "aux_only" and not str(args.aux_base_candidates).strip():
        parser.error("--training_mode aux_only requires auxiliary train-OOF sources")
    if str(args.selection_strategy) == "single_slot" and int(args.protected_prefix) != 9:
        parser.error("single_slot requires --protected_prefix 9")
    if not 0 <= int(args.protected_prefix) <= 9:
        parser.error("--protected_prefix must be between 0 and 9")

    input_dir = Path(args.input_dir).resolve()
    train_y = np.asarray(
        np.load(input_dir / "train.npz", allow_pickle=True)["y_multi_hot"], dtype=np.float32
    )
    val_y = np.asarray(
        np.load(input_dir / "val.npz", allow_pickle=True)["y_multi_hot"], dtype=np.float32
    )
    train_targets = targets_from_matrix(train_y)
    targets = targets_from_matrix(val_y)
    train_meta = pd.read_csv(input_dir / "train_meta.csv", low_memory=False)
    meta = pd.read_csv(input_dir / "val_meta.csv", low_memory=False)
    names = json.loads((input_dir / "precursor_names.json").read_text(encoding="utf-8"))
    names = [str(value) for value in names]
    label_elements, label_groups, label_metals = label_chemistry(names)
    train_seen = np.asarray(train_y.sum(axis=0) > 0)
    length_modes = family_length_modes(train_meta, train_y)
    prior_builder = CandidatePriorBuilder(train_y, train_meta)
    template_builder = TemplatePriorBuilder(train_y, train_meta, names)

    base_rows = load_source(args.base_candidates, len(targets), int(args.base_output_limit))
    expert_paths = dict(parse_named_source(value) for value in args.expert)
    expert_rows = [
        load_source(path, len(targets), int(args.expert_limit))
        for path in expert_paths.values()
    ]

    matsci_builder = None
    train_direct = train_projected = None
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
        train_direct, train_projected = matsci_builder.transform_queries(train_query_views)
        val_direct, val_projected = matsci_builder.transform_queries(val_query_views)

    families = meta["family_signature_primary"].fillna("UNK").astype(str).to_numpy()
    feature_rows: List[np.ndarray] = []
    label_rows: List[np.ndarray] = []
    candidate_rows: List[List[SetKey]] = []
    base_protected_hit = np.zeros(len(targets), dtype=bool)
    for row_index, target in enumerate(targets):
        base = base_rows[row_index]
        base_protected_hit[row_index] = target in set(base[: int(args.protected_prefix)])
        sources = [base[: int(args.expert_limit)]] + [rows[row_index] for rows in expert_rows]
        candidates, features = build_row_candidates_and_features(
            sources,
            json_set(meta.iloc[row_index]["target_cation_elements"]),
            json_set(meta.iloc[row_index]["target_anion_elements"]),
            label_elements,
            label_groups,
            label_metals,
            train_seen,
            int(length_modes.get(str(families[row_index]), length_modes["__GLOBAL__"])),
            int(args.union_limit),
            prior_builder=prior_builder,
            template_prior_builder=template_builder,
            family=str(families[row_index]),
            source_agnostic=bool(args.source_agnostic),
            base_aware=True,
        )
        prefix = set(base[: int(args.protected_prefix)])
        keep = np.asarray([candidate not in prefix for candidate in candidates], dtype=bool)
        candidates = [candidate for candidate, use in zip(candidates, keep) if bool(use)]
        features = features[keep]
        features = append_matsci_features(
            features,
            candidates,
            matsci_builder,
            None if val_direct is None else val_direct[row_index],
            None if val_projected is None else val_projected[row_index],
        )
        candidate_rows.append(candidates)
        feature_rows.append(features)
        label_rows.append(np.asarray([candidate == target for candidate in candidates], dtype=np.int8))
    if not (
        len(candidate_rows) == len(feature_rows) == len(label_rows) == len(targets)
    ):
        raise RuntimeError(
            "validation candidate/feature/label rows must stay exactly aligned"
        )

    aux_feature_rows: List[np.ndarray] = []
    aux_label_rows: List[np.ndarray] = []
    aux_candidate_rows: List[List[SetKey]] = []
    aux_base_protected_hit = np.zeros(0, dtype=bool)
    aux_expert_paths: Dict[str, str] = {}
    if str(args.aux_base_candidates).strip():
        aux_base_rows = load_source(
            args.aux_base_candidates, len(train_targets), int(args.base_output_limit)
        )
        aux_expert_paths = dict(parse_named_source(value) for value in args.aux_expert)
        aux_expert_rows = [
            load_source(path, len(train_targets), int(args.expert_limit))
            for path in aux_expert_paths.values()
        ]
        train_families = (
            train_meta["family_signature_primary"].fillna("UNK").astype(str).to_numpy()
        )
        aux_base_protected_hit = np.zeros(len(train_targets), dtype=bool)
        for row_index, target in enumerate(train_targets):
            base = aux_base_rows[row_index]
            aux_base_protected_hit[row_index] = target in set(
                base[: int(args.protected_prefix)]
            )
            sources = [base[: int(args.expert_limit)]] + [
                rows[row_index] for rows in aux_expert_rows
            ]
            candidates, features = build_row_candidates_and_features(
                sources,
                json_set(train_meta.iloc[row_index]["target_cation_elements"]),
                json_set(train_meta.iloc[row_index]["target_anion_elements"]),
                label_elements,
                label_groups,
                label_metals,
                train_seen,
                int(
                    length_modes.get(
                        str(train_families[row_index]), length_modes["__GLOBAL__"]
                    )
                ),
                int(args.union_limit),
                prior_builder=prior_builder,
                template_prior_builder=template_builder,
                family=str(train_families[row_index]),
                source_agnostic=True,
                base_aware=True,
            )
            prefix = set(base[: int(args.protected_prefix)])
            keep = np.asarray([candidate not in prefix for candidate in candidates], dtype=bool)
            candidates = [candidate for candidate, use in zip(candidates, keep) if bool(use)]
            features = features[keep]
            features = append_matsci_features(
                features,
                candidates,
                matsci_builder,
                None if train_direct is None else train_direct[row_index],
                None if train_projected is None else train_projected[row_index],
            )
            aux_candidate_rows.append(candidates)
            aux_feature_rows.append(features)
            aux_label_rows.append(
                np.asarray([candidate == target for candidate in candidates], dtype=np.int8)
            )
        if aux_feature_rows and aux_feature_rows[0].shape[1] != feature_rows[0].shape[1]:
            raise ValueError(
                "auxiliary and validation feature dimensions differ; use --source_agnostic"
            )

    groups = meta["family_group_key"].fillna("UNK").astype(str).to_numpy()
    splits = formula_group_folds(groups, int(args.folds), int(args.seed))
    aux_matrix = np.zeros((0, feature_rows[0].shape[1]), dtype=np.float32)
    aux_labels = np.zeros(0, dtype=np.int8)
    aux_groups: List[int] = []
    aux_kept: List[int] = []
    if aux_feature_rows:
        aux_indices = np.asarray(
            [
                index
                for index in range(len(aux_feature_rows))
                if not bool(aux_base_protected_hit[index])
            ],
            dtype=np.int32,
        )
        aux_matrix, aux_labels, aux_groups, aux_kept = matrix_for_rows(
            aux_feature_rows, aux_label_rows, aux_indices, require_positive=True
        )
    score_rows = [np.zeros(len(row), dtype=np.float32) for row in candidate_rows]
    fold_reports = []
    if str(args.training_mode) == "aux_only":
        model = fit_model(
            aux_matrix,
            aux_labels,
            aux_groups,
            int(args.seed),
            args,
            np.ones(len(aux_groups), dtype=np.float32),
        )
        query_indices = np.arange(len(targets), dtype=np.int32)
        query_matrix, _, query_groups, query_kept = matrix_for_rows(
            feature_rows, label_rows, query_indices, require_positive=False
        )
        prediction = predict_scores(model, query_matrix, str(args.model_objective))
        offset = 0
        for row_index, size in zip(query_kept, query_groups):
            score_rows[int(row_index)] = prediction[offset : offset + int(size)]
            offset += int(size)
        fold_reports.append(
            {
                "fold": "train_oof_aux_only",
                "train_rows_with_positive": int(len(aux_kept)),
                "query_rows": int(len(targets)),
                "query_base_protected_misses": int((~base_protected_hit).sum()),
            }
        )
    else:
        for fold, (train_indices, query_indices) in enumerate(splits):
            train_indices = np.asarray(
                [
                    index
                    for index in train_indices
                    if not bool(base_protected_hit[int(index)])
                ],
                dtype=np.int32,
            )
            matrix, labels, train_groups, kept = matrix_for_rows(
                feature_rows, label_rows, train_indices, require_positive=True
            )
            if len(aux_groups):
                matrix = np.vstack([matrix, aux_matrix])
                labels = np.concatenate([labels, aux_labels])
                fit_groups = [*train_groups, *aux_groups]
                row_weights = [
                    *np.ones(len(train_groups)),
                    *np.full(len(aux_groups), args.aux_weight),
                ]
            else:
                fit_groups = train_groups
                row_weights = np.ones(len(train_groups), dtype=np.float32)
            model = fit_model(
                matrix,
                labels,
                fit_groups,
                int(args.seed) + fold * 1009,
                args,
                row_weights,
            )
            query_matrix, _, query_groups, query_kept = matrix_for_rows(
                feature_rows, label_rows, query_indices, require_positive=False
            )
            prediction = predict_scores(model, query_matrix, str(args.model_objective))
            offset = 0
            for row_index, size in zip(query_kept, query_groups):
                score_rows[int(row_index)] = prediction[offset : offset + int(size)]
                offset += int(size)
            fold_reports.append(
                {
                    "fold": int(fold),
                    "train_rows_with_positive": int(len(kept)),
                    "aux_train_rows_with_positive": int(len(aux_kept)),
                    "query_rows": int(len(query_indices)),
                    "query_base_protected_misses": int(
                        (~base_protected_hit[query_indices]).sum()
                    ),
                }
            )

    margins = [
        0.0,
        0.0025,
        0.005,
        0.01,
        0.02,
        0.03,
        0.05,
        0.075,
        0.1,
        0.15,
        0.2,
        0.3,
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
        16.0,
        32.0,
        64.0,
        128.0,
        1.0e9,
    ]
    base_metrics = exact_metrics(targets, base_rows)
    base_trial = {
        "strategy": "no_change",
        "margin": None,
        "switched_rows": 0,
        "new_hits_over_base": 0,
        "lost_hits_vs_base": 0,
        **base_metrics,
    }
    trials = [base_trial]
    best = (
        (base_trial["exact_hit@10"], 0, 0),
        base_trial,
    )
    best_rows: List[List[SetKey]] = [list(row) for row in base_rows]
    best_switched = np.zeros(len(targets), dtype=bool)
    base_hits = np.asarray([target in set(row[:10]) for target, row in zip(targets, base_rows)])
    for margin in margins:
        selection = slot_rows if str(args.selection_strategy) == "single_slot" else multi_slot_rows
        rows, switched = selection(
            base_rows, candidate_rows, score_rows, float(margin), int(args.protected_prefix)
        )
        hits = np.asarray([target in set(row[:10]) for target, row in zip(targets, rows)])
        current = {
            "margin": float(margin),
            "switched_rows": int(switched.sum()),
            "new_hits_over_base": int((hits & ~base_hits).sum()),
            "lost_hits_vs_base": int((base_hits & ~hits).sum()),
            **exact_metrics(targets, rows),
        }
        trials.append(current)
        key = (current["exact_hit@10"], -current["lost_hits_vs_base"], -current["switched_rows"])
        if key > best[0]:
            best = (key, current)
            best_rows = rows
            best_switched = switched
    if str(args.training_mode) == "aux_only":
        full_model = model
        full_training_rows = int(len(aux_kept))
    else:
        full_indices = np.asarray(
            [
                index
                for index in range(len(targets))
                if not bool(base_protected_hit[index])
            ],
            dtype=np.int32,
        )
        matrix, labels, train_groups, kept = matrix_for_rows(
            feature_rows, label_rows, full_indices, require_positive=True
        )
        if len(aux_groups):
            matrix = np.vstack([matrix, aux_matrix])
            labels = np.concatenate([labels, aux_labels])
            full_groups = [*train_groups, *aux_groups]
            full_row_weights = [
                *np.ones(len(train_groups)),
                *np.full(len(aux_groups), args.aux_weight),
            ]
        else:
            full_groups = train_groups
            full_row_weights = np.ones(len(train_groups), dtype=np.float32)
        full_model = fit_model(
            matrix,
            labels,
            full_groups,
            int(args.seed) + 100000,
            args,
            full_row_weights,
        )
        full_training_rows = int(len(kept))
    oracle_hits = np.asarray(
        [
            bool(base_hits[index]) or targets[index] in set(candidate_rows[index])
            for index in range(len(targets))
        ],
        dtype=bool,
    )
    final_hits = np.asarray(
        [target in set(row[:10]) for target, row in zip(targets, best_rows)], dtype=bool
    )
    report = {
        "protocol": (
            "train_oof_aux_only_val_heldout_safe_candidate_selector"
            if str(args.training_mode) == "aux_only"
            else "val_formula_group_disjoint_oof_safe_candidate_selector"
        ),
        "config": vars(args),
        "expert_paths": expert_paths,
        "aux_expert_paths": aux_expert_paths,
        "validation": {
            "rows": len(targets),
            "base": base_metrics,
            "base_protected_prefix_exact": float(base_protected_hit.mean()),
            "safe_slot_oracle_exact_hit@10": float(oracle_hits.mean()),
            "rows_with_target_in_slot_pool": int(
                sum(target in set(row) for target, row in zip(targets, candidate_rows))
            ),
            "best": best[1],
            "trials": trials,
            "base_protected_miss_slice": slice_metrics(
                targets, best_rows, ~base_protected_hit
            ),
            "switched_slice": slice_metrics(targets, best_rows, best_switched),
            "final_hit_rows": int(final_hits.sum()),
            "folds": fold_reports,
        },
        "feature_dim": int(feature_rows[0].shape[1]),
        "training_rows_with_positive": full_training_rows,
        "auxiliary": {
            "enabled": bool(aux_feature_rows),
            "rows": int(len(aux_feature_rows)),
            "base_protected_prefix_exact": float(aux_base_protected_hit.mean())
            if len(aux_base_protected_hit)
            else float("nan"),
            "rows_with_target_in_slot_pool": int(
                sum(
                    target in set(row)
                    for target, row in zip(train_targets, aux_candidate_rows)
                )
            ),
            "training_rows_with_positive_after_top9_filter": int(len(aux_kept)),
        },
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
                    {
                        "row_index": row_index,
                        "switched_safe_slot": bool(best_switched[row_index]),
                        "candidate_label_ids": [list(candidate) for candidate in row],
                    }
                )
                + "\n"
            )
    model_output = Path(args.output_model).resolve()
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": full_model,
            "expert_paths": expert_paths,
            "aux_expert_paths": aux_expert_paths,
            "prior_builder": prior_builder,
            "template_builder": template_builder,
            "matsci_builder": matsci_builder,
            "margin": float(best[1]["margin"]),
            "protected_prefix": int(args.protected_prefix),
            "length_modes": length_modes,
        },
        model_output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
