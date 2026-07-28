#!/usr/bin/env python3
"""Rank exact precursor variants while preserving a strong family-template slate.

The model is trained only on the Stage-2 training split.  For validation, the
family key at each Top-10 base slot is held fixed; the model may only choose a
different exact precursor set with the same family key from the base Top-N
pool.  Consequently family-equivalent coverage cannot be manufactured by the
variant ranker and any gain is an exact-set gain.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from training.family.evaluate_stage2_candidate_fusion import load_source
from training.family.evaluate_stage2_oof_chemistry_rescore import (
    chemistry_features_for_candidate,
    family_length_modes,
    json_set,
    label_chemistry,
)
from training.family.evaluate_stage2_precursor_family_slate import (
    family_key,
    precursor_family,
)
from training.family.train_stage2_oof_candidate_stacker import (
    CandidatePriorBuilder,
    MatSciFeatureBuilder,
    TemplatePriorBuilder,
    append_matsci_features,
    load_matsci_views,
    query_route_features,
)


SetKey = Tuple[int, ...]
TOP_K = (1, 3, 5, 10, 20, 50, 100)


def targets_from_matrix(matrix: np.ndarray) -> List[SetKey]:
    return [tuple(np.flatnonzero(row > 0.5).astype(int).tolist()) for row in matrix]


def exact_metrics(targets: Sequence[SetKey], rows: Sequence[Sequence[SetKey]]) -> Dict[str, float]:
    return {
        f"exact_hit@{k}": float(
            np.mean([target in set(row[:k]) for target, row in zip(targets, rows)])
        )
        for k in TOP_K
    }


def build_features(
    candidates: Sequence[SetKey],
    target_cations: set[str],
    target_anions: set[str],
    family: str,
    expected_length: int,
    label_elements: Sequence[set[str]],
    label_groups: Sequence[set[int]],
    label_metals: Sequence[set[str]],
    train_seen: np.ndarray,
    prior_builder: CandidatePriorBuilder,
    template_builder: TemplatePriorBuilder,
    matsci_builder: MatSciFeatureBuilder | None,
    query_direct: np.ndarray | None,
    query_projected: np.ndarray | None,
) -> np.ndarray:
    route = query_route_features(target_cations, target_anions)
    rows = []
    for candidate in candidates:
        rows.append(
            np.concatenate(
                [
                    chemistry_features_for_candidate(
                        candidate,
                        target_cations,
                        target_anions,
                        label_elements,
                        label_groups,
                        label_metals,
                        train_seen,
                        expected_length,
                    ),
                    route,
                    prior_builder.features(candidate, family),
                    template_builder.features(candidate, family, target_anions),
                ]
            ).astype(np.float32)
        )
    base = np.asarray(rows, dtype=np.float32)
    return append_matsci_features(
        base, candidates, matsci_builder, query_direct, query_projected
    )


def family_slot_rerank(
    base: Sequence[SetKey],
    scores: np.ndarray,
    label_families: Sequence[str],
    slate_size: int,
    protected_prefix: int,
) -> List[SetKey]:
    unique = list(dict.fromkeys(base))
    keys = [family_key(candidate, label_families) for candidate in unique]
    by_family: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        by_family[key].append(index)
    for key, indices in by_family.items():
        indices.sort(key=lambda index: (-float(scores[index]), index))

    selected: List[SetKey] = list(unique[: int(protected_prefix)])
    selected_set = set(selected)
    family_offsets: Counter[Tuple[str, ...]] = Counter()

    slot_keys = keys[: int(slate_size)]
    for key in slot_keys[len(selected) :]:
        indices = by_family.get(key, [])
        offset = int(family_offsets[key])
        chosen = None
        while offset < len(indices):
            value = unique[indices[offset]]
            offset += 1
            if value not in selected_set:
                chosen = value
                break
        family_offsets[key] = offset
        if chosen is None:
            chosen = next((value for value in unique if value not in selected_set), None)
        if chosen is not None:
            selected.append(chosen)
            selected_set.add(chosen)
        if len(selected) >= int(slate_size):
            break
    return selected + [candidate for candidate in unique if candidate not in selected_set]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--base_val_candidates", required=True)
    parser.add_argument("--matsci_embeddings", default="")
    parser.add_argument("--candidate_limit", type=int, default=100)
    parser.add_argument("--train_family_pool_limit", type=int, default=200)
    parser.add_argument("--n_estimators", type=int, default=1800)
    parser.add_argument("--num_leaves", type=int, default=127)
    parser.add_argument("--learning_rate", type=float, default=0.02)
    parser.add_argument("--min_child_samples", type=int, default=30)
    parser.add_argument("--matsci_components", type=int, default=32)
    parser.add_argument("--matsci_ridge_alpha", type=float, default=10.0)
    parser.add_argument("--n_jobs", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", required=True)
    parser.add_argument("--output_model", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    train_pack = np.load(input_dir / "train.npz", allow_pickle=True)
    val_pack = np.load(input_dir / "val.npz", allow_pickle=True)
    train_y = np.asarray(train_pack["y_multi_hot"], dtype=np.float32)
    val_y = np.asarray(val_pack["y_multi_hot"], dtype=np.float32)
    train_targets = targets_from_matrix(train_y)
    val_targets = targets_from_matrix(val_y)
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

    matsci_builder = None
    train_direct = train_projected = val_direct = val_projected = None
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

    family_counts: Dict[Tuple[str, ...], Counter[SetKey]] = defaultdict(Counter)
    for target in train_targets:
        family_counts[family_key(target, label_families)][target] += 1
    family_pools = {
        key: [candidate for candidate, _ in counter.most_common()]
        for key, counter in family_counts.items()
    }

    feature_rows = []
    labels = []
    groups = []
    covered_rows = 0
    train_families = train_meta["family_signature_primary"].fillna("UNK").astype(str).to_numpy()
    for row_index, target in enumerate(train_targets):
        key = family_key(target, label_families)
        pool = list(family_pools.get(key, []))[: int(args.train_family_pool_limit)]
        if target not in pool:
            pool.append(target)
        if len(pool) < 2:
            continue
        cations = json_set(train_meta.iloc[row_index]["target_cation_elements"])
        anions = json_set(train_meta.iloc[row_index]["target_anion_elements"])
        family = str(train_families[row_index])
        matrix = build_features(
            pool,
            cations,
            anions,
            family,
            int(length_modes.get(family, length_modes["__GLOBAL__"])),
            label_elements,
            label_groups,
            label_metals,
            train_seen,
            prior_builder,
            template_builder,
            matsci_builder,
            None if train_direct is None else train_direct[row_index],
            None if train_projected is None else train_projected[row_index],
        )
        feature_rows.append(matrix)
        local_labels = np.asarray([candidate == target for candidate in pool], dtype=np.int8)
        labels.append(local_labels)
        groups.append(len(pool))
        covered_rows += 1
    train_matrix = np.vstack(feature_rows)
    train_labels = np.concatenate(labels)
    sample_weight = np.ones(len(train_labels), dtype=np.float32)
    offset = 0
    for size in groups:
        local = train_labels[offset : offset + size]
        positive = np.flatnonzero(local > 0) + offset
        sample_weight[positive] *= max(1, size - 1) ** 0.75
        offset += size

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        num_leaves=int(args.num_leaves),
        min_child_samples=int(args.min_child_samples),
        colsample_bytree=0.9,
        reg_lambda=2.0,
        random_state=int(args.seed),
        n_jobs=int(args.n_jobs),
        verbosity=-1,
    )
    model.fit(train_matrix, train_labels, sample_weight=sample_weight)

    base_rows = load_source(
        args.base_val_candidates, len(val_targets), int(args.candidate_limit)
    )
    val_families = val_meta["family_signature_primary"].fillna("UNK").astype(str).to_numpy()
    score_rows = []
    for row_index, candidates in enumerate(base_rows):
        family = str(val_families[row_index])
        matrix = build_features(
            candidates,
            json_set(val_meta.iloc[row_index]["target_cation_elements"]),
            json_set(val_meta.iloc[row_index]["target_anion_elements"]),
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
        score_rows.append(model.predict_proba(matrix)[:, 1])

    trials = []
    best = None
    best_rows = []
    for protected_prefix in range(0, 11):
        ranked = [
            family_slot_rerank(
                row,
                scores,
                label_families,
                10,
                protected_prefix,
            )
            for row, scores in zip(base_rows, score_rows)
        ]
        current = {"protected_prefix": protected_prefix, **exact_metrics(val_targets, ranked)}
        trials.append(current)
        key = (current["exact_hit@10"], current["exact_hit@5"], current["exact_hit@1"])
        if best is None or key > best[0]:
            best = (key, current)
            best_rows = ranked
    assert best is not None

    report = {
        "protocol": "train_only_within_family_exact_variant_ranker_val_evaluation",
        "config": vars(args),
        "training": {
            "rows": len(train_targets),
            "rows_with_multiple_family_variants": int(covered_rows),
            "candidate_examples": int(len(train_labels)),
            "positive_examples": int(train_labels.sum()),
            "feature_dim": int(train_matrix.shape[1]),
            "family_template_count": int(len(family_pools)),
        },
        "validation": {
            "base": exact_metrics(val_targets, base_rows),
            "best": best[1],
            "trials": trials,
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
                    {"row_index": row_index, "candidate_label_ids": [list(value) for value in row]}
                )
                + "\n"
            )
    model_output = Path(args.output_model).resolve()
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "prior_builder": prior_builder,
            "template_builder": template_builder,
            "matsci_builder": matsci_builder,
            "label_families": label_families,
            "length_modes": length_modes,
            "protected_prefix": int(best[1]["protected_prefix"]),
        },
        model_output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
