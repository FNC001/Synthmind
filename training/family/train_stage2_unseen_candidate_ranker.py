#!/usr/bin/env python3
"""OOF-trained rescue ranker for precursor sets containing train-unseen labels.

The main Stage 2 ranker is strong when every target precursor label has been
observed in training, but it cannot reliably prioritize a vocabulary label that
has zero training frequency.  This script builds an honest training signal for
that case: each Stage 2 training row is scored against the labels visible in
its formula-group-disjoint OOF fold.  Rows containing a fold-unseen target
label train both a query-level risk gate and a chemistry-aware candidate
ranker.  The held-out validation split is used only for normal model/threshold
selection; the frozen test split is never read.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from pymatgen.core import Element
from sklearn.metrics import average_precision_score, roc_auc_score


SetKey = Tuple[int, ...]
ELEMENT_PATTERN = re.compile(r"[A-Z][a-z]?")
TOP_K = (1, 3, 5, 10, 20, 50, 100)


@dataclass(frozen=True)
class CandidateInfo:
    candidate: SetKey
    best_rank: int
    reciprocal_rank_sum: float
    best_score_delta: float
    source_count: int


def json_set(value: object) -> set[str]:
    try:
        parsed = json.loads(str(value))
    except Exception:
        return set()
    return {str(item) for item in parsed}


def key_from_row(row: np.ndarray) -> SetKey:
    return tuple(np.flatnonzero(row > 0.5).astype(int).tolist())


def exact_metrics(targets: Sequence[SetKey], rows: Sequence[Sequence[SetKey]]) -> Dict[str, float]:
    return {
        f"exact_hit@{k}": float(
            np.mean([target in set(candidates[:k]) for target, candidates in zip(targets, rows)])
        )
        for k in TOP_K
    }


def slice_metrics(
    targets: Sequence[SetKey], rows: Sequence[Sequence[SetKey]], mask: np.ndarray
) -> Dict[str, float]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return {"rows": 0, **{f"exact_hit@{k}": 0.0 for k in TOP_K}}
    return {
        "rows": int(len(indices)),
        **exact_metrics([targets[i] for i in indices], [rows[i] for i in indices]),
    }


def build_oof_reference_statistics(
    y: np.ndarray, fold_root: Path
) -> tuple[np.ndarray, List[np.ndarray], List[np.ndarray], List[str]]:
    fold_dirs = sorted(Path(value) for value in glob.glob(str(fold_root / "fold_*")))
    if not fold_dirs:
        raise FileNotFoundError(f"no OOF folds found under {fold_root}")
    row_fold = np.full(len(y), -1, dtype=np.int16)
    fold_seen: List[np.ndarray] = []
    fold_frequency: List[np.ndarray] = []
    names: List[str] = []
    for fold_index, fold_dir in enumerate(fold_dirs):
        held_out = np.load(fold_dir / "val_global_row_indices.npy").astype(np.int64)
        if np.any(row_fold[held_out] >= 0):
            raise ValueError(f"overlapping OOF rows in {fold_dir}")
        row_fold[held_out] = fold_index
        reference_mask = np.ones(len(y), dtype=bool)
        reference_mask[held_out] = False
        frequency = np.asarray(y[reference_mask].sum(axis=0), dtype=np.float32)
        fold_frequency.append(frequency)
        fold_seen.append(frequency > 0)
        names.append(fold_dir.name)
    if np.any(row_fold < 0):
        missing = np.flatnonzero(row_fold < 0)
        raise ValueError(f"OOF folds do not cover {len(missing)} training rows")
    return row_fold, fold_seen, fold_frequency, names


def build_query_features(
    train_x: np.ndarray,
    val_x: np.ndarray,
    train_meta: pd.DataFrame,
    val_meta: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    def numeric(meta: pd.DataFrame) -> np.ndarray:
        rows = []
        for cations, anions in zip(
            meta["target_cation_elements"], meta["target_anion_elements"]
        ):
            cation_set = json_set(cations)
            anion_set = json_set(anions)
            cation_groups = set()
            for symbol in cation_set:
                try:
                    group = Element(symbol).group
                except ValueError:
                    group = None
                if group is not None:
                    cation_groups.add(int(group))
            rows.append(
                [
                    min(len(cation_set), 10) / 10.0,
                    min(len(anion_set), 10) / 10.0,
                    min(len(cation_groups), 10) / 10.0,
                    float("O" in anion_set),
                    float(bool(anion_set & {"Cl", "Br", "I", "F"})),
                    float(bool(anion_set & {"S", "Se", "Te"})),
                    float(bool(anion_set & {"N", "P", "As"})),
                ]
            )
        return np.asarray(rows, dtype=np.float32)

    category_columns = ["source_dataset", "family_signature_primary"]
    combined = pd.concat(
        [
            train_meta[category_columns].fillna("UNK").astype(str),
            val_meta[category_columns].fillna("UNK").astype(str),
        ],
        ignore_index=True,
    )
    category = pd.get_dummies(combined, columns=category_columns, dtype=np.float32)
    category_values = category.to_numpy(dtype=np.float32)
    train_category = category_values[: len(train_meta)]
    val_category = category_values[len(train_meta) :]
    train = np.hstack([np.nan_to_num(train_x), numeric(train_meta), train_category]).astype(np.float32)
    val = np.hstack([np.nan_to_num(val_x), numeric(val_meta), val_category]).astype(np.float32)
    names = [
        *[f"query_x_{index}" for index in range(train_x.shape[1])],
        "target_cation_count",
        "target_anion_count",
        "target_cation_group_count",
        "target_has_oxygen_anion",
        "target_has_halogen_anion",
        "target_has_chalcogen_anion",
        "target_has_pnictogen_anion",
        *[f"query_cat_{value}" for value in category.columns],
    ]
    return train, val, names


def label_chemistry(names: Sequence[str]) -> tuple[List[set[str]], List[set[str]], List[set[int]], np.ndarray]:
    elements: List[set[str]] = []
    metals: List[set[str]] = []
    groups: List[set[int]] = []
    patterns = np.zeros((len(names), 9), dtype=np.float32)
    tokens = ("NO3", "OH", "SO4", "CO3", "Cl", "Br", "PO4", "H2O", "NH4")
    for row, name in enumerate(names):
        current = set(ELEMENT_PATTERN.findall(str(name)))
        current_metals: set[str] = set()
        current_groups: set[int] = set()
        for symbol in current:
            try:
                element = Element(symbol)
            except ValueError:
                continue
            if bool(element.is_metal):
                current_metals.add(symbol)
            if element.group is not None:
                current_groups.add(int(element.group))
        elements.append(current)
        metals.append(current_metals)
        groups.append(current_groups)
        compact = str(name).replace(" ", "")
        patterns[row] = [float(token in compact) for token in tokens]
    return elements, metals, groups, patterns


def stream_candidate_records(
    path: Path,
    selected_rows: set[int],
    limit: int,
) -> Dict[int, tuple[List[SetKey], List[float]]]:
    output: Dict[int, tuple[List[SetKey], List[float]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            row_index = int(record.get("row_index", len(output)))
            if row_index not in selected_rows:
                continue
            raw_candidates = list(record.get("candidate_label_ids", []))[: int(limit)]
            candidates = [tuple(sorted({int(value) for value in row})) for row in raw_candidates]
            raw_scores = list(record.get("scores", []))[: len(candidates)]
            if len(raw_scores) != len(candidates):
                raw_scores = [float(-math.log1p(index)) for index in range(len(candidates))]
            output[row_index] = (candidates, [float(value) for value in raw_scores])
    missing = selected_rows.difference(output)
    if missing:
        raise ValueError(f"{path} is missing {len(missing)} selected rows")
    return output


def merge_candidate_sources(
    rows_by_source: Sequence[Mapping[int, tuple[List[SetKey], List[float]]]],
    row_index: int,
    limit: int,
) -> List[CandidateInfo]:
    merged: Dict[SetKey, List[float]] = {}
    for source_index, source in enumerate(rows_by_source):
        candidates, scores = source[row_index]
        top_score = float(scores[0]) if scores else 0.0
        for rank, candidate in enumerate(candidates, start=1):
            if not candidate:
                continue
            score_delta = float(np.clip(float(scores[rank - 1]) - top_score, -50.0, 0.0))
            current = merged.get(candidate)
            if current is None:
                # best rank, reciprocal sum, best score delta, source count,
                # primary-source rank (large sentinel when absent)
                current = [float(rank), 0.0, -50.0, 0.0, 1e9]
                merged[candidate] = current
            current[0] = min(current[0], float(rank))
            current[1] += 1.0 / (60.0 + float(rank))
            current[2] = max(current[2], score_delta)
            current[3] += 1.0
            if source_index == 0:
                current[4] = float(rank)
    ordered = sorted(
        merged.items(),
        key=lambda item: (
            item[1][4] if item[1][4] < 1e8 else item[1][0] + 100000.0,
            -item[1][1],
            item[0],
        ),
    )[: int(limit)]
    return [
        CandidateInfo(
            candidate=key,
            best_rank=int(values[0]),
            reciprocal_rank_sum=float(values[1]),
            best_score_delta=float(values[2]),
            source_count=int(values[3]),
        )
        for key, values in ordered
    ]


def candidate_features(
    info: CandidateInfo,
    seen: np.ndarray,
    frequency: np.ndarray,
    target_cations: set[str],
    target_anions: set[str],
    query_features: np.ndarray,
    label_elements: Sequence[set[str]],
    label_metals: Sequence[set[str]],
    label_groups: Sequence[set[int]],
    label_patterns: np.ndarray,
) -> np.ndarray:
    labels = np.asarray(info.candidate, dtype=np.int64)
    candidate_elements: set[str] = set()
    candidate_metals: set[str] = set()
    candidate_groups: set[int] = set()
    touching = 0
    for label in labels.tolist():
        candidate_elements.update(label_elements[label])
        candidate_metals.update(label_metals[label])
        candidate_groups.update(label_groups[label])
        touching += int(bool(label_elements[label] & target_cations))
    target_groups: set[int] = set()
    for symbol in target_cations:
        try:
            group = Element(symbol).group
        except ValueError:
            group = None
        if group is not None:
            target_groups.add(int(group))
    unseen_count = int(np.sum(~seen[labels]))
    frequencies = np.asarray(frequency[labels], dtype=np.float32)
    log_frequency = np.log1p(frequencies)
    union_target = target_cations | target_anions
    numeric = np.asarray(
        [
            math.log1p(info.best_rank),
            float(info.reciprocal_rank_sum),
            float(info.best_score_delta),
            min(info.source_count, 10) / 10.0,
            min(len(labels), 10) / 10.0,
            min(unseen_count, 10) / 10.0,
            unseen_count / max(1, len(labels)),
            float(log_frequency.sum()),
            float(log_frequency.min(initial=0.0)),
            float(log_frequency.max(initial=0.0)),
            float(np.mean(frequencies <= 0)),
            len(candidate_elements & target_cations) / max(1, len(target_cations)),
            len(target_cations - candidate_elements) / max(1, len(target_cations)),
            min(len(candidate_metals - target_cations), 10) / 10.0,
            len(candidate_elements & target_anions) / max(1, len(target_anions)),
            touching / max(1, len(labels)),
            (len(labels) - touching) / max(1, len(labels)),
            len(candidate_groups & target_groups) / max(1, len(target_groups)),
            min(len(candidate_elements - union_target), 20) / 20.0,
            len(candidate_elements & union_target) / max(1, len(union_target)),
            *np.max(label_patterns[labels], axis=0).astype(float).tolist(),
        ],
        dtype=np.float32,
    )
    return np.concatenate([numeric, np.asarray(query_features, dtype=np.float32)])


def select_training_candidates(
    infos: Sequence[CandidateInfo],
    target: SetKey,
    seen: np.ndarray,
    negative_count: int,
    seed: int,
) -> List[CandidateInfo]:
    lookup = {value.candidate: value for value in infos}
    positive = lookup.get(target)
    if positive is None:
        return []
    unseen_rows = [
        value for value in infos
        if value.candidate != target and any(not bool(seen[label]) for label in value.candidate)
    ]
    if len(unseen_rows) <= int(negative_count):
        return [positive, *unseen_rows]
    rng = random.Random(int(seed))
    head_count = min(len(unseen_rows), max(32, int(negative_count) // 2))
    selected = unseen_rows[:head_count]
    remaining = unseen_rows[head_count:]
    take = max(0, int(negative_count) - len(selected))
    if take:
        selected.extend(rng.sample(remaining, min(take, len(remaining))))
    return [positive, *selected]


def merge_top10(
    base: Sequence[SetKey], specialist: Sequence[SetKey], specialist_slots: int
) -> List[SetKey]:
    base_slots = max(0, 10 - int(specialist_slots))
    output: List[SetKey] = []
    seen = set()
    for candidate in list(base[:base_slots]) + list(specialist[: int(specialist_slots)]):
        if candidate not in seen:
            seen.add(candidate)
            output.append(candidate)
    for candidate in list(base) + list(specialist):
        if candidate not in seen:
            seen.add(candidate)
            output.append(candidate)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--oof_fold_dir", required=True)
    parser.add_argument("--train_candidates", nargs="+", required=True)
    parser.add_argument("--val_candidates", nargs="+", required=True)
    parser.add_argument("--base_val_candidates", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--train_candidate_limit", type=int, default=2000)
    parser.add_argument("--val_candidate_limit", type=int, default=5000)
    parser.add_argument("--negative_count", type=int, default=127)
    parser.add_argument("--ranker_estimators", type=int, default=1800)
    parser.add_argument("--gate_estimators", type=int, default=1200)
    parser.add_argument("--num_leaves", type=int, default=127)
    parser.add_argument("--learning_rate", type=float, default=0.025)
    parser.add_argument("--n_jobs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    input_dir = Path(args.input_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    train_pack = np.load(input_dir / "train.npz", allow_pickle=True)
    val_pack = np.load(input_dir / "val.npz", allow_pickle=True)
    train_y = np.asarray(train_pack["y_multi_hot"], dtype=np.float32)
    val_y = np.asarray(val_pack["y_multi_hot"], dtype=np.float32)
    train_targets = [key_from_row(row) for row in train_y]
    val_targets = [key_from_row(row) for row in val_y]
    train_meta = pd.read_csv(input_dir / "train_meta.csv", low_memory=False)
    val_meta = pd.read_csv(input_dir / "val_meta.csv", low_memory=False)
    train_query, val_query, query_feature_names = build_query_features(
        np.asarray(train_pack["x"], dtype=np.float32),
        np.asarray(val_pack["x"], dtype=np.float32),
        train_meta,
        val_meta,
    )
    precursor_names = [
        str(value)
        for value in json.loads((input_dir / "precursor_names.json").read_text(encoding="utf-8"))
    ]
    label_elements, label_metals, label_groups, label_patterns = label_chemistry(precursor_names)
    row_fold, fold_seen, fold_frequency, fold_names = build_oof_reference_statistics(
        train_y, Path(args.oof_fold_dir).resolve()
    )
    pseudo_unseen = np.asarray(
        [
            any(not bool(fold_seen[int(row_fold[row])][label]) for label in target)
            for row, target in enumerate(train_targets)
        ],
        dtype=bool,
    )
    (run_dir / "oof_pseudo_unseen_train_rows.json").write_text(
        json.dumps(
            {
                "protocol": "train_formula_group_disjoint_oof_pseudo_unseen_rows",
                "row_indices": np.flatnonzero(pseudo_unseen).astype(int).tolist(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    full_frequency = np.asarray(train_y.sum(axis=0), dtype=np.float32)
    full_seen = full_frequency > 0
    val_unseen = np.asarray(
        [any(not bool(full_seen[label]) for label in target) for target in val_targets],
        dtype=bool,
    )

    gate = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=int(args.gate_estimators),
        learning_rate=float(args.learning_rate),
        num_leaves=int(args.num_leaves),
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=int(args.seed),
        n_jobs=int(args.n_jobs),
        verbosity=-1,
    )
    gate.fit(train_query, pseudo_unseen.astype(np.int32))
    gate_probability = gate.predict_proba(val_query)[:, 1]
    print(json.dumps({
        "stage": "gate_complete",
        "pseudo_unseen_rows": int(pseudo_unseen.sum()),
        "validation_unseen_rows": int(val_unseen.sum()),
        "validation_roc_auc": float(roc_auc_score(val_unseen, gate_probability)),
    }), flush=True)

    selected_train_rows = set(np.flatnonzero(pseudo_unseen).astype(int).tolist())
    train_sources = [
        stream_candidate_records(Path(path).resolve(), selected_train_rows, args.train_candidate_limit)
        for path in args.train_candidates
    ]
    train_feature_rows: List[np.ndarray] = []
    train_labels: List[int] = []
    train_groups: List[int] = []
    covered_train_rows = 0
    for row_index in sorted(selected_train_rows):
        fold_index = int(row_fold[row_index])
        infos = merge_candidate_sources(
            train_sources, row_index, int(args.train_candidate_limit)
        )
        selected = select_training_candidates(
            infos,
            train_targets[row_index],
            fold_seen[fold_index],
            int(args.negative_count),
            int(args.seed) + row_index,
        )
        if not selected:
            continue
        covered_train_rows += 1
        cations = json_set(train_meta.iloc[row_index]["target_cation_elements"])
        anions = json_set(train_meta.iloc[row_index]["target_anion_elements"])
        for info in selected:
            train_feature_rows.append(
                candidate_features(
                    info,
                    fold_seen[fold_index],
                    fold_frequency[fold_index],
                    cations,
                    anions,
                    train_query[row_index],
                    label_elements,
                    label_metals,
                    label_groups,
                    label_patterns,
                )
            )
            train_labels.append(int(info.candidate == train_targets[row_index]))
        train_groups.append(len(selected))
    if not train_groups:
        raise RuntimeError("no OOF pseudo-unseen training positives were covered by candidates")
    train_matrix = np.vstack(train_feature_rows).astype(np.float32)
    print(json.dumps({
        "stage": "ranker_matrix_complete",
        "covered_training_queries": int(covered_train_rows),
        "training_candidates": int(len(train_labels)),
        "feature_count": int(train_matrix.shape[1]),
    }), flush=True)
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=(1, 3, 5, 10),
        lambdarank_truncation_level=20,
        n_estimators=int(args.ranker_estimators),
        learning_rate=float(args.learning_rate),
        num_leaves=int(args.num_leaves),
        min_child_samples=10,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=int(args.seed),
        n_jobs=int(args.n_jobs),
        verbosity=-1,
    )
    ranker.fit(train_matrix, np.asarray(train_labels, dtype=np.int32), group=train_groups)
    print(json.dumps({"stage": "ranker_training_complete"}), flush=True)

    all_val_rows = set(range(len(val_targets)))
    val_sources = [
        stream_candidate_records(Path(path).resolve(), all_val_rows, args.val_candidate_limit)
        for path in args.val_candidates
    ]
    base_source = stream_candidate_records(
        Path(args.base_val_candidates).resolve(), all_val_rows, max(TOP_K)
    )
    base_rows = [base_source[row][0] for row in range(len(val_targets))]
    specialist_rows: List[List[SetKey]] = []
    candidate_coverage = np.zeros(len(val_targets), dtype=bool)
    for row_index in range(len(val_targets)):
        infos = merge_candidate_sources(val_sources, row_index, int(args.val_candidate_limit))
        candidate_coverage[row_index] = val_targets[row_index] in {value.candidate for value in infos}
        infos = [
            value for value in infos
            if any(not bool(full_seen[label]) for label in value.candidate)
        ]
        if not infos:
            specialist_rows.append([])
            continue
        cations = json_set(val_meta.iloc[row_index]["target_cation_elements"])
        anions = json_set(val_meta.iloc[row_index]["target_anion_elements"])
        matrix = np.vstack(
            [
                candidate_features(
                    info,
                    full_seen,
                    full_frequency,
                    cations,
                    anions,
                    val_query[row_index],
                    label_elements,
                    label_metals,
                    label_groups,
                    label_patterns,
                )
                for info in infos
            ]
        ).astype(np.float32)
        scores = ranker.predict(matrix)
        order = np.argsort(-np.asarray(scores), kind="stable")
        specialist_rows.append([infos[int(index)].candidate for index in order[: max(TOP_K)]])
        if (row_index + 1) % 250 == 0:
            print(json.dumps({
                "stage": "validation_scoring",
                "rows_complete": int(row_index + 1),
                "rows_total": int(len(val_targets)),
            }), flush=True)

    threshold_grid = sorted(
        {
            0.0,
            1.0,
            *np.linspace(0.02, 0.98, 49).tolist(),
            *np.quantile(gate_probability, np.linspace(0.01, 0.99, 49)).tolist(),
        }
    )
    trials = []
    best = None
    best_rows: List[List[SetKey]] = []
    for threshold in threshold_grid:
        gate_mask = gate_probability >= float(threshold)
        for specialist_slots in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
            rows = [
                merge_top10(base, specialist, specialist_slots) if use_specialist else list(base)
                for base, specialist, use_specialist in zip(base_rows, specialist_rows, gate_mask)
            ]
            metrics = exact_metrics(val_targets, rows)
            trial = {
                "gate_threshold": float(threshold),
                "specialist_slots": int(specialist_slots),
                "gated_rows": int(gate_mask.sum()),
                **metrics,
            }
            trials.append(trial)
            key = (metrics["exact_hit@10"], metrics["exact_hit@5"], metrics["exact_hit@1"])
            if best is None or key > best[0]:
                best = (key, trial)
                best_rows = rows
    assert best is not None
    oracle_rows = [
        merge_top10(base, specialist, 10) if is_unseen else list(base)
        for base, specialist, is_unseen in zip(base_rows, specialist_rows, val_unseen)
    ]
    feature_names = [
        "log_best_rank",
        "reciprocal_rank_sum",
        "best_score_delta",
        "source_count",
        "set_length",
        "unseen_count",
        "unseen_fraction",
        "log_frequency_sum",
        "log_frequency_min",
        "log_frequency_max",
        "zero_frequency_fraction",
        "target_cation_coverage",
        "target_cation_missing",
        "extra_metal_count",
        "target_anion_overlap",
        "target_touch_fraction",
        "accessory_fraction",
        "target_group_coverage",
        "extra_element_count",
        "target_element_coverage",
        "has_nitrate",
        "has_hydroxide",
        "has_sulfate",
        "has_carbonate",
        "has_chloride",
        "has_bromide",
        "has_phosphate",
        "has_hydrate",
        "has_ammonium",
        *query_feature_names,
    ]
    report = {
        "protocol": "val_formula_disjoint_oof_pseudo_unseen_candidate_rescue",
        "config": vars(args),
        "training": {
            "rows": int(len(train_targets)),
            "pseudo_unseen_rows": int(pseudo_unseen.sum()),
            "pseudo_unseen_candidate_covered_rows": int(covered_train_rows),
            "ranker_training_candidates": int(len(train_labels)),
            "ranker_parameters": ranker.get_params(),
            "gate_parameters": gate.get_params(),
            "oof_folds": fold_names,
        },
        "validation": {
            "rows": int(len(val_targets)),
            "unseen_label_rows": int(val_unseen.sum()),
            "gate_roc_auc": float(roc_auc_score(val_unseen, gate_probability)),
            "gate_average_precision": float(average_precision_score(val_unseen, gate_probability)),
            "candidate_coverage_all": float(candidate_coverage.mean()),
            "candidate_coverage_unseen": float(candidate_coverage[val_unseen].mean()),
            "base": {
                "all": exact_metrics(val_targets, base_rows),
                "seen": slice_metrics(val_targets, base_rows, ~val_unseen),
                "unseen": slice_metrics(val_targets, base_rows, val_unseen),
            },
            "specialist": {
                "all": exact_metrics(val_targets, specialist_rows),
                "seen": slice_metrics(val_targets, specialist_rows, ~val_unseen),
                "unseen": slice_metrics(val_targets, specialist_rows, val_unseen),
            },
            "oracle_unseen_gate": {
                "all": exact_metrics(val_targets, oracle_rows),
                "seen": slice_metrics(val_targets, oracle_rows, ~val_unseen),
                "unseen": slice_metrics(val_targets, oracle_rows, val_unseen),
            },
            "best": best[1],
            "best_slices": {
                "seen": slice_metrics(val_targets, best_rows, ~val_unseen),
                "unseen": slice_metrics(val_targets, best_rows, val_unseen),
            },
            "top_trials": sorted(
                trials,
                key=lambda row: (row["exact_hit@10"], row["exact_hit@5"], row["exact_hit@1"]),
                reverse=True,
            )[:50],
        },
        "feature_names": feature_names,
        "ranker_feature_importance_gain": {
            name: float(value)
            for name, value in sorted(
                zip(feature_names, ranker.booster_.feature_importance(importance_type="gain")),
                key=lambda item: -item[1],
            )
        },
        "gate_feature_importance_gain": {
            name: float(value)
            for name, value in sorted(
                zip(query_feature_names, gate.booster_.feature_importance(importance_type="gain")),
                key=lambda item: -item[1],
            )
        },
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (run_dir / "val_candidates.jsonl").open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(best_rows):
            handle.write(
                json.dumps(
                    {
                        "row_index": row_index,
                        "candidate_label_ids": [list(value) for value in row],
                        "gate_probability": float(gate_probability[row_index]),
                    }
                )
                + "\n"
            )
    gate.booster_.save_model(str(run_dir / "unseen_gate.txt"))
    ranker.booster_.save_model(str(run_dir / "unseen_candidate_ranker.txt"))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
