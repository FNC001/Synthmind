#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from pymatgen.core import Element


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.family.evaluate_stage2_candidate_fusion import load_source  # noqa: E402


SetKey = Tuple[int, ...]
ELEMENT_PATTERN = re.compile(r"[A-Z][a-z]?")
FEATURE_NAMES = (
    "exact_cation_coverage",
    "periodic_group_coverage",
    "extra_metal_penalty",
    "target_anion_coverage",
    "train_unseen_fraction",
    "set_length_gap_penalty",
)
WEIGHT_GRIDS = (
    (0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
    (0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
    (0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
    (0.0, 0.25, 0.5, 1.0, 2.0),
    (0.0, 0.1, 0.25, 0.5, 1.0, 2.0),
    (0.0, 0.1, 0.25, 0.5, 1.0),
)


def json_set(value: object) -> set[str]:
    try:
        return {str(item) for item in json.loads(str(value))}
    except Exception:
        return set()


def family_length_modes(meta: pd.DataFrame, y: np.ndarray) -> Dict[str, int]:
    counts: Dict[str, Counter[int]] = defaultdict(Counter)
    families = meta["family_signature_primary"].fillna("UNK").astype(str).to_numpy()
    for family, row in zip(families, y):
        counts[str(family)][int(np.count_nonzero(row > 0.5))] += 1
    global_mode = Counter(int(np.count_nonzero(row > 0.5)) for row in y).most_common(1)[0][0]
    return {
        "__GLOBAL__": int(global_mode),
        **{family: int(values.most_common(1)[0][0]) for family, values in counts.items()},
    }


def label_chemistry(names: Sequence[str]) -> tuple[List[set[str]], List[set[int]], List[set[str]]]:
    element_sets: List[set[str]] = []
    group_sets: List[set[int]] = []
    metal_sets: List[set[str]] = []
    for name in names:
        elements = set(ELEMENT_PATTERN.findall(str(name)))
        element_sets.append(elements)
        groups = set()
        metals = set()
        for symbol in elements:
            try:
                element = Element(symbol)
            except ValueError:
                continue
            if element.group is not None:
                groups.add(int(element.group))
            if bool(element.is_metal):
                metals.add(symbol)
        group_sets.append(groups)
        metal_sets.append(metals)
    return element_sets, group_sets, metal_sets


def chemistry_features_for_candidate(
    candidate: SetKey,
    target_cations: set[str],
    target_anions: set[str],
    label_elements: Sequence[set[str]],
    label_groups: Sequence[set[int]],
    label_metals: Sequence[set[str]],
    train_seen: np.ndarray,
    expected_length: int,
) -> np.ndarray:
    elements: set[str] = set()
    groups: set[int] = set()
    metals: set[str] = set()
    for label in candidate:
        elements.update(label_elements[int(label)])
        groups.update(label_groups[int(label)])
        metals.update(label_metals[int(label)])
    target_groups = set()
    for symbol in target_cations:
        try:
            element = Element(symbol)
        except ValueError:
            continue
        if element.group is not None:
            target_groups.add(int(element.group))
    exact_coverage = len(elements & target_cations) / max(1, len(target_cations))
    group_coverage = len(groups & target_groups) / max(1, len(target_groups))
    extra_metal = -len(metals - target_cations) / max(1, len(metals))
    anion_coverage = len(elements & target_anions) / max(1, len(target_anions))
    unseen_fraction = sum(not bool(train_seen[int(label)]) for label in candidate) / max(1, len(candidate))
    length_gap = -abs(len(candidate) - int(expected_length))
    return np.asarray(
        [exact_coverage, group_coverage, extra_metal, anion_coverage, unseen_fraction, length_gap],
        dtype=np.float32,
    )


def build_feature_tensor(
    candidates: Sequence[Sequence[SetKey]],
    targets: Sequence[SetKey],
    meta: pd.DataFrame,
    limit: int,
    label_elements: Sequence[set[str]],
    label_groups: Sequence[set[int]],
    label_metals: Sequence[set[str]],
    train_seen: np.ndarray,
    length_modes: Mapping[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rows = len(candidates)
    prior = np.full((n_rows, limit), -np.inf, dtype=np.float32)
    features = np.zeros((n_rows, limit, len(FEATURE_NAMES)), dtype=np.float32)
    positive = np.full(n_rows, -1, dtype=np.int32)
    families = meta["family_signature_primary"].fillna("UNK").astype(str).to_numpy()
    cations = [json_set(value) for value in meta["target_cation_elements"]]
    anions = [json_set(value) for value in meta["target_anion_elements"]]
    for row_index, row in enumerate(candidates):
        values = list(row[:limit])
        expected_length = int(length_modes.get(str(families[row_index]), length_modes["__GLOBAL__"]))
        for rank, candidate in enumerate(values):
            prior[row_index, rank] = -math.log1p(rank)
            features[row_index, rank] = chemistry_features_for_candidate(
                candidate,
                cations[row_index],
                anions[row_index],
                label_elements,
                label_groups,
                label_metals,
                train_seen,
                expected_length,
            )
        try:
            positive[row_index] = values.index(targets[row_index])
        except ValueError:
            pass
    return prior, features, positive


def rank_metrics(
    prior: np.ndarray,
    features: np.ndarray,
    positive: np.ndarray,
    weights: np.ndarray,
    indices: np.ndarray,
) -> tuple[float, float]:
    valid_indices = indices[positive[indices] >= 0]
    if len(valid_indices) == 0:
        return 0.0, 0.0
    scores = prior[valid_indices] + np.sum(
        features[valid_indices] * weights.reshape(1, 1, -1), axis=2
    )
    positions = positive[valid_indices]
    positive_scores = scores[np.arange(len(valid_indices)), positions]
    candidate_positions = np.arange(scores.shape[1], dtype=np.int32).reshape(1, -1)
    ahead = (scores > positive_scores[:, None]) | (
        (scores == positive_scores[:, None]) & (candidate_positions < positions[:, None])
    )
    ranks = 1 + ahead.sum(axis=1)
    denominator = max(1, len(indices))
    return float(np.count_nonzero(ranks <= 10) / denominator), float(np.count_nonzero(ranks <= 50) / denominator)


def coordinate_search(
    prior: np.ndarray,
    features: np.ndarray,
    positive: np.ndarray,
    indices: np.ndarray,
    initial: np.ndarray | None = None,
    passes: int = 2,
) -> tuple[np.ndarray, Dict[str, float]]:
    weights = np.zeros(len(FEATURE_NAMES), dtype=np.float32) if initial is None else initial.copy()
    best_pair = rank_metrics(prior, features, positive, weights, indices)
    for _ in range(int(passes)):
        changed = False
        for feature_index, grid in enumerate(WEIGHT_GRIDS):
            selected_value = float(weights[feature_index])
            selected_pair = best_pair
            selected_complexity = float(np.abs(weights).sum())
            for value in grid:
                trial = weights.copy()
                trial[feature_index] = float(value)
                pair = rank_metrics(prior, features, positive, trial, indices)
                if (pair[0], pair[1], -float(np.abs(trial).sum())) > (
                    selected_pair[0], selected_pair[1], -selected_complexity
                ):
                    selected_value = float(value)
                    selected_pair = pair
                    selected_complexity = float(np.abs(trial).sum())
            if selected_value != float(weights[feature_index]):
                weights[feature_index] = selected_value
                best_pair = selected_pair
                changed = True
        if not changed:
            break
    hit10, hit50 = rank_metrics(prior, features, positive, weights, indices)
    return weights, {"exact_hit@10": hit10, "exact_hit@50": hit50}


def apply_weights(
    candidates: Sequence[Sequence[SetKey]],
    prior: np.ndarray,
    features: np.ndarray,
    families: Sequence[str],
    global_weights: np.ndarray,
    family_weights: Mapping[str, np.ndarray],
) -> List[List[SetKey]]:
    output: List[List[SetKey]] = []
    for row_index, row in enumerate(candidates):
        values = list(row[: prior.shape[1]])
        weights = family_weights.get(str(families[row_index]), global_weights)
        score = prior[row_index, : len(values)] + features[row_index, : len(values)] @ weights
        order = np.argsort(-score, kind="stable")
        output.append([values[int(index)] for index in order])
    return output


def exact_metrics(targets: Sequence[SetKey], rows: Sequence[Sequence[SetKey]]) -> Dict[str, float]:
    return {
        f"exact_hit@{k}": float(np.mean([target in set(row[:k]) for target, row in zip(targets, rows)]))
        for k in (1, 3, 5, 10, 20, 50, 100, 500)
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune chemistry-only candidate rescoring on honest OOF train rankings."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--train_candidates", required=True)
    parser.add_argument("--val_candidates", required=True)
    parser.add_argument("--candidate_limit", type=int, default=500)
    parser.add_argument("--min_family_rows", type=int, default=300)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    packs = {
        split: np.load(input_dir / f"{split}.npz", allow_pickle=True)
        for split in ("train", "val")
    }
    y = {split: np.asarray(packs[split]["y_multi_hot"], dtype=np.float32) for split in packs}
    targets = {
        split: [tuple(np.flatnonzero(row > 0.5).tolist()) for row in y[split]]
        for split in packs
    }
    meta = {
        split: pd.read_csv(input_dir / f"{split}_meta.csv", low_memory=False)
        for split in packs
    }
    candidates = {
        "train": load_source(args.train_candidates, len(y["train"]), int(args.candidate_limit)),
        "val": load_source(args.val_candidates, len(y["val"]), int(args.candidate_limit)),
    }
    names = [
        str(value)
        for value in json.loads((input_dir / "precursor_names.json").read_text(encoding="utf-8"))
    ]
    label_elements, label_groups, label_metals = label_chemistry(names)
    train_seen = np.asarray(y["train"].sum(axis=0) > 0)
    length_modes = family_length_modes(meta["train"], y["train"])
    tensors = {
        split: build_feature_tensor(
            candidates[split], targets[split], meta[split], int(args.candidate_limit),
            label_elements, label_groups, label_metals, train_seen, length_modes,
        )
        for split in ("train", "val")
    }
    train_prior, train_features, train_positive = tensors["train"]
    all_train = np.arange(len(y["train"]), dtype=np.int32)
    global_weights, global_oof = coordinate_search(
        train_prior, train_features, train_positive, all_train, passes=2
    )
    train_families = meta["train"]["family_signature_primary"].fillna("UNK").astype(str).to_numpy()
    family_weights: Dict[str, np.ndarray] = {}
    family_oof: Dict[str, Dict[str, object]] = {}
    for family in sorted(set(train_families)):
        indices = np.flatnonzero(train_families == family).astype(np.int32)
        if len(indices) < int(args.min_family_rows):
            continue
        weights, family_metric = coordinate_search(
            train_prior, train_features, train_positive, indices,
            initial=global_weights, passes=1,
        )
        family_weights[str(family)] = weights
        family_oof[str(family)] = {
            "n_rows": int(len(indices)),
            "weights": dict(zip(FEATURE_NAMES, map(float, weights))),
            **family_metric,
        }
    val_prior, val_features, _ = tensors["val"]
    val_families = meta["val"]["family_signature_primary"].fillna("UNK").astype(str).to_numpy()
    ranked = apply_weights(
        candidates["val"], val_prior, val_features, val_families,
        global_weights, family_weights,
    )
    report = {
        "protocol": "val_formula_disjoint_oof_tuned_chemistry_rescore",
        "config": vars(args),
        "feature_names": list(FEATURE_NAMES),
        "global_weights": dict(zip(FEATURE_NAMES, map(float, global_weights))),
        "global_oof": global_oof,
        "family_oof": family_oof,
        "validation": exact_metrics(targets["val"], ranked),
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with Path(args.output_candidates_jsonl).resolve().open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(ranked):
            handle.write(json.dumps({
                "row_index": row_index,
                "candidate_label_ids": [list(candidate) for candidate in row],
            }) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
