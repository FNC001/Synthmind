#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from pymatgen.core import Element


SetKey = Tuple[int, ...]
ELEMENT_PATTERN = re.compile(r"[A-Z][a-z]?")
CARBOHYDRATE_PATTERN = re.compile(r"^C(?:\d+)?H(?:\d+)?O(?:\d+)?$")


def json_set(value: object) -> set[str]:
    try:
        return {str(item) for item in json.loads(str(value))}
    except Exception:
        return set()


def label_chemistry(names: Sequence[str]) -> tuple[List[set[str]], List[set[str]]]:
    elements: List[set[str]] = []
    metals: List[set[str]] = []
    for name in names:
        current = set(ELEMENT_PATTERN.findall(str(name)))
        elements.append(current)
        current_metals = set()
        for symbol in current:
            try:
                if bool(Element(symbol).is_metal):
                    current_metals.add(symbol)
            except ValueError:
                continue
        metals.append(current_metals)
    return elements, metals


def canonical_pattern_score(name: str, target_elements: set[str]) -> float:
    compact = str(name).replace(" ", "")
    score = 0.0
    patterns = (
        ("NO3", 5.0),
        ("OH", 3.5),
        ("SO4", 3.0),
        ("OAc", 3.0),
        ("C2H3O2", 3.0),
        ("CO3", 2.5),
        ("Cl", 2.0),
        ("Br", 1.5),
        ("PO4", 1.5),
    )
    for pattern, value in patterns:
        if pattern in compact:
            score += value
    if "·" in compact or "H2O" in compact:
        score += 1.0
    if "NH4" in compact or "(NH4)" in compact:
        score += 1.0
    if target_elements == {"C"}:
        elements = set(ELEMENT_PATTERN.findall(compact))
        if elements <= {"C", "H", "O"} and {"C", "H", "O"} <= elements:
            score += 12.0
            if CARBOHYDRATE_PATTERN.match(compact):
                score += 3.0
        if elements & {"N", "S", "P", "Cl", "Br", "I"}:
            score -= 6.0
    return score


def label_score(
    label: int,
    name: str,
    elements: set[str],
    metals: set[str],
    target_elements: set[str],
    target_anions: set[str],
    frequency: np.ndarray,
) -> float:
    covered = len(elements & target_elements)
    if covered == 0:
        return -math.inf
    extra_metals = metals - target_elements
    extra_nonmetal_targets = target_elements - elements
    score = 12.0 * covered / max(1, len(target_elements))
    score -= 8.0 * len(extra_metals)
    score -= 5.0 * len(extra_nonmetal_targets)
    score += 2.0 * len(elements & target_anions)
    score += canonical_pattern_score(name, target_elements)
    score += 0.35 * math.log1p(float(frequency[int(label)]))
    if len(elements) == 1:
        score -= 1.0
    return score


def metrics(targets: Sequence[SetKey], rows: Sequence[Sequence[SetKey]]) -> Dict[str, float]:
    return {
        f"exact_hit@{k}": float(np.mean([target in set(row[:k]) for target, row in zip(targets, rows)]))
        for k in (1, 3, 5, 10, 20, 50, 100, 500)
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chemistry-prior candidates for formula-group extrapolation and train-unseen salts."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--query_families", default="LN,G14")
    parser.add_argument("--per_element", type=int, default=80)
    parser.add_argument("--accessories", type=int, default=16)
    parser.add_argument(
        "--pair_penalty", type=float, default=20.0,
        help="Keep canonical singleton salts ahead of accessory combinations unless strongly supported.",
    )
    parser.add_argument("--candidate_limit", type=int, default=500)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    train_pack = np.load(input_dir / "train.npz", allow_pickle=True)
    query_pack = np.load(input_dir / f"{args.split}.npz", allow_pickle=True)
    train_y = np.asarray(train_pack["y_multi_hot"], dtype=np.float32)
    query_y = np.asarray(query_pack["y_multi_hot"], dtype=np.float32)
    train_meta = pd.read_csv(input_dir / "train_meta.csv", low_memory=False)
    query_meta = pd.read_csv(input_dir / f"{args.split}_meta.csv", low_memory=False)
    names = [str(value) for value in json.loads((input_dir / "precursor_names.json").read_text())]
    elements, metals = label_chemistry(names)
    frequency = np.asarray(train_y.sum(axis=0), dtype=np.float32)
    selected_families = {
        value.strip() for value in str(args.query_families).split(",") if value.strip()
    }
    train_families = train_meta["family_signature_primary"].fillna("UNK").astype(str).to_numpy()
    query_families = query_meta["family_signature_primary"].fillna("UNK").astype(str).to_numpy()
    family_accessories: Dict[str, Counter[int]] = defaultdict(Counter)
    global_accessories: Counter[int] = Counter()
    for row_index, row in enumerate(train_y):
        family = str(train_families[row_index])
        for label in np.flatnonzero(row > 0.5):
            label = int(label)
            if not metals[label]:
                family_accessories[family][label] += 1
                global_accessories[label] += 1

    query_targets = [tuple(np.flatnonzero(row > 0.5).tolist()) for row in query_y]
    target_elements_rows = [json_set(value) for value in query_meta["target_cation_elements"]]
    target_anion_rows = [json_set(value) for value in query_meta["target_anion_elements"]]
    all_candidates: List[List[SetKey]] = []
    all_scores: List[List[float]] = []
    for row_index, (target_elements, target_anions) in enumerate(
        zip(target_elements_rows, target_anion_rows)
    ):
        family = str(query_families[row_index])
        if selected_families and family not in selected_families:
            all_candidates.append([])
            all_scores.append([])
            continue
        ranked_by_element: List[List[tuple[int, float]]] = []
        for target_element in sorted(target_elements):
            values = []
            for label, name in enumerate(names):
                score = label_score(
                    label, name, elements[label], metals[label], {target_element},
                    target_anions, frequency,
                )
                if math.isfinite(score):
                    values.append((label, score))
            values.sort(key=lambda item: (-item[1], names[item[0]]))
            ranked_by_element.append(values[: int(args.per_element)])

        scored: Dict[SetKey, float] = {}
        for values in ranked_by_element:
            for label, score in values:
                scored[(int(label),)] = max(scored.get((int(label),), -math.inf), float(score))
        if ranked_by_element and all(ranked_by_element):
            product_lists = [values[: min(30, len(values))] for values in ranked_by_element]
            for combination in itertools.product(*product_lists):
                key = tuple(sorted({int(label) for label, _ in combination}))
                score = float(sum(value for _, value in combination) / len(combination))
                scored[key] = max(scored.get(key, -math.inf), score)

        accessories = Counter(global_accessories)
        for label, count in family_accessories.get(family, Counter()).items():
            accessories[label] += 8 * count
        accessory_rows = sorted(
            accessories.items(), key=lambda item: (-item[1], names[int(item[0])])
        )[: int(args.accessories)]
        base_rows = sorted(scored.items(), key=lambda item: (-item[1], item[0]))[:100]
        for base, base_score in base_rows:
            for accessory, count in accessory_rows:
                key = tuple(sorted({*base, int(accessory)}))
                if key == base:
                    continue
                score = float(
                    base_score + 0.2 * math.log1p(float(count)) - float(args.pair_penalty)
                )
                scored[key] = max(scored.get(key, -math.inf), score)

        ordered = sorted(scored.items(), key=lambda item: (-item[1], item[0]))[
            : int(args.candidate_limit)
        ]
        all_candidates.append([key for key, _ in ordered])
        all_scores.append([float(score) for _, score in ordered])

    report = {
        "protocol": f"{args.split}_formula_disjoint_canonical_chemistry_candidates",
        "config": vars(args),
        "rows": len(query_targets),
        "selected_rows": int(sum(family in selected_families for family in query_families))
        if selected_families else len(query_targets),
        "metrics": metrics(query_targets, all_candidates),
        "mean_candidates_selected": float(np.mean([
            len(row) for row, family in zip(all_candidates, query_families)
            if not selected_families or family in selected_families
        ])),
    }
    Path(args.output_json).resolve().write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with Path(args.output_candidates_jsonl).resolve().open("w", encoding="utf-8") as handle:
        for row_index, (candidates, scores) in enumerate(zip(all_candidates, all_scores)):
            handle.write(json.dumps({
                "row_index": row_index,
                "candidate_label_ids": [list(value) for value in candidates],
                "scores": scores,
            }, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
