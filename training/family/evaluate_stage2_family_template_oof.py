#!/usr/bin/env python3
"""Generate formula-group-disjoint OOF family-template candidates for Stage 2."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.family.evaluate_stage2_family_template import (  # noqa: E402
    TemplateKey,
    canonical_template,
    instantiate_template,
    json_set,
    metrics,
)


SetKey = Tuple[int, ...]
Bank = Dict[str, object]


def stable_group_fold(value: str, folds: int, seed: int) -> int:
    digest = hashlib.sha1(f"{int(seed)}::{str(value)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % int(folds)


def build_bank(
    indices: Iterable[int],
    meta: pd.DataFrame,
    target_sets: Sequence[SetKey],
    names: Sequence[str],
) -> Bank:
    family_counts: Dict[str, Counter[TemplateKey]] = defaultdict(Counter)
    family_source_counts: Dict[Tuple[str, str], Counter[TemplateKey]] = defaultdict(Counter)
    family_anion_counts: Dict[Tuple[str, str], Counter[TemplateKey]] = defaultdict(Counter)
    route_counts: Dict[Tuple[str, str, str], Counter[TemplateKey]] = defaultdict(Counter)
    formula_counts: Dict[Tuple[str, str], Counter[TemplateKey]] = defaultdict(Counter)
    formula_source_counts: Dict[Tuple[str, str, str], Counter[TemplateKey]] = defaultdict(Counter)
    rows = 0
    for row_index in indices:
        record = meta.iloc[int(row_index)]
        elements = json_set(record.get("target_cation_elements", "[]"))
        family = str(record.get("family_signature_primary", "UNK"))
        source = str(record.get("source_dataset", ""))
        anions = "+".join(sorted(json_set(record.get("target_anion_elements", "[]")))) or "NONE"
        template = canonical_template(
            [names[label] for label in target_sets[int(row_index)]], elements
        )
        target_values = canonical_template([str(record.get("formula", ""))], elements)
        target_template = target_values[0] if target_values else ""
        if not template:
            continue
        weight = max(1, int(round(2.0 * float(record.get("quality_weight", 0.5)))))
        family_counts[family][template] += weight
        family_source_counts[(family, source)][template] += weight
        family_anion_counts[(family, anions)][template] += weight
        route_counts[(family, source, anions)][template] += weight
        if target_template:
            formula_counts[(family, target_template)][template] += weight
            formula_source_counts[(family, source, target_template)][template] += weight
        rows += 1
    return {
        "family": family_counts,
        "family_source": family_source_counts,
        "family_anion": family_anion_counts,
        "route": route_counts,
        "formula": formula_counts,
        "formula_source": formula_source_counts,
        "rows": int(rows),
    }


def rank_from_bank(
    bank: Bank,
    record: pd.Series,
    name_to_id: Dict[str, int],
    candidate_limit: int,
    source_weight: float,
    anion_weight: float,
    route_weight: float,
    formula_weight: float,
    formula_source_weight: float,
) -> tuple[list[SetKey], list[float]]:
    elements = json_set(record.get("target_cation_elements", "[]"))
    family = str(record.get("family_signature_primary", "UNK"))
    source = str(record.get("source_dataset", ""))
    anions = "+".join(sorted(json_set(record.get("target_anion_elements", "[]")))) or "NONE"
    target_values = canonical_template([str(record.get("formula", ""))], elements)
    target_template = target_values[0] if target_values else ""
    family_counts = bank["family"]
    combined: Counter[TemplateKey] = Counter(family_counts.get(family, Counter()))
    additions = (
        (bank["family_source"].get((family, source), Counter()), float(source_weight)),
        (bank["family_anion"].get((family, anions), Counter()), float(anion_weight)),
        (bank["route"].get((family, source, anions), Counter()), float(route_weight)),
        (bank["formula"].get((family, target_template), Counter()), float(formula_weight)),
        (
            bank["formula_source"].get((family, source, target_template), Counter()),
            float(formula_source_weight),
        ),
    )
    for counter, weight in additions:
        for template, count in counter.items():
            combined[template] += float(weight) * count
    candidate_scores: Dict[SetKey, float] = {}
    for template, count in sorted(combined.items(), key=lambda item: (-item[1], item[0])):
        for instantiated in instantiate_template(template, elements):
            if any(name not in name_to_id for name in instantiated):
                continue
            candidate = tuple(sorted({name_to_id[name] for name in instantiated}))
            if candidate:
                candidate_scores[candidate] = max(
                    candidate_scores.get(candidate, -math.inf), math.log1p(float(count))
                )
        if len(candidate_scores) >= int(candidate_limit) * 2:
            break
    ordered = sorted(candidate_scores.items(), key=lambda item: (-item[1], item[0]))[
        : int(candidate_limit)
    ]
    return [candidate for candidate, _ in ordered], [float(score) for _, score in ordered]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=8511)
    parser.add_argument("--candidate_limit", type=int, default=5000)
    parser.add_argument("--source_weight", type=float, default=2.0)
    parser.add_argument("--anion_weight", type=float, default=1.0)
    parser.add_argument("--route_weight", type=float, default=3.0)
    parser.add_argument("--formula_weight", type=float, default=16.0)
    parser.add_argument("--formula_source_weight", type=float, default=24.0)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", required=True)
    args = parser.parse_args()
    if int(args.folds) < 2:
        parser.error("--folds must be at least 2")

    input_dir = Path(args.input_dir).resolve()
    pack = np.load(input_dir / "train.npz", allow_pickle=True)
    target_sets = [
        tuple(np.flatnonzero(row > 0.5).tolist())
        for row in np.asarray(pack["y_multi_hot"], dtype=np.float32)
    ]
    meta = pd.read_csv(input_dir / "train_meta.csv", low_memory=False)
    names = json.loads((input_dir / "precursor_names.json").read_text(encoding="utf-8"))
    name_to_id = {str(name): index for index, name in enumerate(names)}
    groups = meta["family_group_key"].fillna("UNK").astype(str).to_numpy()
    fold_ids = np.asarray(
        [stable_group_fold(group, int(args.folds), int(args.seed)) for group in groups],
        dtype=np.int64,
    )
    banks = [
        build_bank(
            np.flatnonzero(fold_ids != fold).tolist(), meta, target_sets, names
        )
        for fold in range(int(args.folds))
    ]
    rows = []
    scores = []
    for row_index in range(len(target_sets)):
        candidates, values = rank_from_bank(
            banks[int(fold_ids[row_index])],
            meta.iloc[row_index],
            name_to_id,
            int(args.candidate_limit),
            float(args.source_weight),
            float(args.anion_weight),
            float(args.route_weight),
            float(args.formula_weight),
            float(args.formula_source_weight),
        )
        rows.append(candidates)
        scores.append(values)

    report = {
        "protocol": "train_formula_group_disjoint_oof_family_normalized_template",
        "config": vars(args),
        "rows": int(len(target_sets)),
        "formula_groups": int(pd.Series(groups).nunique()),
        "fold_rows": [int((fold_ids == fold).sum()) for fold in range(int(args.folds))],
        "bank_rows": [int(bank["rows"]) for bank in banks],
        "train_oof": metrics(target_sets, rows),
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    candidates_output = Path(args.output_candidates_jsonl).resolve()
    candidates_output.parent.mkdir(parents=True, exist_ok=True)
    with candidates_output.open("w", encoding="utf-8") as handle:
        for row_index, (candidates, values) in enumerate(zip(rows, scores)):
            handle.write(
                json.dumps(
                    {
                        "row_index": int(row_index),
                        "candidate_label_ids": [list(candidate) for candidate in candidates],
                        "scores": values,
                        "oof_fold": int(fold_ids[row_index]),
                    }
                )
                + "\n"
            )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
