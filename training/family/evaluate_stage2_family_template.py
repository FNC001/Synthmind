#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synthmind.chemistry.families import element_group_bucket  # noqa: E402


SetKey = Tuple[int, ...]
TemplateKey = Tuple[str, ...]
ELEMENT_PATTERN = re.compile(r"[A-Z][a-z]?")
PLACEHOLDER_PATTERN = re.compile(r"\{([A-Z0-9]+)_(\d+)\}")


def json_set(value: Any) -> Set[str]:
    try:
        return {str(item) for item in json.loads(str(value))}
    except Exception:
        return set()


def bucket(symbol: str) -> str:
    from pymatgen.core import Element

    return element_group_bucket(Element(symbol))


def grouped(elements: Iterable[str]) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = defaultdict(list)
    for element in sorted(set(elements)):
        result[bucket(element)].append(element)
    return dict(result)


def mapping_variants(elements: Iterable[str], max_variants: int = 48) -> List[Dict[str, str]]:
    groups = grouped(elements)
    options: List[List[Dict[str, str]]] = []
    for group in sorted(groups):
        values = sorted(groups[group])
        group_options = []
        for permutation in itertools.permutations(values):
            group_options.append(
                {element: f"{{{group}_{index}}}" for index, element in enumerate(permutation)}
            )
        options.append(group_options)
    variants: List[Dict[str, str]] = []
    for combination in itertools.product(*options) if options else []:
        merged: Dict[str, str] = {}
        for item in combination:
            merged.update(item)
        variants.append(merged)
        if len(variants) >= max_variants:
            break
    return variants


def normalize_formula(formula: str, mapping: Mapping[str, str]) -> str:
    return ELEMENT_PATTERN.sub(lambda match: mapping.get(match.group(0), match.group(0)), formula)


def canonical_template(names: Sequence[str], elements: Iterable[str]) -> TemplateKey | None:
    variants = mapping_variants(elements)
    if not variants:
        return None
    normalized = [
        tuple(sorted({normalize_formula(name, mapping) for name in names}))
        for mapping in variants
    ]
    return min(normalized)


def instantiate_template(template: TemplateKey, elements: Iterable[str]) -> List[Tuple[str, ...]]:
    groups = grouped(elements)
    placeholders: Dict[str, Set[int]] = defaultdict(set)
    for formula in template:
        for match in PLACEHOLDER_PATTERN.finditer(formula):
            placeholders[match.group(1)].add(int(match.group(2)))
    if set(placeholders) != set(groups):
        return []
    group_options: List[List[Dict[str, str]]] = []
    for group in sorted(placeholders):
        indices = sorted(placeholders[group])
        values = sorted(groups[group])
        if len(indices) != len(values) or indices != list(range(len(indices))):
            return []
        group_options.append(
            [
                {f"{{{group}_{index}}}": value for index, value in zip(indices, permutation)}
                for permutation in itertools.permutations(values)
            ]
        )
    output = []
    for combination in itertools.product(*group_options):
        mapping: Dict[str, str] = {}
        for item in combination:
            mapping.update(item)
        names = tuple(
            sorted(
                {
                    PLACEHOLDER_PATTERN.sub(
                        lambda match: mapping[match.group(0)], formula
                    )
                    for formula in template
                }
            )
        )
        output.append(names)
    return output


def metrics(targets: Sequence[SetKey], rows: Sequence[Sequence[SetKey]]) -> Dict[str, float]:
    result = {}
    for k in (1, 3, 5, 10, 20, 50, 100, 500, 1000):
        result[f"exact_hit@{k}"] = float(
            np.mean([target in set(candidates[:k]) for target, candidates in zip(targets, rows)])
        )
    result["mean_candidates"] = float(np.mean([len(row) for row in rows]))
    result["nonempty_rate"] = float(np.mean([bool(row) for row in rows]))
    result["oracle_recall"] = float(
        np.mean([target in set(candidates) for target, candidates in zip(targets, rows)])
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate exact precursor sets from family-normalized reaction templates.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", required=True)
    parser.add_argument("--candidate_limit", type=int, default=5000)
    parser.add_argument("--source_weight", type=float, default=2.0)
    parser.add_argument("--anion_weight", type=float, default=1.0)
    parser.add_argument("--route_weight", type=float, default=3.0)
    parser.add_argument("--formula_weight", type=float, default=8.0)
    parser.add_argument("--formula_source_weight", type=float, default=12.0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    split_names = tuple(dict.fromkeys(("train", args.split)))
    packs = {
        split: np.load(input_dir / f"{split}.npz", allow_pickle=True)
        for split in split_names
    }
    meta = {
        split: pd.read_csv(input_dir / f"{split}_meta.csv", low_memory=False)
        for split in split_names
    }
    names = [str(value) for value in json.loads((input_dir / "precursor_names.json").read_text())]
    name_to_id = {name: index for index, name in enumerate(names)}
    train_sets = [tuple(np.flatnonzero(row > 0.5).tolist()) for row in packs["train"]["y_multi_hot"]]
    query_sets = [tuple(np.flatnonzero(row > 0.5).tolist()) for row in packs[args.split]["y_multi_hot"]]

    family_counts: Dict[str, Counter[TemplateKey]] = defaultdict(Counter)
    family_source_counts: Dict[Tuple[str, str], Counter[TemplateKey]] = defaultdict(Counter)
    family_anion_counts: Dict[Tuple[str, str], Counter[TemplateKey]] = defaultdict(Counter)
    route_counts: Dict[Tuple[str, str, str], Counter[TemplateKey]] = defaultdict(Counter)
    formula_counts: Dict[Tuple[str, str], Counter[TemplateKey]] = defaultdict(Counter)
    formula_source_counts: Dict[Tuple[str, str, str], Counter[TemplateKey]] = defaultdict(Counter)
    template_rows = 0
    for row, label_set in enumerate(train_sets):
        record = meta["train"].iloc[row]
        elements = json_set(record.get("target_cation_elements", "[]"))
        family = str(record.get("family_signature_primary", "UNK"))
        source = str(record.get("source_dataset", ""))
        anions = "+".join(sorted(json_set(record.get("target_anion_elements", "[]")))) or "NONE"
        template = canonical_template([names[label] for label in label_set], elements)
        target_template_values = canonical_template([str(record.get("formula", ""))], elements)
        target_template = target_template_values[0] if target_template_values else ""
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
        template_rows += 1

    ranked_rows: List[List[SetKey]] = []
    score_rows: List[List[float]] = []
    for row in range(len(query_sets)):
        record = meta[args.split].iloc[row]
        elements = json_set(record.get("target_cation_elements", "[]"))
        family = str(record.get("family_signature_primary", "UNK"))
        source = str(record.get("source_dataset", ""))
        anions = "+".join(sorted(json_set(record.get("target_anion_elements", "[]")))) or "NONE"
        target_template_values = canonical_template([str(record.get("formula", ""))], elements)
        target_template = target_template_values[0] if target_template_values else ""
        combined: Counter[TemplateKey] = Counter(family_counts.get(family, Counter()))
        for template, count in family_source_counts.get((family, source), Counter()).items():
            combined[template] += float(args.source_weight) * count
        for template, count in family_anion_counts.get((family, anions), Counter()).items():
            combined[template] += float(args.anion_weight) * count
        for template, count in route_counts.get((family, source, anions), Counter()).items():
            combined[template] += float(args.route_weight) * count
        for template, count in formula_counts.get((family, target_template), Counter()).items():
            combined[template] += float(args.formula_weight) * count
        for template, count in formula_source_counts.get((family, source, target_template), Counter()).items():
            combined[template] += float(args.formula_source_weight) * count
        candidate_scores: Dict[SetKey, float] = {}
        for template, count in sorted(combined.items(), key=lambda item: (-item[1], item[0])):
            for instantiated in instantiate_template(template, elements):
                if any(name not in name_to_id for name in instantiated):
                    continue
                candidate = tuple(sorted({name_to_id[name] for name in instantiated}))
                if not candidate:
                    continue
                score = math.log1p(float(count))
                candidate_scores[candidate] = max(candidate_scores.get(candidate, -math.inf), score)
            if len(candidate_scores) >= int(args.candidate_limit) * 2:
                break
        ordered = sorted(candidate_scores.items(), key=lambda item: (-item[1], item[0]))[
            : int(args.candidate_limit)
        ]
        ranked_rows.append([candidate for candidate, _ in ordered])
        score_rows.append([score for _, score in ordered])

    report = {
        "protocol": f"{args.split}_formula_disjoint_family_normalized_template",
        "config": vars(args),
        "training": {
            "rows": len(train_sets),
            "rows_with_template": template_rows,
            "families": len(family_counts),
            "unique_templates": int(sum(len(values) for values in family_counts.values())),
        },
        args.split: metrics(query_sets, ranked_rows),
    }
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_candidates = Path(args.output_candidates_jsonl).resolve()
    output_candidates.parent.mkdir(parents=True, exist_ok=True)
    with output_candidates.open("w", encoding="utf-8") as handle:
        for row, (candidates, scores) in enumerate(zip(ranked_rows, score_rows)):
            handle.write(
                json.dumps(
                    {
                        "row_index": row,
                        "candidate_label_ids": [list(candidate) for candidate in candidates],
                        "scores": scores,
                    }
                )
                + "\n"
            )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
