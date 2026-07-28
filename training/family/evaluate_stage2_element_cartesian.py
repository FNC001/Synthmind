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
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from pymatgen.core import Element

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synthmind.chemistry.families import element_group_bucket


SetKey = Tuple[int, ...]
ELEMENT_PATTERN = re.compile(r"[A-Z][a-z]?")


def json_set(value: object) -> Set[str]:
    try:
        return {str(item) for item in json.loads(str(value))}
    except Exception:
        return set()


def exact_metrics(targets: Sequence[SetKey], candidates: Sequence[Sequence[SetKey]]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for k in (1, 3, 5, 10, 20, 50, 100, 500, 1000, 5000):
        result[f"exact_hit@{k}"] = float(np.mean([target in set(row[:k]) for target, row in zip(targets, candidates)]))
    result["mean_candidates"] = float(np.mean([len(row) for row in candidates]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate precursor sets by Cartesian products of element-specific source priors.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", required=True)
    parser.add_argument("--per_element", type=int, default=30)
    parser.add_argument("--accessories", type=int, default=12)
    parser.add_argument(
        "--accessory_combo_pool",
        type=int,
        default=0,
        help=(
            "If positive, form previously unseen accessory bundles from this many "
            "top train-derived accessory labels. Disabled by default."
        ),
    )
    parser.add_argument(
        "--accessory_combo_order",
        type=int,
        default=1,
        help="Maximum accessory subset size used with --accessory_combo_pool.",
    )
    parser.add_argument(
        "--accessory_combo_limit",
        type=int,
        default=5000,
        help="Maximum number of scored accessory subsets retained per query chemistry.",
    )
    parser.add_argument(
        "--accessory_base_rows",
        type=int,
        default=100,
        help="Number of target-bearing base candidates expanded with accessory bundles.",
    )
    parser.add_argument(
        "--same_element_combo_pool",
        type=int,
        default=0,
        help=(
            "If positive, allow multiple train-derived source labels for one target "
            "element using this many top per-element labels. Disabled by default."
        ),
    )
    parser.add_argument(
        "--same_element_max_sources",
        type=int,
        default=1,
        help="Maximum labels selected for one target element.",
    )
    parser.add_argument(
        "--same_element_combo_limit",
        type=int,
        default=5000,
        help="Maximum per-element source subsets retained before the Cartesian product.",
    )
    parser.add_argument("--candidate_limit", type=int, default=5000)
    parser.add_argument("--max_products", type=int, default=200000)
    parser.add_argument("--length_weight", type=float, default=1.0)
    parser.add_argument("--accessory_weight", type=float, default=1.0)
    parser.add_argument("--family_pool_weight", type=float, default=2.0)
    parser.add_argument(
        "--zero_shot_per_element", type=int, default=0,
        help="Reserve this many full-vocabulary, train-unseen labels for each target cation.",
    )
    parser.add_argument(
        "--zero_shot_weight", type=float, default=1.0,
        help="Pseudo-count used to score chemistry-matched train-unseen labels.",
    )
    parser.add_argument(
        "--group_balance_power", type=float, default=0.0,
        help="Inverse training formula/material-group frequency exponent for candidate priors.",
    )
    parser.add_argument(
        "--query_families", default="",
        help="Optional comma-separated query families; other rows are emitted with empty candidates.",
    )
    parser.add_argument(
        "--query_anion_signature",
        default="",
        help="Optional exact sorted anion signature, for example C+O.",
    )
    parser.add_argument("--query_source_dataset", default="")
    args = parser.parse_args()
    input_dir = Path(args.input_dir).resolve()
    split_names = tuple(dict.fromkeys(("train", args.split)))
    packs = {split: np.load(input_dir / f"{split}.npz", allow_pickle=True) for split in split_names}
    meta = {split: pd.read_csv(input_dir / f"{split}_meta.csv", low_memory=False) for split in split_names}
    names = [str(value) for value in json.loads((input_dir / "precursor_names.json").read_text())]
    name_to_id = {name: index for index, name in enumerate(names)}
    label_all_elements = [set(ELEMENT_PATTERN.findall(name)) for name in names]
    label_elements = [{value for value in elements if value not in {"H", "O"}} for elements in label_all_elements]
    train_targets = [json_set(value) for value in meta["train"]["target_cation_elements"]]
    query_targets_elements = [json_set(value) for value in meta[args.split]["target_cation_elements"]]
    train_anions = [json_set(value) for value in meta["train"]["target_anion_elements"]]
    query_anions = [json_set(value) for value in meta[args.split]["target_anion_elements"]]
    train_sets = [tuple(np.flatnonzero(row > 0.5).tolist()) for row in packs["train"]["y_multi_hot"]]
    query_sets = [tuple(np.flatnonzero(row > 0.5).tolist()) for row in packs[args.split]["y_multi_hot"]]
    selected_query_families = {
        value.strip() for value in str(args.query_families).split(",") if value.strip()
    }
    selected_query_anions = "+".join(
        sorted(
            value.strip()
            for value in str(args.query_anion_signature).split("+")
            if value.strip()
        )
    )
    query_family_values = meta[args.split]["family_signature_primary"].fillna("UNK").astype(str).to_numpy()
    by_source_element: Dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    by_element: Dict[str, Counter[int]] = defaultdict(Counter)
    by_source_element_anions: Dict[tuple[str, str, str], Counter[int]] = defaultdict(Counter)
    by_element_anions: Dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    by_route_element: Dict[tuple[str, str, str, str], Counter[int]] = defaultdict(Counter)
    by_cations_anions_element: Dict[tuple[str, str, str], Counter[int]] = defaultdict(Counter)
    by_family_template: Dict[str, Counter[str]] = defaultdict(Counter)
    by_source_family_template: Dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    accessory_by_source: Dict[str, Counter[int]] = defaultdict(Counter)
    accessory_global: Counter[int] = Counter()
    accessory_by_source_anions: Dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    accessory_by_anions: Dict[str, Counter[int]] = defaultdict(Counter)
    accessory_by_route: Dict[tuple[str, str, str], Counter[int]] = defaultdict(Counter)
    accessory_by_cations_anions: Dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    bundle_global: Counter[SetKey] = Counter()
    bundle_by_source: Dict[str, Counter[SetKey]] = defaultdict(Counter)
    bundle_by_anions: Dict[str, Counter[SetKey]] = defaultdict(Counter)
    bundle_by_source_anions: Dict[tuple[str, str], Counter[SetKey]] = defaultdict(Counter)
    bundle_by_cations_anions: Dict[tuple[str, str], Counter[SetKey]] = defaultdict(Counter)
    bundle_by_route: Dict[tuple[str, str, str], Counter[SetKey]] = defaultdict(Counter)
    length_by_source_targets: Dict[tuple[str, int], Counter[int]] = defaultdict(Counter)
    train_label_frequency: Counter[int] = Counter(
        label for labels in train_sets for label in labels
    )
    group_column = (
        meta["train"]["family_group_key"].fillna("UNK").astype(str)
        if "family_group_key" in meta["train"]
        else meta["train"]["formula"].fillna("UNK").astype(str)
    )
    group_counts = group_column.value_counts().to_dict()
    train_row_weights = np.asarray(
        [float(group_counts[value]) ** -float(args.group_balance_power) for value in group_column],
        dtype=np.float64,
    )
    train_row_weights /= max(float(train_row_weights.mean()), 1e-12)
    target_covered = 0
    all_labels_touch_target = 0
    for index, (target_elements, target_anions, labels) in enumerate(zip(train_targets, train_anions, train_sets)):
        row_weight = float(train_row_weights[index])
        source = str(meta["train"].iloc[index].get("source_dataset", ""))
        anion_key = "+".join(sorted(target_anions)) or "NONE"
        cation_key = "+".join(sorted(target_elements)) or "NONE"
        length_by_source_targets[(source, len(target_elements))][len(labels)] += row_weight
        accessory_bundle = tuple(sorted(label for label in labels if not (label_elements[label] & target_elements)))
        if accessory_bundle:
            bundle_global[accessory_bundle] += row_weight
            bundle_by_source[source][accessory_bundle] += row_weight
            bundle_by_anions[anion_key][accessory_bundle] += row_weight
            bundle_by_source_anions[(source, anion_key)][accessory_bundle] += row_weight
            bundle_by_cations_anions[(cation_key, anion_key)][accessory_bundle] += row_weight
            bundle_by_route[(source, cation_key, anion_key)][accessory_bundle] += row_weight
        covered: Set[str] = set()
        touches = True
        for label in labels:
            overlap = label_elements[label] & target_elements
            covered.update(overlap)
            touches = touches and bool(overlap)
            for element in overlap:
                by_source_element[(source, element)][label] += row_weight
                by_element[element][label] += row_weight
                by_source_element_anions[(source, element, anion_key)][label] += row_weight
                by_element_anions[(element, anion_key)][label] += row_weight
                by_route_element[(source, cation_key, anion_key, element)][label] += row_weight
                by_cations_anions_element[(cation_key, anion_key, element)][label] += row_weight
                group = element_group_bucket(Element(element))
                placeholder = f"{{{group}}}"
                template = ELEMENT_PATTERN.sub(
                    lambda match: placeholder if match.group(0) == element else match.group(0),
                    names[label],
                )
                by_family_template[group][template] += row_weight
                by_source_family_template[(source, group)][template] += row_weight
            if not overlap:
                accessory_by_source[source][label] += row_weight
                accessory_global[label] += row_weight
                accessory_by_source_anions[(source, anion_key)][label] += row_weight
                accessory_by_anions[anion_key][label] += row_weight
                accessory_by_route[(source, cation_key, anion_key)][label] += row_weight
                accessory_by_cations_anions[(cation_key, anion_key)][label] += row_weight
        target_covered += int(target_elements <= covered)
        all_labels_touch_target += int(touches)

    all_candidates: List[List[SetKey]] = []
    all_scores: List[List[float]] = []
    # Candidate generation depends on query chemistry and source collection, not
    # on the validation label.  Caching repeated formula groups (for example the
    # many literature routes of one held-out material) avoids recomputing a
    # potentially large Cartesian expansion and cannot leak the query target.
    query_cache: Dict[tuple[str, tuple[str, ...], tuple[str, ...]], tuple[List[SetKey], List[float]]] = {}
    for index, (target_elements, target_anions) in enumerate(zip(query_targets_elements, query_anions)):
        query_source = str(meta[args.split].iloc[index].get("source_dataset", ""))
        query_anion_key = "+".join(sorted(target_anions))
        if (
            (selected_query_families and str(query_family_values[index]) not in selected_query_families)
            or (selected_query_anions and query_anion_key != selected_query_anions)
            or (
                str(args.query_source_dataset).strip()
                and query_source != str(args.query_source_dataset).strip()
            )
        ):
            all_candidates.append([])
            all_scores.append([])
            continue
        source = query_source
        cache_key = (source, tuple(sorted(target_elements)), tuple(sorted(target_anions)))
        cached = query_cache.get(cache_key)
        if cached is not None:
            all_candidates.append(list(cached[0]))
            all_scores.append(list(cached[1]))
            continue
        anion_key = "+".join(sorted(target_anions)) or "NONE"
        cation_key = "+".join(sorted(target_elements)) or "NONE"
        choice_lists: List[List[tuple[SetKey, float]]] = []
        for element in sorted(target_elements):
            merged = Counter(by_element.get(element, Counter()))
            group = element_group_bucket(Element(element))
            placeholder = f"{{{group}}}"
            pooled_templates = Counter(by_family_template.get(group, Counter()))
            for template, count in by_source_family_template.get((source, group), Counter()).items():
                pooled_templates[template] += 2 * count
            for template, count in pooled_templates.items():
                instantiated = template.replace(placeholder, element)
                label = name_to_id.get(instantiated)
                if label is not None:
                    merged[label] += float(args.family_pool_weight) * count
            for label, count in by_source_element.get((source, element), Counter()).items():
                merged[label] += 2 * count
            for label, count in by_element_anions.get((element, anion_key), Counter()).items():
                merged[label] += 3 * count
            for label, count in by_source_element_anions.get((source, element, anion_key), Counter()).items():
                merged[label] += 6 * count
            for label, count in by_cations_anions_element.get((cation_key, anion_key, element), Counter()).items():
                merged[label] += 8 * count
            for label, count in by_route_element.get((source, cation_key, anion_key, element), Counter()).items():
                merged[label] += 12 * count
            ranked = sorted(merged.items(), key=lambda item: (-item[1], item[0]))[: args.per_element]
            if int(args.zero_shot_per_element) > 0:
                # The vocabulary is frozen from the complete database, so a
                # formula-disjoint split can contain chemically valid labels
                # absent from train.  Reserve a small, explicitly chemistry-
                # matched tail instead of making those labels unreachable.
                zero_shot = [
                    label for label, elements in enumerate(label_elements)
                    if element in elements and train_label_frequency.get(label, 0) == 0
                ]
                zero_shot.sort(
                    key=lambda label: (
                        -len(label_all_elements[label] & target_anions),
                        len(label_all_elements[label] - {element, "H", "O"} - target_anions),
                        len(label_all_elements[label]),
                        len(names[label]),
                        names[label],
                    )
                )
                ranked_ids = {label for label, _ in ranked}
                ranked.extend(
                    (label, float(args.zero_shot_weight))
                    for label in zero_shot[: int(args.zero_shot_per_element)]
                    if label not in ranked_ids
                )
            total = float(sum(merged.values()) + max(1, len(merged)))
            singleton_choices = [
                ((label,), math.log((count + 1.0) / total)) for label, count in ranked
            ]
            element_choices = list(singleton_choices)
            combo_pool_size = min(int(args.same_element_combo_pool), len(singleton_choices))
            max_sources = max(1, int(args.same_element_max_sources))
            if combo_pool_size > 0 and max_sources > 1:
                combo_pool = singleton_choices[:combo_pool_size]
                multi_choices: List[tuple[SetKey, float]] = []
                for order in range(2, min(max_sources, combo_pool_size) + 1):
                    for combination in itertools.combinations(combo_pool, order):
                        key = tuple(sorted(label for labels, _ in combination for label in labels))
                        # The average log probability prevents the number of
                        # same-element sources from being penalized twice; the
                        # learned route-length prior below still controls size.
                        score = float(np.mean([value for _, value in combination]))
                        multi_choices.append((key, score))
                multi_choices.sort(key=lambda item: (-item[1], item[0]))
                element_choices.extend(multi_choices[: int(args.same_element_combo_limit)])
            choice_lists.append(element_choices)
        length_counts = length_by_source_targets.get((source, len(target_elements)), Counter())
        length_total = float(sum(length_counts.values()) + 6)

        def length_log_probability(length: int) -> float:
            return math.log((length_counts.get(length, 0) + 1.0) / length_total)

        scored: Dict[SetKey, float] = {}
        if choice_lists and all(choice_lists):
            for product_index, combination in enumerate(itertools.product(*choice_lists)):
                if product_index >= args.max_products:
                    break
                key = tuple(sorted({label for labels, _ in combination for label in labels}))
                source_score = float(sum(value for _, value in combination) / len(combination))
                score = source_score + args.length_weight * length_log_probability(len(key))
                scored[key] = max(scored.get(key, -math.inf), score)
        accessories = Counter(accessory_global)
        for label, count in accessory_by_source.get(source, Counter()).items():
            accessories[label] += 2 * count
        for label, count in accessory_by_anions.get(anion_key, Counter()).items():
            accessories[label] += 3 * count
        for label, count in accessory_by_source_anions.get((source, anion_key), Counter()).items():
            accessories[label] += 6 * count
        for label, count in accessory_by_cations_anions.get((cation_key, anion_key), Counter()).items():
            accessories[label] += 8 * count
        for label, count in accessory_by_route.get((source, cation_key, anion_key), Counter()).items():
            accessories[label] += 12 * count
        bundles = Counter(bundle_global)
        for bundle, count in bundle_by_source.get(source, Counter()).items():
            bundles[bundle] += 2 * count
        for bundle, count in bundle_by_anions.get(anion_key, Counter()).items():
            bundles[bundle] += 3 * count
        for bundle, count in bundle_by_source_anions.get((source, anion_key), Counter()).items():
            bundles[bundle] += 6 * count
        for bundle, count in bundle_by_cations_anions.get((cation_key, anion_key), Counter()).items():
            bundles[bundle] += 8 * count
        for bundle, count in bundle_by_route.get((source, cation_key, anion_key), Counter()).items():
            bundles[bundle] += 12 * count
        # Keep strong singleton fallbacks alongside observed multi-reagent bundles.
        for label, count in accessories.items():
            bundles[(label,)] += count
        bundle_total = float(sum(bundles.values()) + max(1, len(bundles)))
        bundle_choices = [
            (bundle, math.log((count + 1.0) / bundle_total))
            for bundle, count in sorted(bundles.items(), key=lambda item: (-item[1], item[0]))[: max(30, args.accessories * 2)]
        ]
        combo_pool_size = min(int(args.accessory_combo_pool), len(accessories))
        max_accessory_order = max(1, int(args.accessory_combo_order))
        if combo_pool_size > 0 and max_accessory_order > 1:
            accessory_total = float(sum(accessories.values()) + max(1, len(accessories)))
            # A conditionally strong accessory can be globally rare and fall
            # outside the merged top-N list.  Preserve labels already present
            # in a high-ranked observed bundle, then combine them with the
            # broad marginal pool.  This is still train-only: both the bundle
            # and marginal counts were built exclusively from train rows.
            marginal_combo_label_ids = [
                label
                for label, _ in sorted(
                    accessories.items(), key=lambda item: (-item[1], item[0])
                )[:combo_pool_size]
            ]
            combo_label_ids = list(marginal_combo_label_ids)
            combo_label_seen = set(combo_label_ids)
            observed_bundle_label_ids: List[int] = []
            observed_bundle_label_seen: Set[int] = set()
            for bundle, _ in bundle_choices:
                for label in bundle:
                    if label not in observed_bundle_label_seen:
                        observed_bundle_label_seen.add(label)
                        observed_bundle_label_ids.append(label)
                    if label not in combo_label_seen:
                        combo_label_seen.add(label)
                        combo_label_ids.append(label)
            singleton_pool = [
                (label, math.log((accessories.get(label, 0.0) + 1.0) / accessory_total))
                for label in combo_label_ids
            ]
            generated_bundles: List[tuple[SetKey, float]] = []
            for order in range(2, min(max_accessory_order, combo_pool_size) + 1):
                for combination in itertools.combinations(singleton_pool, order):
                    key = tuple(sorted(label for label, _ in combination))
                    score = float(np.mean([value for _, value in combination]))
                    generated_bundles.append((key, score))
            generated_bundles.sort(key=lambda item: (-item[1], item[0]))
            bundle_score_map = {bundle: score for bundle, score in bundle_choices}
            for bundle, score in generated_bundles[: int(args.accessory_combo_limit)]:
                bundle_score_map[bundle] = max(bundle_score_map.get(bundle, -math.inf), score)
            # Score truncation must not erase the cross between a broad
            # marginal accessory and a conditionally strong observed one.
            # Keeping these pairs fixes a common route-support failure while
            # remaining bounded by roughly pool_size × observed_bundle_labels.
            singleton_scores = dict(singleton_pool)
            for left in marginal_combo_label_ids:
                for right in observed_bundle_label_ids:
                    if left == right:
                        continue
                    bundle = tuple(sorted((left, right)))
                    score = float(
                        np.mean([singleton_scores[left], singleton_scores[right]])
                    )
                    bundle_score_map[bundle] = max(
                        bundle_score_map.get(bundle, -math.inf), score
                    )
            bundle_choices = sorted(
                bundle_score_map.items(), key=lambda item: (-item[1], item[0])
            )
        base_rows = sorted(scored.items(), key=lambda item: (-item[1], item[0]))[
            : int(args.accessory_base_rows)
        ]
        for base_key, base_score in base_rows:
            # Remove the base length prior before applying the new candidate's
            # length prior after adding an accessory.
            base_source_score = base_score - args.length_weight * length_log_probability(len(base_key))
            for bundle, accessory_log_probability in bundle_choices:
                key = tuple(sorted({*base_key, *bundle}))
                if key == base_key:
                    continue
                score = (
                    base_source_score
                    + args.accessory_weight * accessory_log_probability
                    + args.length_weight * length_log_probability(len(key))
                )
                scored[key] = max(scored.get(key, -math.inf), score)
        ordered = sorted(scored.items(), key=lambda item: (-item[1], item[0]))[: args.candidate_limit]
        row_candidates = [key for key, _ in ordered]
        row_scores = [score for _, score in ordered]
        query_cache[cache_key] = (row_candidates, row_scores)
        all_candidates.append(list(row_candidates))
        all_scores.append(list(row_scores))
    report = {
        "protocol": f"{args.split}_formula_disjoint_element_source_cartesian",
        "config": vars(args),
        "training_chemistry_audit": {
            "target_cations_covered_by_true_precursors": target_covered / max(1, len(train_sets)),
            "all_true_precursors_touch_target_cation": all_labels_touch_target / max(1, len(train_sets)),
            "group_balance_power": float(args.group_balance_power),
            "sampling_weight_min": float(train_row_weights.min()),
            "sampling_weight_max": float(train_row_weights.max()),
            "query_sets_with_any_train_unseen_label": float(
                np.mean([
                    any(train_label_frequency.get(label, 0) == 0 for label in labels)
                    for labels in query_sets
                ])
            ),
        },
        args.split: exact_metrics(query_sets, all_candidates),
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with Path(args.output_candidates_jsonl).resolve().open("w", encoding="utf-8") as handle:
        for index, (candidates, scores) in enumerate(zip(all_candidates, all_scores)):
            handle.write(json.dumps({"row_index": index, "candidate_label_ids": [list(value) for value in candidates], "scores": scores}) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
