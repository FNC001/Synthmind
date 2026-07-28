#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.family.evaluate_stage2_candidate_fusion import (  # noqa: E402
    fuse_row,
    fuse_row_topk,
    load_source,
)


SetKey = Tuple[int, ...]
ELEMENT_PATTERN = re.compile(r"[A-Z][a-z]?")
ANION_GROUP_BY_SYMBOL = {
    "H": "G01",
    "B": "G13",
    "C": "G14", "Si": "G14", "Ge": "G14",
    "N": "G15", "P": "G15", "As": "G15", "Sb": "G15",
    "O": "G16", "S": "G16", "Se": "G16", "Te": "G16",
    "F": "G17", "Cl": "G17", "Br": "G17", "I": "G17",
}


def parse_named_source(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"expert must be NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    return name.strip(), path.strip()


def parse_count_bins(value: str) -> List[int]:
    bins = sorted({int(item.strip()) for item in str(value).split(",") if item.strip()})
    if any(item < 1 for item in bins):
        raise ValueError("formula element-count bins must be positive integers")
    return bins


def formula_element_count_bucket(formula: str, bins: Sequence[int]) -> str:
    """Observable formula-complexity bucket used by the frozen expert gate."""
    count = len(set(ELEMENT_PATTERN.findall(str(formula))))
    for boundary in bins:
        if count <= int(boundary):
            return f"E<={int(boundary)}"
    return f"E>{int(bins[-1])}" if bins else "E_ALL"


def anion_group_signature(value: Any) -> str:
    """Map stored target anions to periodic groups without using stoichiometry."""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = ELEMENT_PATTERN.findall(value)
    elif isinstance(value, (list, tuple)):
        parsed = value
    else:
        parsed = []
    groups = sorted({ANION_GROUP_BY_SYMBOL.get(str(symbol), "A_OTHER") for symbol in parsed})
    return "+".join(groups) if groups else "A_NONE"


def hit_rate(
    targets: Sequence[SetKey], rows: Sequence[Sequence[SetKey]], indices: np.ndarray, k: int
) -> float:
    return float(np.mean([targets[int(index)] in set(rows[int(index)][:k]) for index in indices]))


def metrics(
    targets: Sequence[SetKey], rows: Sequence[Sequence[SetKey]], indices: np.ndarray
) -> Dict[str, float]:
    return {f"exact_hit@{k}": hit_rate(targets, rows, indices, k) for k in (1, 3, 5, 10, 20, 50, 100)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-frozen routing among several global rankers by family.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--base_candidates", required=True)
    parser.add_argument("--expert", action="append", default=[], help="Repeat NAME=candidates.jsonl")
    parser.add_argument("--source_limit", type=int, default=100)
    parser.add_argument("--min_family_rows", type=int, default=20)
    parser.add_argument("--min_gain_hits", type=int, default=1)
    parser.add_argument(
        "--max_top1_loss_hits", type=int, default=-1,
        help="Optional per-route guard on lost Top-1 hits; -1 disables the guard.",
    )
    parser.add_argument(
        "--max_expert_combo", type=int, choices=(1, 2), default=1,
        help="Allow base-plus-two-expert RRF choices during validation routing.",
    )
    parser.add_argument(
        "--formula_element_count_bins", default="",
        help="Optional comma-separated observable formula-complexity bins, e.g. 2,3,4.",
    )
    parser.add_argument(
        "--stratify_anion_groups", action="store_true",
        help="Also route by target-anion periodic groups (O/S and Cl/Br remain equivalent).",
    )
    parser.add_argument("--routing_json", default="", help="Validation report whose routing must be reused on test")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    pack = np.load(input_dir / f"{args.split}.npz", allow_pickle=True)
    targets = [tuple(np.flatnonzero(row > 0.5).tolist()) for row in pack["y_multi_hot"]]
    count_bins = parse_count_bins(args.formula_element_count_bins)
    meta_columns = [
        "family_signature_primary",
        *(["formula"] if count_bins else []),
        *(["target_anion_elements"] if args.stratify_anion_groups else []),
    ]
    metadata = pd.read_csv(input_dir / f"{args.split}_meta.csv", usecols=meta_columns)
    family_values = metadata["family_signature_primary"].fillna("UNK").astype(str).to_numpy()
    if count_bins:
        formula_buckets = np.asarray(
            [
                formula_element_count_bucket(value, count_bins)
                for value in metadata["formula"].fillna("").astype(str)
            ],
            dtype=object,
        )
        family_values = np.asarray(
            [f"{family}|{bucket}" for family, bucket in zip(family_values, formula_buckets)],
            dtype=object,
        )
    if args.stratify_anion_groups:
        anion_signatures = np.asarray(
            [anion_group_signature(value) for value in metadata["target_anion_elements"]],
            dtype=object,
        )
        family_values = np.asarray(
            [f"{family}|A:{anion}" for family, anion in zip(family_values, anion_signatures)],
            dtype=object,
        )
    base = load_source(args.base_candidates, len(targets), args.source_limit)
    expert_paths = dict(parse_named_source(value) for value in args.expert)
    experts = {
        name: load_source(path, len(targets), args.source_limit) for name, path in expert_paths.items()
    }
    frozen = None
    if args.routing_json:
        frozen = json.loads(Path(args.routing_json).resolve().read_text(encoding="utf-8"))["routing"]
        if args.split != "test":
            raise ValueError("--routing_json is intended for frozen test evaluation")

    routed = [list(row) for row in base]
    routing: Dict[str, Dict[str, Any]] = {}
    for family in sorted(set(family_values)):
        indices = np.flatnonzero(family_values == family)
        base_metric = metrics(targets, base, indices)
        if frozen is not None:
            if family not in frozen:
                continue
            selected = frozen[family]
            if selected["kind"] == "expert":
                source_name = str(selected["source"])
                expert_rows = experts[source_name]
                selected_rows = [expert_rows[int(index)] for index in indices]
            elif selected["kind"] == "rrf":
                source_name = str(selected["source"])
                expert_rows = experts[source_name]
                selected_rows = [
                    fuse_row(
                        [base[int(index)], expert_rows[int(index)]],
                        [1.0, float(selected["expert_weight"])],
                        float(selected["rrf_constant"]),
                    )
                    for index in indices
                ]
            elif selected["kind"] == "rrf_multi":
                source_names = [str(value) for value in selected["sources"]]
                source_rows = [experts[name] for name in source_names]
                selected_rows = [
                    fuse_row(
                        [base[int(index)], *[rows[int(index)] for rows in source_rows]],
                        [1.0, *[float(value) for value in selected["expert_weights"]]],
                        float(selected["rrf_constant"]),
                    )
                    for index in indices
                ]
            else:
                raise ValueError(f"unsupported frozen route kind: {selected['kind']}")
            selected_metric = {
                f"exact_hit@{k}": float(
                    np.mean([targets[int(index)] in set(row[:k]) for index, row in zip(indices, selected_rows)])
                )
                for k in (1, 3, 5, 10, 20, 50, 100)
            }
            selected_metadata = {
                key: value for key, value in selected.items()
                if not key.startswith("exact_hit@")
                and key not in {"n_rows", "base_exact_hit@10", "gain_hits@10"}
            }
            if "base_exact_hit@10" in selected:
                selected_metadata["validation_base_exact_hit@10"] = selected["base_exact_hit@10"]
            if "gain_hits@10" in selected:
                selected_metadata["validation_gain_hits@10"] = selected["gain_hits@10"]
            routing[family] = {
                **selected_metadata,
                **selected_metric,
                "n_rows": int(len(indices)),
            }
        else:
            if len(indices) < int(args.min_family_rows):
                continue
            # Trial search only needs the first 100 predictions.  Keeping every
            # full fused row for every grid point made deep-candidate routing
            # both quadratic in memory and needlessly expensive to sort.
            choices: List[Dict[str, Any]] = []
            for source_name, expert_rows in experts.items():
                direct_rows = [expert_rows[int(index)] for index in indices]
                direct_metrics = {
                    f"exact_hit@{k}": float(
                        np.mean([targets[int(index)] in set(row[:k]) for index, row in zip(indices, direct_rows)])
                    )
                    for k in (1, 3, 5, 10, 20, 50, 100)
                }
                choices.append({"kind": "expert", "source": source_name, **direct_metrics})
                for constant in (1.0, 5.0, 10.0, 20.0, 50.0, 100.0):
                    for expert_weight in (0.25, 0.5, 1.0, 2.0):
                        values = [
                            fuse_row_topk(
                                [base[int(index)], expert_rows[int(index)]],
                                [1.0, expert_weight], constant, 100,
                            )
                            for index in indices
                        ]
                        current_metrics = {
                            f"exact_hit@{k}": float(
                                np.mean([targets[int(index)] in set(row[:k]) for index, row in zip(indices, values)])
                            )
                            for k in (1, 3, 5, 10, 20, 50, 100)
                        }
                        choices.append({
                            "kind": "rrf", "source": source_name,
                            "rrf_constant": constant, "expert_weight": expert_weight,
                            **current_metrics,
                        })
            if int(args.max_expert_combo) >= 2:
                for (first_name, first_rows), (second_name, second_rows) in itertools.combinations(
                    experts.items(), 2
                ):
                    for constant in (1.0, 5.0, 10.0, 20.0, 50.0, 100.0):
                        for first_weight in (0.25, 0.5, 1.0, 2.0):
                            for second_weight in (0.25, 0.5, 1.0, 2.0):
                                values = [
                                    fuse_row_topk(
                                        [
                                            base[int(index)], first_rows[int(index)],
                                            second_rows[int(index)],
                                        ],
                                        [1.0, first_weight, second_weight],
                                        constant, 100,
                                    )
                                    for index in indices
                                ]
                                current_metrics = {
                                    f"exact_hit@{k}": float(
                                        np.mean([
                                            targets[int(index)] in set(row[:k])
                                            for index, row in zip(indices, values)
                                        ])
                                    )
                                    for k in (1, 3, 5, 10, 20, 50, 100)
                                }
                                choices.append({
                                    "kind": "rrf_multi",
                                    "sources": [first_name, second_name],
                                    "rrf_constant": constant,
                                    "expert_weights": [first_weight, second_weight],
                                    **current_metrics,
                                })
            guarded_choices = choices
            if int(args.max_top1_loss_hits) >= 0:
                base_top1_hits = int(round(base_metric["exact_hit@1"] * len(indices)))
                guarded_choices = [
                    item for item in choices
                    if int(round(item["exact_hit@1"] * len(indices)))
                    >= base_top1_hits - int(args.max_top1_loss_hits)
                ]
            if not guarded_choices:
                continue
            selected = max(
                guarded_choices,
                key=lambda item: (
                    item["exact_hit@10"], item["exact_hit@50"], item["exact_hit@1"]
                ),
            )
            base_hits = int(round(base_metric["exact_hit@10"] * len(indices)))
            selected_hits = int(round(selected["exact_hit@10"] * len(indices)))
            if selected_hits - base_hits < int(args.min_gain_hits):
                continue
            if selected["kind"] == "expert":
                expert_rows = experts[str(selected["source"])]
                selected_rows = [expert_rows[int(index)] for index in indices]
            elif selected["kind"] == "rrf":
                expert_rows = experts[str(selected["source"])]
                selected_rows = [
                    fuse_row(
                        [base[int(index)], expert_rows[int(index)]],
                        [1.0, float(selected["expert_weight"])],
                        float(selected["rrf_constant"]),
                    )
                    for index in indices
                ]
            elif selected["kind"] == "rrf_multi":
                source_rows = [experts[str(name)] for name in selected["sources"]]
                selected_rows = [
                    fuse_row(
                        [base[int(index)], *[rows[int(index)] for rows in source_rows]],
                        [1.0, *[float(value) for value in selected["expert_weights"]]],
                        float(selected["rrf_constant"]),
                    )
                    for index in indices
                ]
            else:
                raise ValueError(f"unsupported selected route kind: {selected['kind']}")
            routing[family] = {
                **selected,
                "n_rows": int(len(indices)),
                "base_exact_hit@10": base_metric["exact_hit@10"],
                "gain_hits@10": int(selected_hits - base_hits),
            }
        for index, row in zip(indices, selected_rows):
            routed[int(index)] = row

    all_indices = np.arange(len(targets))
    report = {
        "protocol": f"{args.split}_formula_disjoint_multi_expert_family_routing",
        "base_candidates": args.base_candidates,
        "expert_paths": expert_paths,
        "source_limit": int(args.source_limit),
        "min_family_rows": int(args.min_family_rows),
        "min_gain_hits": int(args.min_gain_hits),
        "max_top1_loss_hits": int(args.max_top1_loss_hits),
        "max_expert_combo": int(args.max_expert_combo),
        "formula_element_count_bins": count_bins,
        "stratify_anion_groups": bool(args.stratify_anion_groups),
        "routing": routing,
        "overall": metrics(targets, routed, all_indices),
    }
    Path(args.output_json).resolve().write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with Path(args.output_candidates_jsonl).resolve().open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(routed):
            handle.write(json.dumps({"row_index": row_index, "candidate_label_ids": [list(key) for key in row]}) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
