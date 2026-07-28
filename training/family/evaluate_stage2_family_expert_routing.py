#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.family.evaluate_stage2_candidate_fusion import fuse_row, load_source


SetKey = Tuple[int, ...]


def hit_rate(targets: Sequence[SetKey], rows: Sequence[Sequence[SetKey]], indices: np.ndarray, k: int) -> float:
    if not len(indices):
        return 0.0
    return float(np.mean([targets[int(index)] in set(rows[int(index)][:k]) for index in indices]))


def metrics(targets: Sequence[SetKey], rows: Sequence[Sequence[SetKey]], indices: np.ndarray) -> Dict[str, float]:
    return {f"exact_hit@{k}": hit_rate(targets, rows, indices, k) for k in (1, 3, 5, 10, 20, 50, 100)}


def parse_expert(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"expert must be FAMILY=PATH, got {value!r}")
    family, path = value.split("=", 1)
    return family.strip(), path.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-calibrated routing between a global ranker and family experts.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--base_candidates", required=True)
    parser.add_argument("--expert", action="append", default=[], help="Repeat FAMILY=expert_candidates.jsonl")
    parser.add_argument("--source_limit", type=int, default=1000)
    parser.add_argument("--routing_json", default="", help="Frozen validation routing for test evaluation")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    pack = np.load(input_dir / f"{args.split}.npz", allow_pickle=True)
    targets = [tuple(np.flatnonzero(row > 0.5).tolist()) for row in pack["y_multi_hot"]]
    family_values = pd.read_csv(input_dir / f"{args.split}_meta.csv", usecols=["family_signature_primary"])[
        "family_signature_primary"
    ].fillna("UNK").astype(str).to_numpy()
    base = load_source(args.base_candidates, len(targets), args.source_limit)
    expert_paths = dict(parse_expert(value) for value in args.expert)
    experts = {family: load_source(path, len(targets), args.source_limit) for family, path in expert_paths.items()}

    frozen_routing: Dict[str, Dict[str, Any]] = {}
    if args.routing_json:
        frozen_routing = json.loads(Path(args.routing_json).read_text(encoding="utf-8"))["routing"]

    routed = [list(row) for row in base]
    routing: Dict[str, Dict[str, Any]] = {}
    for family, expert_rows in experts.items():
        indices = np.flatnonzero(family_values == family)
        if not len(indices):
            continue
        choices: List[tuple[Dict[str, Any], List[List[SetKey]]]] = []
        base_metric = metrics(targets, base, indices)
        choices.append(({"kind": "base", **base_metric}, [base[int(index)] for index in indices]))
        expert_metric = metrics(targets, expert_rows, indices)
        choices.append(({"kind": "expert", **expert_metric}, [expert_rows[int(index)] for index in indices]))
        for constant in (1.0, 5.0, 10.0, 20.0, 50.0, 100.0):
            for expert_weight in (0.25, 0.5, 1.0, 2.0):
                values = [
                    fuse_row([base[int(index)], expert_rows[int(index)]], [1.0, expert_weight], constant)
                    for index in indices
                ]
                current = {
                    "kind": "rrf", "rrf_constant": constant, "expert_weight": expert_weight,
                    **{f"exact_hit@{k}": float(np.mean([targets[int(index)] in set(row[:k]) for index, row in zip(indices, values)])) for k in (1, 3, 5, 10, 20, 50, 100)},
                }
                choices.append((current, values))

        if frozen_routing:
            selected_spec = frozen_routing[family]
            matches = [item for item in choices if all(item[0].get(key) == value for key, value in selected_spec.items() if key in {"kind", "rrf_constant", "expert_weight"})]
            if not matches:
                raise RuntimeError(f"frozen routing choice not found for {family}: {selected_spec}")
            selected, selected_rows = matches[0]
        else:
            selected, selected_rows = max(choices, key=lambda item: (item[0]["exact_hit@10"], item[0]["exact_hit@50"], item[0]["exact_hit@1"]))
        for index, row in zip(indices, selected_rows):
            routed[int(index)] = row
        routing[family] = {
            **selected,
            "n_rows": int(len(indices)),
            "base_exact_hit@10": base_metric["exact_hit@10"],
            "expert_exact_hit@10": expert_metric["exact_hit@10"],
        }

    all_indices = np.arange(len(targets))
    report = {
        "protocol": f"{args.split}_formula_disjoint_family_expert_routing",
        "base_candidates": args.base_candidates,
        "expert_paths": expert_paths,
        "source_limit": int(args.source_limit),
        "routing": routing,
        "overall": metrics(targets, routed, all_indices),
    }
    Path(args.output_json).resolve().write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with Path(args.output_candidates_jsonl).resolve().open("w", encoding="utf-8") as handle:
        for row_index, values in enumerate(routed):
            handle.write(json.dumps({"row_index": row_index, "candidate_label_ids": [list(value) for value in values]}) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
