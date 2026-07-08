#!/usr/bin/env python3
"""
Build a unified route-supervised refined set.

This set is intended for end-to-end route evaluation: Stage2 and Stage3 see
the same train/val/test rows, and every row has structure, main precursors,
and at least one synthesis-condition target.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_REFINED_DIR = Path(
    "/Users/lihonglin/Desktop/Syn_DP/SynPred/data/interim/refined/structdesc_refined_merged_20260609_with_structures"
)
DEFAULT_OUTPUT_DIR = Path(
    "/Users/lihonglin/Desktop/Syn_DP/SynPred/data/interim/refined/"
    "structdesc_refined_route_unified_20260609"
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def path_exists(value: Optional[str]) -> bool:
    return bool(value) and Path(str(value)).exists()


def has_main_precursors(row: Dict[str, Any]) -> bool:
    return bool(row.get("main_precursors") or [])


def has_any_condition(row: Dict[str, Any]) -> bool:
    keys = [
        "temperature_c",
        "time_h",
        "atmosphere",
        "solvent",
        "temperature_c_op",
        "time_h_op",
        "atmosphere_op",
        "solvent_op",
        "temperature_c_fallback",
        "time_h_fallback",
        "atmosphere_fallback",
        "solvent_fallback",
    ]
    return any(row.get(k) is not None for k in keys)


def route_ready(row: Dict[str, Any]) -> bool:
    return path_exists(row.get("poscar_path")) and has_main_precursors(row) and has_any_condition(row)


def dedupe_by_id(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for row in rows:
        key = str(row.get("id") or row.get("synth_uid") or "")
        if not key:
            key = json.dumps(
                [row.get("source_dataset"), row.get("material_id"), row.get("reaction_string")],
                ensure_ascii=False,
                sort_keys=True,
            )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "n_rows": len(rows),
        "n_with_existing_poscar_path": sum(1 for r in rows if path_exists(r.get("poscar_path"))),
        "n_with_main_precursors": sum(1 for r in rows if has_main_precursors(r)),
        "n_with_any_condition": sum(1 for r in rows if has_any_condition(r)),
        "source_dataset_counts": dict(Counter(str(r.get("source_dataset")) for r in rows).most_common()),
        "condition_source_counts": dict(
            Counter(str(r.get("condition_source")) for r in rows if r.get("condition_source") is not None).most_common()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create unified route-supervised refined sets.")
    parser.add_argument("--refined_dir", type=Path, default=DEFAULT_REFINED_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    refined_dir = args.refined_dir
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    stage3_gold_all = read_jsonl(refined_dir / "stage3_gold.jsonl")
    stage3_relaxed_all = read_jsonl(refined_dir / "stage3_train_relaxed.jsonl")

    route_gold = dedupe_by_id(r for r in stage3_gold_all if route_ready(r))
    route_relaxed = dedupe_by_id(r for r in stage3_relaxed_all if route_ready(r))

    written = {
        "route_gold.jsonl": write_jsonl(out_dir / "route_gold.jsonl", route_gold),
        "route_train_relaxed.jsonl": write_jsonl(out_dir / "route_train_relaxed.jsonl", route_relaxed),
        "stage2_gold.jsonl": write_jsonl(out_dir / "stage2_gold.jsonl", route_gold),
        "stage2_train_relaxed.jsonl": write_jsonl(out_dir / "stage2_train_relaxed.jsonl", route_relaxed),
        "stage3_gold.jsonl": write_jsonl(out_dir / "stage3_gold.jsonl", route_gold),
        "stage3_train_relaxed.jsonl": write_jsonl(out_dir / "stage3_train_relaxed.jsonl", route_relaxed),
    }

    summary = {
        "policy": {
            "purpose": "End-to-end route evaluation with aligned Stage2/Stage3 splits.",
            "row_filter": "existing poscar_path + main_precursors + at least one synthesis condition",
            "stage2_files": "identical to route files",
            "stage3_files": "identical to route files",
        },
        "input_counts": {
            "stage3_gold_all": len(stage3_gold_all),
            "stage3_relaxed_all": len(stage3_relaxed_all),
        },
        "outputs": {
            "route_gold": summarize(route_gold),
            "route_train_relaxed": summarize(route_relaxed),
        },
        "written_rows": written,
    }
    write_json(out_dir / "route_unified_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
