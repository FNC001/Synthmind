#!/usr/bin/env python3
"""
Build task-specific refined sets from the common structdesc refined outputs.

Default policy:
- Stage2 precursor prediction uses only the new codex dataset rows with valid
  structure paths.
- Stage3 condition prediction uses both old and new datasets with valid
  condition supervision and precursor inputs.
- Rows that have condition information but no main precursor are archived for
  audit, because the current Stage3 pipeline predicts conditions from precursor
  inputs and cannot use them directly.
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
    "structdesc_refined_task_new_stage2_all_stage3_20260609"
)
DEFAULT_NEW_SOURCE = "codex_final_database_20260608_clean"


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
    if not value:
        return False
    return Path(str(value)).exists()


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


def is_new_source(row: Dict[str, Any], new_source: str) -> bool:
    return row.get("source_dataset") == new_source


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    source = Counter(str(r.get("source_dataset")) for r in rows)
    cond_source = Counter(str(r.get("condition_source")) for r in rows if r.get("condition_source") is not None)
    return {
        "n_rows": len(rows),
        "n_with_poscar_path": sum(1 for r in rows if r.get("poscar_path")),
        "n_with_existing_poscar_path": sum(1 for r in rows if path_exists(r.get("poscar_path"))),
        "n_with_main_precursors": sum(1 for r in rows if has_main_precursors(r)),
        "n_with_any_condition": sum(1 for r in rows if has_any_condition(r)),
        "source_dataset_counts": dict(source.most_common()),
        "condition_source_counts": dict(cond_source.most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create task-specific refined sets for Stage2 and Stage3.")
    parser.add_argument("--refined_dir", type=Path, default=DEFAULT_REFINED_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--new_source", type=str, default=DEFAULT_NEW_SOURCE)
    args = parser.parse_args()

    refined_dir = args.refined_dir
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    stage2_gold_all = read_jsonl(refined_dir / "stage2_gold.jsonl")
    stage2_relaxed_all = read_jsonl(refined_dir / "stage2_train_relaxed.jsonl")
    stage3_gold_all = read_jsonl(refined_dir / "stage3_gold.jsonl")
    stage3_relaxed_all = read_jsonl(refined_dir / "stage3_train_relaxed.jsonl")
    dropped_all = read_jsonl(refined_dir / "dropped_records.jsonl")

    stage2_gold = [
        r for r in stage2_gold_all if is_new_source(r, args.new_source) and path_exists(r.get("poscar_path"))
    ]
    stage2_relaxed = [
        r for r in stage2_relaxed_all if is_new_source(r, args.new_source) and path_exists(r.get("poscar_path"))
    ]

    stage3_gold = [r for r in stage3_gold_all if has_main_precursors(r) and has_any_condition(r)]
    stage3_relaxed = [r for r in stage3_relaxed_all if has_main_precursors(r) and has_any_condition(r)]

    no_precursor_condition_audit = [
        r
        for r in dropped_all
        if (not has_main_precursors(r))
        and has_any_condition(r)
        and ("no_main_precursors" in (r.get("_refine_severe_reasons") or []))
    ]

    write_counts = {
        "stage2_gold.jsonl": write_jsonl(out_dir / "stage2_gold.jsonl", stage2_gold),
        "stage2_train_relaxed.jsonl": write_jsonl(out_dir / "stage2_train_relaxed.jsonl", stage2_relaxed),
        "stage3_gold.jsonl": write_jsonl(out_dir / "stage3_gold.jsonl", stage3_gold),
        "stage3_train_relaxed.jsonl": write_jsonl(out_dir / "stage3_train_relaxed.jsonl", stage3_relaxed),
        "no_precursor_condition_audit.jsonl": write_jsonl(
            out_dir / "no_precursor_condition_audit.jsonl", no_precursor_condition_audit
        ),
    }

    summary = {
        "policy": {
            "new_source_for_stage2": args.new_source,
            "stage2": "new_source rows only, requiring an existing poscar_path",
            "stage3": "old and new source rows, requiring main_precursors and condition supervision",
            "no_precursor_condition_audit": (
                "condition-bearing rows without main_precursors; archived because current Stage3 uses precursor inputs"
            ),
        },
        "input_counts": {
            "stage2_gold_all": len(stage2_gold_all),
            "stage2_relaxed_all": len(stage2_relaxed_all),
            "stage3_gold_all": len(stage3_gold_all),
            "stage3_relaxed_all": len(stage3_relaxed_all),
            "dropped_all": len(dropped_all),
        },
        "outputs": {
            "stage2_gold": summarize(stage2_gold),
            "stage2_train_relaxed": summarize(stage2_relaxed),
            "stage3_gold": summarize(stage3_gold),
            "stage3_train_relaxed": summarize(stage3_relaxed),
            "no_precursor_condition_audit": summarize(no_precursor_condition_audit),
        },
        "written_rows": write_counts,
    }
    write_json(out_dir / "task_specific_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
