#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


TEXT_COLUMNS = [
    "synthesis_text",
    "synthesis_route_text",
    "synthesis_route_text_cn",
    "paragraph",
    "procedure",
    "reaction_string",
    "route_description",
]


def scan_csv(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    cols = list(pd.read_csv(path, nrows=0).columns)
    found = [c for c in TEXT_COLUMNS if c in cols]
    return {"path": str(path), "exists": True, "columns": cols, "text_columns": found}


def main() -> None:
    ap = argparse.ArgumentParser(description="Check route text feature availability for TF-IDF/keyword experiments.")
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--output_dir", default="outputs/auto_improve/synpred_auto_v1_20260612/route_text_features")
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    candidates: List[Path] = [
        root / "data/interim/generative/stage3_condition_targets_v3_20260610/train.csv",
        root / "data/interim/generative/stage3_condition_targets_v3_20260610/val.csv",
        root / "data/interim/generative/stage2_setpred_dataset/descriptor/route_method_stratified_canonical_v4_20260610_relaxed_only/train.csv",
    ]
    scans = [scan_csv(p) for p in candidates]
    found_any = any(s.get("text_columns") for s in scans)
    obj = {"experiment": "route_text_features", "status": "available" if found_any else "no_text_columns_found", "scans": scans}
    (out / "route_text_feature_availability.json").write_text(json.dumps(obj, indent=2), encoding="utf-8")
    (out / "route_text_feature_availability_report.md").write_text(
        "# Route Text Feature Availability\n\n"
        f"Status: `{obj['status']}`.\n\n"
        "No model is trained when required text columns are absent.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(out)}, indent=2))


if __name__ == "__main__":
    main()

