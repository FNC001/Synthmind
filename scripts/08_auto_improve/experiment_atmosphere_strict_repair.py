#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


TEXT_COLS = ["atmosphere", "true_atmosphere", "atmosphere_known_mask", "solvent"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit atmosphere/solvent strict-comparable repair opportunity.")
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--output_dir", default="outputs/auto_improve/synpred_auto_v1_20260612/atmosphere_strict_repair")
    ap.add_argument("--candidate_csv", default="outputs/evaluation/stage3_condition_calibration_v3_final_20260612/test_condition_candidates_calibrated.csv")
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    csv_path = root / args.candidate_csv
    summary: Dict[str, Any] = {"experiment": "atmosphere_strict_repair", "candidate_csv": str(csv_path), "status": "availability_only"}
    if csv_path.exists():
        df = pd.read_csv(csv_path, usecols=lambda c: c in set(TEXT_COLS))
        summary["rows"] = int(len(df))
        for col in df.columns:
            summary[f"{col}_missing_rate"] = float(df[col].isna().mean())
            summary[f"{col}_n_unique"] = int(df[col].nunique(dropna=True))
    else:
        summary["status"] = "missing_candidate_csv"
    (out / "atmosphere_strict_repair_report.md").write_text(
        "# Atmosphere Strict Repair Audit\n\n"
        "This lightweight step audits availability and missingness; it does not alter default labels.\n",
        encoding="utf-8",
    )
    (out / "atmosphere_strict_repair_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(out)}, indent=2))


if __name__ == "__main__":
    main()

