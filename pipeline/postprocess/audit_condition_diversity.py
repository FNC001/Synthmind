#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audit Stage3 condition diversity for pipeline_v3.

This is a lightweight reliability/audit layer adapted from the old V45b benchmark branch.

Purpose:
  - detect missing temperature/time
  - detect clipped low temperature/time
  - detect baseline-seed-only condition outputs
  - summarize condition diversity within each inferred case/material

Input:
  final_top_routes_with_condition_confidence.csv
  or any later route table containing temperature_c and time_h

Outputs:
  condition_diversity_audit.csv
  condition_diversity_audit.md
  condition_diversity_audit_summary.json

Claim boundary:
  This is an internal diagnostic and reliability audit, not experimental validation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--output_md", required=True)
    ap.add_argument("--summary_json", required=True)
    ap.add_argument("--top_n", type=int, default=30)
    ap.add_argument("--baseline_temperature_c", type=float, default=830.0)
    ap.add_argument("--baseline_time_h", type=float, default=14.0)
    ap.add_argument("--clip_low_temperature_c", type=float, default=4.01)
    ap.add_argument("--clip_low_time_h", type=float, default=0.1668)
    return ap.parse_args()


def safe_str(x) -> str:
    if x is None:
        return ""
    try:
        if isinstance(x, float) and math.isnan(x):
            return ""
    except Exception:
        pass
    s = str(x)
    if s.lower() == "nan":
        return ""
    return s.strip()


def to_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, str) and not x.strip():
            return default
        v = float(x)
        if math.isnan(v):
            return default
        return v
    except Exception:
        return default


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def warn_condition_row(
    row: pd.Series,
    temp_col: str,
    time_col: str,
    baseline_temperature_c: float,
    baseline_time_h: float,
    clip_low_temperature_c: float,
    clip_low_time_h: float,
) -> str:
    warnings = []

    t = to_float(row.get(temp_col), np.nan)
    h = to_float(row.get(time_col), np.nan)

    if np.isnan(t):
        warnings.append("missing_temperature")
    elif t <= clip_low_temperature_c:
        warnings.append("clip_low_temperature")

    if np.isnan(h):
        warnings.append("missing_time")
    elif h <= clip_low_time_h:
        warnings.append("clip_low_time")

    if not np.isnan(t) and not np.isnan(h):
        if abs(t - baseline_temperature_c) < 1e-8 and abs(h - baseline_time_h) < 1e-8:
            warnings.append("baseline_seed_condition")

    return ";".join(warnings)


def infer_group_col(df: pd.DataFrame) -> str:
    candidates = [
        "infer_name",
        "material_id",
        "case_id",
        "mp_id",
        "target_formula",
        "formula",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return "__all__"


def main() -> None:
    args = parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    summary_json = Path(args.summary_json)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    if df.empty:
        raise RuntimeError(f"Empty input route table: {input_csv}")

    temp_col = pick_col(df, [
        "temperature_c",
        "candidate_temperature_c",
        "v29_attached_temperature_c",
        "condition_temperature_c",
    ])
    time_col = pick_col(df, [
        "time_h",
        "candidate_time_h",
        "v29_attached_time_h",
        "condition_time_h",
    ])

    if temp_col is None or time_col is None:
        raise RuntimeError(
            f"Missing temperature/time columns. "
            f"temperature column found={temp_col}, time column found={time_col}"
        )

    group_col = infer_group_col(df)
    if group_col == "__all__":
        df[group_col] = "all"

    work = df.copy()
    work["_audit_temperature_c"] = pd.to_numeric(work[temp_col], errors="coerce")
    work["_audit_time_h"] = pd.to_numeric(work[time_col], errors="coerce")

    work["condition_diversity_warning"] = work.apply(
        lambda r: warn_condition_row(
            r,
            temp_col=temp_col,
            time_col=time_col,
            baseline_temperature_c=args.baseline_temperature_c,
            baseline_time_h=args.baseline_time_h,
            clip_low_temperature_c=args.clip_low_temperature_c,
            clip_low_time_h=args.clip_low_time_h,
        ),
        axis=1,
    )

    audit_rows = []

    for key, g in work.groupby(group_col, dropna=False):
        temps = pd.to_numeric(g["_audit_temperature_c"], errors="coerce")
        times = pd.to_numeric(g["_audit_time_h"], errors="coerce")

        temp_unique = sorted(set(round(float(x), 6) for x in temps.dropna().tolist()))
        time_unique = sorted(set(round(float(x), 6) for x in times.dropna().tolist()))

        n_warning_rows = int(g["condition_diversity_warning"].astype(str).str.len().gt(0).sum())
        n_baseline_seed = int(
            g["condition_diversity_warning"].astype(str).str.contains("baseline_seed_condition", regex=False).sum()
        )
        n_clip_low_temperature = int(
            g["condition_diversity_warning"].astype(str).str.contains("clip_low_temperature", regex=False).sum()
        )
        n_clip_low_time = int(
            g["condition_diversity_warning"].astype(str).str.contains("clip_low_time", regex=False).sum()
        )
        n_missing_temperature = int(
            g["condition_diversity_warning"].astype(str).str.contains("missing_temperature", regex=False).sum()
        )
        n_missing_time = int(
            g["condition_diversity_warning"].astype(str).str.contains("missing_time", regex=False).sum()
        )

        if len(temp_unique) <= 1 and len(time_unique) <= 1:
            diversity_status = "no_condition_diversity"
        elif len(temp_unique) <= 1 or len(time_unique) <= 1:
            diversity_status = "partial_condition_diversity"
        else:
            diversity_status = "condition_diversity_present"

        if n_baseline_seed == len(g) and len(g) > 0:
            diversity_status = "no_condition_diversity_baseline_seed_only"

        if n_missing_temperature or n_missing_time:
            audit_level = "major_warning"
        elif n_clip_low_temperature or n_clip_low_time:
            audit_level = "major_warning"
        elif diversity_status in {"no_condition_diversity_baseline_seed_only", "no_condition_diversity"}:
            audit_level = "minor_warning"
        else:
            audit_level = "pass"

        audit_rows.append({
            group_col: key,
            "n_route_rows": int(len(g)),
            "n_unique_temperature": int(len(temp_unique)),
            "n_unique_time": int(len(time_unique)),
            "min_temperature_c": float(temps.min()) if temps.notna().any() else None,
            "mean_temperature_c": float(temps.mean()) if temps.notna().any() else None,
            "max_temperature_c": float(temps.max()) if temps.notna().any() else None,
            "min_time_h": float(times.min()) if times.notna().any() else None,
            "mean_time_h": float(times.mean()) if times.notna().any() else None,
            "max_time_h": float(times.max()) if times.notna().any() else None,
            "n_warning_rows": n_warning_rows,
            "n_baseline_seed_condition": n_baseline_seed,
            "n_clip_low_temperature": n_clip_low_temperature,
            "n_clip_low_time": n_clip_low_time,
            "n_missing_temperature": n_missing_temperature,
            "n_missing_time": n_missing_time,
            "condition_diversity_status": diversity_status,
            "condition_diversity_audit_level": audit_level,
        })

    audit = pd.DataFrame(audit_rows)

    level_order = {
        "major_warning": 0,
        "minor_warning": 1,
        "pass": 2,
    }
    audit["_level_order"] = audit["condition_diversity_audit_level"].map(level_order).fillna(9)
    audit = audit.sort_values(["_level_order", group_col]).drop(columns=["_level_order"])

    audit.to_csv(output_csv, index=False)
    audit.head(args.top_n).to_markdown(output_md, index=False)

    summary = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "output_md": str(output_md),
        "n_input_rows": int(len(df)),
        "group_col": group_col,
        "temperature_col": temp_col,
        "time_col": time_col,
        "n_groups": int(len(audit)),
        "condition_diversity_status_counts": audit["condition_diversity_status"].value_counts(dropna=False).to_dict(),
        "condition_diversity_audit_level_counts": audit["condition_diversity_audit_level"].value_counts(dropna=False).to_dict(),
        "n_major_warning_groups": int((audit["condition_diversity_audit_level"] == "major_warning").sum()),
        "n_minor_warning_groups": int((audit["condition_diversity_audit_level"] == "minor_warning").sum()),
        "n_pass_groups": int((audit["condition_diversity_audit_level"] == "pass").sum()),
        "n_baseline_seed_groups": int((audit["n_baseline_seed_condition"] > 0).sum()),
        "n_clip_low_temperature_groups": int((audit["n_clip_low_temperature"] > 0).sum()),
        "n_clip_low_time_groups": int((audit["n_clip_low_time"] > 0).sum()),
        "n_missing_temperature_groups": int((audit["n_missing_temperature"] > 0).sum()),
        "n_missing_time_groups": int((audit["n_missing_time"] > 0).sum()),
        "baseline_temperature_c": float(args.baseline_temperature_c),
        "baseline_time_h": float(args.baseline_time_h),
        "clip_low_temperature_c": float(args.clip_low_temperature_c),
        "clip_low_time_h": float(args.clip_low_time_h),
        "claim_boundary": "condition_diversity_audit_is_internal_diagnostic_not_experimental_validation",
        "interpretation": (
            "This audit checks whether Stage3 temperature/time candidates show diversity and flags "
            "missing, clipped, or baseline-seed-only condition outputs."
        ),
    }

    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[SAVE] {output_csv}")
    print(f"[SAVE] {output_md}")
    print(f"[SAVE] {summary_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
