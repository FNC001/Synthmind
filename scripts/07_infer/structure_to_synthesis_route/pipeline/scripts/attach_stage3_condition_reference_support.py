#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--reference_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--output_md", required=True)
    ap.add_argument("--summary_json", required=True)
    ap.add_argument("--top_n", type=int, default=30)

    ap.add_argument("--temperature_col", default="temperature_c")
    ap.add_argument("--time_col", default="time_h")

    ap.add_argument("--reference_temperature_col", default="candidate_temperature_c")
    ap.add_argument("--reference_time_col", default="candidate_time_h")

    ap.add_argument("--min_width_temperature", type=float, default=50.0)
    ap.add_argument("--min_width_log_time", type=float, default=0.25)

    return ap.parse_args()


def safe_read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing CSV: {p}")
    return pd.read_csv(p)


def robust_center_width(values: pd.Series, min_width: float) -> tuple[float, float, str]:
    x = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(x) == 0:
        return float("nan"), float("nan"), "empty"

    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    width = 1.4826 * mad

    if not np.isfinite(width) or width <= 1e-12:
        q25 = float(np.percentile(x, 25))
        q75 = float(np.percentile(x, 75))
        width = (q75 - q25) / 1.349 if q75 > q25 else 0.0
        method = "iqr_fallback"
    else:
        method = "median_mad"

    if not np.isfinite(width) or width < min_width:
        std = float(np.std(x))
        if np.isfinite(std) and std > width:
            width = std
            method = method + "_std_fallback"

    if not np.isfinite(width) or width < min_width:
        width = float(min_width)
        method = method + "_min_width"

    return med, width, method


def support_from_z(z: float) -> float:
    if not np.isfinite(z):
        return 0.0
    return float(math.exp(-0.5 * z * z))


def level_from_support(score: float) -> str:
    if score >= 0.70:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"


def warning_from_support(score: float) -> str:
    if score < 0.20:
        return "major_warning"
    if score < 0.35:
        return "minor_warning"
    return "no_warning"


def recommendation_from_support(score: float) -> str:
    if score < 0.20:
        return "review_required"
    if score < 0.70:
        return "recommended_with_validation"
    return "recommended"


def main():
    args = parse_args()

    input_csv = Path(args.input_csv)
    reference_csv = Path(args.reference_csv)
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    summary_json = Path(args.summary_json)

    df = safe_read_csv(str(input_csv))
    ref = safe_read_csv(str(reference_csv))

    required_input = [args.temperature_col, args.time_col]
    required_ref = [args.reference_temperature_col, args.reference_time_col]

    missing_input = [c for c in required_input if c not in df.columns]
    missing_ref = [c for c in required_ref if c not in ref.columns]

    if missing_input:
        raise RuntimeError(f"Input CSV missing required columns: {missing_input}")

    if missing_ref:
        raise RuntimeError(f"Reference CSV missing required columns: {missing_ref}")

    ref_temp = pd.to_numeric(ref[args.reference_temperature_col], errors="coerce")
    ref_time = pd.to_numeric(ref[args.reference_time_col], errors="coerce")
    ref_log_time = np.log1p(ref_time.clip(lower=0))

    temp_center, temp_width, temp_method = robust_center_width(
        ref_temp,
        min_width=args.min_width_temperature,
    )
    log_time_center, log_time_width, log_time_method = robust_center_width(
        pd.Series(ref_log_time),
        min_width=args.min_width_log_time,
    )

    out = df.copy()

    temp = pd.to_numeric(out[args.temperature_col], errors="coerce")
    time_h = pd.to_numeric(out[args.time_col], errors="coerce")
    log_time = np.log1p(time_h.clip(lower=0))

    temp_z = (temp - temp_center) / temp_width
    log_time_z = (log_time - log_time_center) / log_time_width

    temp_support = temp_z.apply(support_from_z)
    time_support = log_time_z.apply(support_from_z)

    # Temperature usually dominates solid-state route plausibility,
    # but time still matters. Keep the weight transparent and conservative.
    support = 0.65 * temp_support + 0.35 * time_support

    out["real_stage3_condition_reference_support_score"] = support.round(6)
    out["real_stage3_condition_reference_temperature_center_c"] = round(temp_center, 6)
    out["real_stage3_condition_reference_temperature_width_c"] = round(temp_width, 6)
    out["real_stage3_condition_reference_log_time_center"] = round(log_time_center, 6)
    out["real_stage3_condition_reference_log_time_width"] = round(log_time_width, 6)
    out["real_stage3_condition_reference_temperature_z"] = temp_z.round(6)
    out["real_stage3_condition_reference_log_time_z"] = log_time_z.round(6)
    out["real_stage3_condition_reference_temperature_support"] = temp_support.round(6)
    out["real_stage3_condition_reference_time_support"] = time_support.round(6)

    out["real_stage3_condition_reference_level"] = out[
        "real_stage3_condition_reference_support_score"
    ].apply(level_from_support)

    out["real_stage3_condition_reference_warning_level"] = out[
        "real_stage3_condition_reference_support_score"
    ].apply(warning_from_support)

    out["real_stage3_condition_reference_recommendation_status"] = out[
        "real_stage3_condition_reference_support_score"
    ].apply(recommendation_from_support)

    out["real_stage3_condition_reference_source"] = "v30_real_stage3_mdn_flow_reference"
    out["real_stage3_condition_reference_claim_boundary"] = (
        "reference_support_is_based_on_real_stage3_generated_condition_distribution_not_experimental_validation"
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    preview_cols = [
        "precursor_set",
        args.temperature_col,
        args.time_col,
        "real_stage3_condition_reference_support_score",
        "real_stage3_condition_reference_level",
        "real_stage3_condition_reference_warning_level",
        "real_stage3_condition_reference_recommendation_status",
        "real_stage3_condition_reference_temperature_z",
        "real_stage3_condition_reference_log_time_z",
    ]
    preview_cols = [c for c in preview_cols if c in out.columns]
    out[preview_cols].head(args.top_n).to_markdown(output_md, index=False)

    summary = {
        "input_csv": str(input_csv),
        "reference_csv": str(reference_csv),
        "output_csv": str(output_csv),
        "output_md": str(output_md),
        "n_routes": int(len(out)),
        "n_reference_rows": int(len(ref)),
        "n_reference_temperature_nonnull": int(ref_temp.notna().sum()),
        "n_reference_time_nonnull": int(ref_time.notna().sum()),
        "temperature_center_c": temp_center,
        "temperature_width_c": temp_width,
        "temperature_width_method": temp_method,
        "log_time_center": log_time_center,
        "log_time_width": log_time_width,
        "log_time_width_method": log_time_method,
        "mean_real_stage3_condition_reference_support_score": float(
            out["real_stage3_condition_reference_support_score"].mean()
        ) if len(out) else None,
        "min_real_stage3_condition_reference_support_score": float(
            out["real_stage3_condition_reference_support_score"].min()
        ) if len(out) else None,
        "max_real_stage3_condition_reference_support_score": float(
            out["real_stage3_condition_reference_support_score"].max()
        ) if len(out) else None,
        "real_stage3_condition_reference_level_counts": out[
            "real_stage3_condition_reference_level"
        ].value_counts(dropna=False).to_dict(),
        "real_stage3_condition_reference_warning_level_counts": out[
            "real_stage3_condition_reference_warning_level"
        ].value_counts(dropna=False).to_dict(),
        "real_stage3_condition_reference_recommendation_status_counts": out[
            "real_stage3_condition_reference_recommendation_status"
        ].value_counts(dropna=False).to_dict(),
        "claim_boundary": (
            "real_stage3_condition_reference_support_is_internal_distribution_support_from_mdn_flow_candidates_not_experimental_validation"
        ),
        "interpretation": (
            "This step compares route temperature/time candidates against a real Stage3 MDN/Flow condition reference library."
        ),
    }

    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[SAVE] {output_csv}")
    print(f"[SAVE] {output_md}")
    print(f"[SAVE] {summary_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
