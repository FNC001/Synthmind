#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def to_float(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def robust_center_width(values: pd.Series, min_width: float) -> tuple[float, float, dict]:
    """
    Robust center/width using median and MAD.
    Falls back to std/IQR when needed.
    """
    x = pd.to_numeric(values, errors="coerce").dropna()
    meta = {
        "n_valid": int(len(x)),
        "method": "median_mad",
    }

    if len(x) == 0:
        return float("nan"), float("nan"), meta

    center = float(x.median())
    mad = float((x - center).abs().median())
    width = 1.4826 * mad

    if not np.isfinite(width) or width < min_width:
        std = float(x.std(ddof=0)) if len(x) > 1 else 0.0
        if np.isfinite(std) and std >= min_width:
            width = std
            meta["method"] = "median_std"
        else:
            q75 = float(x.quantile(0.75))
            q25 = float(x.quantile(0.25))
            iqr_width = (q75 - q25) / 1.349 if q75 >= q25 else 0.0
            if np.isfinite(iqr_width) and iqr_width >= min_width:
                width = iqr_width
                meta["method"] = "median_iqr"
            else:
                width = min_width
                meta["method"] = "median_min_width"

    return center, float(width), meta


def level_from_score_01(score: float) -> str:
    if not np.isfinite(score):
        return "unknown"
    if score >= 0.75:
        return "high"
    if score >= 0.50:
        return "medium"
    return "low"


def exp_support(z: float) -> float:
    if not np.isfinite(z):
        return 0.5
    z = max(0.0, float(z))
    return float(math.exp(-0.5 * z * z))


def choose_confidence_score(row: pd.Series) -> float:
    """
    Prefer existing 0-1 confidence score if present.
    Fall back to learned/v43 scores, then medium baseline.
    """
    candidates = [
        "route_confidence_score",
        "stage35_v43_template_chemonly_score",
        "v3_learned_ranker_score",
        "stage3_score",
    ]

    for c in candidates:
        if c in row.index:
            v = to_float(row.get(c), np.nan)
            if np.isfinite(v):
                # Some historical scores may be 0-100.
                if v > 1.5:
                    v = v / 100.0
                return max(0.0, min(1.0, v))

    return 0.5


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Add condition-distribution confidence to route table."
    )
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--output_md", default=None)
    ap.add_argument("--summary_json", default=None)
    ap.add_argument("--top_n", type=int, default=30)

    ap.add_argument("--temperature_col", default="temperature_c")
    ap.add_argument("--time_col", default="time_h")

    ap.add_argument("--min_temp_width", type=float, default=50.0)
    ap.add_argument("--min_log_time_width", type=float, default=0.35)
    ap.add_argument("--warning_z", type=float, default=2.5)
    ap.add_argument("--low_support_threshold", type=float, default=0.35)
    ap.add_argument("--small_n_threshold", type=int, default=5)

    args = ap.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md) if args.output_md else None
    summary_json = Path(args.summary_json) if args.summary_json else None

    if not input_csv.exists():
        raise FileNotFoundError(input_csv)

    df = pd.read_csv(input_csv)
    if df.empty:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        if output_md:
            output_md.write_text("# Condition Distribution Confidence\n\nEmpty input.\n", encoding="utf-8")
        if summary_json:
            summary_json.write_text(
                json.dumps(
                    {
                        "input_csv": str(input_csv),
                        "output_csv": str(output_csv),
                        "n_routes": 0,
                        "status": "empty_input",
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        print(f"[SAVE] {output_csv}")
        return

    n_routes = len(df)

    temp = pd.to_numeric(df.get(args.temperature_col, pd.Series(dtype=float)), errors="coerce")
    time_h = pd.to_numeric(df.get(args.time_col, pd.Series(dtype=float)), errors="coerce")
    log_time = np.log1p(time_h.clip(lower=0))

    temp_center, temp_width, temp_meta = robust_center_width(temp, args.min_temp_width)
    log_time_center, log_time_width, log_time_meta = robust_center_width(
        pd.Series(log_time), args.min_log_time_width
    )

    temp_z = (temp - temp_center).abs() / temp_width
    log_time_z = (log_time - log_time_center).abs() / log_time_width

    temp_z = temp_z.replace([np.inf, -np.inf], np.nan)
    log_time_z = log_time_z.replace([np.inf, -np.inf], np.nan)

    distance_z = pd.concat([temp_z, log_time_z], axis=1).max(axis=1, skipna=True)
    distance_z = distance_z.fillna(0.0)

    distance_support = distance_z.apply(exp_support)

    # Width support: if candidate distribution is too narrow/small, confidence should not be overclaimed.
    sample_support = 1.0 if n_routes >= args.small_n_threshold else 0.75
    distribution_support = (distance_support * sample_support).clip(lower=0.0, upper=1.0)

    condition_scores = []
    condition_levels = []
    warning_levels = []
    warning_reasons = []
    adjusted_scores = []

    for idx, row in df.iterrows():
        reasons = []

        tz = to_float(temp_z.iloc[idx], 0.0)
        ltz = to_float(log_time_z.iloc[idx], 0.0)
        support = to_float(distribution_support.iloc[idx], 0.5)

        if n_routes < args.small_n_threshold:
            reasons.append("insufficient_distribution_samples")
        if tz > args.warning_z:
            reasons.append("temperature_far_from_distribution")
        if ltz > args.warning_z:
            reasons.append("time_far_from_distribution")
        if support < args.low_support_threshold:
            reasons.append("low_condition_distribution_support")

        base_conf = choose_confidence_score(row)

        # Attenuation layer: never boosts above base confidence.
        multiplier = 0.65 + 0.35 * support
        penalty = 0.0
        if "temperature_far_from_distribution" in reasons:
            penalty += 0.08
        if "time_far_from_distribution" in reasons:
            penalty += 0.08
        if "low_condition_distribution_support" in reasons:
            penalty += 0.05
        if "insufficient_distribution_samples" in reasons:
            penalty += 0.03

        adjusted = max(0.0, min(1.0, base_conf * multiplier - penalty))

        if reasons:
            warn_level = "minor_warning"
        else:
            warn_level = "no_warning"

        if support < 0.25 or "temperature_far_from_distribution" in reasons or "time_far_from_distribution" in reasons:
            if adjusted < 0.5:
                warn_level = "major_warning"

        condition_scores.append(round(float(support), 6))
        condition_levels.append(level_from_score_01(float(support)))
        warning_levels.append(warn_level)
        warning_reasons.append(";".join(reasons))
        adjusted_scores.append(round(float(adjusted), 6))

    out = df.copy()

    out["condition_temperature_center_c"] = round(float(temp_center), 6) if np.isfinite(temp_center) else np.nan
    out["condition_temperature_width_c"] = round(float(temp_width), 6) if np.isfinite(temp_width) else np.nan
    out["condition_time_log_center"] = round(float(log_time_center), 6) if np.isfinite(log_time_center) else np.nan
    out["condition_time_log_width"] = round(float(log_time_width), 6) if np.isfinite(log_time_width) else np.nan

    out["condition_temperature_z_distance"] = temp_z.round(6)
    out["condition_time_z_distance"] = log_time_z.round(6)
    out["condition_distribution_z_distance"] = distance_z.round(6)

    out["condition_distance_support_score"] = distance_support.round(6)
    out["condition_distribution_support_score"] = condition_scores
    out["condition_distribution_confidence_level"] = condition_levels
    out["condition_distribution_warning_level"] = warning_levels
    out["condition_distribution_warning_reason"] = warning_reasons
    out["condition_adjusted_confidence_score"] = adjusted_scores

    # Optional status propagation, conservative:
    # condition warnings do not immediately force review_required unless already severe.
    existing_status_col = None
    for c in ["route_recommendation_status", "recommendation_status"]:
        if c in out.columns:
            existing_status_col = c
            break

    if existing_status_col:
        final_status = out[existing_status_col].fillna("").astype(str).copy()
    else:
        final_status = pd.Series(["recommended"] * len(out), index=out.index)

    major_mask = out["condition_distribution_warning_level"].astype(str).eq("major_warning")
    minor_mask = out["condition_distribution_warning_level"].astype(str).eq("minor_warning")

    final_status.loc[minor_mask & final_status.eq("recommended")] = "recommended_with_validation"
    final_status.loc[major_mask] = "review_required"

    out["condition_distribution_recommendation_status"] = final_status

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    preview_cols = [
        "precursor_set",
        "temperature_c",
        "time_h",
        "condition_distribution_support_score",
        "condition_distribution_confidence_level",
        "condition_distribution_warning_level",
        "condition_distribution_warning_reason",
        "condition_adjusted_confidence_score",
        "condition_distribution_recommendation_status",
    ]
    preview_cols = [c for c in preview_cols if c in out.columns]

    if output_md:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        preview = out[preview_cols].head(args.top_n)
        text = []
        text.append("# Condition Distribution Confidence\n")
        text.append(f"- input_csv: `{input_csv}`")
        text.append(f"- n_routes: {n_routes}")
        text.append(f"- temperature_center_c: {temp_center:.6g}")
        text.append(f"- temperature_width_c: {temp_width:.6g}")
        text.append(f"- log_time_center: {log_time_center:.6g}")
        text.append(f"- log_time_width: {log_time_width:.6g}")
        text.append("")
        text.append(preview.to_markdown(index=False))
        text.append("")
        output_md.write_text("\n".join(text), encoding="utf-8")

    summary = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "output_md": str(output_md) if output_md else None,
        "n_routes": int(n_routes),
        "temperature_center_c": float(temp_center) if np.isfinite(temp_center) else None,
        "temperature_width_c": float(temp_width) if np.isfinite(temp_width) else None,
        "temperature_width_method": temp_meta.get("method"),
        "log_time_center": float(log_time_center) if np.isfinite(log_time_center) else None,
        "log_time_width": float(log_time_width) if np.isfinite(log_time_width) else None,
        "log_time_width_method": log_time_meta.get("method"),
        "mean_condition_distribution_support_score": float(out["condition_distribution_support_score"].mean()),
        "min_condition_distribution_support_score": float(out["condition_distribution_support_score"].min()),
        "max_condition_distribution_support_score": float(out["condition_distribution_support_score"].max()),
        "condition_distribution_confidence_level_counts": out["condition_distribution_confidence_level"].value_counts(dropna=False).to_dict(),
        "condition_distribution_warning_level_counts": out["condition_distribution_warning_level"].value_counts(dropna=False).to_dict(),
        "condition_distribution_recommendation_status_counts": out["condition_distribution_recommendation_status"].value_counts(dropna=False).to_dict(),
        "claim_boundary": "condition_distribution_confidence_is_internal_distribution_support_not_experimental_validation",
    }

    if summary_json:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[SAVE] {output_csv}")
    if output_md:
        print(f"[SAVE] {output_md}")
    if summary_json:
        print(f"[SAVE] {summary_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
