#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from collections import Counter
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


def to_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def clip01(x: float, default=0.0) -> float:
    if not np.isfinite(x):
        return default
    return max(0.0, min(1.0, float(x)))


def get_first_existing(row: pd.Series, cols: list[str], default=""):
    for c in cols:
        if c in row.index:
            v = row.get(c)
            if pd.notna(v) and str(v).strip() != "":
                return v
    return default


def split_flags(s: str) -> list[str]:
    s = to_str(s)
    if not s:
        return []
    return [x.strip() for x in s.split(";") if x.strip()]


def get_condition_support(row: pd.Series) -> tuple[float, str]:
    """
    Prefer real Stage3 MDN/Flow reference support.
    Fall back to internal route distribution support.
    """
    candidates = [
        "real_stage3_condition_reference_support_score",
        "condition_distribution_support_score",
    ]

    for c in candidates:
        if c in row.index:
            v = to_float(row.get(c), np.nan)
            if np.isfinite(v):
                return clip01(v), c

    return 1.0, "default_no_condition_support_column"


def get_condition_warning_and_status(row: pd.Series) -> tuple[str, str, str, str]:
    """
    Prefer real Stage3 reference warning/status if available.
    Fall back to internal condition distribution warning/status.
    """
    warning_cols = [
        "real_stage3_condition_reference_warning_level",
        "condition_distribution_warning_level",
    ]
    status_cols = [
        "real_stage3_condition_reference_recommendation_status",
        "condition_distribution_recommendation_status",
    ]

    warning = ""
    warning_col = ""
    for c in warning_cols:
        if c in row.index:
            v = to_str(row.get(c)).lower()
            if v:
                warning = v
                warning_col = c
                break

    status = ""
    status_col = ""
    for c in status_cols:
        if c in row.index:
            v = to_str(row.get(c)).lower()
            if v:
                status = v
                status_col = c
                break

    return warning, status, warning_col, status_col


def get_base_score(row: pd.Series) -> tuple[float, str]:
    """
    Prefer the final safe-strict/template score.
    Historical rule scores above 1 are squashed into 0-1.
    """
    candidates = [
        "stage35_v43_safe_strict_score",
        "stage35_v43_template_chemonly_score",
        "v3_learned_ranker_score",
        "v3_joint_feature_score",
        "route_confidence_score",
        "condition_adjusted_confidence_score",
        "stage3_score",
    ]

    for c in candidates:
        if c in row.index:
            v = to_float(row.get(c), np.nan)
            if np.isfinite(v):
                if v > 1.5:
                    v = v / (v + 1.0)
                return clip01(v), c

    return 0.5, "default_no_base_score_column"


def route_family_condition_sanity(row: pd.Series) -> dict:
    """
    Legacy V28-style broad route-family sanity check.

    This is secondary. It should not override real Stage3 reference support
    unless the condition is missing or outside global Stage3 bounds.
    """
    template = to_str(
        get_first_existing(
            row,
            ["route_template_primary", "route_template_secondary", "v43_route_template"],
            "",
        )
    ).lower()

    temp = to_float(row.get("temperature_c"), np.nan)
    time_h = to_float(row.get("time_h"), np.nan)

    default_temp_min, default_temp_max = 100.0, 1400.0
    default_time_min, default_time_max = 0.1, 240.0

    ranges = {
        "oxide_route": (400.0, 1200.0, 0.5, 72.0),
        "phosphate_route": (400.0, 1200.0, 0.5, 72.0),
        "selenite_selenate_route": (300.0, 1100.0, 0.5, 96.0),
        "sulfate_route": (300.0, 1100.0, 0.5, 96.0),
        "halide_route": (100.0, 900.0, 0.1, 120.0),
        "nitride_route": (500.0, 1400.0, 0.1, 240.0),
        "carbonate_route": (100.0, 900.0, 0.1, 120.0),
    }

    tmin, tmax, hmin, hmax = (
        default_temp_min,
        default_temp_max,
        default_time_min,
        default_time_max,
    )

    for key, vals in ranges.items():
        if key in template:
            tmin, tmax, hmin, hmax = vals
            break

    reasons = []
    score_parts = []

    if not np.isfinite(temp):
        reasons.append("missing_temperature")
        score_parts.append(0.0)
    elif temp < tmin:
        dist = (tmin - temp) / max(tmax - tmin, 1.0)
        score_parts.append(max(0.0, 1.0 - 2.0 * dist))
        reasons.append("temperature_below_range")
    elif temp > tmax:
        dist = (temp - tmax) / max(tmax - tmin, 1.0)
        score_parts.append(max(0.0, 1.0 - 2.0 * dist))
        reasons.append("temperature_above_range")
    else:
        score_parts.append(1.0)

    if not np.isfinite(time_h):
        reasons.append("missing_time")
        score_parts.append(0.0)
    elif time_h < hmin:
        dist = (hmin - time_h) / max(hmax - hmin, 1.0)
        score_parts.append(max(0.0, 1.0 - 2.0 * dist))
        reasons.append("time_below_range")
    elif time_h > hmax:
        dist = (time_h - hmax) / max(hmax - hmin, 1.0)
        score_parts.append(max(0.0, 1.0 - 2.0 * dist))
        reasons.append("time_above_range")
    else:
        score_parts.append(1.0)

    sanity_score = float(np.mean(score_parts)) if score_parts else 1.0

    extreme_reasons = []

    if not np.isfinite(temp):
        extreme_reasons.append("missing_temperature")
    elif temp < 4.0 or temp > 1525.0:
        extreme_reasons.append("temperature_outside_global_stage3_bounds")

    if not np.isfinite(time_h):
        extreme_reasons.append("missing_time")
    elif time_h < 0.1667 or time_h > 126.0:
        extreme_reasons.append("time_outside_global_stage3_bounds")

    return {
        "final_audit_condition_sanity_score": round(sanity_score, 6),
        "final_audit_condition_warning_reason": ";".join(reasons),
        "final_audit_extreme_condition_reason": ";".join(extreme_reasons),
        "final_audit_route_family_temp_min": tmin,
        "final_audit_route_family_temp_max": tmax,
        "final_audit_route_family_time_min": hmin,
        "final_audit_route_family_time_max": hmax,
    }


def audit_row(row: pd.Series, prev_rank: int | None, prev_score: float | None) -> dict:
    flags: list[str] = []

    rank = to_float(row.get("final_recommendation_rank"), np.nan)
    score = to_float(row.get("final_recommendation_score"), np.nan)
    status = to_str(row.get("final_recommendation_status")).lower()

    base_score, base_score_col = get_base_score(row)
    condition_support, condition_support_col = get_condition_support(row)
    condition_warning, condition_status, condition_warning_col, condition_status_col = (
        get_condition_warning_and_status(row)
    )

    penalty = to_float(row.get("final_recommendation_penalty"), 0.0)
    penalty_reason = to_str(row.get("final_recommendation_penalty_reason"))

    precursor_qc_level = to_str(row.get("precursor_qc_level")).lower()
    safe_bucket = to_str(row.get("stage35_v43_safe_bucket")).lower()
    safe_reason = to_str(row.get("stage35_v43_safe_reason"))

    sanity = route_family_condition_sanity(row)
    extreme_reason = sanity["final_audit_extreme_condition_reason"]
    family_warning_reason = sanity["final_audit_condition_warning_reason"]

    if not np.isfinite(rank):
        flags.append("missing_final_rank")

    if not np.isfinite(score):
        flags.append("missing_final_score")
    elif score < 0 or score > 1:
        flags.append("final_score_out_of_range")

    if status not in {"recommended", "recommended_with_validation", "review_required"}:
        flags.append("invalid_final_status")

    if prev_rank is not None and np.isfinite(rank) and int(rank) <= prev_rank:
        flags.append("rank_not_strictly_increasing")

    if prev_score is not None and np.isfinite(score) and score > prev_score + 1e-9:
        flags.append("score_not_monotonic_decreasing")

    # Finalizer may blend route score with adjusted confidence and condition support.
    # This is not a bug; keep it as a minor diagnostic flag only.
    if np.isfinite(score) and np.isfinite(base_score) and score > base_score + 0.05:
        flags.append("final_score_boosted_by_condition_support")

    if condition_warning == "major_warning":
        if "real_stage3" in condition_warning_col:
            flags.append("real_stage3_condition_major_warning")
        else:
            flags.append("condition_major_warning")
    elif condition_warning == "minor_warning":
        if "real_stage3" in condition_warning_col:
            flags.append("real_stage3_condition_minor_warning")
        else:
            flags.append("condition_minor_warning")

    if condition_status == "review_required":
        flags.append("condition_status_review_required")

    if condition_support_col != "default_no_condition_support_column" and condition_support < 0.25:
        flags.append("very_low_condition_support")

    # Extreme global bounds remain major regardless of reference support.
    if extreme_reason:
        flags.append("extreme_condition_global_bounds_warning")

    # Route-family sanity is only secondary consistency warning.
    if family_warning_reason and not extreme_reason:
        flags.append("route_family_condition_sanity_warning")

    if precursor_qc_level in {"major_warning", "review_required"}:
        flags.append("precursor_major_warning")
    elif precursor_qc_level == "minor_warning":
        flags.append("precursor_minor_warning")

    if safe_bucket and safe_bucket not in {"safe_candidate", "nan", "none"}:
        flags.append(f"safe_strict_bucket={safe_bucket}")

    major_triggers = {
        "real_stage3_condition_major_warning",
        "condition_major_warning",
        "condition_status_review_required",
        "very_low_condition_support",
        "extreme_condition_global_bounds_warning",
        "precursor_major_warning",
    }

    has_major = any(f in major_triggers or f.startswith("safe_strict_bucket=") for f in flags)
    has_minor = bool(flags)

    if has_major and status != "review_required":
        flags.append("major_warning_but_status_not_review_required")

    if (not has_major) and (not has_minor) and status == "review_required":
        flags.append("review_required_without_major_reason")

    final_flags = flags.copy()

    hard_consistency_flags = {
        "missing_final_rank",
        "missing_final_score",
        "final_score_out_of_range",
        "invalid_final_status",
        "rank_not_strictly_increasing",
        "score_not_monotonic_decreasing",
        "major_warning_but_status_not_review_required",
    }

    if any(f in hard_consistency_flags for f in final_flags):
        audit_level = "major_warning"
    elif any(f in major_triggers or f.startswith("safe_strict_bucket=") for f in final_flags):
        audit_level = "major_warning"
    elif final_flags:
        audit_level = "minor_warning"
    else:
        audit_level = "pass"

    return {
        "final_audit_level": audit_level,
        "final_audit_flags": ";".join(final_flags),
        "final_audit_base_score_col": base_score_col,
        "final_audit_base_score": round(float(base_score), 6),
        "final_audit_condition_support_col": condition_support_col,
        "final_audit_condition_support": round(float(condition_support), 6),
        "final_audit_condition_warning_col": condition_warning_col,
        "final_audit_condition_warning_level": condition_warning,
        "final_audit_condition_status_col": condition_status_col,
        "final_audit_condition_status": condition_status,
        "final_audit_penalty": round(float(penalty), 6) if np.isfinite(penalty) else np.nan,
        "final_audit_penalty_reason": penalty_reason,
        "final_audit_safe_bucket": safe_bucket,
        "final_audit_safe_reason": safe_reason,
        **sanity,
    }


def count_flags(series: pd.Series) -> dict:
    counter: Counter[str] = Counter()

    for value in series.fillna("").astype(str).tolist():
        for flag in split_flags(value):
            counter[flag] += 1

    return dict(counter)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Audit final_recommended_routes.csv for score/rank/status consistency "
            "and condition-reference support."
        )
    )
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--output_md", required=True)
    ap.add_argument("--summary_json", required=True)
    ap.add_argument("--top_n", type=int, default=30)
    args = ap.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    summary_json = Path(args.summary_json)

    if not input_csv.exists():
        raise FileNotFoundError(input_csv)

    df = pd.read_csv(input_csv)

    if df.empty:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        output_md.write_text(
            "# Final Recommended Routes Audit\n\nEmpty input.\n",
            encoding="utf-8",
        )
        summary_json.write_text(
            json.dumps(
                {
                    "input_csv": str(input_csv),
                    "output_csv": str(output_csv),
                    "n_routes": 0,
                    "audit_status": "empty_input",
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"[SAVE] {output_csv}")
        print(f"[SAVE] {output_md}")
        print(f"[SAVE] {summary_json}")
        return

    if "final_recommendation_rank" in df.columns:
        df["final_recommendation_rank"] = pd.to_numeric(
            df["final_recommendation_rank"],
            errors="coerce",
        )
        df = df.sort_values(
            "final_recommendation_rank",
            ascending=True,
            na_position="last",
        ).reset_index(drop=True)

    audit_records = []
    prev_rank = None
    prev_score = None

    for _, row in df.iterrows():
        rec = audit_row(row, prev_rank, prev_score)
        audit_records.append(rec)

        rank = to_float(row.get("final_recommendation_rank"), np.nan)
        score = to_float(row.get("final_recommendation_score"), np.nan)

        if np.isfinite(rank):
            prev_rank = int(rank)

        if np.isfinite(score):
            prev_score = float(score)

    audit_df = pd.DataFrame(audit_records)
    out = pd.concat([df.reset_index(drop=True), audit_df], axis=1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    preview_cols = [
        "final_recommendation_rank",
        "final_recommendation_score",
        "final_recommendation_status",
        "final_audit_level",
        "final_audit_flags",
        "final_audit_condition_support_col",
        "final_audit_condition_support",
        "final_audit_condition_warning_col",
        "final_audit_condition_warning_level",
        "final_audit_condition_sanity_score",
        "final_audit_condition_warning_reason",
        "precursor_qc_level",
        "precursor_set",
        "temperature_c",
        "time_h",
        "route_template_primary",
    ]
    preview_cols = [c for c in preview_cols if c in out.columns]

    lines = []
    lines.append("# Final Recommended Routes Audit\n")
    lines.append(f"- input_csv: `{input_csv}`")
    lines.append(f"- n_routes: {len(out)}")
    lines.append(f"- top_n: {args.top_n}")
    lines.append("")
    if preview_cols:
        lines.append(out[preview_cols].head(args.top_n).to_markdown(index=False))
    else:
        lines.append("No preview columns available.")
    lines.append("")

    output_md.write_text("\n".join(lines), encoding="utf-8")

    level_counts = out["final_audit_level"].value_counts(dropna=False).to_dict()

    status_counts = (
        out["final_recommendation_status"].value_counts(dropna=False).to_dict()
        if "final_recommendation_status" in out.columns
        else {}
    )

    support_col_counts = (
        out["final_audit_condition_support_col"].value_counts(dropna=False).to_dict()
        if "final_audit_condition_support_col" in out.columns
        else {}
    )

    warning_col_counts = (
        out["final_audit_condition_warning_col"].value_counts(dropna=False).to_dict()
        if "final_audit_condition_warning_col" in out.columns
        else {}
    )

    flag_counts = (
        count_flags(out["final_audit_flags"])
        if "final_audit_flags" in out.columns
        else {}
    )

    n_major = int((out["final_audit_level"] == "major_warning").sum())
    n_minor = int((out["final_audit_level"] == "minor_warning").sum())
    n_pass = int((out["final_audit_level"] == "pass").sum())

    n_condition_reference_used = (
        int(
            out["final_audit_condition_support_col"]
            .astype(str)
            .eq("real_stage3_condition_reference_support_score")
            .sum()
        )
        if "final_audit_condition_support_col" in out.columns
        else 0
    )

    n_condition_sanity_warning = (
        int(
            out["final_audit_condition_warning_reason"]
            .fillna("")
            .astype(str)
            .str.len()
            .gt(0)
            .sum()
        )
        if "final_audit_condition_warning_reason" in out.columns
        else 0
    )

    n_real_stage3_condition_warning = 0
    if (
        "final_audit_condition_warning_col" in out.columns
        and "final_audit_condition_warning_level" in out.columns
    ):
        real_stage3_warning_mask = (
            out["final_audit_condition_warning_col"]
            .astype(str)
            .str.contains("real_stage3_condition_reference", na=False)
        ) & (
            out["final_audit_condition_warning_level"]
            .astype(str)
            .str.lower()
            .isin(["minor_warning", "major_warning"])
        )

        n_real_stage3_condition_warning = int(real_stage3_warning_mask.sum())

    top1 = out.iloc[0]

    summary = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "output_md": str(output_md),
        "n_routes": int(len(out)),
        "top_n": int(args.top_n),
        "rank_continuous": bool(
            out["final_recommendation_rank"].is_monotonic_increasing
            if "final_recommendation_rank" in out.columns
            else False
        ),
        "final_audit_level_counts": level_counts,
        "final_audit_flag_counts": flag_counts,
        "n_major_warning": n_major,
        "n_minor_warning": n_minor,
        "n_pass": n_pass,
        "n_condition_sanity_warning": n_condition_sanity_warning,
        "n_condition_reference_used": n_condition_reference_used,
        "n_real_stage3_condition_warning": n_real_stage3_condition_warning,
        "condition_support_col_counts": support_col_counts,
        "condition_warning_col_counts": warning_col_counts,
        "final_status_counts": status_counts,
        "top1_precursor_set": str(top1.get("precursor_set", "")),
        "top1_final_recommendation_score": float(
            top1.get("final_recommendation_score", np.nan)
        ),
        "top1_final_audit_level": str(top1.get("final_audit_level", "")),
        "top1_final_audit_flags": str(top1.get("final_audit_flags", "")),
        "top1_condition_support_col": str(
            top1.get("final_audit_condition_support_col", "")
        ),
        "top1_condition_support": float(
            top1.get("final_audit_condition_support", np.nan)
        ),
        "top1_condition_warning_col": str(
            top1.get("final_audit_condition_warning_col", "")
        ),
        "top1_condition_warning_level": str(
            top1.get("final_audit_condition_warning_level", "")
        ),
        "claim_boundary": "final_audit_is_internal_consistency_check_not_experimental_validation",
        "interpretation": (
            "This audit validates final recommendation score/rank/status consistency. "
            "It prioritizes real Stage3 MDN/Flow condition-reference support when available. "
            "Route-family temperature/time sanity, precursor minor QC, and final score boosting by condition support "
            "are treated as secondary minor diagnostic signals unless they violate global bounds or trigger review-required conditions."
        ),
    }

    summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[SAVE] {output_csv}")
    print(f"[SAVE] {output_md}")
    print(f"[SAVE] {summary_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
