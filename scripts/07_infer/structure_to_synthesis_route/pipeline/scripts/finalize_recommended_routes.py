#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FINALIZER_NAME = "v43_template_plus_condition_qc_finalizer"


def to_float(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def to_str(x) -> str:
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def normalize_01(v: float) -> float:
    if not np.isfinite(v):
        return np.nan

    # Historical confidence scores may be 0--100.
    if v > 1.5 and v <= 100.0:
        v = v / 100.0

    # Some historical rule scores may be >1 but not 0--100.
    elif v > 1.5:
        v = v / (v + 1.0)

    return max(0.0, min(1.0, float(v)))


def first_existing_col(row: pd.Series, candidates: list[str]) -> tuple[str | None, object]:
    for c in candidates:
        if c in row.index:
            v = row.get(c)
            try:
                if pd.notna(v):
                    return c, v
            except Exception:
                return c, v
    return None, None


def choose_base_score(row: pd.Series) -> tuple[float, str]:
    """
    Prefer v43 score, then v3 learned score, then v3 joint feature score,
    then confidence / Stage3 scores.

    Returns:
      base_score_0_1, base_score_col_used
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
                v01 = normalize_01(v)
                if np.isfinite(v01):
                    return v01, c

    return 0.5, "fallback_default_0.5"


def choose_condition_support(row: pd.Series) -> tuple[float, str]:
    """
    Prefer the real Stage3 MDN/Flow condition reference support.

    Fallback:
      condition_distribution_support_score

    Final fallback:
      1.0, meaning no extra attenuation when condition support is absent.
    """
    candidates = [
        "real_stage3_condition_reference_support_score",
        "condition_distribution_support_score",
    ]

    for c in candidates:
        if c in row.index:
            v = to_float(row.get(c), np.nan)
            if np.isfinite(v):
                return max(0.0, min(1.0, float(v))), c

    return 1.0, "fallback_no_condition_support"


def choose_adjusted_confidence(row: pd.Series, base: float) -> tuple[float, str]:
    """
    Prefer condition-adjusted confidence, then route confidence, then base.
    """
    candidates = [
        "condition_adjusted_confidence_score",
        "route_confidence_score",
        "confidence_score",
        "v27_confidence_score",
    ]

    for c in candidates:
        if c in row.index:
            v = to_float(row.get(c), np.nan)
            if np.isfinite(v):
                v01 = normalize_01(v)
                if np.isfinite(v01):
                    return v01, c

    return max(0.0, min(1.0, base)), "fallback_base_score"


def choose_condition_warning(row: pd.Series) -> tuple[str, str]:
    """
    Prefer real Stage3 reference warning level if present.
    Fallback to internal condition distribution warning.
    """
    candidates = [
        "real_stage3_condition_reference_warning_level",
        "condition_distribution_warning_level",
    ]

    c, v = first_existing_col(row, candidates)
    if c is None:
        return "", "none"

    return to_str(v).strip().lower(), c


def choose_condition_status(row: pd.Series) -> tuple[str, str]:
    """
    Prefer real Stage3 reference recommendation status if present.
    Fallback to internal condition distribution status.
    """
    candidates = [
        "real_stage3_condition_reference_recommendation_status",
        "condition_distribution_recommendation_status",
    ]

    c, v = first_existing_col(row, candidates)
    if c is None:
        return "", "none"

    return to_str(v).strip().lower(), c


def compute_final_score(row: pd.Series) -> dict:
    base, base_col = choose_base_score(row)

    condition_support, condition_support_col = choose_condition_support(row)

    adjusted_conf, adjusted_conf_col = choose_adjusted_confidence(row, base)

    condition_warning, condition_warning_col = choose_condition_warning(row)
    condition_status, condition_status_col = choose_condition_status(row)

    precursor_qc_level = to_str(row.get("precursor_qc_level", "")).strip().lower()
    route_recommendation_status = to_str(row.get("route_recommendation_status", "")).strip().lower()
    recommendation_status = to_str(row.get("recommendation_status", "")).strip().lower()

    safe_bucket = to_str(row.get("stage35_v43_safe_bucket", "")).strip().lower()
    safe_reason = to_str(row.get("stage35_v43_safe_reason", "")).strip().lower()

    penalty = 0.0
    reasons: list[str] = []

    # Condition/reference support penalties.
    if condition_warning == "major_warning":
        penalty += 0.35
        reasons.append("condition_major_warning")
    elif condition_warning == "minor_warning":
        penalty += 0.08
        reasons.append("condition_minor_warning")

    if condition_status == "review_required":
        penalty += 0.25
        reasons.append("condition_review_required")
    elif condition_status == "recommended_with_validation":
        penalty += 0.05
        reasons.append("condition_recommended_with_validation")

    # Precursor QC penalties.
    if precursor_qc_level in {"major_warning", "review_required"}:
        penalty += 0.25
        reasons.append("precursor_major_warning")
    elif precursor_qc_level == "minor_warning":
        penalty += 0.05
        reasons.append("precursor_minor_warning")

    # v43 safe-strict gate penalties.
    if safe_bucket in {"review_required", "unsafe_candidate", "strict_extra_element", "missing_core"}:
        penalty += 0.25
        reasons.append("safe_strict_review_required")
    elif safe_reason and "upstream_qc=minor_warning" in safe_reason:
        # Do not double-count too strongly; precursor_qc_level may already add 0.05.
        if "precursor_minor_warning" not in reasons:
            penalty += 0.05
            reasons.append("safe_strict_upstream_minor_warning")

    # Existing route recommendation status — soft penalty only.
    if route_recommendation_status == "review_required" or recommendation_status == "review_required":
        penalty += 0.05
        reasons.append("route_review_required")

    # Support factor: strong support keeps score, weak support attenuates.
    # Minimum factor is 0.70 to avoid over-dominating v43/template ranking
    # unless explicit warnings/status penalties are present.
    support_factor = 0.70 + 0.30 * condition_support

    # Blend v43/template score with condition-adjusted confidence.
    blended = 0.65 * base + 0.35 * adjusted_conf

    final_score = blended * support_factor - penalty
    final_score = max(0.0, min(1.0, final_score))

    # Tiered status: hard reasons + score-based thresholds.
    hard_review_reasons = {
        "condition_major_warning",
        "condition_review_required",
        "precursor_major_warning",
        "safe_strict_review_required",
    }

    if any(r in hard_review_reasons for r in reasons):
        if final_score >= 0.45 and condition_support >= 0.85:
            final_status = "recommended_with_validation"
        else:
            final_status = "review_required"
    elif reasons:
        if final_score >= 0.45:
            final_status = "recommended"
        elif final_score >= 0.25:
            final_status = "recommended_with_validation"
        else:
            final_status = "review_required"
    else:
        final_status = "recommended"

    return {
        "final_recommendation_score": round(float(final_score), 6),
        "final_recommendation_status": final_status,
        "final_recommendation_source": FINALIZER_NAME,
        "final_recommendation_penalty": round(float(penalty), 6),
        "final_recommendation_penalty_reason": ";".join(reasons),

        "final_recommendation_base_score": round(float(base), 6),
        "final_recommendation_base_score_col": base_col,

        "final_recommendation_condition_support": round(float(condition_support), 6),
        "final_recommendation_condition_support_col": condition_support_col,

        "final_recommendation_adjusted_confidence": round(float(adjusted_conf), 6),
        "final_recommendation_adjusted_confidence_col": adjusted_conf_col,

        "final_recommendation_condition_warning_col": condition_warning_col,
        "final_recommendation_condition_status_col": condition_status_col,

        "final_recommendation_support_factor": round(float(support_factor), 6),
    }


def write_empty_outputs(
    df: pd.DataFrame,
    input_csv: Path,
    output_csv: Path,
    output_md: Path,
    summary_json: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    output_md.write_text("# Final Recommended Routes\n\nEmpty input.\n", encoding="utf-8")
    summary_json.write_text(
        json.dumps(
            {
                "input_csv": str(input_csv),
                "output_csv": str(output_csv),
                "output_md": str(output_md),
                "n_routes": 0,
                "status": "empty_input",
                "claim_boundary": "final_recommendation_score_is_internal_ranking_score_not_experimental_validation",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[SAVE] {output_csv}")
    print(f"[SAVE] {output_md}")
    print(f"[SAVE] {summary_json}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Finalize recommended routes using v43 score plus real Stage3 "
            "condition-reference support, condition/QC penalties, and safety gates."
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
        write_empty_outputs(df, input_csv, output_csv, output_md, summary_json)
        return

    records = []
    for _, row in df.iterrows():
        records.append(compute_final_score(row))

    score_df = pd.DataFrame(records)
    out = pd.concat([df.reset_index(drop=True), score_df], axis=1)

    # Per-sample sort and rank
    if "sample_id" in out.columns and out["sample_id"].nunique() > 1:
        out = out.sort_values(
            ["sample_id", "final_recommendation_score"],
            ascending=[True, False],
            na_position="last",
        ).reset_index(drop=True)
        out["final_recommendation_rank"] = out.groupby("sample_id").cumcount() + 1
    else:
        out = out.sort_values(
            ["final_recommendation_score"],
            ascending=[False],
            na_position="last",
        ).reset_index(drop=True)
        out["final_recommendation_rank"] = np.arange(1, len(out) + 1)

    # Put final columns in front while preserving all original audit columns.
    front_cols = [
        "final_recommendation_rank",
        "final_recommendation_score",
        "final_recommendation_status",
        "final_recommendation_source",
        "final_recommendation_penalty",
        "final_recommendation_penalty_reason",

        "final_recommendation_base_score",
        "final_recommendation_base_score_col",
        "final_recommendation_condition_support",
        "final_recommendation_condition_support_col",
        "final_recommendation_adjusted_confidence",
        "final_recommendation_adjusted_confidence_col",
        "final_recommendation_support_factor",
        "final_recommendation_condition_warning_col",
        "final_recommendation_condition_status_col",

        "precursor_set",
        "temperature_c",
        "time_h",

        "stage35_v43_safe_strict_rank",
        "stage35_v43_safe_strict_score",
        "stage35_v43_safe_bucket",
        "stage35_v43_safe_reason",

        "stage35_v43_template_chemonly_rank",
        "stage35_v43_template_chemonly_score",
        "stage35_v43_template_chemonly_win_rate",
        "stage35_v43_template_chemonly_mean_prob",

        "route_template_primary",
        "route_template_secondary",
        "route_template_type_signature",

        "real_stage3_condition_reference_support_score",
        "real_stage3_condition_reference_level",
        "real_stage3_condition_reference_warning_level",
        "real_stage3_condition_reference_recommendation_status",
        "real_stage3_condition_reference_temperature_z",
        "real_stage3_condition_reference_log_time_z",
        "real_stage3_condition_reference_temperature_support",
        "real_stage3_condition_reference_time_support",
        "real_stage3_condition_reference_source",

        "condition_distribution_support_score",
        "condition_distribution_confidence_level",
        "condition_distribution_warning_level",
        "condition_distribution_warning_reason",
        "condition_adjusted_confidence_score",
        "condition_distribution_recommendation_status",

        "precursor_qc_level",
        "precursor_qc_status",
    ]

    front_cols = [c for c in front_cols if c in out.columns]
    rest_cols = [c for c in out.columns if c not in front_cols]
    out = out[front_cols + rest_cols]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    preview_cols = [
        "final_recommendation_rank",
        "final_recommendation_score",
        "final_recommendation_status",
        "final_recommendation_penalty_reason",
        "precursor_set",
        "temperature_c",
        "time_h",
        "stage35_v43_safe_strict_score",
        "stage35_v43_template_chemonly_score",
        "real_stage3_condition_reference_support_score",
        "real_stage3_condition_reference_level",
        "real_stage3_condition_reference_warning_level",
        "condition_distribution_support_score",
        "condition_distribution_warning_level",
        "condition_adjusted_confidence_score",
        "route_template_primary",
    ]
    preview_cols = [c for c in preview_cols if c in out.columns]

    preview = out[preview_cols].head(args.top_n)

    condition_support_col_used_counts = (
        out["final_recommendation_condition_support_col"]
        .value_counts(dropna=False)
        .to_dict()
        if "final_recommendation_condition_support_col" in out.columns
        else {}
    )

    base_score_col_used_counts = (
        out["final_recommendation_base_score_col"]
        .value_counts(dropna=False)
        .to_dict()
        if "final_recommendation_base_score_col" in out.columns
        else {}
    )

    top1 = out.iloc[0]

    lines = []
    lines.append("# Final Recommended Routes\n")
    lines.append(f"- input_csv: `{input_csv}`")
    lines.append(f"- n_routes: {len(out)}")
    lines.append(f"- top_n: {args.top_n}")
    lines.append(f"- finalizer: `{FINALIZER_NAME}`")
    lines.append(f"- top1_precursor_set: `{top1.get('precursor_set', '')}`")
    lines.append(f"- top1_final_recommendation_score: `{top1.get('final_recommendation_score', '')}`")
    lines.append(f"- condition_support_col_used_counts: `{json.dumps(condition_support_col_used_counts, ensure_ascii=False)}`")
    lines.append("")
    lines.append(preview.to_markdown(index=False))
    lines.append("")

    output_md.write_text("\n".join(lines), encoding="utf-8")

    top1_real_ref_support = None
    if "real_stage3_condition_reference_support_score" in out.columns:
        v = to_float(top1.get("real_stage3_condition_reference_support_score", np.nan), np.nan)
        top1_real_ref_support = float(v) if np.isfinite(v) else None

    top1_condition_distribution_support = None
    if "condition_distribution_support_score" in out.columns:
        v = to_float(top1.get("condition_distribution_support_score", np.nan), np.nan)
        top1_condition_distribution_support = float(v) if np.isfinite(v) else None

    top1_condition_support = to_float(
        top1.get("final_recommendation_condition_support", np.nan),
        np.nan,
    )

    top1_v43_safe_score = None
    if "stage35_v43_safe_strict_score" in out.columns:
        v = to_float(top1.get("stage35_v43_safe_strict_score", np.nan), np.nan)
        top1_v43_safe_score = float(v) if np.isfinite(v) else None

    top1_v43_template_score = None
    if "stage35_v43_template_chemonly_score" in out.columns:
        v = to_float(top1.get("stage35_v43_template_chemonly_score", np.nan), np.nan)
        top1_v43_template_score = float(v) if np.isfinite(v) else None

    summary = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "output_md": str(output_md),
        "n_routes": int(len(out)),
        "top_n": int(args.top_n),

        "finalizer": FINALIZER_NAME,

        "condition_support_col_used_counts": condition_support_col_used_counts,
        "base_score_col_used_counts": base_score_col_used_counts,
        "condition_support_col_used_top1": str(top1.get("final_recommendation_condition_support_col", "")),
        "base_score_col_used_top1": str(top1.get("final_recommendation_base_score_col", "")),
        "condition_warning_col_used_top1": str(top1.get("final_recommendation_condition_warning_col", "")),
        "condition_status_col_used_top1": str(top1.get("final_recommendation_condition_status_col", "")),

        "top1_precursor_set": str(top1.get("precursor_set", "")),
        "top1_final_recommendation_score": float(top1.get("final_recommendation_score", np.nan)),
        "top1_final_recommendation_status": str(top1.get("final_recommendation_status", "")),
        "top1_final_recommendation_penalty_reason": str(top1.get("final_recommendation_penalty_reason", "")),

        "top1_stage35_v43_safe_strict_score": top1_v43_safe_score,
        "top1_stage35_v43_template_chemonly_score": top1_v43_template_score,

        "top1_condition_support_score": float(top1_condition_support) if np.isfinite(top1_condition_support) else None,
        "top1_real_stage3_condition_reference_support_score": top1_real_ref_support,
        "top1_condition_distribution_support_score": top1_condition_distribution_support,

        "final_recommendation_status_counts": out["final_recommendation_status"].value_counts(dropna=False).to_dict(),
        "n_penalized": int((out["final_recommendation_penalty"] > 0).sum()),

        "claim_boundary": "final_recommendation_score_is_internal_ranking_score_not_experimental_validation",
        "interpretation": (
            "Final recommendations prioritize v43/safe-strict route ranking and use real Stage3 MDN/Flow "
            "condition-reference support when available; internal condition-distribution support is used as fallback."
        ),
    }

    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[SAVE] {output_csv}")
    print(f"[SAVE] {output_md}")
    print(f"[SAVE] {summary_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
