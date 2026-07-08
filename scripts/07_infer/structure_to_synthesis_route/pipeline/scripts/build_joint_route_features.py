#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def clean(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def has_token(text: str, token: str) -> int:
    return int(token.lower() in clean(text).lower())


def parse_precursors(s: str) -> list[str]:
    s = clean(s)
    if not s:
        return []
    return [x.strip() for x in re.split(r";|,", s) if x.strip()]


def precursor_role_features(precursor_set: str) -> dict:
    ps = parse_precursors(precursor_set)
    joined = ";".join(ps)

    n = len(ps)
    n_oxide = sum(("O" in p and "NO3" not in p and "CO3" not in p) for p in ps)
    n_nitrate = sum(("NO3" in p or "Nitrate" in p) for p in ps)
    n_carbonate = sum(("CO3" in p or "Carbonate" in p) for p in ps)
    n_elemental = 0

    # conservative elemental check: single element-like token without digits/parentheses/O/C/N groups
    for p in ps:
        pp = p.strip()
        if re.fullmatch(r"[A-Z][a-z]?", pp):
            n_elemental += 1

    return {
        "v3_n_precursors": n,
        "v3_n_oxide_precursors": int(n_oxide),
        "v3_n_nitrate_precursors": int(n_nitrate),
        "v3_n_carbonate_precursors": int(n_carbonate),
        "v3_n_elemental_precursors": int(n_elemental),
        "v3_has_nitrate": int(n_nitrate > 0),
        "v3_has_carbonate": int(n_carbonate > 0),
        "v3_has_elemental": int(n_elemental > 0),
        "v3_has_oxide": int(n_oxide > 0),
    }


def safe_float(row, col: str, default: float = 0.0) -> float:
    try:
        return float(row.get(col, default))
    except Exception:
        return default


def build_features(row) -> dict:
    f = {}

    # Existing numerical signals
    f["v3_stage35_v21_score"] = safe_float(row, "stage35_v21_score")
    f["v3_stage35_v2_prob"] = safe_float(row, "stage35_v2_prob")
    f["v3_stage3_score"] = safe_float(row, "stage3_score")
    f["v3_element_coverage"] = safe_float(row, "element_coverage")
    f["v3_missing_count"] = safe_float(row, "missing_count")
    f["v3_extra_element_penalty"] = safe_float(row, "extra_element_penalty")
    f["v3_temperature_c"] = safe_float(row, "temperature_c")
    f["v3_time_h"] = safe_float(row, "time_h")
    f["v3_route_confidence_score"] = safe_float(row, "route_confidence_score")

    # Rank-like features
    f["v3_final_route_rank"] = safe_float(row, "final_route_rank")
    f["v3_precursor_rank"] = safe_float(row, "precursor_rank")
    f["v3_stage35_v21_rank"] = safe_float(row, "stage35_v21_rank")

    # Condition source
    condition_source = clean(row.get("condition_source", ""))
    f["v3_condition_is_top_component_mean"] = int(condition_source == "top_component_mean")
    f["v3_condition_is_flow_sample"] = int(condition_source == "flow_sample")
    f["v3_condition_is_lgbm_top1"] = int(condition_source == "lgbm_top1")

    # LightGBM stage3 discrete predictions
    pred_atmosphere = clean(row.get("pred_atmosphere", ""))
    f["v3_pred_atmosphere_nonoxidizing"] = int(pred_atmosphere == "non-oxidizing")
    f["v3_pred_atmosphere_proba"] = safe_float(row, "pred_atmosphere_proba")
    pred_time_bucket = clean(row.get("pred_time_bucket", ""))
    f["v3_pred_time_bucket_short"] = int(pred_time_bucket == "short")
    f["v3_pred_time_bucket_medium"] = int(pred_time_bucket == "medium")

    # QC / confidence / warning signals
    qc_level = clean(row.get("precursor_qc_level", ""))
    warning_level = clean(row.get("route_warning_level", ""))
    rec_status = clean(row.get("route_recommendation_status", ""))
    conf_level = clean(row.get("route_confidence_level", ""))

    f["v3_precursor_qc_pass"] = int(qc_level == "pass")
    f["v3_precursor_qc_minor"] = int(qc_level == "minor_warning")
    f["v3_precursor_qc_major"] = int(qc_level == "major_warning")

    f["v3_route_warning_none"] = int(warning_level == "no_warning")
    f["v3_route_warning_minor"] = int(warning_level == "minor_warning")
    f["v3_route_warning_major"] = int(warning_level == "major_warning")

    f["v3_status_recommended"] = int(rec_status == "recommended")
    f["v3_status_recommended_with_validation"] = int(rec_status == "recommended_with_validation")
    f["v3_status_review_required"] = int(rec_status == "review_required")

    f["v3_confidence_high"] = int(conf_level == "high_confidence")
    f["v3_confidence_medium"] = int(conf_level == "medium_confidence")
    f["v3_confidence_low"] = int(conf_level == "low_confidence")

    # Warnings text features
    route_warnings = clean(row.get("route_warnings", ""))
    precursor_qc_warnings = clean(row.get("precursor_qc_warnings", ""))

    f["v3_warn_high_temperature"] = has_token(route_warnings, "high_temperature")
    f["v3_warn_extra_non_target_elements"] = has_token(precursor_qc_warnings, "extra_non_target_elements")
    f["v3_warn_elemental_precursors"] = has_token(precursor_qc_warnings, "elemental_precursors")

    # Precursor role features
    f.update(precursor_role_features(clean(row.get("precursor_set", ""))))

    # Simple joint score for v3 bootstrap.
    # This is not the final learned ranker; it is a transparent initial route-level feature score.
    score = 0.0
    score += 2.0 * f["v3_element_coverage"]
    score += 1.5 * f["v3_route_confidence_score"]
    score += 1.0 * f["v3_stage3_score"]
    score += 0.8 * f["v3_condition_is_top_component_mean"]
    score += 0.8 * f["v3_condition_is_lgbm_top1"]
    score += 0.5 * f["v3_precursor_qc_pass"]
    score -= 0.4 * f["v3_precursor_qc_minor"]
    score -= 1.2 * f["v3_precursor_qc_major"]
    score -= 0.5 * f["v3_warn_high_temperature"]
    score -= 0.3 * f["v3_has_nitrate"]
    score -= 0.2 * f["v3_has_carbonate"]
    score -= 0.2 * f["v3_has_elemental"]

    f["v3_joint_feature_score"] = float(score)

    return f


def main():
    ap = argparse.ArgumentParser()
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

    if args.top_n and len(df) > args.top_n:
        if "sample_id" in df.columns:
            df = df.groupby("sample_id", sort=False).head(args.top_n).reset_index(drop=True)
        else:
            df = df.head(args.top_n).copy()

    feature_rows = []
    for _, row in df.iterrows():
        feature_rows.append(build_features(row))

    feat = pd.DataFrame(feature_rows)

    out = pd.concat([df.reset_index(drop=True), feat.reset_index(drop=True)], axis=1)

    out = out.sort_values(
        by=["v3_joint_feature_score", "route_confidence_score"],
        ascending=[False, False],
        kind="mergesort",
    ).reset_index(drop=True)

    out["v3_joint_feature_rank"] = np.arange(1, len(out) + 1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    view_cols = [
        "v3_joint_feature_rank",
        "final_route_rank",
        "precursor_set",
        "temperature_c",
        "time_h",
        "route_confidence_score",
        "route_confidence_level",
        "route_warning_level",
        "route_recommendation_status",
        "precursor_qc_level",
        "v3_joint_feature_score",
        "v3_element_coverage",
        "v3_stage3_score",
        "v3_condition_is_top_component_mean",
        "v3_has_oxide",
        "v3_has_carbonate",
        "v3_has_nitrate",
        "v3_has_elemental",
    ]
    view_cols = [c for c in view_cols if c in out.columns]
    output_md.write_text(out[view_cols].to_markdown(index=False), encoding="utf-8")

    summary = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "output_md": str(output_md),
        "n_routes": int(len(out)),
        "n_v3_feature_columns": int(len([c for c in out.columns if c.startswith("v3_")])),
        "top_joint_feature_score": float(out["v3_joint_feature_score"].max()) if len(out) else None,
        "mean_joint_feature_score": float(out["v3_joint_feature_score"].mean()) if len(out) else None,
        "claim_boundary": "v3_joint_feature_score_is_bootstrap_rule_score_not_learned_ranker",
    }

    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[SAVE] {output_csv}")
    print(f"[SAVE] {output_md}")
    print(f"[SAVE] {summary_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
