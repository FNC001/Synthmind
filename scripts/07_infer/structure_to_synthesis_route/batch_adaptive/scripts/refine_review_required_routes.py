#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
import re
import pandas as pd


def get_float(row, col, default=None):
    if col not in row.index:
        return default
    try:
        v = row[col]
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def get_text(row, col, default=""):
    if col not in row.index:
        return default
    v = row[col]
    if pd.isna(v):
        return default
    return str(v)


def has_suspicious_precursor(precursor_set: str) -> bool:
    """
    先用轻量规则标记明显异常的 precursor 字符串。
    后续可以扩展为化学式解析器。
    """
    suspicious_patterns = [
        r"\bP4P4\b",
        r"\bCa2CO3\b",
        r"\bCo2O3\b",
        r"\bCuCO3-Cu\(OH\)2\b",
    ]

    for pat in suspicious_patterns:
        if re.search(pat, precursor_set):
            return True
    return False


def classify_route(row) -> tuple[str, str]:
    precursor = get_text(row, "precursor_set")
    final_score = get_float(row, "final_recommendation_score", 0.0)
    safe_score = get_float(row, "stage35_v43_safe_strict_score", 0.0)
    ref_support = get_float(row, "real_stage3_condition_reference_support_score", None)
    dist_support = get_float(row, "condition_distribution_support_score", None)

    reasons = []

    suspicious = has_suspicious_precursor(precursor)
    if suspicious:
        reasons.append("suspicious_precursor_formula")

    if ref_support is None:
        reasons.append("missing_condition_reference_support")
    elif ref_support < 0.60:
        reasons.append("very_low_condition_reference_support")
    elif ref_support < 0.75:
        reasons.append("low_condition_reference_support")

    if dist_support is None:
        reasons.append("missing_condition_distribution_support")
    elif dist_support < 0.40:
        reasons.append("very_low_condition_distribution_support")
    elif dist_support < 0.70:
        reasons.append("moderate_condition_distribution_support")

    if final_score is None or final_score < 0.40:
        reasons.append("low_final_recommendation_score")
    elif final_score < 0.55:
        reasons.append("moderate_final_recommendation_score")

    if suspicious:
        return "needs_manual_check", ";".join(reasons)

    if ref_support is not None and ref_support >= 0.90:
        if dist_support is not None and dist_support >= 0.70 and final_score >= 0.55:
            return "high_confidence_review", ";".join(reasons) if reasons else "high_support_review_only"

    if ref_support is not None and ref_support >= 0.75 and final_score >= 0.45:
        return "medium_confidence_review", ";".join(reasons) if reasons else "medium_support_review_only"

    return "low_confidence_review", ";".join(reasons) if reasons else "low_combined_support"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_root", default="/Users/wyc/SynPred/outputs/batch_adaptive")
    ap.add_argument("--batch_name", default="batch_001")
    ap.add_argument("--input_name", default="batch_recommended_routes_topn.csv")
    args = ap.parse_args()

    batch_dir = Path(args.batch_root) / args.batch_name
    input_csv = batch_dir / args.input_name

    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input csv: {input_csv}")

    df = pd.read_csv(input_csv)

    if len(df) == 0:
        raise RuntimeError("Input route table is empty.")

    labels = []
    reasons = []
    for _, row in df.iterrows():
        label, reason = classify_route(row)
        labels.append(label)
        reasons.append(reason)

    df["review_severity"] = labels
    df["review_reason_refined"] = reasons

    # 排序：优先高置信，然后中等，再低置信，最后人工检查
    priority = {
        "high_confidence_review": 0,
        "medium_confidence_review": 1,
        "low_confidence_review": 2,
        "needs_manual_check": 3,
    }
    df["_review_priority"] = df["review_severity"].map(priority).fillna(9)

    score_col = "final_recommendation_score" if "final_recommendation_score" in df.columns else None
    sort_cols = ["case_id", "_review_priority"]
    ascending = [True, True]

    if score_col:
        sort_cols.append(score_col)
        ascending.append(False)

    df = df.sort_values(sort_cols, ascending=ascending).drop(columns=["_review_priority"])

    out_csv = batch_dir / "batch_recommended_routes_topn_review_refined.csv"
    out_md = batch_dir / "batch_recommended_routes_topn_review_refined.md"
    summary_csv = batch_dir / "batch_review_refined_summary.csv"
    summary_md = batch_dir / "batch_review_refined_summary.md"

    df.to_csv(out_csv, index=False)

    summary = (
        df.groupby(["case_id", "review_severity"])
        .size()
        .reset_index(name="n_routes")
    )
    summary.to_csv(summary_csv, index=False)

    preview_cols = [
        "case_id",
        "review_severity",
        "review_reason_refined",
        "precursor_set",
        "final_recommendation_score",
        "real_stage3_condition_reference_support_score",
        "condition_distribution_support_score",
        "final_recommendation_status",
    ]
    preview_cols = [c for c in preview_cols if c in df.columns]

    lines = []
    lines.append("# Refined Review Required Routes")
    lines.append("")
    lines.append(f"- Input: `{input_csv}`")
    lines.append(f"- Total routes: `{len(df)}`")
    lines.append("")
    lines.append("## Review severity counts")
    lines.append("")
    counts = df["review_severity"].value_counts().reset_index()
    counts.columns = ["review_severity", "n_routes"]
    lines.append(counts.to_markdown(index=False))
    lines.append("")
    lines.append("## Route preview")
    lines.append("")
    lines.append(df[preview_cols].to_markdown(index=False))
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    s_lines = []
    s_lines.append("# Batch Review Refined Summary")
    s_lines.append("")
    s_lines.append(summary.to_markdown(index=False))
    summary_md.write_text("\n".join(s_lines) + "\n", encoding="utf-8")

    print("[SAVE]", out_csv)
    print("[SAVE]", out_md)
    print("[SAVE]", summary_csv)
    print("[SAVE]", summary_md)
    print()
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
