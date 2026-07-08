#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
from pathlib import Path
import pandas as pd


ROUTE_SUBDIR = "routes_flow_fallback_retrieval_baseline_element_reranked"


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
    Route-level precursor sanity check.
    这不是严格化学式解析器，只是先标记明显异常或需要复核的前驱体字符串。
    """
    if not precursor_set:
        return False

    suspicious_patterns = [
        r"\bP4P4\b",
        r"\bCa2CO3\b",
        r"\bCo2O3\b",
        r"\bCuCO3-Cu\(OH\)2\b",
        r"\bC60C60\b",
        r"\bNaCO3\b",
        r"\bAl_HO_3\b",
        r"\bMgH8_ClO6_2\b",
        r"\bSnH22C6_NCl2_2\b",
        r"\bK2CaNi_NO2_6\b",
    ]

    for pat in suspicious_patterns:
        if re.search(pat, precursor_set):
            return True

    # 单一有机分子作为唯一前驱体时，先标记为需要检查。
    # 对有机材料不一定错误，但在当前 inorganic synthesis route 场景下应人工复核。
    tokens = [x.strip() for x in precursor_set.split(";") if x.strip()]
    if len(tokens) == 1:
        only = tokens[0]
        organic_like = re.fullmatch(r"C\d*H\d*(O\d*)?(N\d*)?", only)
        if organic_like:
            return True

    return False


def classify_route(row) -> tuple[str, str]:
    """
    Per-route refined status.
    注意：这个是每一条 route 的分级，不是 case-level 分级。
    """
    precursor = get_text(row, "precursor_set")
    final_score = get_float(row, "final_recommendation_score", 0.0)
    safe_score = get_float(row, "stage35_v43_safe_strict_score", 0.0)

    ref_support = get_float(row, "real_stage3_condition_reference_support_score", None)
    if ref_support is None:
        ref_support = get_float(row, "condition_support_score", None)

    dist_support = get_float(row, "condition_distribution_support_score", None)

    final_status = get_text(row, "final_recommendation_status")
    warning_level = get_text(row, "real_stage3_condition_reference_warning_level")
    audit_level = get_text(row, "final_audit_level")

    reasons = []

    suspicious = has_suspicious_precursor(precursor)
    if suspicious:
        reasons.append("suspicious_precursor_formula")

    if "major" in warning_level.lower() or "major" in audit_level.lower():
        reasons.append("major_warning")

    if ref_support is None:
        reasons.append("missing_condition_reference_support")
    elif ref_support < 0.50:
        reasons.append("very_low_condition_reference_support")
    elif ref_support < 0.75:
        reasons.append("low_condition_reference_support")
    elif ref_support < 0.90:
        reasons.append("moderate_condition_reference_support")

    if dist_support is None:
        reasons.append("missing_condition_distribution_support")
    elif dist_support < 0.40:
        reasons.append("very_low_condition_distribution_support")
    elif dist_support < 0.70:
        reasons.append("moderate_condition_distribution_support")

    if final_score is None:
        reasons.append("missing_final_recommendation_score")
    elif final_score < 0.40:
        reasons.append("low_final_recommendation_score")
    elif final_score < 0.55:
        reasons.append("moderate_final_recommendation_score")

    if safe_score is not None and safe_score < 0.80:
        reasons.append("low_safe_strict_score")

    # hard flags
    if suspicious or "major_warning" in reasons:
        return "route_needs_manual_check", ";".join(reasons)

    # strong route
    if (
        ref_support is not None and ref_support >= 0.90
        and dist_support is not None and dist_support >= 0.70
        and final_score is not None and final_score >= 0.55
    ):
        return "route_high_confidence", ";".join(reasons) if reasons else "route_high_support"

    # usable route
    if (
        ref_support is not None and ref_support >= 0.75
        and final_score is not None and final_score >= 0.45
    ):
        return "route_medium_confidence", ";".join(reasons) if reasons else "route_medium_support"

    return "route_low_confidence", ";".join(reasons) if reasons else "route_low_combined_support"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default="/Users/wyc/SynPred")
    ap.add_argument("--batch_name", default="batch_001")
    ap.add_argument("--batch_root", default="/Users/wyc/SynPred/outputs/batch_adaptive")
    ap.add_argument("--top_n_per_case", type=int, default=5)
    args = ap.parse_args()

    project_root = Path(args.project_root)
    batch_dir = Path(args.batch_root) / args.batch_name
    master_csv = batch_dir / "master_status.csv"

    if not master_csv.exists():
        raise FileNotFoundError(f"Missing master_status.csv: {master_csv}")

    master = pd.read_csv(master_csv)
    rows = []

    for _, rec in master.iterrows():
        case_id = rec["case_id"]

        final_csv = (
            project_root
            / "outputs"
            / "inference"
            / case_id
            / ROUTE_SUBDIR
            / "final_recommended_routes.csv"
        )

        if not final_csv.exists():
            print(f"[SKIP] missing final routes: {case_id}")
            continue

        try:
            df = pd.read_csv(final_csv)
        except Exception as e:
            print(f"[SKIP] failed to read {final_csv}: {e}")
            continue

        if len(df) == 0:
            print(f"[SKIP] empty final routes: {case_id}")
            continue

        df = df.head(args.top_n_per_case).copy()

        # Case-level status from master_status.csv
        df.insert(0, "case_id", case_id)
        df.insert(1, "case_status", rec.get("final_case_status", ""))
        df.insert(2, "case_refined_status", rec.get("refined_case_status", ""))
        df.insert(3, "case_refined_reason", rec.get("refined_case_reason", ""))
        df.insert(4, "case_refined_condition_support_score", rec.get("refined_condition_support_score", ""))
        df.insert(5, "case_refined_final_score", rec.get("refined_final_score", ""))
        df.insert(6, "problem_type", rec.get("problem_type", ""))
        df.insert(7, "input_poscar", rec.get("input_poscar", ""))

        # Route-level status computed per row
        route_status = []
        route_reason = []
        for _, row in df.iterrows():
            s, r = classify_route(row)
            route_status.append(s)
            route_reason.append(r)

        df.insert(8, "route_refined_status", route_status)
        df.insert(9, "route_refined_reason", route_reason)

        rows.append(df)

    if rows:
        out = pd.concat(rows, ignore_index=True)
    else:
        out = pd.DataFrame()

    out_csv = batch_dir / "batch_recommended_routes_topn.csv"
    out_md = batch_dir / "batch_recommended_routes_topn.md"
    route_summary_csv = batch_dir / "batch_route_refined_summary.csv"
    route_summary_md = batch_dir / "batch_route_refined_summary.md"

    out.to_csv(out_csv, index=False)

    if len(out) > 0 and "route_refined_status" in out.columns:
        route_summary = (
            out.groupby(["case_id", "route_refined_status"])
            .size()
            .reset_index(name="n_routes")
        )
    else:
        route_summary = pd.DataFrame(columns=["case_id", "route_refined_status", "n_routes"])

    route_summary.to_csv(route_summary_csv, index=False)

    md_lines = []
    md_lines.append("# Batch Recommended Routes")
    md_lines.append("")
    md_lines.append(f"- Batch name: `{args.batch_name}`")
    md_lines.append(f"- Top N per case: `{args.top_n_per_case}`")
    md_lines.append(f"- Total routes exported: `{len(out)}`")
    md_lines.append("")

    if len(out) == 0:
        md_lines.append("No routes found.")
    else:
        preview_cols = [
            "case_id",
            "case_status",
            "case_refined_status",
            "route_refined_status",
            "route_refined_reason",
            "problem_type",
            "precursor_set",
            "final_recommendation_score",
            "stage35_v43_safe_strict_score",
            "real_stage3_condition_reference_support_score",
            "condition_distribution_support_score",
            "final_recommendation_status",
        ]
        preview_cols = [c for c in preview_cols if c in out.columns]
        md_lines.append(out[preview_cols].to_markdown(index=False))

    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    s_lines = []
    s_lines.append("# Batch Route Refined Summary")
    s_lines.append("")
    if len(route_summary) > 0:
        counts = out["route_refined_status"].value_counts().reset_index()
        counts.columns = ["route_refined_status", "n_routes"]
        s_lines.append("## Overall route-level status counts")
        s_lines.append("")
        s_lines.append(counts.to_markdown(index=False))
        s_lines.append("")
        s_lines.append("## Per-case route-level status counts")
        s_lines.append("")
        s_lines.append(route_summary.to_markdown(index=False))
    else:
        s_lines.append("No route summary available.")

    route_summary_md.write_text("\n".join(s_lines) + "\n", encoding="utf-8")

    print("[SAVE]", out_csv)
    print("[SAVE]", out_md)
    print("[SAVE]", route_summary_csv)
    print("[SAVE]", route_summary_md)
    print("[INFO] n_routes =", len(out))


if __name__ == "__main__":
    main()
