#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def clean(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def pick_input_csv(route_dir: Path) -> Path:
    candidates = [
        route_dir / "final_top_routes_v3_joint_reranked.csv",
        route_dir / "final_top_routes_with_joint_features.csv",
        route_dir / "final_top_routes_with_confidence.csv",
        route_dir / "final_top_routes_with_precursor_qc.csv",
        route_dir / "final_top_routes.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"No final route csv found in {route_dir}")


def rename_for_report(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "v3_joint_rerank_rank": "V3 rank",
        "v3_joint_feature_rank": "Feature rank",
        "final_route_rank": "Original rank",
        "precursor_set": "Precursors",
        "temperature_c": "Temp. (°C)",
        "time_h": "Time (h)",
        "v3_joint_feature_score": "V3 joint score",
        "route_confidence_score": "Confidence",
        "route_confidence_level": "Confidence level",
        "route_warning_level": "Warning level",
        "route_recommendation_status": "Status",
        "precursor_qc_level": "Precursor QC",
        "precursor_qc_warnings": "Precursor QC warnings",
        "route_warnings": "Route warnings",
        "condition_source": "Condition source",
        "stage3_score": "Stage3 score",
        "stage35_v21_score": "Stage35 score",
    }
    return df.rename(columns={k: v for k, v in rename.items() if k in df.columns})


def table_md(df: pd.DataFrame, cols: list[str], top_n: int) -> str:
    cols = [c for c in cols if c in df.columns]
    if not cols or df.empty:
        return "_No routes available._\n"
    return df[cols].head(top_n).to_markdown(index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route_out_dir", required=True)
    ap.add_argument("--output_md", required=True)
    ap.add_argument("--top_n", type=int, default=10)
    args = ap.parse_args()

    route_dir = Path(args.route_out_dir)
    output_md = Path(args.output_md)

    input_csv = pick_input_csv(route_dir)
    df = pd.read_csv(input_csv)

    # Keep ordering from v3 rerank file if available.
    if "v3_joint_rerank_rank" in df.columns:
        df = df.sort_values("v3_joint_rerank_rank", ascending=True, kind="mergesort")
    elif "v3_joint_feature_rank" in df.columns:
        df = df.sort_values("v3_joint_feature_rank", ascending=True, kind="mergesort")
    elif "final_route_rank" in df.columns:
        df = df.sort_values("final_route_rank", ascending=True, kind="mergesort")

    report_df = rename_for_report(df)

    lines = []
    lines.append("# Structure-to-Synthesis Route Report v3\n")
    lines.append("## Claim boundary\n")
    lines.append("- These routes are model-suggested candidates, not experimentally validated protocols.\n")
    lines.append("- The v3 joint reranker is a transparent rule-based bootstrap reranker, not a learned ranker.\n")
    lines.append("- Routes marked as `review_required` should not be treated as ready-to-use synthesis protocols.\n")

    lines.append("\n## Input source\n")
    lines.append(f"- Route table: `{input_csv}`\n")
    lines.append(f"- Total routes: {len(df)}\n")

    if "Status" in report_df.columns:
        lines.append("\n## Recommendation status\n")
        status_counts = report_df["Status"].fillna("NA").value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        lines.append(status_counts.to_markdown(index=False))
        lines.append("")

    if "Precursor QC" in report_df.columns:
        lines.append("\n## Precursor QC levels\n")
        qc_counts = report_df["Precursor QC"].fillna("NA").value_counts().reset_index()
        qc_counts.columns = ["Precursor QC", "Count"]
        lines.append(qc_counts.to_markdown(index=False))
        lines.append("")

    common_cols = [
        "V3 rank",
        "Feature rank",
        "Original rank",
        "Precursors",
        "Temp. (°C)",
        "Time (h)",
        "V3 joint score",
        "Confidence",
        "Confidence level",
        "Warning level",
        "Status",
        "Precursor QC",
        "Precursor QC warnings",
        "Route warnings",
    ]

    if "Status" in report_df.columns:
        recommended = report_df[report_df["Status"] == "recommended"].copy()
        validation = report_df[report_df["Status"] == "recommended_with_validation"].copy()
        review = report_df[report_df["Status"] == "review_required"].copy()
    else:
        recommended = report_df.iloc[0:0].copy()
        validation = report_df.copy()
        review = report_df.iloc[0:0].copy()

    lines.append("\n## Recommended routes\n")
    lines.append(table_md(recommended, common_cols, args.top_n))

    lines.append("\n## Recommended with validation\n")
    lines.append(table_md(validation, common_cols, args.top_n))

    lines.append("\n## Review required routes\n")
    lines.append(table_md(review, common_cols, args.top_n))

    lines.append("\n## Notes\n")
    lines.append("- `V3 joint score` is a bootstrap score integrating route confidence, Stage3 score, precursor QC, warning level, and precursor role features.\n")
    lines.append("- Carbonates, nitrates, and elemental precursors are not automatically invalid, but they introduce decomposition, by-product, atmosphere, or mass-balance considerations.\n")
    lines.append("- Future v3.x versions may replace the bootstrap score with learned structure-aware or evidence-aware ranking.\n")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"[SAVE] {output_md}")
    print(f"[INFO] input_csv = {input_csv}")
    print(f"[INFO] total_routes = {len(df)}")


if __name__ == "__main__":
    main()
