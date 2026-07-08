#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


ROUTE_LEVELS = [
    "route_high_confidence",
    "route_medium_confidence",
    "route_low_confidence",
    "route_needs_manual_check",
]


def write_md(df: pd.DataFrame, path: Path, title: str) -> None:
    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- Number of routes: `{len(df)}`")
    lines.append("")

    if len(df) == 0:
        lines.append("No routes.")
    else:
        cols = [
            "case_id",
            "case_refined_status",
            "route_refined_status",
            "route_refined_reason",
            "precursor_set",
            "final_recommendation_score",
            "stage35_v43_safe_strict_score",
            "real_stage3_condition_reference_support_score",
            "condition_distribution_support_score",
            "final_recommendation_status",
            "input_poscar",
        ]
        cols = [c for c in cols if c in df.columns]
        lines.append(df[cols].to_markdown(index=False))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_root", default="/Users/wyc/SynPred/outputs/batch_adaptive")
    ap.add_argument("--batch_name", default="batch_001")
    ap.add_argument("--input_name", default="batch_recommended_routes_topn.csv")
    args = ap.parse_args()

    batch_dir = Path(args.batch_root) / args.batch_name
    input_csv = batch_dir / args.input_name

    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input route table: {input_csv}")

    df = pd.read_csv(input_csv)

    if "route_refined_status" not in df.columns:
        raise ValueError(
            "Input table does not contain route_refined_status. "
            "Please rerun export_batch_recommendations.py first."
        )

    split_dir = batch_dir / "route_splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for level in ROUTE_LEVELS:
        sub = df[df["route_refined_status"] == level].copy()

        out_csv = split_dir / f"{level}.csv"
        out_md = split_dir / f"{level}.md"

        sub.to_csv(out_csv, index=False)
        write_md(sub, out_md, level)

        summary_rows.append({
            "route_refined_status": level,
            "n_routes": len(sub),
            "csv": str(out_csv),
            "md": str(out_md),
        })

        print("[SAVE]", out_csv)
        print("[SAVE]", out_md)

    # 额外输出一个推荐用的 merged 表：高可信 + 中可信
    recommended = df[
        df["route_refined_status"].isin([
            "route_high_confidence",
            "route_medium_confidence",
        ])
    ].copy()

    recommended_csv = split_dir / "route_recommended_high_medium.csv"
    recommended_md = split_dir / "route_recommended_high_medium.md"

    recommended.to_csv(recommended_csv, index=False)
    write_md(recommended, recommended_md, "Recommended High + Medium Confidence Routes")

    summary_rows.append({
        "route_refined_status": "route_recommended_high_medium",
        "n_routes": len(recommended),
        "csv": str(recommended_csv),
        "md": str(recommended_md),
    })

    summary = pd.DataFrame(summary_rows)
    summary_csv = split_dir / "route_split_summary.csv"
    summary_md = split_dir / "route_split_summary.md"

    summary.to_csv(summary_csv, index=False)

    lines = []
    lines.append("# Route Split Summary")
    lines.append("")
    lines.append(summary[["route_refined_status", "n_routes"]].to_markdown(index=False))
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[SAVE]", recommended_csv)
    print("[SAVE]", recommended_md)
    print("[SAVE]", summary_csv)
    print("[SAVE]", summary_md)
    print()
    print(summary[["route_refined_status", "n_routes"]].to_string(index=False))


if __name__ == "__main__":
    main()
