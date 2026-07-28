#!/usr/bin/env python
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


def fmt_float(x, ndigits=3):
    try:
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return ""


def find_input_csv(route_out_dir: Path) -> Path:
    candidates = [
        route_out_dir / "final_top_routes_with_confidence.csv",
        route_out_dir / "final_top_routes_with_precursor_qc.csv",
        route_out_dir / "final_top_routes.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "No final route csv found. Tried:\n" + "\n".join(str(p) for p in candidates)
    )


def make_table(df: pd.DataFrame, top_n: int) -> str:
    if df.empty:
        return "_No routes in this category._\n"

    cols = [
        "final_route_rank",
        "precursor_set",
        "temperature_c",
        "time_h",
        "route_confidence_score",
        "route_confidence_level",
        "route_warning_level",
        "route_recommendation_status",
        "precursor_qc_level",
        "precursor_qc_warnings",
        "route_warnings",
    ]
    cols = [c for c in cols if c in df.columns]

    view = df[cols].head(top_n).copy()

    rename = {
        "final_route_rank": "Rank",
        "precursor_set": "Precursors",
        "temperature_c": "Temp. (°C)",
        "time_h": "Time (h)",
        "route_confidence_score": "Confidence",
        "route_confidence_level": "Confidence level",
        "route_warning_level": "Warning level",
        "route_recommendation_status": "Status",
        "precursor_qc_level": "Precursor QC",
        "precursor_qc_warnings": "Precursor QC warnings",
        "route_warnings": "Route warnings",
    }

    for c in ["temperature_c", "time_h", "route_confidence_score"]:
        if c in view.columns:
            nd = 1 if c == "temperature_c" else 3
            view[c] = view[c].apply(lambda x: fmt_float(x, nd))

    view = view.rename(columns=rename)
    return view.to_markdown(index=False) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--route_out_dir", required=True)
    ap.add_argument("--output_md", required=True)
    ap.add_argument("--top_n", type=int, default=10)
    args = ap.parse_args()

    route_out_dir = Path(args.route_out_dir)
    output_md = Path(args.output_md)

    input_csv = find_input_csv(route_out_dir)
    df = pd.read_csv(input_csv)

    if "final_route_rank" in df.columns:
        df = df.sort_values("final_route_rank", kind="mergesort").reset_index(drop=True)

    n_total = len(df)

    status_counts = (
        df["route_recommendation_status"].value_counts(dropna=False).to_dict()
        if "route_recommendation_status" in df.columns else {}
    )
    confidence_counts = (
        df["route_confidence_level"].value_counts(dropna=False).to_dict()
        if "route_confidence_level" in df.columns else {}
    )
    warning_counts = (
        df["route_warning_level"].value_counts(dropna=False).to_dict()
        if "route_warning_level" in df.columns else {}
    )
    qc_counts = (
        df["precursor_qc_level"].value_counts(dropna=False).to_dict()
        if "precursor_qc_level" in df.columns else {}
    )

    if "route_recommendation_status" in df.columns:
        recommended = df[df["route_recommendation_status"] == "recommended"]
        validation = df[df["route_recommendation_status"] == "recommended_with_validation"]
        review = df[df["route_recommendation_status"] == "review_required"]
    else:
        recommended = df.iloc[0:0]
        validation = df
        review = df.iloc[0:0]

    lines = []
    lines.append("# Final Structure-to-Synthesis Route Report\n")
    lines.append("## Report scope\n")
    lines.append(f"- Input route table: `{input_csv}`")
    lines.append(f"- Total final routes: **{n_total}**")
    lines.append("- Claim boundary: these routes are model-suggested candidates, not experimentally validated protocols.\n")

    lines.append("## Reliability summary\n")
    lines.append("### Recommendation status\n")
    if status_counts:
        for k, v in status_counts.items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- No route recommendation status column found.")

    lines.append("\n### Confidence levels\n")
    if confidence_counts:
        for k, v in confidence_counts.items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- No route confidence column found.")

    lines.append("\n### Warning levels\n")
    if warning_counts:
        for k, v in warning_counts.items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- No route warning column found.")

    lines.append("\n### Precursor QC levels\n")
    if qc_counts:
        for k, v in qc_counts.items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- No precursor QC column found.")

    lines.append("\n## Recommended routes\n")
    lines.append(make_table(recommended, args.top_n))

    lines.append("\n## Recommended with validation\n")
    lines.append(make_table(validation, args.top_n))

    lines.append("\n## Review required routes\n")
    lines.append(make_table(review, args.top_n))

    lines.append("\n## Notes for human review\n")
    lines.append(
        "- `extra_non_target_elements` means that the precursor introduces elements outside the target formula, "
        "such as C or N from carbonates/nitrates. These may be chemically acceptable, but require validation of "
        "decomposition pathways, atmosphere, by-products, and mass balance."
    )
    lines.append(
        "- `elemental_precursors` indicates that an elemental precursor is used. This may be valid for some systems, "
        "but usually requires checking volatility, oxidation state, and compatibility with the proposed synthesis temperature."
    )
    lines.append(
        "- `review_required` routes should not be treated as directly recommended recipes. They are retained as model-generated candidates for expert inspection."
    )

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[SAVE] {output_md}")
    print(f"[INFO] input_csv = {input_csv}")
    print(f"[INFO] total_routes = {n_total}")


if __name__ == "__main__":
    main()
