#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd


def clean(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def merge_warnings(*items):
    out = []
    seen = set()
    for item in items:
        s = clean(item)
        if not s:
            continue
        for part in [x.strip() for x in s.split(";") if x.strip()]:
            if part not in seen:
                seen.add(part)
                out.append(part)
    return "; ".join(out) if out else ""


def score_to_level(x):
    try:
        x = float(x)
    except Exception:
        x = 0.0
    if x >= 0.78:
        return "high_confidence"
    if x >= 0.50:
        return "medium_confidence"
    return "low_confidence"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--output_md", default="")
    args = ap.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md) if args.output_md else output_csv.with_suffix(".md")

    if not input_csv.exists():
        raise FileNotFoundError(input_csv)

    df = pd.read_csv(input_csv)

    if "precursor_qc_level" not in df.columns:
        raise ValueError(
            "input csv has no precursor_qc_level column; "
            "please make sure confidence input is final_top_routes_with_precursor_qc.csv"
        )

    if "route_confidence_score" not in df.columns:
        df["route_confidence_score"] = 0.5
    df["route_confidence_score"] = pd.to_numeric(
        df["route_confidence_score"], errors="coerce"
    ).fillna(0.0)

    if "route_confidence_level" not in df.columns:
        df["route_confidence_level"] = df["route_confidence_score"].apply(score_to_level)

    if "route_warning_level" not in df.columns:
        df["route_warning_level"] = "no_warning"

    if "route_recommendation_status" not in df.columns:
        df["route_recommendation_status"] = "recommended_with_validation"

    if "route_warnings" not in df.columns:
        df["route_warnings"] = ""

    for col in [
        "route_confidence_level",
        "route_warning_level",
        "route_recommendation_status",
        "route_warnings",
    ]:
        if col in df.columns:
            df[col] = df[col].apply(clean).astype(object)

    changed_major = 0
    changed_minor = 0

    for idx, row in df.iterrows():
        qc_level = clean(row.get("precursor_qc_level", ""))
        qc_warnings = clean(row.get("precursor_qc_warnings", ""))

        if qc_level == "major_warning":
            df.at[idx, "route_confidence_score"] = max(
                0.0,
                float(df.at[idx, "route_confidence_score"]) - 0.15,
            )
            df.at[idx, "route_warning_level"] = "major_warning"
            df.at[idx, "route_recommendation_status"] = "review_required"

            msg = "precursor_qc_major_warning"
            if qc_warnings:
                msg += f":{qc_warnings}"

            df.at[idx, "route_warnings"] = merge_warnings(
                row.get("route_warnings", ""),
                msg,
            )
            changed_major += 1

        elif qc_level == "minor_warning":
            df.at[idx, "route_confidence_score"] = max(
                0.0,
                float(df.at[idx, "route_confidence_score"]) - 0.05,
            )

            old_warning = clean(row.get("route_warning_level", "no_warning"))
            if old_warning == "no_warning":
                df.at[idx, "route_warning_level"] = "minor_warning"

            old_status = clean(row.get("route_recommendation_status", "recommended"))
            if old_status == "recommended":
                df.at[idx, "route_recommendation_status"] = "recommended_with_validation"

            msg = "precursor_qc_minor_warning"
            if qc_warnings:
                msg += f":{qc_warnings}"

            df.at[idx, "route_warnings"] = merge_warnings(
                row.get("route_warnings", ""),
                msg,
            )
            changed_minor += 1

    df["route_confidence_level"] = df["route_confidence_score"].apply(score_to_level)

    df["route_warnings"] = df["route_warnings"].apply(clean)
    df.loc[df["route_warnings"] == "", "route_warnings"] = pd.NA

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    view_cols = [
        "final_route_rank",
        "precursor_set",
        "temperature_c",
        "time_h",
        "precursor_qc_score",
        "precursor_qc_level",
        "precursor_qc_status",
        "precursor_qc_warnings",
        "route_confidence_score",
        "route_confidence_level",
        "route_warning_level",
        "route_recommendation_status",
        "route_warnings",
    ]
    view_cols = [c for c in view_cols if c in df.columns]
    output_md.write_text(df[view_cols].to_markdown(index=False), encoding="utf-8")

    bad = (
        (df.get("precursor_qc_level", "") == "major_warning")
        & (df.get("route_recommendation_status", "") != "review_required")
    ).sum()

    print(f"[SAVE] {output_csv}")
    print(f"[SAVE] {output_md}")
    print(f"[OK] minor QC propagated: {changed_minor}")
    print(f"[OK] major QC propagated: {changed_major}")
    print(f"[CHECK] major QC but not review_required: {int(bad)}")


if __name__ == "__main__":
    main()
