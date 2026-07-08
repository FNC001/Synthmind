#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


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

    if "v3_joint_feature_score" not in df.columns:
        raise ValueError("Missing required column: v3_joint_feature_score")

    if "route_confidence_score" not in df.columns:
        df["route_confidence_score"] = 0.0

    if "stage35_v21_score" not in df.columns:
        df["stage35_v21_score"] = 0.0

    # Conservative bootstrap rerank:
    # primary: v3 joint feature score
    # secondary: confidence score
    # tertiary: inherited Stage35 score
    if "sample_id" in df.columns and df["sample_id"].nunique() > 1:
        df = df.sort_values(
            by=["sample_id", "v3_joint_feature_score", "route_confidence_score", "stage35_v21_score"],
            ascending=[True, False, False, False],
            kind="mergesort",
        ).reset_index(drop=True)
    else:
        df = df.sort_values(
            by=["v3_joint_feature_score", "route_confidence_score", "stage35_v21_score"],
            ascending=[False, False, False],
            kind="mergesort",
        ).reset_index(drop=True)

    if args.top_n and len(df) > args.top_n:
        if "sample_id" in df.columns:
            df = df.groupby("sample_id", sort=False).head(args.top_n).reset_index(drop=True)
        else:
            df = df.head(args.top_n).copy()

    if "sample_id" in df.columns and df["sample_id"].nunique() > 1:
        df["v3_joint_rerank_rank"] = df.groupby("sample_id").cumcount() + 1
    else:
        df["v3_joint_rerank_rank"] = np.arange(1, len(df) + 1)
    df["v3_joint_rerank_source"] = "bootstrap_joint_feature_score"
    df["v3_joint_rerank_claim_boundary"] = "rule-based bootstrap rerank, not learned ranker"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    view_cols = [
        "v3_joint_rerank_rank",
        "v3_joint_feature_rank",
        "final_route_rank",
        "precursor_set",
        "temperature_c",
        "time_h",
        "v3_joint_feature_score",
        "route_confidence_score",
        "route_confidence_level",
        "route_warning_level",
        "route_recommendation_status",
        "precursor_qc_level",
        "precursor_qc_warnings",
        "condition_source",
        "stage3_score",
        "stage35_v21_score",
    ]
    view_cols = [c for c in view_cols if c in df.columns]
    output_md.write_text(df[view_cols].to_markdown(index=False), encoding="utf-8")

    summary = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "output_md": str(output_md),
        "n_routes": int(len(df)),
        "top_n": int(args.top_n),
        "top_precursor_set": str(df.iloc[0].get("precursor_set", "")) if len(df) else "",
        "top_v3_joint_feature_score": float(df.iloc[0]["v3_joint_feature_score"]) if len(df) else None,
        "claim_boundary": "rule-based bootstrap rerank, not learned ranker",
    }

    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[SAVE] {output_csv}")
    print(f"[SAVE] {output_md}")
    print(f"[SAVE] {summary_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
