#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
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


def bool01(series: pd.Series) -> pd.Series:
    return series.fillna(0).astype(float).clip(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--feature_cols_json", required=True)
    ap.add_argument("--summary_json", required=True)
    ap.add_argument("--top_positive_frac", type=float, default=0.25)
    ap.add_argument("--bottom_negative_frac", type=float, default=0.35)
    args = ap.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    feature_cols_json = Path(args.feature_cols_json)
    summary_json = Path(args.summary_json)

    if not input_csv.exists():
        raise FileNotFoundError(input_csv)

    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError(f"Empty input csv: {input_csv}")

    # Ensure rank ordering.
    if "v3_joint_rerank_rank" in df.columns:
        df = df.sort_values("v3_joint_rerank_rank", ascending=True, kind="mergesort")
    elif "v3_joint_feature_rank" in df.columns:
        df = df.sort_values("v3_joint_feature_rank", ascending=True, kind="mergesort")
    elif "final_route_rank" in df.columns:
        df = df.sort_values("final_route_rank", ascending=True, kind="mergesort")

    df = df.reset_index(drop=True)

    n = len(df)
    n_pos = max(1, int(round(n * args.top_positive_frac)))
    n_neg = max(1, int(round(n * args.bottom_negative_frac)))

    labels = np.full(n, 0.5, dtype=float)
    labels[:n_pos] = 1.0
    labels[max(n - n_neg, n_pos):] = 0.0

    df["v3_train_label"] = labels
    df["v3_train_label_type"] = "middle_uncertain"
    df.loc[df.index < n_pos, "v3_train_label_type"] = "positive_top_ranked"
    df.loc[df.index >= max(n - n_neg, n_pos), "v3_train_label_type"] = "negative_bottom_ranked"

    # Pairwise/listwise ranking target.
    if "v3_joint_rerank_rank" in df.columns:
        rank = pd.to_numeric(df["v3_joint_rerank_rank"], errors="coerce").fillna(n)
    else:
        rank = pd.Series(np.arange(1, n + 1), index=df.index)

    df["v3_rank_target"] = 1.0 - (rank - 1.0) / max(n - 1.0, 1.0)
    df["v3_rank_target"] = df["v3_rank_target"].clip(0, 1)

    # Candidate feature columns.
    feature_cols = []

    preferred_numeric = [
        "stage35_v21_score",
        "stage35_v2_prob",
        "stage3_score",
        "temperature_c",
        "time_h",
        "element_coverage",
        "missing_count",
        "extra_element_penalty",
        "route_confidence_score",
        "precursor_qc_score",
        "v3_joint_feature_score",
    ]

    for c in preferred_numeric:
        if c in df.columns:
            feature_cols.append(c)

    # Add all v3_ numerical feature columns except labels/ranks that leak too directly.
    blocked = {
        "v3_joint_rerank_rank",
        "v3_joint_feature_rank",
        "v3_train_label",
        "v3_rank_target",
    }

    for c in df.columns:
        if c.startswith("v3_") and c not in blocked and c not in feature_cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                feature_cols.append(c)

    # Encode categorical safety/status columns into simple numeric indicators.
    if "route_warning_level" in df.columns:
        df["feat_warning_major"] = (df["route_warning_level"].astype(str) == "major_warning").astype(int)
        df["feat_warning_minor"] = (df["route_warning_level"].astype(str) == "minor_warning").astype(int)
        feature_cols.extend(["feat_warning_major", "feat_warning_minor"])

    if "route_recommendation_status" in df.columns:
        df["feat_status_recommended"] = (df["route_recommendation_status"].astype(str) == "recommended").astype(int)
        df["feat_status_validation"] = (df["route_recommendation_status"].astype(str) == "recommended_with_validation").astype(int)
        df["feat_status_review"] = (df["route_recommendation_status"].astype(str) == "review_required").astype(int)
        feature_cols.extend(["feat_status_recommended", "feat_status_validation", "feat_status_review"])

    if "precursor_qc_level" in df.columns:
        df["feat_qc_pass"] = (df["precursor_qc_level"].astype(str) == "pass").astype(int)
        df["feat_qc_minor"] = (df["precursor_qc_level"].astype(str) == "minor_warning").astype(int)
        df["feat_qc_major"] = (df["precursor_qc_level"].astype(str) == "major_warning").astype(int)
        feature_cols.extend(["feat_qc_pass", "feat_qc_minor", "feat_qc_major"])

    # Deduplicate feature columns while preserving order.
    seen = set()
    feature_cols = [c for c in feature_cols if not (c in seen or seen.add(c))]

    # Fill numeric feature values.
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    feature_cols_json.parent.mkdir(parents=True, exist_ok=True)
    feature_cols_json.write_text(
        json.dumps(feature_cols, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "feature_cols_json": str(feature_cols_json),
        "n_rows": int(len(df)),
        "n_features": int(len(feature_cols)),
        "n_positive": int((df["v3_train_label"] == 1.0).sum()),
        "n_negative": int((df["v3_train_label"] == 0.0).sum()),
        "n_uncertain": int((df["v3_train_label"] == 0.5).sum()),
        "top_positive_frac": args.top_positive_frac,
        "bottom_negative_frac": args.bottom_negative_frac,
        "claim_boundary": "weak_labels_from_v3_bootstrap_ranking_not_experimental_ground_truth",
    }

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[SAVE] {output_csv}")
    print(f"[SAVE] {feature_cols_json}")
    print(f"[SAVE] {summary_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
