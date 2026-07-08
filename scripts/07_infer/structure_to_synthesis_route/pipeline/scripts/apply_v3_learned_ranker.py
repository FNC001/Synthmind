#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


def clean(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def score_to_level(x: float) -> str:
    try:
        x = float(x)
    except Exception:
        x = 0.0

    if x >= 0.78:
        return "high_learned_score"
    if x >= 0.50:
        return "medium_learned_score"
    return "low_learned_score"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--feature_cols_json", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--output_md", required=True)
    ap.add_argument("--summary_json", required=True)
    ap.add_argument("--top_n", type=int, default=30)
    args = ap.parse_args()

    input_csv = Path(args.input_csv)
    model_path = Path(args.model_path)
    feature_cols_json = Path(args.feature_cols_json)
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    summary_json = Path(args.summary_json)

    for p in [input_csv, model_path, feature_cols_json]:
        if not p.exists():
            raise FileNotFoundError(p)

    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError(f"Empty input csv: {input_csv}")

    feature_cols = json.loads(feature_cols_json.read_text(encoding="utf-8"))
    if not isinstance(feature_cols, list) or not feature_cols:
        raise ValueError(f"Invalid feature cols json: {feature_cols_json}")

    # Keep exactly the same feature schema as training.
    # Missing columns are filled with 0.0 instead of being dropped,
    # because sklearn models require the same number/order of features at predict time.
    missing_cols = [c for c in feature_cols if c not in df.columns]

    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    used_cols = list(feature_cols)

    if not used_cols:
        raise ValueError("No usable feature columns found in feature_cols_json.")

    model = joblib.load(model_path)

    x = df[used_cols].values

    if hasattr(model, "predict_proba"):
        pred = model.predict_proba(x)[:, 1]
    else:
        pred = model.predict(x)

    df["v3_learned_ranker_score"] = pred
    df["v3_learned_ranker_score_level"] = df["v3_learned_ranker_score"].apply(score_to_level)

    # Conservative safety override:
    # learned score should not promote review_required routes above safe routes.
    if "route_recommendation_status" in df.columns:
        df["v3_learned_safety_bucket"] = df["route_recommendation_status"].map({
            "recommended": 0,
            "recommended_with_validation": 1,
            "review_required": 2,
        }).fillna(1).astype(int)
    else:
        df["v3_learned_safety_bucket"] = 1

    # Sort by safety first, then learned score, then existing v3 rank.
    sort_cols = ["v3_learned_safety_bucket", "v3_learned_ranker_score"]
    ascending = [True, False]

    if "v3_joint_rerank_rank" in df.columns:
        sort_cols.append("v3_joint_rerank_rank")
        ascending.append(True)
    elif "final_route_rank" in df.columns:
        sort_cols.append("final_route_rank")
        ascending.append(True)

    if "sample_id" in df.columns and df["sample_id"].nunique() > 1:
        df = df.sort_values(["sample_id"] + sort_cols, ascending=[True] + ascending, kind="mergesort").reset_index(drop=True)
        df["v3_learned_rerank_rank"] = df.groupby("sample_id").cumcount() + 1
    else:
        df = df.sort_values(sort_cols, ascending=ascending, kind="mergesort").reset_index(drop=True)
        df["v3_learned_rerank_rank"] = range(1, len(df) + 1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    view_cols = [
        "v3_learned_rerank_rank",
        "v3_learned_ranker_score",
        "v3_learned_ranker_score_level",
        "v3_joint_rerank_rank",
        "v3_joint_feature_score",
        "final_route_rank",
        "precursor_set",
        "temperature_c",
        "time_h",
        "route_confidence_score",
        "route_warning_level",
        "route_recommendation_status",
        "precursor_qc_level",
        "route_warnings",
    ]
    view_cols = [c for c in view_cols if c in df.columns]

    output_md.write_text(
        df[view_cols].head(args.top_n).to_markdown(index=False),
        encoding="utf-8",
    )

    summary = {
        "input_csv": str(input_csv),
        "model_path": str(model_path),
        "feature_cols_json": str(feature_cols_json),
        "output_csv": str(output_csv),
        "output_md": str(output_md),
        "n_routes": int(len(df)),
        "top_n": int(args.top_n),
        "n_used_features": int(len(used_cols)),
        "n_missing_features": int(len(missing_cols)),
        "missing_features": missing_cols[:50],
        "top_precursor_set": clean(df.iloc[0].get("precursor_set", "")),
        "top_learned_ranker_score": float(df.iloc[0]["v3_learned_ranker_score"]),
        "claim_boundary": (
            "learned_ranker_application_smoke_test; "
            "model_trained_on_weak_labels; "
            "not_experimental_ground_truth; "
            "not_generalization_claim"
        ),
    }

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[SAVE] {output_csv}")
    print(f"[SAVE] {output_md}")
    print(f"[SAVE] {summary_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
