#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Apply Stage35 v4.3 template-aware chemonly pairwise ranker.

Input:
  synthesis_routes_stage35_v43_template_features.csv

Output:
  synthesis_routes_stage35_v43_template_chemonly_reranked.csv/md/json

Ranking:
  all-vs-all pairwise comparisons.
  For each candidate i against j:
    p(i better than j) = model.predict_proba(features_i - features_j)
  Aggregate by:
    wins, losses, mean_prob, win_rate, final score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in cols:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            out[c] = 0.0
    return out.astype(float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--feature_cols_json", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--output_md", required=True)
    ap.add_argument("--summary_json", required=True)
    ap.add_argument("--score_prefix", default="stage35_v43_template_chemonly")
    ap.add_argument("--top_n", type=int, default=30)
    args = ap.parse_args()

    input_csv = Path(args.input_csv)
    model_path = Path(args.model_path)
    feature_cols_json = Path(args.feature_cols_json)

    df = pd.read_csv(input_csv)
    if df.empty:
        raise SystemExit(f"[ERROR] empty input csv: {input_csv}")

    model = joblib.load(model_path)
    feature_cols = json.loads(feature_cols_json.read_text(encoding="utf-8"))

    # feature_cols are diff__xxx, so base cols are xxx.
    base_cols = []
    for c in feature_cols:
        if not c.startswith("diff__"):
            raise SystemExit(f"[ERROR] feature column does not start with diff__: {c}")
        base_cols.append(c.replace("diff__", "", 1))

    X_base = ensure_numeric(df, base_cols)

    # Per-sample pairwise ranking if sample_id exists
    if "sample_id" in df.columns:
        group_col = "sample_id"
        groups = df[group_col].values
        unique_groups = df[group_col].unique()
    else:
        groups = np.zeros(len(df), dtype=int)
        unique_groups = [0]

    n = len(df)
    wins = np.zeros(n, dtype=float)
    losses = np.zeros(n, dtype=float)
    prob_sum = np.zeros(n, dtype=float)
    n_comp = np.zeros(n, dtype=float)

    for g in unique_groups:
        idx = np.where(groups == g)[0]
        ng = len(idx)
        if ng <= 1:
            continue

        for ii in range(ng):
            i = idx[ii]
            batch_rows = []
            opponents = []

            xi = X_base.iloc[i]
            for jj in range(ng):
                if ii == jj:
                    continue
                j = idx[jj]
                xj = X_base.iloc[j]
                diff = xi.values - xj.values
                batch_rows.append(diff)
                opponents.append(j)

            X_pair = pd.DataFrame(batch_rows, columns=feature_cols)
            prob = model.predict_proba(X_pair)[:, 1]

            for p, j in zip(prob, opponents):
                prob_sum[i] += float(p)
                n_comp[i] += 1.0
                if p >= 0.5:
                    wins[i] += 1.0
                    losses[j] += 1.0

    mean_prob = prob_sum / np.maximum(n_comp, 1.0)
    win_rate = wins / np.maximum(n_comp, 1.0)

    # Final score combines probability and win-rate.
    score = 0.7 * mean_prob + 0.3 * win_rate

    out = df.copy()
    out[f"{args.score_prefix}_wins"] = wins
    out[f"{args.score_prefix}_losses"] = losses
    out[f"{args.score_prefix}_mean_prob"] = mean_prob
    out[f"{args.score_prefix}_win_rate"] = win_rate
    out[f"{args.score_prefix}_score"] = score

    if "sample_id" in out.columns and out["sample_id"].nunique() > 1:
        out = out.sort_values(
            ["sample_id", f"{args.score_prefix}_score", f"{args.score_prefix}_mean_prob", f"{args.score_prefix}_wins"],
            ascending=[True, False, False, False],
        ).reset_index(drop=True)
        out[f"{args.score_prefix}_rank"] = out.groupby("sample_id").cumcount() + 1
    else:
        out = out.sort_values(
            [f"{args.score_prefix}_score", f"{args.score_prefix}_mean_prob", f"{args.score_prefix}_wins"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        out[f"{args.score_prefix}_rank"] = np.arange(1, len(out) + 1, dtype=int)

    # Put key columns in front when possible.
    front_cols = [
        f"{args.score_prefix}_rank",
        f"{args.score_prefix}_score",
        f"{args.score_prefix}_mean_prob",
        f"{args.score_prefix}_win_rate",
        f"{args.score_prefix}_wins",
        f"{args.score_prefix}_losses",
        "precursor_set",
        "temperature_c",
        "time_h",
        "condition_source",
        "target_elements_v33",
        "precursor_elements_v33",
        "target_element_coverage_v33",
        "target_extra_count_v33",
        "n_warnings",
        "route_warning_penalty",
        "route_template_primary",
        "route_template_matches_target_anion",
        "route_template_is_common_solid_state",
        "route_template_is_overly_elemental",
        "route_template_confidence",
        "stage35_v42_pairwise_foreignaware_score",
        "stage35_v33_chemonly_prob",
        "stage35_v32_light_score",
    ]
    front_cols = [c for c in front_cols if c in out.columns]
    other_cols = [c for c in out.columns if c not in front_cols]
    out = out[front_cols + other_cols]

    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    summary_json = Path(args.summary_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    out.head(args.top_n).to_markdown(output_md, index=False)

    # Per-sample top_n output
    if "sample_id" in out.columns:
        out_trunc = out.groupby("sample_id", sort=False).head(args.top_n).reset_index(drop=True)
    else:
        out_trunc = out.head(args.top_n).copy()
    out_trunc.to_csv(output_csv, index=False)

    top1 = out.iloc[0]
    summary = {
        "input_csv": str(input_csv),
        "model_path": str(model_path),
        "feature_cols_json": str(feature_cols_json),
        "output_csv": str(output_csv),
        "output_md": str(output_md),
        "rows_input": int(len(df)),
        "rows_output": int(min(args.top_n, len(out))),
        "n_features": int(len(feature_cols)),
        "n_pair_comparisons": int(n * (n - 1)),
        "score_prefix": args.score_prefix,
        "top_n": int(args.top_n),
        "top1_precursor_set": str(top1.get("precursor_set", "")),
        "top1_score": safe_float(top1.get(f"{args.score_prefix}_score", 0.0)),
        "top1_mean_prob": safe_float(top1.get(f"{args.score_prefix}_mean_prob", 0.0)),
        "top1_win_rate": safe_float(top1.get(f"{args.score_prefix}_win_rate", 0.0)),
        "top1_template": str(top1.get("route_template_primary", "")),
        "top1_template_match": safe_float(top1.get("route_template_matches_target_anion", 0.0)),
        "top1_common_solid_state": safe_float(top1.get("route_template_is_common_solid_state", 0.0)),
        "top1_overly_elemental": safe_float(top1.get("route_template_is_overly_elemental", 0.0)),
        "note": "Stage35 v4.3 template-aware chemonly pairwise reranker.",
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[SAVE]", output_csv)
    print("[SAVE]", output_md)
    print("[SAVE]", summary_json)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
