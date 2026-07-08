#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build Stage35 v4.3 template-aware pairwise preference dataset.

Input:
  benchmark_run_status.tsv
  per-benchmark synthesis_routes_stage35_v43_template_features.csv

Output:
  Pairwise CSV with route_a / route_b features and weak preference label.

Design:
  - Same-target pairwise preferences.
  - Preference score combines coverage, warnings, foreign-cation penalty,
    template match, common solid-state template, and overly-elemental soft penalty.
  - This script does not train a model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x)
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def compute_template_quality(row: pd.Series) -> float:
    """
    Weak route quality for pair construction only.
    Higher is better.

    Do not treat overly_elemental as hard invalid.
    It is a soft penalty.
    """
    coverage = safe_float(row.get("target_element_coverage_v33", 1.0), 1.0)
    missing = safe_float(row.get("target_missing_count_v33", row.get("element_missing_v33", 0.0)), 0.0)
    extra = safe_float(row.get("target_extra_count_v33", 0.0), 0.0)
    warnings = safe_float(row.get("n_warnings", 0.0), 0.0)
    warning_penalty = safe_float(row.get("route_warning_penalty", 0.0), 0.0)
    foreign_cation = safe_float(row.get("q_foreign_cation_count", 0.0), 0.0)
    extra_nontrivial = safe_float(row.get("q_extra_nontrivial_count", 0.0), 0.0)

    template_match = safe_float(row.get("route_template_matches_target_anion", 0.0), 0.0)
    common_solid = safe_float(row.get("route_template_is_common_solid_state", 0.0), 0.0)
    overly_elemental = safe_float(row.get("route_template_is_overly_elemental", 0.0), 0.0)
    template_conf = safe_float(row.get("route_template_confidence", 0.0), 0.0)

    v42_score = safe_float(row.get("stage35_v42_pairwise_foreignaware_score", 0.0), 0.0)
    v33_prob = safe_float(row.get("stage35_v33_chemonly_prob", 0.0), 0.0)
    v32_score = safe_float(row.get("stage35_v32_light_score", 0.0), 0.0)

    score = 0.0

    # Hardest correctness signals.
    score += 4.0 * coverage
    score -= 2.0 * missing
    score -= 0.25 * extra
    score -= 1.5 * foreign_cation
    score -= 0.5 * extra_nontrivial

    # Route diagnostics.
    score -= 0.6 * warnings
    score -= 0.8 * warning_penalty

    # Template features.
    score += 0.8 * template_match
    score += 0.4 * common_solid
    score += 0.2 * template_conf
    score -= 0.35 * overly_elemental

    # Existing ranking priors, weakly included.
    score += 0.4 * v42_score
    score += 0.2 * v33_prob
    score += 0.1 * v32_score

    return float(score)


def numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {
        "route_quality_v43_weak",
    }
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def build_pairs_for_group(
    df: pd.DataFrame,
    infer_name: str,
    max_pairs_per_group: int,
    min_quality_gap: float,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    d = df.copy().reset_index(drop=True)
    d["route_quality_v43_weak"] = d.apply(compute_template_quality, axis=1)
    d["route_local_index"] = np.arange(len(d), dtype=int)

    # Prefer comparing good top candidates against lower-quality tail candidates.
    d_sorted = d.sort_values("route_quality_v43_weak", ascending=False).reset_index(drop=True)

    high_pool = d_sorted.head(min(10, len(d_sorted))).copy()
    low_pool = d_sorted.tail(min(15, len(d_sorted))).copy()

    rows = []
    feat_cols = numeric_feature_cols(d)

    for _, a in high_pool.iterrows():
        for _, b in low_pool.iterrows():
            if int(a["route_local_index"]) == int(b["route_local_index"]):
                continue

            qa = float(a["route_quality_v43_weak"])
            qb = float(b["route_quality_v43_weak"])
            gap = qa - qb

            if abs(gap) < min_quality_gap:
                continue

            if gap > 0:
                better = a
                worse = b
                label = 1
                quality_gap = gap
            else:
                better = b
                worse = a
                label = 1
                quality_gap = -gap

            rec: dict[str, Any] = {
                "infer_name": infer_name,
                "target_group": infer_name,
                "label": label,
                "quality_gap": quality_gap,
                "better_precursor_set": safe_str(better.get("precursor_set", "")),
                "worse_precursor_set": safe_str(worse.get("precursor_set", "")),
                "better_template": safe_str(better.get("route_template_primary", "")),
                "worse_template": safe_str(worse.get("route_template_primary", "")),
                "better_quality": float(better["route_quality_v43_weak"]),
                "worse_quality": float(worse["route_quality_v43_weak"]),
            }

            # Difference features: better - worse.
            for c in feat_cols:
                rec[f"diff__{c}"] = safe_float(better.get(c, 0.0)) - safe_float(worse.get(c, 0.0))

            rows.append(rec)

            if len(rows) >= max_pairs_per_group:
                break
        if len(rows) >= max_pairs_per_group:
            break

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark_name", default="benchmark_30_clean_v2_final_shell_v11_v33")
    ap.add_argument("--project_root", default="/Users/wyc/SynPred")
    ap.add_argument("--status_tsv", default="")
    ap.add_argument("--input_filename", default="synthesis_routes_stage35_v43_template_features.csv")
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--output_group_summary_md", default="")
    ap.add_argument("--summary_json", default="")
    ap.add_argument("--max_pairs_per_group", type=int, default=80)
    ap.add_argument("--min_quality_gap", type=float, default=0.25)
    args = ap.parse_args()

    project_root = Path(args.project_root)
    status_tsv = Path(args.status_tsv) if args.status_tsv else project_root / "outputs" / "benchmark" / args.benchmark_name / "benchmark_run_status.tsv"

    status = pd.read_csv(status_tsv, sep="\t")

    all_pairs = []
    group_rows = []

    for _, r in status.iterrows():
        infer = str(r["infer_name"])
        route_dir = Path(str(r["final_md"])).parent
        input_csv = route_dir / args.input_filename

        if not input_csv.exists():
            group_rows.append({
                "infer_name": infer,
                "status": "missing_input",
                "source_file": str(input_csv),
                "n_routes": 0,
                "n_pairs": 0,
            })
            continue

        df = pd.read_csv(input_csv)
        pairs = build_pairs_for_group(
            df=df,
            infer_name=infer,
            max_pairs_per_group=args.max_pairs_per_group,
            min_quality_gap=args.min_quality_gap,
        )

        if not pairs.empty:
            all_pairs.append(pairs)

        group_rows.append({
            "infer_name": infer,
            "status": "ok",
            "source_file": str(input_csv),
            "n_routes": int(len(df)),
            "n_pairs": int(len(pairs)),
        })

    out = pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame()
    group_summary = pd.DataFrame(group_rows)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    if args.output_group_summary_md:
        md_path = Path(args.output_group_summary_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        group_summary.to_markdown(md_path, index=False)

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "benchmark_name": args.benchmark_name,
            "status_tsv": str(status_tsv),
            "input_filename": args.input_filename,
            "n_groups": int(len(group_summary)),
            "n_ok_groups": int((group_summary["status"] == "ok").sum()),
            "n_total_pairs": int(len(out)),
            "mean_pairs_per_group": float(group_summary["n_pairs"].mean()) if len(group_summary) else 0.0,
            "min_pairs_per_group": int(group_summary["n_pairs"].min()) if len(group_summary) else 0,
            "max_pairs_per_group": int(group_summary["n_pairs"].max()) if len(group_summary) else 0,
            "max_pairs_per_group_setting": int(args.max_pairs_per_group),
            "min_quality_gap": float(args.min_quality_gap),
            "output_csv": str(output_csv),
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[SAVE]", output_csv)
    if args.output_group_summary_md:
        print("[SAVE]", args.output_group_summary_md)
    if args.summary_json:
        print("[SAVE]", args.summary_json)
    print(group_summary.to_string(index=False))


if __name__ == "__main__":
    main()
