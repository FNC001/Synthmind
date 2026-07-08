#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from _common import load_config, get_paths, read_status_table, write_table_and_md


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_one_case(case_id: str, flow_csv: Path) -> pd.DataFrame:
    if not flow_csv.exists():
        return pd.DataFrame([{
            "case_id": case_id,
            "normalize_status": "missing_flow_flat_csv",
            "source_csv": str(flow_csv),
        }])

    try:
        df = pd.read_csv(flow_csv)
    except Exception as e:
        return pd.DataFrame([{
            "case_id": case_id,
            "normalize_status": f"read_error:{e}",
            "source_csv": str(flow_csv),
        }])

    if len(df) == 0:
        return pd.DataFrame([{
            "case_id": case_id,
            "normalize_status": "empty_flow_flat_csv",
            "source_csv": str(flow_csv),
        }])

    precursor_col = pick_col(df, [
        "precursor_set",
        "precursors",
        "precursor_names",
        "pred_precursor_set",
    ])
    score_col = pick_col(df, [
        "final_recommendation_score",
        "stage3_score",
        "score",
        "route_score",
        "condition_score",
    ])
    support_col = pick_col(df, [
        "real_stage3_condition_reference_support_score",
        "condition_support_score",
        "condition_distribution_support_score",
        "support_score",
    ])
    rank_col = pick_col(df, [
        "global_rank",
        "rank",
        "route_rank",
        "candidate_rank",
    ])

    out = df.copy()
    out.insert(0, "case_id", case_id)
    out.insert(1, "normalize_status", "ok")
    out.insert(2, "source_csv", str(flow_csv))

    out["normalized_precursor_set"] = out[precursor_col].astype(str) if precursor_col else ""
    out["normalized_score"] = pd.to_numeric(out[score_col], errors="coerce") if score_col else pd.NA
    out["normalized_support_score"] = pd.to_numeric(out[support_col], errors="coerce") if support_col else pd.NA

    if rank_col:
        out["normalized_rank"] = pd.to_numeric(out[rank_col], errors="coerce")
    else:
        out["normalized_rank"] = range(1, len(out) + 1)

    out["source_precursor_col"] = precursor_col or ""
    out["source_score_col"] = score_col or ""
    out["source_support_col"] = support_col or ""
    out["source_rank_col"] = rank_col or ""

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = get_paths(cfg)
    project_root = paths["project_root"]

    df = read_status_table(cfg)

    targets = df[
        df["final_case_status"].astype(str).isin([
            "needs_stage3_recovery",
            "needs_condition_reexport",
            "needs_route_finalization_recovery",
        ])
    ].copy()

    rows = []
    for _, r in targets.iterrows():
        case_id = str(r["case_id"])
        flow_csv = (
            project_root
            / "outputs/inference"
            / case_id
            / "stage3_condition_predictions_flow_fallback_retrieval_baseline_element_reranked"
            / "test_candidates_flat.csv"
        )
        rows.append(normalize_one_case(case_id, flow_csv))

    if rows:
        out = pd.concat(rows, ignore_index=True)
    else:
        out = pd.DataFrame(columns=[
            "case_id",
            "normalize_status",
            "source_csv",
            "normalized_precursor_set",
            "normalized_score",
            "normalized_support_score",
            "normalized_rank",
        ])

    out_dir = paths["reliability_root"] / "normalize_recovered_candidates"

    write_table_and_md(
        out,
        out_dir / "normalized_recovered_candidates.csv",
        out_dir / "normalized_recovered_candidates.md",
        "Normalized Recovered Stage3 Candidates",
        [
            "case_id",
            "normalize_status",
            "normalized_rank",
            "normalized_precursor_set",
            "normalized_score",
            "normalized_support_score",
            "source_csv",
        ],
    )

    if len(out) > 0:
        summary = (
            out.groupby(["case_id", "normalize_status"])
            .size()
            .reset_index(name="n_rows")
        )
    else:
        summary = pd.DataFrame(columns=["case_id", "normalize_status", "n_rows"])

    write_table_and_md(
        summary,
        out_dir / "normalize_recovered_candidates_summary.csv",
        out_dir / "normalize_recovered_candidates_summary.md",
        "Normalize Recovered Candidates Summary",
    )

    print("[SAVE]", out_dir / "normalized_recovered_candidates.csv")
    print("[SAVE]", out_dir / "normalize_recovered_candidates_summary.md")


if __name__ == "__main__":
    main()
