#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
from pathlib import Path
from _common import load_config, get_paths, read_status_table, read_compact_watchlist, write_table_and_md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = get_paths(cfg)
    df = read_status_table(cfg)
    compact_watchlist = read_compact_watchlist(cfg)

    out_dir = paths["reliability_root"] / "audit_batch"
    out_dir.mkdir(parents=True, exist_ok=True)

    target_statuses = cfg.get("target_statuses", [])
    abnormal = df[df["final_case_status"].astype(str).isin(target_statuses)].copy() if "final_case_status" in df.columns else df.iloc[0:0].copy()

    normal = df[~df.index.isin(abnormal.index)].copy()

    cols = [
        "case_id", "layer1_status", "final_case_status", "refined_case_status",
        "has_stage2_candidates", "has_stage3_conditions", "has_final_recommendation",
        "condition_support_score", "audit_level", "problem_type", "recovery_action",
        "top1_precursor_set", "top1_final_score", "input_poscar"
    ]

    write_table_and_md(
        df,
        out_dir / "all_cases_status.csv",
        out_dir / "all_cases_status.md",
        "All Cases Status",
        cols,
    )

    write_table_and_md(
        abnormal,
        out_dir / "abnormal_cases.csv",
        out_dir / "abnormal_cases.md",
        "Abnormal Cases for Reliability Recovery",
        cols,
    )

    write_table_and_md(
        normal,
        out_dir / "normal_cases.csv",
        out_dir / "normal_cases.md",
        "Normal or Review-only Cases",
        cols,
    )

    watch_cols = [c for c in cols if c in compact_watchlist.columns]
    if len(compact_watchlist) > 0 and watch_cols:
        compact_watchlist_out = compact_watchlist[watch_cols].copy()
    else:
        compact_watchlist_out = compact_watchlist.copy()

    write_table_and_md(
        compact_watchlist_out,
        out_dir / "compact_watchlist_imported.csv",
        out_dir / "compact_watchlist_imported.md",
        "Imported Compact Watchlist from batch_adaptive",
    )

    summary = df["final_case_status"].value_counts(dropna=False).reset_index()
    summary.columns = ["final_case_status", "n_cases"]
    write_table_and_md(
        summary,
        out_dir / "status_summary.csv",
        out_dir / "status_summary.md",
        "Batch Status Summary",
    )

    print("[SAVE]", out_dir / "abnormal_cases.csv")
    print("[SAVE]", out_dir / "status_summary.md")


if __name__ == "__main__":
    main()
