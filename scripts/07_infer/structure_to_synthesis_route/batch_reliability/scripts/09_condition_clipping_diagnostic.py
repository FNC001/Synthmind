#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import pandas as pd
from _common import load_config, get_paths, read_status_table, write_table_and_md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = get_paths(cfg)
    df = read_status_table(cfg)
    th = float(cfg.get("thresholds", {}).get("medium_condition_support", 0.75))

    if "condition_support_score" in df.columns:
        low = df[pd.to_numeric(df["condition_support_score"], errors="coerce") < th].copy()
    else:
        low = df.iloc[0:0].copy()

    low["clipping_diagnostic_status"] = "needs_check_condition_distribution_or_clip_range"

    out_dir = paths["reliability_root"] / "clipping_diagnostic"
    write_table_and_md(
        low,
        out_dir / "clipping_diagnostic_cases.csv",
        out_dir / "clipping_diagnostic_cases.md",
        "Clipping Diagnostic Cases",
        ["case_id", "condition_support_score", "top1_final_score", "final_case_status", "problem_type", "clipping_diagnostic_status"]
    )

    print("[SAVE]", out_dir / "clipping_diagnostic_cases.csv")


if __name__ == "__main__":
    main()
