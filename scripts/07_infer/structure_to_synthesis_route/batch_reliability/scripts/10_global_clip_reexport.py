#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from _common import load_config, get_paths, write_table_and_md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = get_paths(cfg)

    diag_csv = paths["reliability_root"] / "clipping_diagnostic" / "clipping_diagnostic_cases.csv"
    if diag_csv.exists():
        df = pd.read_csv(diag_csv)
    else:
        df = pd.DataFrame()

    if len(df) > 0:
        df["global_clip_export_status"] = "planned_global_clip_reexport"
    else:
        df["global_clip_export_status"] = []

    out_dir = paths["reliability_root"] / "global_clip_export"
    write_table_and_md(
        df,
        out_dir / "global_clip_export_plan.csv",
        out_dir / "global_clip_export_plan.md",
        "Global Clip Export Plan",
    )

    print("[SAVE]", out_dir / "global_clip_export_plan.csv")


if __name__ == "__main__":
    main()
