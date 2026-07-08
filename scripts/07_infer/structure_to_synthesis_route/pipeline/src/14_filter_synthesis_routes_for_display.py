#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter synthesis route table for display.")
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--output_md", required=True)
    ap.add_argument("--min_temperature_c", type=float, default=300.0)
    ap.add_argument("--max_temperature_c", type=float, default=1600.0)
    ap.add_argument("--min_time_h", type=float, default=0.1)
    ap.add_argument("--max_time_h", type=float, default=240.0)
    ap.add_argument("--top_n", type=int, default=20)
    ap.add_argument("--prefer_top_component_mean", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)

    for c in ["temperature_c", "time_h", "condition_score", "stage3_score"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    keep = (
        (df["temperature_c"] >= args.min_temperature_c)
        & (df["temperature_c"] <= args.max_temperature_c)
        & (df["time_h"] >= args.min_time_h)
        & (df["time_h"] <= args.max_time_h)
    )
    out = df[keep].copy()

    if args.prefer_top_component_mean and "condition_source" in out.columns:
        out["source_priority"] = (~out["condition_source"].isin(["top_component_mean", "lgbm_top1"])).astype(int)
    else:
        out["source_priority"] = 0

    sort_cols = []
    ascending = []

    if "precursor_rank" in out.columns:
        sort_cols.append("precursor_rank")
        ascending.append(True)
    sort_cols.append("source_priority")
    ascending.append(True)

    if "condition_rank" in out.columns:
        sort_cols.append("condition_rank")
        ascending.append(True)

    if sort_cols:
        out = out.sort_values(sort_cols, ascending=ascending)

    # Per-sample top_n (not global top_n)
    if "sample_id" in out.columns:
        out = out.groupby("sample_id", sort=False).head(int(args.top_n)).reset_index(drop=True)
    else:
        out = out.head(int(args.top_n)).copy()

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(out.to_markdown(index=False))

    print("[SAVE]", args.output_csv)
    print("[SAVE]", args.output_md)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
