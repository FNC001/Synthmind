#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError


def infer_source(df: pd.DataFrame) -> str:
    if "is_retrieval_candidate" in df.columns:
        try:
            if df["is_retrieval_candidate"].fillna(False).astype(bool).any():
                return "retrieval"
        except Exception:
            pass

    if "is_composition_fallback" in df.columns:
        try:
            if df["is_composition_fallback"].fillna(False).astype(bool).any():
                return "composition_fallback_or_original"
        except Exception:
            pass

    return "stage2_model"


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge Stage2 candidate CSVs from multiple sources.")
    ap.add_argument("--input_csvs", nargs="+", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--summary_json", default="")
    ap.add_argument("--precursor_col", default="precursor_set")
    args = ap.parse_args()

    output_csv = Path(args.output_csv).expanduser().resolve()
    summary_json = (
        Path(args.summary_json).expanduser().resolve()
        if args.summary_json
        else output_csv.with_suffix(".summary.json")
    )

    parts = []
    source_stats = []

    for p in args.input_csvs:
        path = Path(p).expanduser().resolve()
        if not path.exists():
            print(f"[WARN] missing: {path}")
            continue

        try:
            df = pd.read_csv(path)
        except EmptyDataError:
            print(f"[WARN] empty/no columns: {path}")
            continue

        if df.empty:
            print(f"[WARN] empty: {path}")
            continue

        df["candidate_source_file"] = str(path)

        if "candidate_source" not in df.columns:
            df["candidate_source"] = infer_source(df)

        parts.append(df)
        source_stats.append({
            "file": str(path),
            "rows": int(len(df)),
            "source": str(df["candidate_source"].iloc[0]) if "candidate_source" in df.columns else "unknown",
        })

    if not parts:
        raise RuntimeError("No input CSV loaded.")

    merged = pd.concat(parts, ignore_index=True, sort=False)

    if args.precursor_col not in merged.columns:
        raise KeyError(f"Missing precursor column: {args.precursor_col}")

    merged["__set_key_norm"] = (
        merged[args.precursor_col]
        .astype(str)
        .str.replace(r"\s+", "", regex=True)
        .str.replace("||", ";", regex=False)
    )

    # Keep first occurrence for exact duplicate precursor sets.
    merged = merged.drop_duplicates("__set_key_norm", keep="first").drop(columns=["__set_key_norm"])
    merged["rank"] = range(1, len(merged) + 1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    summary = {
        "input_csvs": source_stats,
        "output_csv": str(output_csv),
        "output_rows": int(len(merged)),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[SAVE]", output_csv)
    print("[SAVE]", summary_json)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
