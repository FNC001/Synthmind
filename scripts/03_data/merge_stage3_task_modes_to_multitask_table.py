#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge Stage3 task-mode tables into a single multitask table.

Typical usage:
    python merge_stage3_task_modes_to_multitask_table.py \
        --input_dirs /path/to/stage3_task_mode_1 /path/to/stage3_task_mode_2 \
        --output_csv /path/to/stage3_multitask_table.csv

Or scan one root directory:
    python merge_stage3_task_modes_to_multitask_table.py \
        --input_root /Users/wyc/SynPred/data/interim/stage3_condition_datasets \
        --output_csv /Users/wyc/SynPred/data/interim/stage3_multitask_table.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd


PREFERRED_FILES = [
    "task_table.csv",
    "stage3_task_table.csv",
    "condition_table.csv",
    "stage3_condition_table.csv",
    "dataset.csv",
    "data.csv",
]


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in [".tsv", ".txt"]:
        return pd.read_csv(path, sep="\t")
    if path.suffix.lower() == ".jsonl":
        return pd.read_json(path, lines=True)
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        if isinstance(obj, dict):
            return pd.DataFrame(obj)
    raise ValueError(f"Unsupported table format: {path}")


def find_table_files(input_root: Path) -> List[Path]:
    files: List[Path] = []

    for preferred in PREFERRED_FILES:
        files.extend(input_root.rglob(preferred))

    if not files:
        files.extend(input_root.rglob("*.csv"))

    unique = []
    seen = set()
    for f in files:
        if f.resolve() not in seen:
            unique.append(f)
            seen.add(f.resolve())

    return sorted(unique)


def infer_task_name(path: Path, df: pd.DataFrame) -> str:
    for col in ["task", "task_name", "condition_task", "target_task"]:
        if col in df.columns and df[col].notna().any():
            vals = df[col].dropna().astype(str).unique()
            if len(vals) == 1:
                return vals[0]

    parts = [p.lower() for p in path.parts]
    candidates = [
        "temperature",
        "temp",
        "time",
        "atmosphere",
        "pressure",
        "heating_rate",
        "route",
        "solvent",
        "ph",
    ]
    for c in candidates:
        if any(c in p for p in parts):
            return c

    return path.parent.name


def infer_mode_name(path: Path, df: pd.DataFrame) -> str:
    for col in ["mode", "train_mode", "task_mode", "split_mode"]:
        if col in df.columns and df[col].notna().any():
            vals = df[col].dropna().astype(str).unique()
            if len(vals) == 1:
                return vals[0]

    known_modes = [
        "relaxed_only",
        "gold_only",
        "curriculum",
        "curriculum_phase1",
        "curriculum_phase2",
        "mixed",
        "infer",
        "test",
        "val",
        "train",
    ]

    parts = [p.lower() for p in path.parts]
    for m in known_modes:
        if any(m in p for p in parts):
            return m

    return "unknown"


def normalize_columns(df: pd.DataFrame, source_file: Path) -> pd.DataFrame:
    df = df.copy()

    # Ensure identifiers.
    if "sample_id" not in df.columns:
        for alt in ["id", "material_id", "target_id", "mp_id", "recipe_id"]:
            if alt in df.columns:
                df["sample_id"] = df[alt].astype(str)
                break

    if "sample_id" not in df.columns:
        df["sample_id"] = [
            f"{source_file.parent.name}_{i:07d}" for i in range(len(df))
        ]

    # Add task/mode metadata.
    if "task_name" not in df.columns:
        df["task_name"] = infer_task_name(source_file, df)

    if "task" not in df.columns:
        df["task"] = df["task_name"]

    if "task_mode" not in df.columns:
        df["task_mode"] = infer_mode_name(source_file, df)

    if "source_file" not in df.columns:
        df["source_file"] = str(source_file)

    # Try to standardize common condition-label column names.
    label_candidates = [
        "label",
        "y",
        "target",
        "value",
        "condition_value",
        "temperature",
        "temp",
        "time",
        "atmosphere",
        "pressure",
        "heating_rate",
    ]

    if "label_value" not in df.columns:
        for col in label_candidates:
            if col in df.columns:
                df["label_value"] = df[col]
                break

    # Keep original columns. Do not aggressively drop anything.
    return df


def merge_tables(table_files: List[Path]) -> pd.DataFrame:
    frames = []

    for f in table_files:
        try:
            df = read_table(f)
        except Exception as e:
            print(f"[WARN] Failed to read {f}: {e}")
            continue

        if df.empty:
            print(f"[WARN] Empty table skipped: {f}")
            continue

        df = normalize_columns(df, f)
        frames.append(df)
        print(f"[OK] Loaded {f} | rows={len(df)} | cols={len(df.columns)}")

    if not frames:
        raise RuntimeError("No valid tables were loaded.")

    merged = pd.concat(frames, axis=0, ignore_index=True, sort=False)

    # Remove exact duplicate rows only.
    before = len(merged)
    merged = merged.drop_duplicates()
    after = len(merged)
    print(f"[INFO] Dropped exact duplicates: {before - after}")

    # Stable ordering.
    sort_cols = [c for c in ["task_name", "task_mode", "sample_id"] if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols).reset_index(drop=True)

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge Stage3 task-mode tables into a multitask table."
    )
    parser.add_argument(
        "--input_root",
        type=str,
        default=None,
        help="Root directory to recursively scan for Stage3 task tables.",
    )
    parser.add_argument(
        "--input_dirs",
        type=str,
        nargs="*",
        default=None,
        help="Specific directories to scan for Stage3 task tables.",
    )
    parser.add_argument(
        "--input_files",
        type=str,
        nargs="*",
        default=None,
        help="Specific table files to merge.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Output merged multitask CSV file.",
    )
    parser.add_argument(
        "--summary_json",
        type=str,
        default=None,
        help="Optional output summary JSON file.",
    )

    args = parser.parse_args()

    table_files: List[Path] = []

    if args.input_files:
        table_files.extend([Path(x).expanduser().resolve() for x in args.input_files])

    if args.input_dirs:
        for d in args.input_dirs:
            dpath = Path(d).expanduser().resolve()
            if not dpath.exists():
                print(f"[WARN] Input dir not found: {dpath}")
                continue
            table_files.extend(find_table_files(dpath))

    if args.input_root:
        root = Path(args.input_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"input_root not found: {root}")
        table_files.extend(find_table_files(root))

    # Deduplicate table files.
    unique_files = []
    seen = set()
    for f in table_files:
        if f.exists() and f.resolve() not in seen:
            unique_files.append(f.resolve())
            seen.add(f.resolve())

    if not unique_files:
        raise RuntimeError(
            "No input tables found. Please provide --input_root, --input_dirs, or --input_files."
        )

    print(f"[INFO] Found {len(unique_files)} table files.")
    merged = merge_tables(unique_files)

    output_csv = Path(args.output_csv).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    print(f"[DONE] Saved merged multitask table: {output_csv}")
    print(f"[DONE] rows={len(merged)} cols={len(merged.columns)}")

    summary = {
        "n_input_files": len(unique_files),
        "input_files": [str(x) for x in unique_files],
        "n_rows": int(len(merged)),
        "n_columns": int(len(merged.columns)),
        "columns": list(merged.columns),
    }

    if "task_name" in merged.columns:
        summary["rows_by_task_name"] = (
            merged["task_name"].astype(str).value_counts().to_dict()
        )

    if "task_mode" in merged.columns:
        summary["rows_by_task_mode"] = (
            merged["task_mode"].astype(str).value_counts().to_dict()
        )

    if args.summary_json:
        summary_json = Path(args.summary_json).expanduser().resolve()
    else:
        summary_json = output_csv.with_suffix(".summary.json")

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Saved summary: {summary_json}")


if __name__ == "__main__":
    main()
