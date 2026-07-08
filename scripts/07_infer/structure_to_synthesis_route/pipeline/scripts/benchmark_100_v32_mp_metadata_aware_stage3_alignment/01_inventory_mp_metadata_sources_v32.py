#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ID_COL_CANDIDATES = [
    "material_id",
    "mp_id",
    "task_id",
    "material_key",
    "mpid",
    "id",
]

FORMULA_COL_CANDIDATES = [
    "formula",
    "pretty_formula",
    "reduced_formula",
    "target_formula",
    "composition",
    "composition_reduced",
]

ELEMENT_COL_CANDIDATES = [
    "elements",
    "target_elements",
    "element_set",
    "chemsys",
    "chemical_system",
]

FAMILY_COL_CANDIDATES = [
    "target_family",
    "v26_target_family",
    "v28_target_family",
    "anion_type",
    "target_anion_type",
    "family",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_files", type=int, default=5000)
    ap.add_argument("--preview_rows", type=int, default=5)
    return ap.parse_args()


def detect_source_class(path: Path, columns: list[str]) -> str:
    cols = set(columns)

    has_id = any(c in cols for c in ID_COL_CANDIDATES)
    has_formula = any(c in cols for c in FORMULA_COL_CANDIDATES)
    has_elements = any(c in cols for c in ELEMENT_COL_CANDIDATES)
    has_family = any(c in cols for c in FAMILY_COL_CANDIDATES)

    path_s = str(path).lower()

    if has_id and has_formula and has_elements:
        return "strong_mp_metadata_candidate"
    if has_id and has_formula:
        return "formula_metadata_candidate"
    if has_id and has_elements:
        return "element_metadata_candidate"
    if has_id and ("stage3" in path_s or "material" in path_s or "mp" in path_s):
        return "id_only_possible_metadata_candidate"
    if has_formula or has_elements:
        return "weak_metadata_candidate"
    return "not_metadata_candidate"


def safe_read_table(path: Path, preview_rows: int):
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, nrows=preview_rows)
        elif path.suffix.lower() == ".tsv":
            df = pd.read_csv(path, sep="\t", nrows=preview_rows)
        elif path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
            df = df.head(preview_rows)
        elif path.suffix.lower() == ".json":
            try:
                df = pd.read_json(path)
                df = df.head(preview_rows)
            except Exception:
                obj = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(obj, list):
                    df = pd.DataFrame(obj).head(preview_rows)
                elif isinstance(obj, dict):
                    # Common case: dict of records or metadata summary.
                    df = pd.json_normalize(obj).head(preview_rows)
                else:
                    return None, "json_not_table_like"
        elif path.suffix.lower() == ".jsonl":
            df = pd.read_json(path, lines=True, nrows=preview_rows)
        else:
            return None, "unsupported_suffix"

        return df, ""

    except Exception as e:
        return None, repr(e)


def main():
    args = parse_args()

    root = Path(args.project_root).resolve()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    search_roots = [
        root / "data",
        root / "outputs",
    ]

    suffixes = {".csv", ".tsv", ".json", ".jsonl", ".parquet"}

    files = []
    for sr in search_roots:
        if not sr.exists():
            continue
        for p in sr.rglob("*"):
            if p.is_file() and p.suffix.lower() in suffixes:
                files.append(p)
                if len(files) >= args.max_files:
                    break
        if len(files) >= args.max_files:
            break

    rows = []

    for p in files:
        df, err = safe_read_table(p, args.preview_rows)

        if df is None:
            rows.append({
                "path": str(p),
                "suffix": p.suffix,
                "n_preview_rows": 0,
                "n_preview_cols": 0,
                "columns": "",
                "source_class": "read_error",
                "id_columns": "",
                "formula_columns": "",
                "element_columns": "",
                "family_columns": "",
                "read_error": err,
            })
            continue

        cols = list(df.columns)
        id_cols = [c for c in cols if c in ID_COL_CANDIDATES]
        formula_cols = [c for c in cols if c in FORMULA_COL_CANDIDATES]
        element_cols = [c for c in cols if c in ELEMENT_COL_CANDIDATES]
        family_cols = [c for c in cols if c in FAMILY_COL_CANDIDATES]

        source_class = detect_source_class(p, cols)

        rows.append({
            "path": str(p),
            "suffix": p.suffix,
            "n_preview_rows": int(len(df)),
            "n_preview_cols": int(len(cols)),
            "columns": json.dumps(cols, ensure_ascii=False),
            "source_class": source_class,
            "id_columns": ";".join(id_cols),
            "formula_columns": ";".join(formula_cols),
            "element_columns": ";".join(element_cols),
            "family_columns": ";".join(family_cols),
            "read_error": "",
        })

    inv = pd.DataFrame(rows)

    # Prioritize likely useful metadata sources.
    priority = {
        "strong_mp_metadata_candidate": 0,
        "formula_metadata_candidate": 1,
        "element_metadata_candidate": 2,
        "id_only_possible_metadata_candidate": 3,
        "weak_metadata_candidate": 4,
        "not_metadata_candidate": 5,
        "read_error": 6,
    }
    inv["priority"] = inv["source_class"].map(priority).fillna(9).astype(int)
    inv = inv.sort_values(["priority", "path"]).reset_index(drop=True)

    out_csv = out_dir / "v32_mp_metadata_source_inventory.csv"
    out_md = out_dir / "v32_mp_metadata_source_inventory_preview.md"
    summary_json = out_dir / "v32_mp_metadata_source_inventory_summary.json"

    inv.to_csv(out_csv, index=False)
    inv.head(100).to_markdown(out_md, index=False)

    summary = {
        "project_root": str(root),
        "n_scanned_files": int(len(inv)),
        "source_class_counts": inv["source_class"].value_counts().to_dict() if len(inv) else {},
        "n_strong_mp_metadata_candidate": int((inv["source_class"] == "strong_mp_metadata_candidate").sum()) if len(inv) else 0,
        "n_formula_metadata_candidate": int((inv["source_class"] == "formula_metadata_candidate").sum()) if len(inv) else 0,
        "n_element_metadata_candidate": int((inv["source_class"] == "element_metadata_candidate").sum()) if len(inv) else 0,
        "output_inventory_csv": str(out_csv),
        "status": "pass" if len(inv) > 0 else "review",
        "interpretation": (
            "V32 inventory scans CSV/TSV/JSON/JSONL/Parquet files for possible mp-* metadata sources "
            "containing material identifiers, formula columns, element columns, and target-family columns."
        ),
    }

    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[SAVE] {out_csv}")
    print(f"[SAVE] {out_md}")
    print(f"[SAVE] {summary_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
