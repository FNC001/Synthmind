#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_precursor_set(v: Any) -> str:
    if isinstance(v, list):
        return "; ".join(map(str, v))
    if pd.isna(v):
        return ""
    s = str(v).strip()
    if not s:
        return ""
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return "; ".join(map(str, obj))
    except Exception:
        pass
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize structure-to-synthesis inference outputs into a readable route table.")
    ap.add_argument("--stage3_flat_csv", required=True, help="Path to stage3_condition_predictions/test_candidates_flat.csv")
    ap.add_argument("--output_dir", required=True, help="Output directory for readable route files")
    ap.add_argument("--top_n", type=int, default=50, help="Number of rows to export to markdown preview")
    ap.add_argument("--sort_by", default="parent_precursor_rank,condition_rank,stage3_score")
    args = ap.parse_args()

    stage3_flat_csv = Path(args.stage3_flat_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    if not stage3_flat_csv.exists():
        raise FileNotFoundError(stage3_flat_csv)

    df = pd.read_csv(stage3_flat_csv)

    if "parent_precursor_set" in df.columns:
        df["precursor_set"] = df["parent_precursor_set"].apply(parse_precursor_set)
    elif "precursor_set" in df.columns:
        df["precursor_set"] = df["precursor_set"].apply(parse_precursor_set)
    else:
        df["precursor_set"] = ""

    # Friendly aliases, while preserving existing columns.
    rename_map = {
        "parent_precursor_rank": "precursor_rank",
        "stage3_score": "condition_score",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    preferred_cols: List[str] = [
        "material_id",
        "sample_id",
        "precursor_rank",
        "precursor_set",
        "condition_rank",
        "condition_score",
        "temperature_c",
        "time_h",
        "synthesis_type",
        "mixture_index",
        "stage3_model",
    ]

    cols = [c for c in preferred_cols if c in df.columns]
    extra_cols = [
        c for c in df.columns
        if c not in cols
        and c not in {"parent_precursor_set", "parent_precursor_set_key", "cont_conditions", "cont_conditions_normalized"}
    ]

    view = df[cols + extra_cols].copy()

    sort_cols = [c.strip() for c in args.sort_by.split(",") if c.strip()]
    sort_cols = [
        {"parent_precursor_rank": "precursor_rank", "stage3_score": "condition_score"}.get(c, c)
        for c in sort_cols
    ]
    sort_cols = [c for c in sort_cols if c in view.columns]

    if sort_cols:
        ascending = []
        for c in sort_cols:
            if c in {"condition_score"}:
                ascending.append(False)
            else:
                ascending.append(True)
        view = view.sort_values(sort_cols, ascending=ascending)

    out_csv = output_dir / "synthesis_routes_readable.csv"
    out_md = output_dir / "synthesis_routes_readable.md"
    out_json = output_dir / "synthesis_routes_summary.json"

    view.to_csv(out_csv, index=False)

    preview = view.head(args.top_n)
    try:
        out_md.write_text(preview.to_markdown(index=False), encoding="utf-8")
    except Exception:
        out_md.write_text(preview.to_string(index=False), encoding="utf-8")

    summary = {
        "input_csv": str(stage3_flat_csv),
        "output_csv": str(out_csv),
        "output_md": str(out_md),
        "n_rows": int(len(view)),
        "n_preview_rows": int(len(preview)),
        "columns": list(view.columns),
    }
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[SAVE]", out_csv)
    print("[SAVE]", out_md)
    print("[SAVE]", out_json)
    print(preview.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
