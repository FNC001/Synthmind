#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_summarize_stage2_gflownet_candidates_v2.py
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


CANDIDATE_SET_COLS = [
    "pred_labels",
    "pred_precursors",
    "predicted_precursors",
    "predicted_set",
    "candidate_set",
    "precursor_set",
    "main_precursors",
    "selected_precursors",
    "decoded_precursors",
]

SCORE_COLS_PREFERRED = [
    "score",
    "sample_score",
    "log_prob",
    "prob",
    "set_score",
    "rank_score",
    "sample_rank",
]

META_COLS_PREFERRED = [
    "sample_id",
    "formula",
    "formula_x",
    "formula_y",
    "doi",
    "split_group",
    "decode_method",
    "temperature",
    "top_k",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def try_parse_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    s = str(v).strip()
    if not s:
        return []
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass
    if ";" in s:
        return [x.strip() for x in s.split(";") if x.strip()]
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    return [s]


def infer_set_from_multihot_style_row(row: pd.Series) -> List[str]:
    label_cols = [c for c in row.index if str(c).startswith("pred__") or str(c).startswith("label__")]
    if not label_cols:
        return []
    picked = []
    for c in label_cols:
        try:
            val = float(row[c])
        except Exception:
            continue
        if val > 0.5:
            picked.append(c.split("__", 1)[-1])
    return picked


def canonicalize_precursor_set(items: List[str]) -> Tuple[str, str]:
    cleaned = [x.strip() for x in items if str(x).strip()]
    cleaned = sorted(set(cleaned))
    key = " || ".join(cleaned)
    pretty = "; ".join(cleaned)
    return key, pretty


def detect_set_column(df: pd.DataFrame) -> Optional[str]:
    for c in CANDIDATE_SET_COLS:
        if c in df.columns:
            return c
    return None


def detect_score_cols(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in SCORE_COLS_PREFERRED:
        if c in df.columns:
            cols.append(c)
    return cols


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize stage2 GFlowNet candidate precursor sets.")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--top_n", type=int, default=20)
    args = parser.parse_args()

    input_csv = Path(args.input_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input_csv: {input_csv}")

    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError(f"Input csv is empty: {input_csv}")

    set_col = detect_set_column(df)
    score_cols = detect_score_cols(df)

    rows = []
    for i, row in df.iterrows():
        if set_col is not None:
            items = try_parse_list(row[set_col])
        else:
            items = infer_set_from_multihot_style_row(row)

        key, pretty = canonicalize_precursor_set(items)
        if not key:
            key = f"__EMPTY__ROW_{i}"
            pretty = ""

        record = {
            "row_idx": int(i),
            "set_key": key,
            "precursor_set": pretty,
            "n_precursors": 0 if not pretty else len(pretty.split("; ")),
        }

        for c in META_COLS_PREFERRED:
            if c in row.index:
                record[c] = row[c]

        for c in score_cols:
            record[c] = row[c]

        rows.append(record)

    rdf = pd.DataFrame(rows)

    grouped = []
    for set_key, g in rdf.groupby("set_key", dropna=False):
        rep = g.iloc[0].to_dict()
        out = {
            "set_key": set_key,
            "precursor_set": rep.get("precursor_set", ""),
            "n_precursors": int(rep.get("n_precursors", 0)),
            "count": int(len(g)),
            "frequency": float(len(g) / len(rdf)),
        }

        for c in META_COLS_PREFERRED:
            if c in g.columns:
                out[c] = rep.get(c)

        if "decode_method" in g.columns:
            out["decode_methods_seen"] = sorted(set(str(x) for x in g["decode_method"].tolist()))

        if "sample_rank" in g.columns:
            sr = pd.to_numeric(g["sample_rank"], errors="coerce")
            out["sample_rank_min"] = float(sr.min())
            out["sample_rank_mean"] = float(sr.mean())

        for c in score_cols:
            try:
                vals = pd.to_numeric(g[c], errors="coerce")
                out[f"{c}_mean"] = float(vals.mean())
                out[f"{c}_max"] = float(vals.max())
                out[f"{c}_min"] = float(vals.min())
            except Exception:
                pass

        grouped.append(out)

    out_df = pd.DataFrame(grouped)
    if not out_df.empty:
        sort_cols = ["count", "frequency"]
        ascending = [False, False]
        if "sample_rank_min" in out_df.columns:
            sort_cols.append("sample_rank_min")
            ascending.append(True)
        out_df = out_df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
        out_df.insert(0, "rank", range(1, len(out_df) + 1))

    out_csv = output_dir / "unique_sets_ranked.csv"
    out_df.to_csv(out_csv, index=False)

    summary = {
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "n_rows": int(len(df)),
        "n_unique_sets": int(len(out_df)),
        "detected_set_column": set_col,
        "detected_score_cols": score_cols,
        "top_n_requested": int(args.top_n),
        "top_preview": out_df.head(args.top_n).to_dict(orient="records"),
        "artifacts": {
            "unique_sets_ranked_csv": str(out_csv),
        },
    }
    write_json(output_dir / "summary.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
