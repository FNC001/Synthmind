#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05c_build_stage3_conditioned_feature_table_infer_v2.py

作用
----
为 stage3 conditioned MDN inference 构造 x_raw(1133) 的 inference 表。

输入
----
1. infer_hybrid_csv:
   例如 hybrid_stage2_dual/stage2_test_hybrid.csv
2. stage2_candidates_csv:
   例如 stage2_gflownet_candidates_dual/test_samples.csv
3. schema_json:
   例如 data/interim/generative/stage3_condition_dataset/hybrid_mixed_v1/schema.json

输出
----
infer_stage3_conditioned_x.csv
每行对应一个 (sample_id, parent_precursor_rank, parent_precursor_set)，
其余列完全按 schema["feature_cols"] 的 1133 维顺序展开。
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd


CANDIDATE_SET_COLS = ["pred_labels", "precursor_set", "main_precursors", "selected_precursors"]


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


def canonicalize_labels(labels: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in labels:
        s = str(x).strip()
        if s and s not in seen:
            out.append(s)
            seen.add(s)
    return out


def normalize_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "sample_id" not in out.columns:
        if "id" in out.columns:
            out = out.rename(columns={"id": "sample_id"})
        elif "material_id" in out.columns:
            out["sample_id"] = out["material_id"].astype(str)
        else:
            raise ValueError(f"Cannot infer sample_id from columns: {out.columns.tolist()}")
    return out


def alias_row_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(row)
    if "formula" not in d:
        if "formula_x" in d:
            d["formula"] = d.get("formula_x")
        elif "formula_y" in d:
            d["formula"] = d.get("formula_y")
    if "material_id" not in d:
        if "material_id_x" in d:
            d["material_id"] = d.get("material_id_x")
        elif "material_id_y" in d:
            d["material_id"] = d.get("material_id_y")
    return d


def safe_numeric(v: Any) -> float:
    try:
        if pd.isna(v):
            return 0.0
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        return 0.0


def pick_stage2_candidates(
    stage2_candidates_csv: Path,
    max_stage2_candidates: int,
) -> Dict[str, List[Dict[str, Any]]]:
    df = pd.read_csv(stage2_candidates_csv)
    df = normalize_id_columns(df)
    df["__input_order"] = range(len(df))

    # Preserve final Stage2 candidate order for inference.
    # 1) If rerank output has final rank, use rank ascending.
    # 2) Else if element_rerank_score exists, use score descending.
    # 3) Else keep file order within each sample.
    if "rank" in df.columns:
        df["__rank_num"] = pd.to_numeric(df["rank"], errors="coerce")
        df = df.sort_values(["sample_id", "__rank_num", "__input_order"],
                            ascending=[True, True, True],
                            kind="mergesort").copy()
    elif "element_rerank_score" in df.columns:
        df["__score_num"] = pd.to_numeric(df["element_rerank_score"], errors="coerce").fillna(-1e30)
        df = df.sort_values(["sample_id", "__score_num", "__input_order"],
                            ascending=[True, False, True],
                            kind="mergesort").copy()
    elif "sample_rank" in df.columns:
        df["__sample_rank_num"] = pd.to_numeric(df["sample_rank"], errors="coerce")
        df = df.sort_values(["sample_id", "__sample_rank_num", "__input_order"],
                            ascending=[True, True, True],
                            kind="mergesort").copy()
    else:
        df = df.sort_values(["sample_id", "__input_order"],
                            ascending=[True, True],
                            kind="mergesort").copy()

    out: Dict[str, List[Dict[str, Any]]] = {}
    seen_keys: Dict[str, set] = {}
    counts: Dict[str, int] = {}

    for _, row in df.iterrows():
        sid = str(row["sample_id"])
        if counts.get(sid, 0) >= max_stage2_candidates:
            continue

        labels: List[str] = []
        for c in CANDIDATE_SET_COLS:
            if c in row.index:
                labels = try_parse_list(row[c])
                if labels:
                    break
        labels = canonicalize_labels(labels)
        if not labels:
            continue

        key = " || ".join(sorted(labels))
        if sid not in seen_keys:
            seen_keys[sid] = set()
        if key in seen_keys[sid]:
            continue
        seen_keys[sid].add(key)

        rank = counts.get(sid, 0)
        counts[sid] = rank + 1

        rec = row.to_dict()
        rec["sample_id"] = sid
        rec["parent_precursor_rank"] = int(rank)
        rec["parent_precursor_set"] = labels
        rec["parent_precursor_set_key"] = key
        out.setdefault(sid, []).append(rec)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stage3 conditioned x_raw inference table aligned to schema feature_cols.")
    parser.add_argument("--infer_hybrid_csv", type=str, required=True)
    parser.add_argument("--stage2_candidates_csv", type=str, required=True)
    parser.add_argument("--schema_json", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--max_stage2_candidates", type=int, default=3)
    args = parser.parse_args()

    infer_hybrid_csv = Path(args.infer_hybrid_csv).expanduser().resolve()
    stage2_candidates_csv = Path(args.stage2_candidates_csv).expanduser().resolve()
    schema_json = Path(args.schema_json).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()

    for p in [infer_hybrid_csv, stage2_candidates_csv, schema_json]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required path: {p}")

    schema = json.load(open(schema_json, "r", encoding="utf-8"))
    feature_cols: List[str] = list(schema["feature_cols"])
    sample_id_col = str(schema.get("sample_id_col", "material_id"))
    precursor_vocab = list(schema.get("precursor_vocab", []))

    infer_df = pd.read_csv(infer_hybrid_csv)
    infer_df = normalize_id_columns(infer_df)

    infer_map: Dict[str, Dict[str, Any]] = {}
    for _, row in infer_df.iterrows():
        d = alias_row_dict(row.to_dict())
        sid = str(d["sample_id"])
        infer_map[sid] = d
        mid = d.get("material_id")
        if mid is not None and str(mid).strip():
            infer_map[str(mid)] = d

    stage2_map = pick_stage2_candidates(stage2_candidates_csv, max_stage2_candidates=int(args.max_stage2_candidates))

    rows_out: List[Dict[str, Any]] = []
    ordinary_matched_cols = set()
    label_activated_cols = set()

    for sid, candidates in stage2_map.items():
        base_row = infer_map.get(sid)
        if base_row is None:
            base_row = infer_map.get(str(sid))
        if base_row is None:
            continue

        base_row = alias_row_dict(base_row)
        material_id_val = base_row.get("material_id", sid)

        for cand in candidates:
            precursor_set = canonicalize_labels(cand["parent_precursor_set"])
            precursor_set_lookup = set(precursor_set)

            out = {
                "sample_id": sid,
                "material_id": material_id_val,
                "parent_precursor_rank": int(cand["parent_precursor_rank"]),
                "parent_precursor_set": json.dumps(precursor_set, ensure_ascii=False),
                "parent_precursor_set_key": cand["parent_precursor_set_key"],
            }

            for col in feature_cols:
                if col in base_row:
                    out[col] = safe_numeric(base_row[col])
                    ordinary_matched_cols.add(col)
                elif str(col).startswith("label_prec__"):
                    raw_name = str(col)[len("label_prec__"):]
                    val = 1.0 if raw_name in precursor_set_lookup else 0.0
                    out[col] = val
                    if val > 0.5:
                        label_activated_cols.add(col)
                else:
                    out[col] = 0.0

            rows_out.append(out)

    ensure_dir(output_csv.parent)
    out_df = pd.DataFrame(rows_out)

    ordered_cols = [
        "sample_id",
        "material_id",
        "parent_precursor_rank",
        "parent_precursor_set",
        "parent_precursor_set_key",
    ] + feature_cols
    out_df = out_df.reindex(columns=ordered_cols)
    out_df.to_csv(output_csv, index=False)

    summary = {
        "infer_hybrid_csv": str(infer_hybrid_csv),
        "stage2_candidates_csv": str(stage2_candidates_csv),
        "schema_json": str(schema_json),
        "output_csv": str(output_csv),
        "sample_id_col_from_schema": sample_id_col,
        "n_feature_cols": int(len(feature_cols)),
        "n_precursor_vocab": int(len(precursor_vocab)),
        "n_label_prec_cols_in_feature_cols": int(sum(str(c).startswith("label_prec__") for c in feature_cols)),
        "n_rows_output": int(len(out_df)),
        "matched_non_label_feature_cols_count": int(len([c for c in ordinary_matched_cols if not str(c).startswith("label_prec__")])),
        "matched_non_label_feature_cols_preview": [c for c in feature_cols if c in ordinary_matched_cols and not str(c).startswith("label_prec__")][:80],
        "activated_label_prec_cols_count": int(len(label_activated_cols)),
        "activated_label_prec_cols_preview": sorted(label_activated_cols)[:80],
    }
    write_json(output_csv.with_name(output_csv.stem + "_summary.json"), summary)

    print(f"[DONE] output_csv -> {output_csv}")
    print(f"[DONE] summary    -> {output_csv.with_name(output_csv.stem + '_summary.json')}")


if __name__ == "__main__":
    main()

