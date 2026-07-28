#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
37_build_stage2_set_candidates_common.py

公共 stage2 set-candidate exporter / assembler
------------------------------------------------
目标：
1) 统一把不同 stage2 模型的输出转成 strong joint 需要的 precursor_set candidates
2) 兼容两大类输入
   A. 已经是“集合候选”：
      - pred_labels / precursor_set / candidate_set / pred_precursors ...
   B. 只有“单标签排序”：
      - candidate_label + score + rank
      -> 自动组装 top-k precursor_set candidates

输出统一格式：
{
  "sample_id": "...",
  "material_id": "mp-17677",
  "material_key": "mp-17677",
  "precursor_rank": 0,
  "precursor_set": ["NH4H2PO4", "Nb2O5"],
  "n_precursors": 2,
  "stage2_score": 0.93,
  "stage2_model": "setpred_commonized_v1",
  "source_split": "val",
  "source_mode": "assembled_from_label_ranking",
  "formula": "...",
  "doi": "...",
  "source_dataset": "...",
  "synthesis_type": "...",
  "exact_match": null,
  "reward": null
}

推荐放置：
/Users/wyc/MP_exp_doi/scripts/03_data/37_build_stage2_set_candidates_common.py
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_builtin(obj), f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_builtin(row), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL 解析失败: {path} 第 {line_no} 行: {e}") from e
    return rows


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.DataFrame(read_jsonl(path))
    raise ValueError(f"不支持的文件格式: {path}")


def normalize_split_name(split: str) -> str:
    s = str(split).strip().lower()
    mapping = {
        "tr": "train",
        "train": "train",
        "va": "val",
        "valid": "val",
        "validation": "val",
        "val": "val",
        "te": "test",
        "test": "test",
    }
    if s not in mapping:
        raise ValueError(f"不支持的 split: {split}")
    return mapping[s]


def normalize_split_list(splits: Sequence[str]) -> List[str]:
    return [normalize_split_name(s) for s in splits]


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except Exception:
        return default


def resolve_input_file(directory: Path, split: str) -> Path:
    split = normalize_split_name(split)
    candidates = [
        directory / f"{split}_candidates.jsonl",
        directory / f"{split}_candidates.csv",
        directory / f"{split}_predictions.jsonl",
        directory / f"{split}_predictions.csv",
        directory / f"pred_{split}.csv",
        directory / f"{split}.jsonl",
        directory / f"{split}.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"在 {directory} 下找不到 split={split} 的输入文件。尝试过:\n" +
        "\n".join(str(x) for x in candidates)
    )


_MP_RE = re.compile(r"(mp-\d+)")
_LABEL_PREFIX_RE = re.compile(r"^label_prec__")


def extract_mp_id(value: Any) -> str:
    s = str(value) if value is not None else ""
    m = _MP_RE.search(s)
    if m:
        return m.group(1)
    return s.strip()


def strip_label_prefix(x: str) -> str:
    return _LABEL_PREFIX_RE.sub("", str(x)).strip()


def parse_listlike(value: Any) -> Optional[List[str]]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, list):
        return [strip_label_prefix(str(x)) for x in value if str(x).strip()]
    if isinstance(value, tuple):
        return [strip_label_prefix(str(x)) for x in value if str(x).strip()]

    s = str(value).strip()
    if not s:
        return None

    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return [strip_label_prefix(str(x)) for x in arr if str(x).strip()]
        except Exception:
            pass
        try:
            arr = ast.literal_eval(s)
            if isinstance(arr, list):
                return [strip_label_prefix(str(x)) for x in arr if str(x).strip()]
        except Exception:
            pass

    for sep in ["||", ";", "|"]:
        if sep in s:
            vals = [strip_label_prefix(x.strip()) for x in s.split(sep) if x.strip()]
            if vals:
                return vals

    return None


# ---------------------------------------------------------------------
# schema detection
# ---------------------------------------------------------------------
SET_COLUMNS = [
    "precursor_set",
    "candidate_set",
    "pred_precursor_set",
    "generated_set",
    "generated_precursor_set",
    "candidate_precursor_set",
    "precursors",
    "pred_precursors",
    "candidate_labels",
    "pred_labels",
    "output_tokens",
]

SINGLE_LABEL_COLUMNS = [
    "candidate_label",
    "label",
    "pred_label",
]

ID_COLUMNS = ["sample_id", "id", "row_id", "material_id", "entry_id"]
RANK_COLUMNS = ["precursor_rank", "candidate_rank", "rank"]
SCORE_COLUMNS = ["stage2_score", "score", "candidate_score", "prob", "logprob", "reward"]
MODEL_COLUMNS = ["stage2_model", "source_model", "model_name", "source"]


def find_first_existing_column(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    for c in names:
        if c in df.columns:
            return c
    return None


def detect_set_column(df: pd.DataFrame) -> Optional[str]:
    for c in SET_COLUMNS:
        if c in df.columns:
            sample_vals = df[c].dropna().head(20).tolist()
            ok = False
            for v in sample_vals:
                parsed = parse_listlike(v)
                if parsed is not None and len(parsed) >= 1:
                    ok = True
                    break
            if ok:
                return c
    return None


def detect_single_label_column(df: pd.DataFrame) -> Optional[str]:
    for c in SINGLE_LABEL_COLUMNS:
        if c in df.columns:
            return c
    return None


def detect_mode(df: pd.DataFrame) -> str:
    set_col = detect_set_column(df)
    if set_col is not None:
        return "set_candidates"
    single_label_col = detect_single_label_column(df)
    if single_label_col is not None:
        return "single_label_ranking"

    # 兜底：如果有很多像 label_prec__* 这种宽表列，也算可组装
    prefix_cols = [c for c in df.columns if c.startswith("label_prec__")]
    if len(prefix_cols) >= 2:
        return "wide_label_scores"

    raise ValueError(
        "无法识别输入模式。既没有检测到 set candidates，也没有检测到单标签排序或宽表标签分数。"
    )


# ---------------------------------------------------------------------
# common metadata helpers
# ---------------------------------------------------------------------
def row_id_value(row: pd.Series, id_col: Optional[str]) -> str:
    if id_col is not None and pd.notna(row.get(id_col)):
        return str(row.get(id_col)).strip()
    for c in ID_COLUMNS:
        if c in row and pd.notna(row.get(c)):
            return str(row.get(c)).strip()
    return ""


def row_material_id(row: pd.Series, id_col: Optional[str]) -> str:
    if "material_id" in row and pd.notna(row.get("material_id")):
        return extract_mp_id(row.get("material_id"))
    return extract_mp_id(row_id_value(row, id_col))


def normalize_meta(row: pd.Series, id_col: Optional[str]) -> Dict[str, Any]:
    return {
        "sample_id": row_id_value(row, id_col),
        "material_id": row_material_id(row, id_col),
        "material_key": row_material_id(row, id_col),
        "formula": row.get("formula"),
        "doi": row.get("doi"),
        "source_dataset": row.get("source_dataset"),
        "synthesis_type": row.get("synthesis_type"),
        "exact_match": row.get("exact_match"),
        "reward": safe_float(row.get("reward")),
    }


# ---------------------------------------------------------------------
# mode A: already set candidates
# ---------------------------------------------------------------------
def normalize_from_set_candidates(df: pd.DataFrame, split: str, model_name: str) -> List[Dict[str, Any]]:
    set_col = detect_set_column(df)
    if set_col is None:
        raise ValueError("未检测到 set candidate 列。")

    id_col = find_first_existing_column(df, ID_COLUMNS)
    rank_col = find_first_existing_column(df, RANK_COLUMNS)
    score_col = find_first_existing_column(df, SCORE_COLUMNS)
    model_col = find_first_existing_column(df, MODEL_COLUMNS)

    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        precursor_set = parse_listlike(row.get(set_col))
        if precursor_set is None or len(precursor_set) == 0:
            continue

        try:
            precursor_rank = int(row.get(rank_col)) if rank_col is not None else len(rows)
        except Exception:
            continue

        meta = normalize_meta(row, id_col)
        rows.append({
            **meta,
            "precursor_rank": precursor_rank,
            "precursor_set": precursor_set,
            "n_precursors": int(len(precursor_set)),
            "stage2_score": safe_float(row.get(score_col)) if score_col is not None else None,
            "stage2_model": str(row.get(model_col)) if model_col is not None and pd.notna(row.get(model_col)) else model_name,
            "source_split": normalize_split_name(split),
            "source_mode": "direct_set_candidates",
        })

    if not rows:
        raise RuntimeError("set candidate 标准化后没有可用行。")
    return rows


# ---------------------------------------------------------------------
# mode B: single label ranking -> set assembly
# ---------------------------------------------------------------------
def assemble_group_from_single_labels(
    g: pd.DataFrame,
    label_col: str,
    rank_col: str,
    score_col: Optional[str],
    id_col: Optional[str],
    split: str,
    model_name: str,
    max_set_size: int,
    min_set_size: int,
    use_cumulative_topn: bool,
    threshold_list: Sequence[float],
) -> List[Dict[str, Any]]:
    if score_col is not None:
        g = g.sort_values([score_col, rank_col], ascending=[False, True]).reset_index(drop=True)
    else:
        g = g.sort_values([rank_col], ascending=[True]).reset_index(drop=True)

    labels: List[str] = []
    scores: List[float] = []

    for _, row in g.iterrows():
        raw_label = row.get(label_col)
        if raw_label is None or (isinstance(raw_label, float) and np.isnan(raw_label)):
            continue
        label = strip_label_prefix(raw_label)
        if not label:
            continue
        labels.append(label)
        scores.append(float(safe_float(row.get(score_col), 0.0)) if score_col is not None else 0.0)

    if not labels:
        return []

    # 去重，保留第一个（通常分数最高）
    uniq_labels: List[str] = []
    uniq_scores: List[float] = []
    seen = set()
    for lbl, sc in zip(labels, scores):
        if lbl in seen:
            continue
        seen.add(lbl)
        uniq_labels.append(lbl)
        uniq_scores.append(sc)

    labels = uniq_labels
    scores = uniq_scores

    max_n = min(int(max_set_size), len(labels))
    min_n = min(int(min_set_size), max_n)
    candidates: List[Tuple[List[str], float, str]] = []

    if use_cumulative_topn:
        for n in range(max(min_n, 1), max_n + 1):
            cand = labels[:n]
            sc = float(np.mean(scores[:n])) if n > 0 else 0.0
            candidates.append((cand, sc, "assembled_topn"))

    for thr in threshold_list:
        idx = [i for i, s in enumerate(scores) if s >= float(thr)]
        if idx:
            cand = [labels[i] for i in idx[:max_n]]
            if len(cand) >= min_n:
                sc = float(np.mean([scores[i] for i in idx[:max_n]]))
                candidates.append((cand, sc, f"assembled_threshold_{thr:g}"))

    # 去重（集合内容相同）
    dedup: Dict[Tuple[str, ...], Tuple[List[str], float, str]] = {}
    for cand, sc, tag in candidates:
        key = tuple(cand)
        if key not in dedup or sc > dedup[key][1]:
            dedup[key] = (cand, sc, tag)

    meta = normalize_meta(g.iloc[0], id_col)
    out_rows: List[Dict[str, Any]] = []
    for rank, (cand, sc, tag) in enumerate(sorted(dedup.values(), key=lambda x: x[1], reverse=True), start=1):
        out_rows.append({
            **meta,
            "precursor_rank": int(rank),
            "precursor_set": cand,
            "n_precursors": int(len(cand)),
            "stage2_score": float(sc),
            "stage2_model": model_name,
            "source_split": normalize_split_name(split),
            "source_mode": tag,
        })
    return out_rows


def normalize_from_single_label_ranking(
    df: pd.DataFrame,
    split: str,
    model_name: str,
    max_set_size: int,
    min_set_size: int,
    use_cumulative_topn: bool,
    threshold_list: Sequence[float],
) -> List[Dict[str, Any]]:
    label_col = detect_single_label_column(df)
    if label_col is None:
        raise ValueError("未检测到 single label 列。")

    id_col = find_first_existing_column(df, ID_COLUMNS)
    rank_col = find_first_existing_column(df, RANK_COLUMNS)
    if rank_col is None:
        raise ValueError(f"未检测到 rank 列。候选列: {RANK_COLUMNS}")
    score_col = find_first_existing_column(df, SCORE_COLUMNS)

    group_col = id_col if id_col is not None else "material_id"
    rows: List[Dict[str, Any]] = []
    for _, g in df.groupby(group_col, sort=False):
        rows.extend(
            assemble_group_from_single_labels(
                g=g,
                label_col=label_col,
                rank_col=rank_col,
                score_col=score_col,
                id_col=id_col,
                split=split,
                model_name=model_name,
                max_set_size=max_set_size,
                min_set_size=min_set_size,
                use_cumulative_topn=use_cumulative_topn,
                threshold_list=threshold_list,
            )
        )

    if not rows:
        raise RuntimeError("single label ranking 组装后没有可用行。")
    return rows


# ---------------------------------------------------------------------
# mode C: wide label-score table -> set assembly
# ---------------------------------------------------------------------
def detect_wide_label_score_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("label_prec__")]
    return cols


def normalize_from_wide_label_scores(
    df: pd.DataFrame,
    split: str,
    model_name: str,
    max_set_size: int,
    min_set_size: int,
    use_cumulative_topn: bool,
    threshold_list: Sequence[float],
) -> List[Dict[str, Any]]:
    label_score_cols = detect_wide_label_score_columns(df)
    if not label_score_cols:
        raise ValueError("未检测到宽表标签分数字段（label_prec__*）。")

    id_col = find_first_existing_column(df, ID_COLUMNS)
    if id_col is None:
        raise ValueError(f"未检测到样本 ID 列。候选列: {ID_COLUMNS}")

    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        label_scores = []
        for c in label_score_cols:
            sc = safe_float(row.get(c))
            if sc is None:
                continue
            label_scores.append((strip_label_prefix(c), float(sc)))
        label_scores.sort(key=lambda x: x[1], reverse=True)

        labels = [x[0] for x in label_scores]
        scores = [x[1] for x in label_scores]

        candidates: List[Tuple[List[str], float, str]] = []
        max_n = min(int(max_set_size), len(labels))
        min_n = min(int(min_set_size), max_n)

        if use_cumulative_topn:
            for n in range(max(min_n, 1), max_n + 1):
                cand = labels[:n]
                sc = float(np.mean(scores[:n])) if n > 0 else 0.0
                candidates.append((cand, sc, "assembled_topn"))

        for thr in threshold_list:
            idx = [i for i, s in enumerate(scores) if s >= float(thr)]
            if idx:
                cand = [labels[i] for i in idx[:max_n]]
                if len(cand) >= min_n:
                    sc = float(np.mean([scores[i] for i in idx[:max_n]]))
                    candidates.append((cand, sc, f"assembled_threshold_{thr:g}"))

        dedup: Dict[Tuple[str, ...], Tuple[List[str], float, str]] = {}
        for cand, sc, tag in candidates:
            key = tuple(cand)
            if key not in dedup or sc > dedup[key][1]:
                dedup[key] = (cand, sc, tag)

        meta = normalize_meta(row, id_col)
        for rank, (cand, sc, tag) in enumerate(sorted(dedup.values(), key=lambda x: x[1], reverse=True), start=1):
            rows.append({
                **meta,
                "precursor_rank": int(rank),
                "precursor_set": cand,
                "n_precursors": int(len(cand)),
                "stage2_score": float(sc),
                "stage2_model": model_name,
                "source_split": normalize_split_name(split),
                "source_mode": tag,
            })

    if not rows:
        raise RuntimeError("wide label-score 组装后没有可用行。")
    return rows


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Common builder for stage2 precursor_set candidates.")
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--splits", type=str, default="val,test")
    p.add_argument("--view_name", type=str, default="hybrid_core_strong")
    p.add_argument("--model_name", type=str, default="")

    p.add_argument("--max_set_size", type=int, default=6)
    p.add_argument("--min_set_size", type=int, default=1)
    p.add_argument("--no_cumulative_topn", action="store_true")
    p.add_argument("--thresholds", type=str, default="0.95,0.75,0.5,0.25")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    splits = normalize_split_list([x.strip() for x in str(args.splits).split(",") if x.strip()])
    threshold_list = [float(x.strip()) for x in str(args.thresholds).split(",") if x.strip()]
    use_cumulative_topn = not bool(args.no_cumulative_topn)

    summary: Dict[str, Any] = {
        "view_name": str(args.view_name),
        "mode": "stage2_set_candidates_common",
        "splits": splits,
        "config": {
            "max_set_size": int(args.max_set_size),
            "min_set_size": int(args.min_set_size),
            "use_cumulative_topn": bool(use_cumulative_topn),
            "thresholds": threshold_list,
        },
        "split_stats": {},
    }

    for split in splits:
        in_path = resolve_input_file(input_dir, split)
        df = read_table(in_path)
        model_name = str(args.model_name).strip() or input_dir.name

        mode = detect_mode(df)
        if mode == "set_candidates":
            rows = normalize_from_set_candidates(df, split=split, model_name=model_name)
        elif mode == "single_label_ranking":
            rows = normalize_from_single_label_ranking(
                df=df,
                split=split,
                model_name=model_name,
                max_set_size=int(args.max_set_size),
                min_set_size=int(args.min_set_size),
                use_cumulative_topn=use_cumulative_topn,
                threshold_list=threshold_list,
            )
        elif mode == "wide_label_scores":
            rows = normalize_from_wide_label_scores(
                df=df,
                split=split,
                model_name=model_name,
                max_set_size=int(args.max_set_size),
                min_set_size=int(args.min_set_size),
                use_cumulative_topn=use_cumulative_topn,
                threshold_list=threshold_list,
            )
        else:
            raise RuntimeError(f"未知 mode: {mode}")

        out_path = output_dir / f"{split}_candidates.jsonl"
        write_jsonl(out_path, rows)

        summary["split_stats"][split] = {
            "input_path": str(in_path),
            "output_path": str(out_path),
            "detected_mode": mode,
            "n_input_rows": int(len(df)),
            "n_output_rows": int(len(rows)),
            "n_unique_materials": int(len(set(r["material_key"] for r in rows))),
            "mean_candidates_per_material": float(
                len(rows) / max(len(set(r["material_key"] for r in rows)), 1)
            ),
        }

        print(
            f"[{split}] mode={mode} input={len(df)} output={len(rows)} "
            f"unique_materials={summary['split_stats'][split]['n_unique_materials']}"
        )

    schema = {
        "view_name": str(args.view_name),
        "mode": "stage2_set_candidates_common",
        "required_fields": [
            "sample_id",
            "material_id",
            "material_key",
            "precursor_rank",
            "precursor_set",
            "n_precursors",
            "stage2_score",
            "stage2_model",
            "source_split",
            "source_mode",
        ],
        "supported_input_modes": [
            "set_candidates",
            "single_label_ranking",
            "wide_label_scores",
        ],
        "notes": [
            "若输入已是完整 set candidates，则直接标准化。",
            "若输入是单 precursor label 排序，则自动组装成 top-k precursor_set candidates。",
            "若输入是宽表标签分数，也会自动组装成 precursor_set candidates。",
            "所有 label_prec__ 前缀都会自动剥离。",
        ],
    }

    write_json(output_dir / "candidate_schema.json", schema)
    write_json(output_dir / "candidate_summary.json", summary)

    print(f"Saved candidate_schema.json -> {output_dir / 'candidate_schema.json'}")
    print(f"Saved candidate_summary.json -> {output_dir / 'candidate_summary.json'}")


if __name__ == "__main__":
    main()
