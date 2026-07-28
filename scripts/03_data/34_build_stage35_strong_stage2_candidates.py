#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


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


def safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


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
    "pred_labels",       # 关键补充：适配 ar_setgen_commonized_v1
    "output_tokens",
]

ID_COLUMNS = ["sample_id", "id", "row_id", "material_id", "entry_id"]
RANK_COLUMNS = ["precursor_rank", "candidate_rank", "rank"]
SCORE_COLUMNS = ["stage2_score", "score", "candidate_score", "prob", "logprob", "reward"]
MODEL_COLUMNS = ["stage2_model", "source_model", "model_name", "source"]


def parse_listlike(value: Any) -> Optional[List[str]]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, list):
        return [strip_label_prefix(str(x)) for x in value]
    if isinstance(value, tuple):
        return [strip_label_prefix(str(x)) for x in value]

    s = str(value).strip()
    if not s:
        return None

    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return [strip_label_prefix(str(x)) for x in arr]
        except Exception:
            pass
        try:
            arr = ast.literal_eval(s)
            if isinstance(arr, list):
                return [strip_label_prefix(str(x)) for x in arr]
        except Exception:
            pass

    for sep in ["||", ";", "|"]:
        if sep in s:
            vals = [strip_label_prefix(x.strip()) for x in s.split(sep) if x.strip()]
            if vals:
                return vals

    return None


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


def assert_not_single_label_mode(df: pd.DataFrame) -> None:
    if "candidate_label" in df.columns and detect_set_column(df) is None:
        raise ValueError(
            "当前 stage2 输入看起来是“单 precursor label 排序”而不是完整 precursor set candidates。\n"
            "这类输入只能用于 weak joint，不满足 strong joint 数据线要求。\n"
            "请改用能导出完整 precursor_set 的 stage2 run（如 AR set generator）。"
        )


def normalize_rows(df: pd.DataFrame, split: str, model_name: str) -> List[Dict[str, Any]]:
    split = normalize_split_name(split)

    assert_not_single_label_mode(df)
    set_col = detect_set_column(df)
    if set_col is None:
        raise ValueError(
            "未检测到完整 precursor set 列。当前文件不适合作为 strong joint 的 stage2 输入。\n"
            f"可识别的 set 列包括: {SET_COLUMNS}"
        )

    id_col = find_first_existing_column(df, ID_COLUMNS)
    if id_col is None:
        raise ValueError(f"未检测到样本 ID 列。候选列: {ID_COLUMNS}")

    rank_col = find_first_existing_column(df, RANK_COLUMNS)
    if rank_col is None:
        raise ValueError(f"未检测到 rank 列。候选列: {RANK_COLUMNS}")

    score_col = find_first_existing_column(df, SCORE_COLUMNS)
    model_col = find_first_existing_column(df, MODEL_COLUMNS)

    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        precursor_set = parse_listlike(row.get(set_col))
        if precursor_set is None or len(precursor_set) == 0:
            continue

        sample_id = str(row.get(id_col)).strip()
        material_id = extract_mp_id(row.get("material_id", row.get(id_col)))

        try:
            precursor_rank = int(row.get(rank_col))
        except Exception:
            continue

        stage2_score = safe_float(row.get(score_col)) if score_col is not None else None
        stage2_model = str(row.get(model_col)) if model_col is not None and pd.notna(row.get(model_col)) else model_name

        rows.append({
            "sample_id": sample_id,
            "material_id": material_id,
            "material_key": material_id,
            "precursor_rank": precursor_rank,
            "precursor_set": precursor_set,
            "n_precursors": int(len(precursor_set)),
            "stage2_score": stage2_score,
            "stage2_model": stage2_model,
            "source_split": split,
            "formula": row.get("formula"),
            "doi": row.get("doi"),
            "source_dataset": row.get("source_dataset"),
            "synthesis_type": row.get("synthesis_type"),
            "exact_match": row.get("exact_match"),
        })

    if not rows:
        raise RuntimeError("标准化后没有可用行。请检查输入文件是否真的包含完整 precursor_set candidates。")
    return rows


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Normalize stage2 full precursor-set candidates for strong joint stage35.")
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--splits", type=str, default="val,test")
    p.add_argument("--view_name", type=str, default="hybrid_core_strong")
    p.add_argument("--model_name", type=str, default="")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    splits = normalize_split_list([x.strip() for x in str(args.splits).split(",") if x.strip()])

    summary: Dict[str, Any] = {
        "view_name": str(args.view_name),
        "mode": "strong_stage2_candidates",
        "splits": splits,
        "split_stats": {},
    }

    for split in splits:
        in_path = resolve_input_file(input_dir, split)
        df = read_table(in_path)
        model_name = str(args.model_name).strip() or input_dir.name

        rows = normalize_rows(df, split=split, model_name=model_name)
        out_path = output_dir / f"{split}_candidates.jsonl"
        write_jsonl(out_path, rows)

        summary["split_stats"][split] = {
            "input_path": str(in_path),
            "output_path": str(out_path),
            "n_input_rows": int(len(df)),
            "n_output_rows": int(len(rows)),
            "n_unique_materials": int(len(set(r["material_key"] for r in rows))),
        }

        print(
            f"[{split}] input={len(df)} output={len(rows)} "
            f"unique_materials={summary['split_stats'][split]['n_unique_materials']}"
        )

    schema = {
        "view_name": str(args.view_name),
        "mode": "strong_stage2_candidates",
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
        ],
        "supported_set_columns": SET_COLUMNS,
        "notes": [
            "接受完整 precursor_set candidates。",
            "会自动去掉 pred_labels 中的 label_prec__ 前缀。",
            "若输入只是单 precursor label 排序，本脚本会直接拒绝。",
        ],
    }

    write_json(output_dir / "candidate_schema.json", schema)
    write_json(output_dir / "candidate_summary.json", summary)

    print(f"Saved candidate_schema.json -> {output_dir / 'candidate_schema.json'}")
    print(f"Saved candidate_summary.json -> {output_dir / 'candidate_summary.json'}")


if __name__ == "__main__":
    main()
