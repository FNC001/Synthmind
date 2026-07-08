#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
38_build_stage35_strong_stage3_candidates.py

把 stage3 的“precursor-conditioned condition candidates”标准化到 strong joint 所需格式。

强约束：
- 输入必须显式表明这个条件候选对应哪个 precursor candidate
- 也就是必须能解析出:
    parent_precursor_rank   (优先)
  或
    precursor_rank          (可作为别名)
- 若没有 parent_precursor_rank / precursor_rank，本脚本会直接报错
- 输出写到：
    data/interim/generative/stage35_strong_stage3_candidates/<view_name>/{split}_candidates.jsonl

推荐放置：
/Users/wyc/MP_exp_doi/scripts/03_data/38_build_stage35_strong_stage3_candidates.py

理想输入字段（任意兼容子集）：
{
  "sample_id": "solid_state_00000001__mp-17677" 或 "mp-17677",
  "material_id": "mp-17677",
  "parent_precursor_rank": 0,
  "candidate_rank": 2,
  "pred_temperature_c": 850.0,
  "pred_time_h": 12.0,
  "generator_score": -1.23
}

输出标准格式：
{
  "sample_id": "...",
  "material_id": "mp-17677",
  "material_key": "mp-17677",
  "parent_precursor_rank": 0,
  "condition_rank": 2,
  "disc_conditions": {...},
  "cont_conditions": {"temperature_c": 850.0, "time_h": 12.0},
  "stage3_score": -1.23,
  "stage3_model": "...",
  "source_split": "val"
}
"""

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


def extract_mp_id(value: Any) -> str:
    s = str(value) if value is not None else ""
    m = _MP_RE.search(s)
    if m:
        return m.group(1)
    return s.strip()


ID_COLUMNS = ["sample_id", "id", "row_id", "material_id", "entry_id"]
PARENT_RANK_COLUMNS = ["parent_precursor_rank", "precursor_rank", "parent_rank"]
COND_RANK_COLUMNS = ["condition_rank", "candidate_rank", "rank"]
SCORE_COLUMNS = ["stage3_score", "generator_score", "score", "candidate_score", "prob", "logprob"]
MODEL_COLUMNS = ["stage3_model", "source_model", "model_name", "candidate_tag", "source"]


def find_first_existing_column(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    for c in names:
        if c in df.columns:
            return c
    return None


def parse_dictlike(value: Any) -> Optional[Dict[str, Any]]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, dict):
        return value
    s = str(value).strip()
    if not s:
        return None
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def build_disc_conditions(row: pd.Series) -> Dict[str, Any]:
    # 先尝试嵌套 dict
    for key in ["disc_conditions", "pred_discrete", "discrete_conditions"]:
        if key in row and pd.notna(row.get(key)):
            parsed = parse_dictlike(row.get(key))
            if parsed is not None:
                return dict(parsed)

    # 再尝试扁平列
    disc = {}
    flat_map = {
        "pred_atmosphere": "atmosphere",
        "pred_solvent": "solvent",
        "pred_synthesis_type": "synthesis_type",
        "pred_condition_source": "condition_source",
        "atmosphere": "atmosphere",
        "solvent": "solvent",
        "synthesis_type": "synthesis_type",
        "condition_source": "condition_source",
    }
    for src, dst in flat_map.items():
        if src in row and pd.notna(row.get(src)):
            disc[dst] = row.get(src)
    return disc


def build_cont_conditions(row: pd.Series) -> Dict[str, Any]:
    # 先尝试嵌套 dict
    for key in ["cont_conditions", "pred_continuous", "continuous_conditions"]:
        if key in row and pd.notna(row.get(key)):
            parsed = parse_dictlike(row.get(key))
            if parsed is not None:
                return {str(k): safe_float(v) if safe_float(v) is not None else v for k, v in parsed.items()}

    # 再尝试扁平列
    cont = {}
    flat_map = {
        "pred_temperature_c": "temperature_c",
        "pred_time_h": "time_h",
        "pred_n_main_precursors": "n_main_precursors",
        "pred_n_aux_precursors": "n_aux_precursors",
        "pred_n_heatlike_ops": "n_heatlike_ops",
        "temperature_c": "temperature_c",
        "time_h": "time_h",
        "n_main_precursors": "n_main_precursors",
        "n_aux_precursors": "n_aux_precursors",
        "n_heatlike_ops": "n_heatlike_ops",
    }
    for src, dst in flat_map.items():
        if src in row and pd.notna(row.get(src)):
            val = safe_float(row.get(src))
            cont[dst] = val if val is not None else row.get(src)
    return cont


def normalize_rows(df: pd.DataFrame, split: str, model_name: str) -> List[Dict[str, Any]]:
    split = normalize_split_name(split)

    id_col = find_first_existing_column(df, ID_COLUMNS)
    if id_col is None:
        raise ValueError(f"未检测到样本 ID 列。候选列: {ID_COLUMNS}")

    parent_rank_col = find_first_existing_column(df, PARENT_RANK_COLUMNS)
    if parent_rank_col is None:
        raise ValueError(
            "当前 stage3 输入缺少 parent_precursor_rank / precursor_rank。\n"
            "这说明它不是显式 precursor-conditioned candidates，不能直接进入 strong joint。\n"
            "请先修改 stage3 exporter，让它在导出候选时带上 parent_precursor_rank。"
        )

    cond_rank_col = find_first_existing_column(df, COND_RANK_COLUMNS)
    if cond_rank_col is None:
        raise ValueError(f"未检测到 condition rank 列。候选列: {COND_RANK_COLUMNS}")

    score_col = find_first_existing_column(df, SCORE_COLUMNS)
    model_col = find_first_existing_column(df, MODEL_COLUMNS)

    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        sample_id = str(row.get(id_col)).strip()
        material_id = extract_mp_id(row.get("material_id", row.get(id_col)))

        try:
            parent_precursor_rank = int(row.get(parent_rank_col))
            condition_rank = int(row.get(cond_rank_col))
        except Exception:
            continue

        disc_conditions = build_disc_conditions(row)
        cont_conditions = build_cont_conditions(row)

        rows.append({
            "sample_id": sample_id,
            "material_id": material_id,
            "material_key": material_id,
            "parent_precursor_rank": parent_precursor_rank,
            "condition_rank": condition_rank,
            "disc_conditions": disc_conditions,
            "cont_conditions": cont_conditions,
            "stage3_score": safe_float(row.get(score_col)) if score_col is not None else None,
            "stage3_model": str(row.get(model_col)) if model_col is not None and pd.notna(row.get(model_col)) else model_name,
            "source_split": split,
        })

    if not rows:
        raise RuntimeError(
            "标准化后没有可用行。请检查输入文件是否真的包含 precursor-conditioned condition candidates。"
        )
    return rows


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Normalize precursor-conditioned stage3 candidates for strong joint stage35.")
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
        "mode": "strong_stage3_candidates",
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
        "mode": "strong_stage3_candidates",
        "required_fields": [
            "sample_id",
            "material_id",
            "material_key",
            "parent_precursor_rank",
            "condition_rank",
            "disc_conditions",
            "cont_conditions",
            "stage3_score",
            "stage3_model",
            "source_split",
        ],
        "notes": [
            "只接受显式 precursor-conditioned stage3 candidates。",
            "若缺少 parent_precursor_rank / precursor_rank，会直接拒绝。",
        ],
    }

    write_json(output_dir / "candidate_schema.json", schema)
    write_json(output_dir / "candidate_summary.json", summary)

    print(f"Saved candidate_schema.json -> {output_dir / 'candidate_schema.json'}")
    print(f"Saved candidate_summary.json -> {output_dir / 'candidate_summary.json'}")


if __name__ == "__main__":
    main()
