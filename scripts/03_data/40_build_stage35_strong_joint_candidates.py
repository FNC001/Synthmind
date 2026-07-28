#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
40_build_stage35_strong_joint_candidates.py

把已经标准化好的 strong stage2 candidates 与 strong stage3 candidates
按 (material_id/material_key, precursor_rank == parent_precursor_rank) 拼接成
真正的 recipe-level strong joint candidates。

输入
----
1) strong stage2 candidates:
   data/interim/generative/stage35_strong_stage2_candidates/<model_name>/{split}_candidates.jsonl
2) strong stage3 candidates:
   data/interim/generative/stage35_strong_stage3_candidates/<model_name>/{split}_candidates.jsonl

输出
----
data/interim/generative/stage35_strong_joint_candidates/<view_name>/
  - val_candidates.jsonl
  - test_candidates.jsonl
  - candidate_schema.json
  - candidate_summary.json

每条输出记录
------------
{
  "sample_id": "...",
  "material_id": "mp-17677",
  "material_key": "mp-17677",
  "recipe_id": "mp-17677__p3__c2",
  "split": "val",

  "precursor_rank": 3,
  "parent_precursor_rank": 3,
  "condition_rank": 2,

  "precursor_set": ["NH4H2PO4", "Nb2O5"],
  "n_precursors": 2,

  "disc_conditions": {...},
  "disc_condition_indices": {...},
  "cont_conditions": {"temperature_c": 850.0, "time_h": 12.0},

  "stage2_score": ...,
  "stage3_score": ...,
  "joint_prior_score": null,

  "stage2_model": "...",
  "stage3_model": "...",

  "formula": "...",
  "doi": "...",
  "source_dataset": "...",
  "synthesis_type": "...",

  "candidate_source": "strong_joint__stage2_set__stage3_precursor_conditioned"
}
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def read_table(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl(path)
    if suffix == ".csv":
        return pd.read_csv(path).to_dict(orient="records")
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


def resolve_candidates_file(directory: Path, split: str) -> Path:
    split = normalize_split_name(split)
    candidates = [
        directory / f"{split}_candidates.jsonl",
        directory / f"{split}_candidates.csv",
        directory / f"{split}.jsonl",
        directory / f"{split}.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"在 {directory} 下找不到 split={split} 的 candidates 文件。尝试过:\n" +
        "\n".join(str(x) for x in candidates)
    )


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except Exception:
        return default


def recipe_id(material_key: str, precursor_rank: int, condition_rank: int) -> str:
    return f"{material_key}__p{int(precursor_rank)}__c{int(condition_rank)}"


# ---------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------
def normalize_stage2_row(row: Dict[str, Any]) -> Dict[str, Any]:
    material_key = str(
        row.get("material_key")
        or row.get("material_id")
        or row.get("sample_id")
        or ""
    ).strip()
    precursor_rank = safe_int(row.get("precursor_rank"), None)
    if not material_key or precursor_rank is None:
        raise ValueError(f"非法 strong stage2 row，缺少 material_key/precursor_rank: {row}")

    precursor_set = row.get("precursor_set", [])
    if not isinstance(precursor_set, list):
        precursor_set = [str(precursor_set)]

    return {
        "sample_id": row.get("sample_id"),
        "material_id": row.get("material_id", material_key),
        "material_key": material_key,
        "precursor_rank": precursor_rank,
        "precursor_set": precursor_set,
        "n_precursors": safe_int(row.get("n_precursors"), len(precursor_set)) or len(precursor_set),
        "stage2_score": row.get("stage2_score"),
        "stage2_model": row.get("stage2_model"),
        "formula": row.get("formula"),
        "doi": row.get("doi"),
        "source_dataset": row.get("source_dataset"),
        "synthesis_type": row.get("synthesis_type"),
        "_raw": row,
    }


def normalize_stage3_row(row: Dict[str, Any]) -> Dict[str, Any]:
    material_key = str(
        row.get("material_key")
        or row.get("material_id")
        or row.get("sample_id")
        or ""
    ).strip()
    parent_precursor_rank = safe_int(row.get("parent_precursor_rank", row.get("precursor_rank")), None)
    condition_rank = safe_int(row.get("condition_rank", row.get("candidate_rank")), None)
    if not material_key or parent_precursor_rank is None or condition_rank is None:
        raise ValueError(f"非法 strong stage3 row，缺少 material_key/parent_precursor_rank/condition_rank: {row}")

    disc_conditions = row.get("disc_conditions", {}) or {}
    disc_condition_indices = row.get("disc_condition_indices", {}) or {}
    cont_conditions = row.get("cont_conditions", {}) or {}

    return {
        "sample_id": row.get("sample_id"),
        "material_id": row.get("material_id", material_key),
        "material_key": material_key,
        "parent_precursor_rank": parent_precursor_rank,
        "condition_rank": condition_rank,
        "disc_conditions": disc_conditions,
        "disc_condition_indices": disc_condition_indices,
        "cont_conditions": cont_conditions,
        "stage3_score": row.get("stage3_score"),
        "stage3_model": row.get("stage3_model"),
        "_raw": row,
    }


# ---------------------------------------------------------------------
# build
# ---------------------------------------------------------------------
def build_strong_joint_candidates(
    split: str,
    stage2_rows: Sequence[Dict[str, Any]],
    stage3_rows: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    split = normalize_split_name(split)

    s2 = [normalize_stage2_row(r) for r in stage2_rows]
    s3 = [normalize_stage3_row(r) for r in stage3_rows]

    stage2_by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for r in s2:
        key = (r["material_key"], int(r["precursor_rank"]))
        # stage2 这里默认一对一；若冲突，保留 precursor_set 更长或 score 更高的
        if key not in stage2_by_key:
            stage2_by_key[key] = r
        else:
            old = stage2_by_key[key]
            old_len = len(old.get("precursor_set", []))
            new_len = len(r.get("precursor_set", []))
            old_score = old.get("stage2_score")
            new_score = r.get("stage2_score")
            old_score = -1e18 if old_score is None else float(old_score)
            new_score = -1e18 if new_score is None else float(new_score)
            if new_len > old_len or (new_len == old_len and new_score > old_score):
                stage2_by_key[key] = r

    stage3_by_key: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in s3:
        key = (r["material_key"], int(r["parent_precursor_rank"]))
        stage3_by_key[key].append(r)
    for k in stage3_by_key:
        stage3_by_key[k].sort(key=lambda x: int(x["condition_rank"]))

    output_rows: List[Dict[str, Any]] = []
    n_stage2_keys = len(stage2_by_key)
    n_stage3_keys = len(stage3_by_key)
    matched_keys = 0

    for key, s2_row in sorted(stage2_by_key.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        s3_list = stage3_by_key.get(key, [])
        if not s3_list:
            continue
        matched_keys += 1

        for s3_row in s3_list:
            out = {
                "sample_id": s2_row.get("sample_id") or s3_row.get("sample_id"),
                "material_id": s2_row.get("material_id") or s3_row.get("material_id"),
                "material_key": s2_row["material_key"],
                "recipe_id": recipe_id(
                    s2_row["material_key"],
                    int(s2_row["precursor_rank"]),
                    int(s3_row["condition_rank"]),
                ),
                "split": split,

                "precursor_rank": int(s2_row["precursor_rank"]),
                "parent_precursor_rank": int(s3_row["parent_precursor_rank"]),
                "condition_rank": int(s3_row["condition_rank"]),

                "precursor_set": s2_row["precursor_set"],
                "n_precursors": int(s2_row["n_precursors"]),

                "disc_conditions": s3_row["disc_conditions"],
                "disc_condition_indices": s3_row["disc_condition_indices"],
                "cont_conditions": s3_row["cont_conditions"],

                "stage2_score": s2_row.get("stage2_score"),
                "stage3_score": s3_row.get("stage3_score"),
                "joint_prior_score": None,

                "stage2_model": s2_row.get("stage2_model"),
                "stage3_model": s3_row.get("stage3_model"),

                "formula": s2_row.get("formula"),
                "doi": s2_row.get("doi"),
                "source_dataset": s2_row.get("source_dataset"),
                "synthesis_type": s2_row.get("synthesis_type"),

                "candidate_source": "strong_joint__stage2_set__stage3_precursor_conditioned",
            }
            output_rows.append(out)

    summary = {
        "split": split,
        "n_stage2_rows": int(len(s2)),
        "n_stage3_rows": int(len(s3)),
        "n_stage2_keys": int(n_stage2_keys),
        "n_stage3_keys": int(n_stage3_keys),
        "n_matched_keys": int(matched_keys),
        "n_joint_recipe_candidates": int(len(output_rows)),
        "mean_stage3_candidates_per_matched_key": float(len(output_rows) / matched_keys) if matched_keys > 0 else None,
    }
    return output_rows, summary


# ---------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build recipe-level strong joint candidates from strong stage2 and stage3 candidates.")
    p.add_argument("--stage2_candidates_dir", type=str, required=True)
    p.add_argument("--stage3_candidates_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--splits", type=str, default="val,test")
    p.add_argument("--view_name", type=str, default="hybrid_core_strong")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    stage2_dir = Path(args.stage2_candidates_dir).expanduser().resolve()
    stage3_dir = Path(args.stage3_candidates_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    splits = normalize_split_list([x.strip() for x in str(args.splits).split(",") if x.strip()])

    summary: Dict[str, Any] = {
        "view_name": str(args.view_name),
        "mode": "strong_joint_candidates",
        "splits": splits,
        "split_stats": {},
    }

    for split in splits:
        stage2_path = resolve_candidates_file(stage2_dir, split)
        stage3_path = resolve_candidates_file(stage3_dir, split)

        stage2_rows = read_table(stage2_path)
        stage3_rows = read_table(stage3_path)

        joint_rows, split_summary = build_strong_joint_candidates(
            split=split,
            stage2_rows=stage2_rows,
            stage3_rows=stage3_rows,
        )

        out_path = output_dir / f"{normalize_split_name(split)}_candidates.jsonl"
        write_jsonl(out_path, joint_rows)

        summary["split_stats"][normalize_split_name(split)] = {
            **split_summary,
            "stage2_candidates_path": str(stage2_path),
            "stage3_candidates_path": str(stage3_path),
            "output_candidates_path": str(out_path),
        }

        print(
            f"[{normalize_split_name(split)}] "
            f"stage2={len(stage2_rows)} "
            f"stage3={len(stage3_rows)} "
            f"matched_keys={split_summary['n_matched_keys']} "
            f"joint={len(joint_rows)}"
        )

    schema = {
        "view_name": str(args.view_name),
        "mode": "strong_joint_candidates",
        "required_fields": [
            "sample_id",
            "material_id",
            "material_key",
            "recipe_id",
            "split",
            "precursor_rank",
            "parent_precursor_rank",
            "condition_rank",
            "precursor_set",
            "n_precursors",
            "disc_conditions",
            "disc_condition_indices",
            "cont_conditions",
            "stage2_score",
            "stage3_score",
            "stage2_model",
            "stage3_model",
        ],
        "join_rule": "(material_key, precursor_rank) <-> (material_key, parent_precursor_rank)",
        "notes": [
            "这是 strong joint 的 recipe-level candidates。",
            "它要求 stage2 是完整 precursor_set candidates，stage3 是 precursor-conditioned candidates。",
        ],
    }

    write_json(output_dir / "candidate_schema.json", schema)
    write_json(output_dir / "candidate_summary.json", summary)

    print(f"Saved candidate_schema.json -> {output_dir / 'candidate_schema.json'}")
    print(f"Saved candidate_summary.json -> {output_dir / 'candidate_summary.json'}")


if __name__ == "__main__":
    main()
