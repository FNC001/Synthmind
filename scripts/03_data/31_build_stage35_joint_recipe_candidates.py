#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
31_build_stage35_joint_recipe_candidates.py

把 stage2 的 precursor top-k 候选，与 stage3 的 condition top-m 候选，
按 sample_id + precursor_rank 对齐，拼成完整的 joint recipe candidate 池。

推荐目录：
- 脚本:
  /Users/wyc/MP_exp_doi/scripts/03_data/31_build_stage35_joint_recipe_candidates.py
- 输出:
  /Users/wyc/MP_exp_doi/data/interim/generative/stage35_joint_recipe_candidates/<view_name>/

推荐输入：
1) base table（可选，但强烈建议提供）
2) stage2 candidates dir
3) stage3 candidates dir

输出文件：
- train_candidates.jsonl / val_candidates.jsonl / test_candidates.jsonl
- candidate_schema.json
- candidate_summary.json
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
# 基础工具
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


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_builtin(row), ensure_ascii=False) + "\n")


def maybe_read_csv_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl(path)
    if suffix == ".csv":
        df = pd.read_csv(path)
        return df.to_dict(orient="records")
    raise ValueError(f"暂不支持的文件格式: {path}")


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


def parse_precursor_set(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, tuple):
        return [str(x) for x in value]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                arr = json.loads(s)
                if isinstance(arr, list):
                    return [str(x) for x in arr]
            except Exception:
                pass
        if "||" in s:
            return [x.strip() for x in s.split("||") if x.strip()]
        if ";" in s:
            return [x.strip() for x in s.split(";") if x.strip()]
        if "|" in s:
            return [x.strip() for x in s.split("|") if x.strip()]
        return [s]
    return [str(value)]


def safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def candidate_recipe_id(sample_id: str, precursor_rank: int, condition_rank: int) -> str:
    return f"{sample_id}__p{int(precursor_rank)}__c{int(condition_rank)}"


# ---------------------------------------------------------------------
# 文件解析
# ---------------------------------------------------------------------
def resolve_candidate_file(directory: Path, split: str, preferred_prefix: str) -> Path:
    split = normalize_split_name(split)
    candidates = [
        directory / f"{split}_candidates.jsonl",
        directory / f"{split}_candidates.csv",
        directory / f"{split}_predictions.jsonl",
        directory / f"{split}_predictions.csv",
        directory / f"{preferred_prefix}_{split}.jsonl",
        directory / f"{preferred_prefix}_{split}.csv",
        directory / f"{split}.jsonl",
        directory / f"{split}.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"在 {directory} 下找不到 split={split} 的候选文件。尝试过:\n" +
        "\n".join(str(x) for x in candidates)
    )


def resolve_base_table(path_or_dir: Optional[str], split: str) -> Optional[Path]:
    if not path_or_dir or not str(path_or_dir).strip():
        return None
    p = Path(path_or_dir).expanduser().resolve()
    split = normalize_split_name(split)

    if p.is_file():
        return p

    candidates = [
        p / f"{split}.jsonl",
        p / f"{split}.csv",
        p / f"{split}_base.jsonl",
        p / f"{split}_base.csv",
        p / f"stage35_{split}_base.jsonl",
        p / f"stage35_{split}_base.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"未找到 split={split} 的 base table。尝试过:\n" +
        "\n".join(str(x) for x in candidates)
    )


# ---------------------------------------------------------------------
# 记录标准化
# ---------------------------------------------------------------------
def normalize_stage2_row(row: Dict[str, Any], default_model_name: str) -> Dict[str, Any]:
    sample_id = str(
        row.get("sample_id")
        or row.get("id")
        or row.get("row_id")
        or row.get("material_id")
        or ""
    ).strip()
    if not sample_id:
        raise ValueError(f"stage2 candidate 缺少 sample_id: {row}")

    precursor_rank = row.get("precursor_rank", row.get("candidate_rank", row.get("rank", 0)))
    try:
        precursor_rank = int(precursor_rank)
    except Exception as e:
        raise ValueError(f"stage2 candidate 的 precursor rank 无法解析: {row}") from e

    precursor_set = parse_precursor_set(
        row.get("precursor_set", row.get("pred_precursors", row.get("precursors")))
    )

    precursor_multihot = row.get("precursor_multihot")
    if isinstance(precursor_multihot, str):
        try:
            precursor_multihot = json.loads(precursor_multihot)
        except Exception:
            precursor_multihot = None

    return {
        "sample_id": sample_id,
        "precursor_rank": precursor_rank,
        "precursor_set": precursor_set,
        "precursor_multihot": precursor_multihot,
        "n_precursors": int(len(precursor_set)),
        "stage2_score": safe_float(
            row.get("stage2_score", row.get("score", row.get("candidate_score", row.get("prob"))))
        ),
        "stage2_model": str(row.get("stage2_model", row.get("model_name", default_model_name))),
        "_raw": row,
    }


def normalize_stage3_row(row: Dict[str, Any], default_model_name: str) -> Dict[str, Any]:
    sample_id = str(
        row.get("sample_id")
        or row.get("id")
        or row.get("row_id")
        or row.get("material_id")
        or ""
    ).strip()
    if not sample_id:
        raise ValueError(f"stage3 candidate 缺少 sample_id: {row}")

    parent_precursor_rank = row.get(
        "parent_precursor_rank",
        row.get("precursor_rank", row.get("candidate_rank", row.get("parent_rank", 0)))
    )
    condition_rank = row.get("condition_rank", row.get("rank", row.get("candidate_rank", 0)))

    try:
        parent_precursor_rank = int(parent_precursor_rank)
        condition_rank = int(condition_rank)
    except Exception as e:
        raise ValueError(f"stage3 candidate rank 无法解析: {row}") from e

    disc_conditions = row.get("disc_conditions")
    if disc_conditions is None:
        disc_conditions = {}
        if "pred_discrete" in row and isinstance(row["pred_discrete"], dict):
            disc_conditions = row["pred_discrete"]
        else:
            for key in ("atmosphere", "solvent", "synthesis_type", "condition_source"):
                if key in row:
                    disc_conditions[key] = row[key]
    if isinstance(disc_conditions, str):
        try:
            disc_conditions = json.loads(disc_conditions)
        except Exception:
            disc_conditions = {}

    cont_conditions = row.get("cont_conditions")
    if cont_conditions is None:
        cont_conditions = {}
        if "pred_continuous" in row and isinstance(row["pred_continuous"], dict):
            cont_conditions = row["pred_continuous"]
        else:
            for key in ("temperature_c", "time_h", "n_main_precursors", "n_aux_precursors", "n_heatlike_ops"):
                if key in row:
                    cont_conditions[key] = row[key]
    if isinstance(cont_conditions, str):
        try:
            cont_conditions = json.loads(cont_conditions)
        except Exception:
            cont_conditions = {}

    return {
        "sample_id": sample_id,
        "parent_precursor_rank": parent_precursor_rank,
        "condition_rank": condition_rank,
        "disc_conditions": dict(disc_conditions),
        "cont_conditions": dict(cont_conditions),
        "stage3_score": safe_float(row.get("stage3_score", row.get("score", row.get("candidate_score", row.get("prob"))))),
        "stage3_model": str(row.get("stage3_model", row.get("model_name", default_model_name))),
        "_raw": row,
    }


def build_base_index(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        sample_id = str(
            row.get("sample_id")
            or row.get("id")
            or row.get("row_id")
            or row.get("material_id")
            or ""
        ).strip()
        if not sample_id:
            continue
        idx[sample_id] = row
    return idx


# ---------------------------------------------------------------------
# 主拼接逻辑
# ---------------------------------------------------------------------
def combine_candidates(
    split: str,
    base_rows: Optional[Sequence[Dict[str, Any]]],
    stage2_rows: Sequence[Dict[str, Any]],
    stage3_rows: Sequence[Dict[str, Any]],
    topk_stage2: int,
    topm_stage3: int,
    candidate_source: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    split = normalize_split_name(split)

    base_index = build_base_index(base_rows or [])
    stage2_norm = [normalize_stage2_row(r, default_model_name="stage2_unknown") for r in stage2_rows]
    stage3_norm = [normalize_stage3_row(r, default_model_name="stage3_unknown") for r in stage3_rows]

    stage2_by_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in stage2_norm:
        stage2_by_sample[row["sample_id"]].append(row)
    for sample_id in stage2_by_sample:
        stage2_by_sample[sample_id].sort(key=lambda x: x["precursor_rank"])

    stage3_by_pair: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in stage3_norm:
        stage3_by_pair[(row["sample_id"], row["parent_precursor_rank"])] .append(row)
    for k in stage3_by_pair:
        stage3_by_pair[k].sort(key=lambda x: x["condition_rank"])

    output_rows: List[Dict[str, Any]] = []
    n_samples = 0
    n_with_stage3 = 0
    missing_stage3_pairs = 0

    for sample_id, s2_list in sorted(stage2_by_sample.items(), key=lambda kv: kv[0]):
        n_samples += 1
        s2_keep = s2_list[:topk_stage2]

        has_any_stage3 = False
        for s2 in s2_keep:
            key = (sample_id, s2["precursor_rank"])
            s3_list = stage3_by_pair.get(key, [])
            if not s3_list:
                missing_stage3_pairs += 1
                continue

            has_any_stage3 = True
            s3_keep = s3_list[:topm_stage3]

            base = base_index.get(sample_id, {})
            for s3 in s3_keep:
                recipe_id = candidate_recipe_id(sample_id, s2["precursor_rank"], s3["condition_rank"])
                row = {
                    "sample_id": sample_id,
                    "material_id": base.get("material_id", base.get("entry_id", base.get("id"))),
                    "target_formula": base.get("target_formula", base.get("formula")),
                    "split": split,
                    "recipe_id": recipe_id,
                    "precursor_rank": int(s2["precursor_rank"]),
                    "condition_rank": int(s3["condition_rank"]),
                    "joint_rank_seed": int(s3["condition_rank"]),
                    "precursor_set": s2["precursor_set"],
                    "precursor_multihot": s2["precursor_multihot"],
                    "n_precursors": int(s2["n_precursors"]),
                    "disc_conditions": s3["disc_conditions"],
                    "cont_conditions": s3["cont_conditions"],
                    "stage2_score": s2["stage2_score"],
                    "stage3_score": s3["stage3_score"],
                    "joint_prior_score": None,
                    "stage2_model": s2["stage2_model"],
                    "stage3_model": s3["stage3_model"],
                    "structure_text": base.get("structure_text"),
                    "source": base.get("source"),
                    "reaction_id": base.get("reaction_id"),
                    "entry_id": base.get("entry_id"),
                    "synth_uid": base.get("synth_uid"),
                    "is_observed_recipe": base.get("is_observed_recipe", -1),
                    "candidate_source": candidate_source,
                }
                output_rows.append(row)

        if has_any_stage3:
            n_with_stage3 += 1

    summary = {
        "split": split,
        "n_samples": int(n_samples),
        "n_stage2_candidates_input": int(len(stage2_norm)),
        "n_stage3_candidates_input": int(len(stage3_norm)),
        "n_joint_recipe_candidates": int(len(output_rows)),
        "n_samples_with_stage3_match": int(n_with_stage3),
        "missing_stage3_pairs": int(missing_stage3_pairs),
        "topk_stage2": int(topk_stage2),
        "topm_stage3": int(topm_stage3),
        "candidate_source": candidate_source,
    }
    return output_rows, summary


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build stage35 joint recipe candidates from stage2 + stage3 outputs.")
    p.add_argument("--base_table", type=str, default="", help="base table 文件或目录；可选")
    p.add_argument("--stage2_candidates_dir", type=str, required=True)
    p.add_argument("--stage3_candidates_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--topk_stage2", type=int, default=16)
    p.add_argument("--topm_stage3", type=int, default=4)
    p.add_argument("--splits", type=str, default="train,val,test", help="逗号分隔，如 train,val,test")
    p.add_argument("--view_name", type=str, default="hybrid_core")
    p.add_argument("--stage2_prefix", type=str, default="stage2_candidates")
    p.add_argument("--stage3_prefix", type=str, default="stage3_candidates")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    stage2_dir = Path(args.stage2_candidates_dir).expanduser().resolve()
    stage3_dir = Path(args.stage3_candidates_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    splits = normalize_split_list([x.strip() for x in str(args.splits).split(",") if x.strip()])
    overall_summary: Dict[str, Any] = {
        "view_name": str(args.view_name),
        "topk_stage2": int(args.topk_stage2),
        "topm_stage3": int(args.topm_stage3),
        "splits": splits,
        "split_stats": {},
    }

    all_disc_keys = set()
    all_cont_keys = set()

    for split in splits:
        split = normalize_split_name(split)

        base_path = resolve_base_table(args.base_table, split)
        base_rows = maybe_read_csv_or_jsonl(base_path) if base_path is not None else []

        stage2_path = resolve_candidate_file(stage2_dir, split, preferred_prefix=args.stage2_prefix)
        stage3_path = resolve_candidate_file(stage3_dir, split, preferred_prefix=args.stage3_prefix)

        stage2_rows = maybe_read_csv_or_jsonl(stage2_path)
        stage3_rows = maybe_read_csv_or_jsonl(stage3_path)

        joint_rows, split_summary = combine_candidates(
            split=split,
            base_rows=base_rows,
            stage2_rows=stage2_rows,
            stage3_rows=stage3_rows,
            topk_stage2=args.topk_stage2,
            topm_stage3=args.topm_stage3,
            candidate_source="stage2_topk__stage3_topm",
        )

        out_path = output_dir / f"{split}_candidates.jsonl"
        write_jsonl(out_path, joint_rows)

        for row in joint_rows:
            all_disc_keys.update(row.get("disc_conditions", {}).keys())
            all_cont_keys.update(row.get("cont_conditions", {}).keys())

        overall_summary["split_stats"][split] = {
            **split_summary,
            "base_table_path": str(base_path) if base_path else None,
            "stage2_candidates_path": str(stage2_path),
            "stage3_candidates_path": str(stage3_path),
            "output_candidates_path": str(out_path),
        }

        print(
            f"[{split}] "
            f"base={len(base_rows)} "
            f"stage2={len(stage2_rows)} "
            f"stage3={len(stage3_rows)} "
            f"joint={len(joint_rows)}"
        )

    candidate_schema = {
        "view_name": str(args.view_name),
        "topk_stage2": int(args.topk_stage2),
        "topm_stage3": int(args.topm_stage3),
        "discrete_condition_keys": sorted(all_disc_keys),
        "continuous_condition_keys": sorted(all_cont_keys),
        "stage2_fields": ["sample_id", "precursor_rank", "precursor_set", "precursor_multihot", "stage2_score"],
        "stage3_fields": ["sample_id", "parent_precursor_rank", "condition_rank", "disc_conditions", "cont_conditions", "stage3_score"],
        "recipe_id_format": "{sample_id}__p{precursor_rank}__c{condition_rank}",
    }

    write_json(output_dir / "candidate_schema.json", candidate_schema)
    write_json(output_dir / "candidate_summary.json", overall_summary)

    print(f"Saved candidate_schema.json -> {output_dir / 'candidate_schema.json'}")
    print(f"Saved candidate_summary.json -> {output_dir / 'candidate_summary.json'}")


if __name__ == "__main__":
    main()
