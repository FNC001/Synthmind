#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
41_build_stage35_strong_joint_dataset.py

从 strong joint recipe-level candidates 构建可训练的 strong joint dataset。

输入:
1) strong joint candidates:
   data/interim/generative/stage35_strong_joint_candidates/<view_name>/{split}_candidates.jsonl
2) stage2 原始 source（用于恢复 true precursor set）:
   runs/stage2/<model_run>/{split}_candidates.csv
3) stage3 原始 source（用于恢复 true continuous conditions）:
   runs/stage3/<source_dir>/{split}_candidates.csv

输出:
- {split}.jsonl
- schema.json
- summary.json

标签定义（strong joint 版）
-------------------------
- true_precursor_labels: 从 stage2 原始 source 的 true_labels 恢复
- precursor_exact_match: candidate precursor_set 与 true_precursor_labels 的集合是否完全一致
- precursor_overlap_count
- precursor_jaccard
- temp_abs_err / time_abs_err
- cont_match
- joint_label: precursor_exact_match AND cont_match
- joint_soft_score: precursor_jaccard 与 cont_score 的加权组合

推荐放置:
  /Users/wyc/MP_exp_doi/scripts/03_data/41_build_stage35_strong_joint_dataset.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
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


def resolve_file(directory: Path, split: str, preferred_names: Sequence[str]) -> Path:
    split = normalize_split_name(split)
    tried: List[Path] = []
    for name in preferred_names:
        p = directory / name.format(split=split)
        tried.append(p)
        if p.exists():
            return p
    raise FileNotFoundError(
        f"在 {directory} 下找不到 split={split} 的文件。尝试过:\n" +
        "\n".join(str(x) for x in tried)
    )


def resolve_joint_candidates_file(directory: Path, split: str) -> Path:
    return resolve_file(directory, split, [
        "{split}_candidates.jsonl",
        "{split}_candidates.csv",
        "{split}.jsonl",
        "{split}.csv",
    ])


def resolve_stage2_source_file(directory: Path, split: str) -> Path:
    return resolve_file(directory, split, [
        "{split}_candidates.csv",
        "{split}_candidates.jsonl",
        "pred_{split}.csv",
        "{split}.csv",
        "{split}.jsonl",
    ])


def resolve_stage3_source_file(directory: Path, split: str) -> Path:
    return resolve_file(directory, split, [
        "{split}_candidates.csv",
        "{split}_candidates.jsonl",
        "{split}_predictions.csv",
        "{split}_predictions.jsonl",
        "pred_{split}.csv",
        "{split}.csv",
        "{split}.jsonl",
    ])


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


def parse_true_labels(value: Any) -> List[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return [strip_label_prefix(str(x)) for x in value]
    s = str(value).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return [strip_label_prefix(str(x)) for x in arr]
        except Exception:
            pass
    # fallback
    for sep in ["||", ";", "|"]:
        if sep in s:
            return [strip_label_prefix(x.strip()) for x in s.split(sep) if x.strip()]
    return [strip_label_prefix(s)]


# ---------------------------------------------------------------------
# truth maps
# ---------------------------------------------------------------------
def build_stage2_truth_map(stage2_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    从 stage2 原始 source 里恢复 true precursor set。
    优先按 material_id 聚合；同一 material 若多行 true_labels 不一致，保留第一次并记录 warning 风险由 summary 体现。
    """
    out: Dict[str, Dict[str, Any]] = {}
    for _, row in stage2_df.iterrows():
        material_id = ""
        if "material_id" in row and pd.notna(row["material_id"]):
            material_id = extract_mp_id(row["material_id"])
        elif "id" in row and pd.notna(row["id"]):
            material_id = extract_mp_id(row["id"])
        if not material_id:
            continue

        true_labels = parse_true_labels(row.get("true_labels"))
        if material_id not in out:
            out[material_id] = {
                "true_precursor_labels": true_labels,
                "formula": row.get("formula"),
                "doi": row.get("doi"),
                "source_dataset": row.get("source_dataset"),
                "synthesis_type": row.get("synthesis_type"),
            }
    return out


def build_stage3_truth_map(stage3_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for _, row in stage3_df.iterrows():
        material_id = ""
        for c in ["sample_id", "material_id", "entry_id", "id"]:
            if c in row and pd.notna(row[c]):
                material_id = extract_mp_id(row[c])
                if material_id:
                    break
        if not material_id:
            continue

        if material_id not in out:
            out[material_id] = {
                "true_temperature_c": safe_float(row.get("true_temperature_c")),
                "mask_temperature_c": safe_float(row.get("mask_temperature_c")),
                "true_time_h": safe_float(row.get("true_time_h")),
                "mask_time_h": safe_float(row.get("mask_time_h")),
            }
    return out


# ---------------------------------------------------------------------
# metrics / labels
# ---------------------------------------------------------------------
def precursor_set_metrics(pred_set: Sequence[str], true_set: Sequence[str]) -> Dict[str, Any]:
    pred = set(str(x) for x in (pred_set or []))
    true = set(str(x) for x in (true_set or []))

    if not true and not pred:
        return {
            "precursor_exact_match": True,
            "precursor_overlap_count": 0,
            "precursor_precision": 1.0,
            "precursor_recall": 1.0,
            "precursor_jaccard": 1.0,
        }

    inter = pred & true
    union = pred | true

    precision = float(len(inter) / len(pred)) if len(pred) > 0 else 0.0
    recall = float(len(inter) / len(true)) if len(true) > 0 else 0.0
    jaccard = float(len(inter) / len(union)) if len(union) > 0 else 0.0

    return {
        "precursor_exact_match": bool(pred == true),
        "precursor_overlap_count": int(len(inter)),
        "precursor_precision": precision,
        "precursor_recall": recall,
        "precursor_jaccard": jaccard,
    }


def capped_score(abs_err: Optional[float], tol: float) -> Optional[float]:
    if abs_err is None or tol <= 0:
        return None
    return max(0.0, 1.0 - float(abs_err) / float(tol))


def compute_cont_metrics(
    cont_pred: Dict[str, Any],
    truth: Dict[str, Any],
    temperature_tol: float,
    time_tol: float,
) -> Dict[str, Any]:
    pred_temp = safe_float(cont_pred.get("temperature_c"))
    pred_time = safe_float(cont_pred.get("time_h"))

    true_temp = safe_float(truth.get("true_temperature_c"))
    true_time = safe_float(truth.get("true_time_h"))
    mask_temp = safe_float(truth.get("mask_temperature_c"))
    mask_time = safe_float(truth.get("mask_time_h"))

    # FIX:
    # 对 strong joint 来说，只要真值存在，就强制参与误差与匹配计算；
    # 不再让 mask_temperature_c / mask_time_h 决定是否跳过该目标。
    use_temp = (pred_temp is not None) and (true_temp is not None)
    use_time = (pred_time is not None) and (true_time is not None)

    temp_abs_err = abs(pred_temp - true_temp) if use_temp else None
    time_abs_err = abs(pred_time - true_time) if use_time else None

    temp_match = None if temp_abs_err is None else bool(temp_abs_err <= temperature_tol)
    time_match = None if time_abs_err is None else bool(time_abs_err <= time_tol)

    temp_score = capped_score(temp_abs_err, temperature_tol)
    time_score = capped_score(time_abs_err, time_tol)

    score_parts = [x for x in [temp_score, time_score] if x is not None]
    cont_score = float(np.mean(score_parts)) if score_parts else None

    cont_match_parts = [x for x in [temp_match, time_match] if x is not None]
    cont_match = bool(all(cont_match_parts)) if cont_match_parts else None

    return {
        "true_temperature_c": true_temp,
        "true_time_h": true_time,
        "mask_temperature_c": mask_temp,
        "mask_time_h": mask_time,
        "temp_abs_err": temp_abs_err,
        "time_abs_err": time_abs_err,
        "temp_match": temp_match,
        "time_match": time_match,
        "cont_score": cont_score,
        "cont_match": cont_match,
    }

def build_joint_labels(
    precursor_exact_match: Optional[bool],
    cont_match: Optional[bool],
    precursor_weight: float,
    condition_weight: float,
    precursor_jaccard: Optional[float],
    cont_score: Optional[float],
) -> Dict[str, Any]:
    if precursor_exact_match is None or cont_match is None:
        joint_label = -1
    else:
        joint_label = int(bool(precursor_exact_match and cont_match))

    precursor_part = precursor_jaccard
    condition_part = cont_score

    if precursor_part is None and condition_part is None:
        joint_soft_score = None
    elif precursor_part is None:
        joint_soft_score = float(condition_part)
    elif condition_part is None:
        joint_soft_score = float(precursor_part)
    else:
        wsum = float(precursor_weight + condition_weight)
        if wsum <= 0:
            joint_soft_score = 0.5 * float(precursor_part) + 0.5 * float(condition_part)
        else:
            joint_soft_score = (
                float(precursor_weight) * float(precursor_part) +
                float(condition_weight) * float(condition_part)
            ) / wsum

    return {
        "joint_label": joint_label,
        "joint_soft_score": joint_soft_score,
    }


# ---------------------------------------------------------------------
# main build
# ---------------------------------------------------------------------
def build_dataset_rows(
    split: str,
    joint_candidates: Sequence[Dict[str, Any]],
    stage2_truth_map: Dict[str, Dict[str, Any]],
    stage3_truth_map: Dict[str, Dict[str, Any]],
    temperature_tol: float,
    time_tol: float,
    precursor_weight: float,
    condition_weight: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    split = normalize_split_name(split)
    rows: List[Dict[str, Any]] = []

    n_with_stage2_truth = 0
    n_with_stage3_truth = 0
    n_precursor_exact = 0
    n_cont_match = 0
    n_joint_positive = 0

    for cand in joint_candidates:
        material_id = extract_mp_id(cand.get("material_key", cand.get("material_id", cand.get("sample_id"))))
        stage2_truth = stage2_truth_map.get(material_id, {})
        stage3_truth = stage3_truth_map.get(material_id, {})

        true_precursor_labels = stage2_truth.get("true_precursor_labels", [])
        pred_precursor_set = [strip_label_prefix(str(x)) for x in (cand.get("precursor_set", []) or [])]

        precursor_metrics = precursor_set_metrics(pred_precursor_set, true_precursor_labels)
        if true_precursor_labels is not None:
            n_with_stage2_truth += 1
            if precursor_metrics["precursor_exact_match"]:
                n_precursor_exact += 1

        cont_metrics = compute_cont_metrics(
            cont_pred=cand.get("cont_conditions", {}),
            truth=stage3_truth,
            temperature_tol=temperature_tol,
            time_tol=time_tol,
        )
        if cont_metrics["cont_match"] is not None:
            n_with_stage3_truth += 1
            if cont_metrics["cont_match"]:
                n_cont_match += 1

        label_info = build_joint_labels(
            precursor_exact_match=precursor_metrics["precursor_exact_match"],
            cont_match=cont_metrics["cont_match"],
            precursor_weight=precursor_weight,
            condition_weight=condition_weight,
            precursor_jaccard=precursor_metrics["precursor_jaccard"],
            cont_score=cont_metrics["cont_score"],
        )
        if label_info["joint_label"] == 1:
            n_joint_positive += 1

        row = {
            **cand,
            "group_id": f"{material_id}__p{int(cand.get('precursor_rank', 0))}",
            "true_precursor_labels": true_precursor_labels,
            **precursor_metrics,
            **cont_metrics,
            **label_info,
        }
        rows.append(row)

    summary = {
        "split": split,
        "n_rows": int(len(rows)),
        "n_with_stage2_truth": int(n_with_stage2_truth),
        "n_with_stage3_truth": int(n_with_stage3_truth),
        "n_precursor_exact": int(n_precursor_exact),
        "n_cont_match": int(n_cont_match),
        "n_joint_positive": int(n_joint_positive),
        "temperature_tol": float(temperature_tol),
        "time_tol": float(time_tol),
    }
    return rows, summary


# ---------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build labeled strong joint dataset from strong joint candidates.")
    p.add_argument("--joint_candidates_dir", type=str, required=True)
    p.add_argument("--stage2_source_dir", type=str, required=True)
    p.add_argument("--stage3_source_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--splits", type=str, default="val,test")
    p.add_argument("--view_name", type=str, default="hybrid_core_strong")

    p.add_argument("--temperature_tol", type=float, default=150.0)
    p.add_argument("--time_tol", type=float, default=24.0)
    p.add_argument("--precursor_weight", type=float, default=1.0)
    p.add_argument("--condition_weight", type=float, default=1.0)
    return p


def main() -> None:
    args = build_argparser().parse_args()

    joint_candidates_dir = Path(args.joint_candidates_dir).expanduser().resolve()
    stage2_source_dir = Path(args.stage2_source_dir).expanduser().resolve()
    stage3_source_dir = Path(args.stage3_source_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    splits = normalize_split_list([x.strip() for x in str(args.splits).split(",") if x.strip()])

    overall_summary: Dict[str, Any] = {
        "view_name": str(args.view_name),
        "mode": "strong_joint_labeled",
        "splits": splits,
        "temperature_tol": float(args.temperature_tol),
        "time_tol": float(args.time_tol),
        "precursor_weight": float(args.precursor_weight),
        "condition_weight": float(args.condition_weight),
        "split_stats": {},
    }

    for split in splits:
        split = normalize_split_name(split)

        joint_path = resolve_joint_candidates_file(joint_candidates_dir, split)
        stage2_path = resolve_stage2_source_file(stage2_source_dir, split)
        stage3_path = resolve_stage3_source_file(stage3_source_dir, split)

        joint_candidates = read_jsonl(joint_path) if joint_path.suffix.lower() == ".jsonl" else read_table(joint_path).to_dict(orient="records")
        stage2_df = read_table(stage2_path)
        stage3_df = read_table(stage3_path)

        stage2_truth_map = build_stage2_truth_map(stage2_df)
        stage3_truth_map = build_stage3_truth_map(stage3_df)

        dataset_rows, split_summary = build_dataset_rows(
            split=split,
            joint_candidates=joint_candidates,
            stage2_truth_map=stage2_truth_map,
            stage3_truth_map=stage3_truth_map,
            temperature_tol=float(args.temperature_tol),
            time_tol=float(args.time_tol),
            precursor_weight=float(args.precursor_weight),
            condition_weight=float(args.condition_weight),
        )

        out_path = output_dir / f"{split}.jsonl"
        write_jsonl(out_path, dataset_rows)

        overall_summary["split_stats"][split] = {
            **split_summary,
            "joint_candidates_path": str(joint_path),
            "stage2_source_path": str(stage2_path),
            "stage3_source_path": str(stage3_path),
            "output_dataset_path": str(out_path),
        }

        print(
            f"[{split}] "
            f"joint={len(joint_candidates)} "
            f"dataset={len(dataset_rows)} "
            f"joint_positive={split_summary['n_joint_positive']}"
        )

    schema = {
        "view_name": str(args.view_name),
        "mode": "strong_joint_labeled",
        "key_fields": [
            "sample_id",
            "material_id",
            "material_key",
            "recipe_id",
            "group_id",
            "precursor_rank",
            "condition_rank",
            "precursor_set",
            "disc_conditions",
            "cont_conditions",
            "stage2_score",
            "stage3_score",
            "true_precursor_labels",
            "precursor_exact_match",
            "precursor_overlap_count",
            "precursor_precision",
            "precursor_recall",
            "precursor_jaccard",
            "true_temperature_c",
            "true_time_h",
            "temp_abs_err",
            "time_abs_err",
            "cont_match",
            "joint_label",
            "joint_soft_score",
        ],
        "label_semantics": {
            "precursor_exact_match": "1 if candidate precursor_set matches true precursor set exactly",
            "precursor_jaccard": "Jaccard overlap between candidate precursor_set and true precursor set",
            "cont_match": "1 if continuous conditions satisfy the tolerance thresholds",
            "joint_label": "1 if precursor_exact_match and cont_match are both true; -1 if unavailable",
            "joint_soft_score": "weighted combination of precursor_jaccard and continuous score",
        },
        "temperature_tol": float(args.temperature_tol),
        "time_tol": float(args.time_tol),
    }

    write_json(output_dir / "schema.json", schema)
    write_json(output_dir / "summary.json", overall_summary)

    print(f"Saved schema.json -> {output_dir / 'schema.json'}")
    print(f"Saved summary.json -> {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
