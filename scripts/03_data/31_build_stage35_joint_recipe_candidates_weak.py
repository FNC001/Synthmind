#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def resolve_candidate_file(directory: Path, split: str) -> Path:
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
        f"在 {directory} 下找不到 split={split} 的文件。尝试过:\n" +
        "\n".join(str(x) for x in candidates)
    )


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return pd.DataFrame(rows)
    raise ValueError(f"不支持的文件格式: {path}")


_MP_RE = re.compile(r"(mp-\d+)")


def extract_mp_id(value: Any) -> str:
    s = str(value) if value is not None else ""
    m = _MP_RE.search(s)
    if m:
        return m.group(1)
    return s.strip()


def stage2_material_id_from_row(row: pd.Series) -> str:
    if "material_id" in row and pd.notna(row["material_id"]):
        return extract_mp_id(row["material_id"])
    if "id" in row and pd.notna(row["id"]):
        return extract_mp_id(row["id"])
    if "entry_id" in row and pd.notna(row["entry_id"]):
        return extract_mp_id(row["entry_id"])
    return ""


def stage3_material_id_from_row(row: pd.Series) -> str:
    for c in ["sample_id", "material_id", "entry_id", "id"]:
        if c in row and pd.notna(row[c]):
            return extract_mp_id(row[c])
    return ""


def recipe_id(material_id: str, precursor_rank: int, condition_rank: int) -> str:
    return f"{material_id}__p{int(precursor_rank)}__c{int(condition_rank)}"


def normalize_stage2_setpred(df: pd.DataFrame) -> pd.DataFrame:
    if "candidate_label" not in df.columns:
        raise ValueError("stage2 文件缺少 candidate_label，当前弱联合版预期是单 precursor label 排序结果。")
    out = df.copy()
    out["material_id_norm"] = out.apply(stage2_material_id_from_row, axis=1)
    out["precursor_rank"] = out["rank"].astype(int)
    out["precursor_label"] = out["candidate_label"].astype(str)
    out["stage2_score"] = out["score"].apply(safe_float)
    return out


def build_disc_conditions_from_stage3_row(row: pd.Series) -> Dict[str, Any]:
    disc = {}
    for k in ["pred_atmosphere", "pred_solvent", "pred_synthesis_type", "pred_condition_source"]:
        if k in row and pd.notna(row[k]):
            disc[k.replace("pred_", "")] = row[k]
    return disc


def build_cont_conditions_from_stage3_row(row: pd.Series) -> Dict[str, Any]:
    cont = {}
    mapping = {
        "pred_temperature_c": "temperature_c",
        "pred_time_h": "time_h",
        "pred_n_main_precursors": "n_main_precursors",
        "pred_n_aux_precursors": "n_aux_precursors",
        "pred_n_heatlike_ops": "n_heatlike_ops",
    }
    for src, dst in mapping.items():
        if src in row and pd.notna(row[src]):
            cont[dst] = safe_float(row[src])
    return cont


def normalize_stage3_candidates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["material_id_norm"] = out.apply(stage3_material_id_from_row, axis=1)
    if "candidate_rank" not in out.columns:
        raise ValueError("stage3 文件缺少 candidate_rank。")
    out["condition_rank"] = out["candidate_rank"].astype(int)
    if "generator_score" in out.columns:
        out["stage3_score"] = out["generator_score"].apply(safe_float)
    else:
        out["stage3_score"] = None
    return out


def combine_weak(
    split: str,
    stage2_df: pd.DataFrame,
    stage3_df: pd.DataFrame,
    topk_stage2: int,
    topm_stage3: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    split = normalize_split_name(split)

    s2 = normalize_stage2_setpred(stage2_df)
    s3 = normalize_stage3_candidates(stage3_df)

    s2 = s2[s2["material_id_norm"].astype(str) != ""].copy()
    s3 = s3[s3["material_id_norm"].astype(str) != ""].copy()

    s2_groups = {str(mid): g.sort_values("precursor_rank").head(int(topk_stage2)) for mid, g in s2.groupby("material_id_norm", sort=True)}
    s3_groups = {str(mid): g.sort_values("condition_rank").head(int(topm_stage3)) for mid, g in s3.groupby("material_id_norm", sort=True)}

    mids2 = set(s2_groups.keys())
    mids3 = set(s3_groups.keys())
    overlap = sorted(mids2 & mids3)

    rows: List[Dict[str, Any]] = []
    for mid in overlap:
        g2 = s2_groups[mid]
        g3 = s3_groups[mid]
        meta2 = g2.iloc[0].to_dict()
        meta3 = g3.iloc[0].to_dict()

        for _, r2 in g2.iterrows():
            precursor_rank = int(r2["precursor_rank"])
            precursor_label = str(r2["precursor_label"])
            stage2_score = safe_float(r2["stage2_score"])

            for _, r3 in g3.iterrows():
                condition_rank = int(r3["condition_rank"])
                stage3_score = safe_float(r3["stage3_score"])
                disc_conditions = build_disc_conditions_from_stage3_row(r3)
                cont_conditions = build_cont_conditions_from_stage3_row(r3)

                rows.append({
                    "sample_id": mid,
                    "material_id": mid,
                    "target_formula": meta2.get("formula"),
                    "split": split,
                    "recipe_id": recipe_id(mid, precursor_rank, condition_rank),
                    "precursor_rank": precursor_rank,
                    "condition_rank": condition_rank,
                    "joint_rank_seed": condition_rank,
                    "precursor_label": precursor_label,
                    "precursor_set": [precursor_label],
                    "precursor_multihot": None,
                    "n_precursors": 1,
                    "disc_conditions": disc_conditions,
                    "cont_conditions": cont_conditions,
                    "stage2_score": stage2_score,
                    "stage3_score": stage3_score,
                    "joint_prior_score": None,
                    "stage2_model": str(meta2.get("source_model", "stage2_unknown")),
                    "stage3_model": "stage3_candidates",
                    "source_dataset": meta2.get("source_dataset"),
                    "synthesis_type": meta2.get("synthesis_type"),
                    "doi": meta2.get("doi"),
                    "split_group": meta2.get("split_group"),
                    "stage3_sample_index": meta3.get("sample_index"),
                    "stage3_candidate_tag": r3.get("candidate_tag"),
                    "is_observed_recipe": -1,
                    "candidate_source": "weak_joint__stage2_label_topk__stage3_cond_topm",
                })

    summary = {
        "split": split,
        "n_stage2_rows": int(len(s2)),
        "n_stage3_rows": int(len(s3)),
        "n_stage2_materials": int(len(mids2)),
        "n_stage3_materials": int(len(mids3)),
        "n_overlap_materials": int(len(overlap)),
        "n_joint_recipe_candidates": int(len(rows)),
        "topk_stage2": int(topk_stage2),
        "topm_stage3": int(topm_stage3),
        "mode": "weak_joint",
    }
    return rows, summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build weak stage35 joint recipe candidates from stage2 label-ranking + stage3 condition candidates.")
    p.add_argument("--stage2_candidates_dir", type=str, required=True)
    p.add_argument("--stage3_candidates_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--topk_stage2", type=int, default=16)
    p.add_argument("--topm_stage3", type=int, default=4)
    p.add_argument("--splits", type=str, default="val,test")
    p.add_argument("--view_name", type=str, default="hybrid_core_weak")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    stage2_dir = Path(args.stage2_candidates_dir).expanduser().resolve()
    stage3_dir = Path(args.stage3_candidates_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    splits = normalize_split_list([x.strip() for x in str(args.splits).split(",") if x.strip()])

    schema = {
        "view_name": str(args.view_name),
        "mode": "weak_joint",
        "topk_stage2": int(args.topk_stage2),
        "topm_stage3": int(args.topm_stage3),
        "stage2_semantics": "single precursor label ranking",
        "stage3_semantics": "condition candidate ranking",
        "recipe_id_format": "{material_id}__p{precursor_rank}__c{condition_rank}",
        "discrete_condition_keys": [],
        "continuous_condition_keys": ["temperature_c", "time_h"],
    }

    summary = {
        "view_name": str(args.view_name),
        "mode": "weak_joint",
        "splits": splits,
        "split_stats": {},
    }

    for split in splits:
        split = normalize_split_name(split)
        s2_path = resolve_candidate_file(stage2_dir, split)
        s3_path = resolve_candidate_file(stage3_dir, split)

        s2_df = read_table(s2_path)
        s3_df = read_table(s3_path)

        out_rows, split_summary = combine_weak(
            split=split,
            stage2_df=s2_df,
            stage3_df=s3_df,
            topk_stage2=args.topk_stage2,
            topm_stage3=args.topm_stage3,
        )

        out_path = output_dir / f"{split}_candidates.jsonl"
        write_jsonl(out_path, out_rows)

        summary["split_stats"][split] = {
            **split_summary,
            "stage2_candidates_path": str(s2_path),
            "stage3_candidates_path": str(s3_path),
            "output_candidates_path": str(out_path),
        }

        print(
            f"[{split}] "
            f"stage2={len(s2_df)} "
            f"stage3={len(s3_df)} "
            f"overlap_materials={split_summary['n_overlap_materials']} "
            f"joint={len(out_rows)}"
        )

    write_json(output_dir / "candidate_schema.json", schema)
    write_json(output_dir / "candidate_summary.json", summary)

    print(f"Saved candidate_schema.json -> {output_dir / 'candidate_schema.json'}")
    print(f"Saved candidate_summary.json -> {output_dir / 'candidate_summary.json'}")


if __name__ == "__main__":
    main()
