#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_build_hybrid_features_infer.py

用途
----
给 inference 场景使用的 hybrid feature 构建脚本。

它兼容两种模式：
1. 训练/原始命名模式：
   - descriptor_dir 下有 stage2_test_ml.csv / stage2_val_ml.csv / ...
   - embedding_dir 下有 stage2_test_graph_embed.csv / ...
2. inference 直连模式：
   - 直接传 --infer_descriptor_csv
   - 直接传 --infer_embedding_csv
   - 自动输出 stage2_test_hybrid.csv，并可选复制到 train/val

这版放在 07_infer 下，避免改坏原始训练脚本。

示例
----
python 05_build_hybrid_features_infer.py \
  --task stage2 \
  --infer_descriptor_csv data/interim/infer/demo1/infer_structdesc.csv \
  --infer_embedding_csv data/interim/infer/demo1/graph_embed/infer_graph_embed.csv \
  --embedding_prefix chgnet \
  --output_dir data/interim/infer/demo1/hybrid_stage2 \
  --replicate_to_train_val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


ID_CANDIDATES = ["sample_id", "id", "material_id"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    return pd.read_csv(path)


def resolve_id_col(df: pd.DataFrame, kind: str) -> str:
    for c in ID_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find id column in {kind} csv. Available columns={df.columns.tolist()}")


def dedup_by_id(df: pd.DataFrame, id_col: str) -> Tuple[pd.DataFrame, int]:
    dup = int(df.duplicated(subset=[id_col]).sum())
    out = df.drop_duplicates(subset=[id_col], keep="first").copy()
    return out, dup


def choose_feature_cols(df: pd.DataFrame, id_col: str, non_feature_cols: List[str]) -> List[str]:
    cols: List[str] = []
    banned = set(non_feature_cols + [id_col])
    for c in df.columns:
        if c in banned:
            continue
        cols.append(c)
    return cols


def normalize_embedding_prefix(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    把 embedding 列统一成 graph_emb_* 形式前缀。
    输入可能是:
      chgnet_graph_emb_0
      cgcnn_graph_emb_0
      graph_emb_0
    输出保持原样，不强制改名；这里只做检查。
    """
    emb_cols = [c for c in df.columns if "graph_emb_" in c]
    if not emb_cols:
        raise ValueError(f"No graph embedding columns found. Available columns={df.columns.tolist()}")
    return df


def merge_one_split(
    desc_csv: Path,
    emb_csv: Path,
    output_csv: Path,
    task: str,
    split_name: str,
    descriptor_kind: str,
    embedding_prefix: str,
) -> Dict[str, Any]:
    desc_df = read_csv_safe(desc_csv)
    emb_df = read_csv_safe(emb_csv)

    desc_id = resolve_id_col(desc_df, "descriptor")
    emb_id = resolve_id_col(emb_df, "embedding")

    desc_df, desc_dup = dedup_by_id(desc_df, desc_id)
    emb_df, emb_dup = dedup_by_id(emb_df, emb_id)

    emb_df = normalize_embedding_prefix(emb_df, prefix=embedding_prefix)

    # 用统一 join key
    if desc_id != "sample_id":
        desc_df = desc_df.rename(columns={desc_id: "sample_id"})
    if emb_id != "sample_id":
        emb_df = emb_df.rename(columns={emb_id: "sample_id"})

    desc_non_feature = ["material_id", "formula", "source_path", "poscar_path"]
    emb_non_feature = ["material_id"]

    desc_feat_cols = choose_feature_cols(desc_df, "sample_id", desc_non_feature)
    emb_feat_cols = choose_feature_cols(emb_df, "sample_id", emb_non_feature)

    left_cols = ["sample_id"] + [c for c in ["material_id", "formula", "source_path", "poscar_path"] if c in desc_df.columns] + desc_feat_cols
    right_cols = ["sample_id"] + [c for c in ["material_id"] if c in emb_df.columns and c not in left_cols] + emb_feat_cols

    merged = desc_df[left_cols].merge(emb_df[right_cols], on="sample_id", how="inner")
    ensure_dir(output_csv.parent)
    merged.to_csv(output_csv, index=False)

    return {
        "task": task,
        "split_name": split_name,
        "descriptor_kind": descriptor_kind,
        "descriptor_input_csv": str(desc_csv),
        "embedding_input_csv": str(emb_csv),
        "output_csv": str(output_csv),
        "descriptor_rows_before_dedup": int(len(read_csv_safe(desc_csv))),
        "descriptor_duplicate_ids": int(desc_dup),
        "descriptor_rows_after_dedup": int(len(desc_df)),
        "embedding_rows_before_dedup": int(len(read_csv_safe(emb_csv))),
        "embedding_duplicate_ids": int(emb_dup),
        "embedding_rows_after_dedup": int(len(emb_df)),
        "merged_rows": int(len(merged)),
        "n_descriptor_features": int(len(desc_feat_cols)),
        "n_embedding_features": int(len(emb_feat_cols)),
        "merged_columns": merged.columns.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build hybrid feature tables for inference or training-style inputs."
    )
    parser.add_argument("--task", type=str, default="stage2", choices=["stage2", "stage3"])
    parser.add_argument("--output_dir", type=str, required=True)

    # 原始目录式接口
    parser.add_argument("--descriptor_dir", type=str, default="")
    parser.add_argument("--embedding_dirs", type=str, nargs="*", default=[])
    parser.add_argument("--embedding_prefixes", type=str, nargs="*", default=[])
    parser.add_argument("--descriptor_kind", type=str, default="auto", choices=["auto", "ml", "raw"])

    # inference 直连接口
    parser.add_argument("--infer_descriptor_csv", type=str, default="")
    parser.add_argument("--infer_embedding_csv", type=str, default="")
    parser.add_argument("--embedding_prefix", type=str, default="chgnet")
    parser.add_argument(
        "--replicate_to_train_val",
        action="store_true",
        help="For inference, also copy/produce stage2_train_hybrid.csv and stage2_val_hybrid.csv using the same single input."
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    summary: Dict[str, Any] = {
        "config": {
            "task": args.task,
            "output_dir": str(output_dir),
            "descriptor_dir": args.descriptor_dir,
            "embedding_dirs": args.embedding_dirs,
            "embedding_prefixes": args.embedding_prefixes,
            "descriptor_kind": args.descriptor_kind,
            "infer_descriptor_csv": args.infer_descriptor_csv,
            "infer_embedding_csv": args.infer_embedding_csv,
            "embedding_prefix": args.embedding_prefix,
            "replicate_to_train_val": bool(args.replicate_to_train_val),
        },
        "splits": {},
        "mode": "",
    }

    # Inference direct mode
    if args.infer_descriptor_csv and args.infer_embedding_csv:
        summary["mode"] = "inference_direct"
        desc_csv = Path(args.infer_descriptor_csv).expanduser().resolve()
        emb_csv = Path(args.infer_embedding_csv).expanduser().resolve()

        for split_name in (["test", "train", "val"] if args.replicate_to_train_val else ["test"]):
            out_csv = output_dir / f"{args.task}_{split_name}_hybrid.csv"
            summary["splits"][split_name] = merge_one_split(
                desc_csv=desc_csv,
                emb_csv=emb_csv,
                output_csv=out_csv,
                task=args.task,
                split_name=split_name,
                descriptor_kind="ml" if args.task == "stage2" else "raw",
                embedding_prefix=args.embedding_prefix,
            )

        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    # Training-style directory mode
    if not args.descriptor_dir or not args.embedding_dirs:
        raise ValueError(
            "Either provide both --infer_descriptor_csv and --infer_embedding_csv, "
            "or provide --descriptor_dir and --embedding_dirs."
        )

    summary["mode"] = "directory_style"

    descriptor_kind = args.descriptor_kind
    if descriptor_kind == "auto":
        descriptor_kind = "ml" if args.task == "stage2" else "raw"

    desc_dir = Path(args.descriptor_dir).expanduser().resolve()
    embedding_dirs = [Path(x).expanduser().resolve() for x in args.embedding_dirs]
    embedding_prefixes = args.embedding_prefixes or [args.embedding_prefix] * len(embedding_dirs)
    if len(embedding_prefixes) != len(embedding_dirs):
        raise ValueError("embedding_prefixes length must match embedding_dirs length")

    for split_name in ["train", "val", "test"]:
        desc_csv = desc_dir / f"{args.task}_{split_name}_{descriptor_kind}.csv"

        # 当前版本先支持一个 embedding_dir；若多个则按列拼接前先内部 merge
        emb_dfs = []
        emb_info = []
        for embed_dir, prefix in zip(embedding_dirs, embedding_prefixes):
            graph_csv = embed_dir / f"{args.task}_{split_name}_graph_embed.csv"
            df = read_csv_safe(graph_csv)
            df = normalize_embedding_prefix(df, prefix)
            emb_dfs.append((df, prefix, graph_csv))
            emb_info.append({"prefix": prefix, "path": str(graph_csv)})

        # 把多个 embedding csv 按 sample_id 合并成临时表
        merged_emb = None
        for df, prefix, graph_csv in emb_dfs:
            emb_id = resolve_id_col(df, f"embedding({prefix})")
            if emb_id != "sample_id":
                df = df.rename(columns={emb_id: "sample_id"})
            if merged_emb is None:
                merged_emb = df
            else:
                extra_cols = [c for c in df.columns if c == "sample_id" or c not in merged_emb.columns]
                merged_emb = merged_emb.merge(df[extra_cols], on="sample_id", how="inner")

        temp_emb_csv = output_dir / f"__temp_{args.task}_{split_name}_graph_embed.csv"
        merged_emb.to_csv(temp_emb_csv, index=False)

        out_csv = output_dir / f"{args.task}_{split_name}_hybrid.csv"
        split_summary = merge_one_split(
            desc_csv=desc_csv,
            emb_csv=temp_emb_csv,
            output_csv=out_csv,
            task=args.task,
            split_name=split_name,
            descriptor_kind=descriptor_kind,
            embedding_prefix="multi",
        )
        split_summary["embedding_sources"] = emb_info
        summary["splits"][split_name] = split_summary

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
