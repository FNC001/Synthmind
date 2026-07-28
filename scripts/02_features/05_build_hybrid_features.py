#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def dedup_by_id(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if "id" not in df.columns:
        raise ValueError("Input table must contain an 'id' column.")
    dup = int(df["id"].duplicated().sum())
    if dup > 0:
        df = df.drop_duplicates(subset=["id"], keep="first").copy()
    return df, dup


def normalize_embed_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Rename any embedding-like columns to a unified prefix form:
      graph_emb_0 -> {prefix}_graph_emb_0
      cgcnn_graph_emb_0 -> {prefix}_graph_emb_0
      alignn_graph_emb_0 -> {prefix}_graph_emb_0
      chgnet_graph_emb_0 -> {prefix}_graph_emb_0
    """
    rename_map = {}
    for c in df.columns:
        if c == "id":
            continue
        if c.endswith("_graph_emb_0"):
            # e.g. cgcnn_graph_emb_0 -> prefix_graph_emb_0
            suffix = c.split("_graph_emb_", 1)[1]
            rename_map[c] = f"{prefix}_graph_emb_{suffix}"
        elif "_graph_emb_" in c:
            suffix = c.split("_graph_emb_", 1)[1]
            rename_map[c] = f"{prefix}_graph_emb_{suffix}"
        elif c.startswith("graph_emb_"):
            suffix = c.split("graph_emb_", 1)[1]
            rename_map[c] = f"{prefix}_graph_emb_{suffix}"
    if rename_map:
        df = df.rename(columns=rename_map).copy()
    return df


def collect_embed_feature_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    return [c for c in df.columns if c.startswith(f"{prefix}_graph_emb_")]


def build_one_split(
    task: str,
    split_name: str,
    desc_dir: Path,
    embed_dirs: List[Path],
    embed_prefixes: List[str],
    out_dir: Path,
    descriptor_kind: str,
) -> Dict[str, Any]:
    if task not in {"stage2", "stage3"}:
        raise ValueError("task must be 'stage2' or 'stage3'")

    if descriptor_kind == "auto":
        descriptor_kind = "ml" if task == "stage2" else "raw"

    desc_path = desc_dir / f"{task}_{split_name}_{descriptor_kind}.csv"
    desc_df = load_csv(desc_path)
    desc_rows = len(desc_df)
    desc_df, desc_dup = dedup_by_id(desc_df)

    merged = desc_df.copy()

    per_embed_summary: Dict[str, Any] = {}
    total_graph_features = 0

    for embed_dir, prefix in zip(embed_dirs, embed_prefixes):
        graph_path = embed_dir / f"{task}_{split_name}_graph_embed.csv"
        if not graph_path.exists():
            raise FileNotFoundError(
                f"Missing embedding file for prefix='{prefix}', split='{split_name}': {graph_path}"
            )
        graph_df = load_csv(graph_path)
        graph_rows = len(graph_df)
        graph_df, graph_dup = dedup_by_id(graph_df)

        # keep id + metadata + embedding columns
        graph_df = normalize_embed_columns(graph_df, prefix=prefix)

        meta_cols = [c for c in ["id", "material_id", "formula", "doi", "split_group"] if c in graph_df.columns]
        embed_cols = collect_embed_feature_cols(graph_df, prefix=prefix)
        graph_keep_cols = meta_cols + embed_cols
        graph_df = graph_df[graph_keep_cols].copy()

        merged_before = len(merged)
        merged = pd.merge(
            merged,
            graph_df,
            on="id",
            how="inner",
            suffixes=("", f"_{prefix}"),
        )

        per_embed_summary[prefix] = {
            "graph_input_csv": str(graph_path),
            "graph_rows_before_dedup": graph_rows,
            "graph_duplicate_ids": graph_dup,
            "graph_rows_after_dedup": int(len(graph_df)),
            "n_graph_features": int(len(embed_cols)),
            "rows_before_merge": int(merged_before),
            "rows_after_merge": int(len(merged)),
        }
        total_graph_features += len(embed_cols)

    out_path = out_dir / f"{task}_{split_name}_hybrid.csv"
    merged.to_csv(out_path, index=False)

    desc_feat_cols = [c for c in merged.columns if c.startswith("feat_")]
    if task == "stage2":
        label_cols = [c for c in merged.columns if c.startswith("label_prec__")]
    else:
        label_cols = [c for c in merged.columns if c.startswith("label_")]

    summary = {
        "task": task,
        "split": split_name,
        "descriptor_input_csv": str(desc_path),
        "output_csv": str(out_path),
        "descriptor_rows_before_dedup": desc_rows,
        "descriptor_duplicate_ids": desc_dup,
        "descriptor_rows_after_dedup": int(len(desc_df)),
        "merged_rows": int(len(merged)),
        "n_descriptor_features": int(len(desc_feat_cols)),
        "n_graph_features_total": int(total_graph_features),
        "n_hybrid_features": int(len(desc_feat_cols) + total_graph_features),
        "n_labels": int(len(label_cols)),
        "graph_sources": per_embed_summary,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build hybrid feature tables by merging descriptor tables with one or more embedding tables."
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["stage2", "stage3"],
        required=True,
        help="Which task to build hybrid features for.",
    )
    parser.add_argument(
        "--descriptor_dir",
        type=str,
        required=True,
        help="Directory containing descriptor CSVs.",
    )
    parser.add_argument(
        "--embedding_dirs",
        type=str,
        nargs="+",
        required=True,
        help="One or more directories containing *_graph_embed.csv files.",
    )
    parser.add_argument(
        "--embedding_prefixes",
        type=str,
        nargs="+",
        required=True,
        help="Prefixes for embedding sources, e.g. cgcnn alignn chgnet",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for hybrid CSVs.",
    )
    parser.add_argument(
        "--descriptor_kind",
        type=str,
        default="auto",
        choices=["auto", "ml", "raw"],
        help="Use 'ml' for stage2, 'raw' for stage3, or let script infer automatically.",
    )
    args = parser.parse_args()

    if len(args.embedding_dirs) != len(args.embedding_prefixes):
        raise ValueError("--embedding_dirs and --embedding_prefixes must have the same length.")

    desc_dir = Path(args.descriptor_dir)
    embed_dirs = [Path(p) for p in args.embedding_dirs]
    embed_prefixes = args.embedding_prefixes
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    summary: Dict[str, Any] = {
        "config": {
            "task": args.task,
            "descriptor_dir": str(desc_dir),
            "embedding_dirs": [str(p) for p in embed_dirs],
            "embedding_prefixes": embed_prefixes,
            "output_dir": str(out_dir),
            "descriptor_kind": args.descriptor_kind,
        },
        "splits": {},
    }

    for split_name in ["train", "val", "test", "gold_train_holdout"]:
        summary["splits"][split_name] = build_one_split(
            task=args.task,
            split_name=split_name,
            desc_dir=desc_dir,
            embed_dirs=embed_dirs,
            embed_prefixes=embed_prefixes,
            out_dir=out_dir,
            descriptor_kind=args.descriptor_kind,
        )

    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
