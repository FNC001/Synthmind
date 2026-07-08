#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_build_stage2_gflownet_infer_npz.py

把 inference 生成的 stage2_test_hybrid.csv 转成 GFlowNet 采样脚本可直接读取的 test.npz。

输入
----
- infer_hybrid_csv: 例如 /Users/wyc/SynPred/infer_runs/demo1/hybrid_stage2/stage2_test_hybrid.csv
- template_dir:     训练时的 GFlowNet 数据目录，例如
                    /Users/wyc/SynPred/data/interim/generative/stage2_gflownet_dataset/hybrid/gold_only

输出
----
output_dir/
  ├── test.npz
  ├── test_meta.csv
  ├── action_to_id.json
  ├── action_vocab.json
  ├── feature_cols.json
  ├── feature_mean.npy
  ├── feature_std.npy
  ├── label_cols.json
  ├── precursor_names.json
  └── summary.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd

JSON_FILES_TO_COPY = [
    "action_to_id.json",
    "action_vocab.json",
    "feature_cols.json",
    "label_cols.json",
    "precursor_names.json",
]
NPY_FILES_TO_COPY = [
    "feature_mean.npy",
    "feature_std.npy",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def resolve_id_col(df: pd.DataFrame) -> str:
    for c in ["sample_id", "id", "material_id"]:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find id column in infer_hybrid_csv, columns={df.columns.tolist()}")


def build_meta_csv(df: pd.DataFrame, output_path: Path) -> List[str]:
    preferred = [
        "sample_id",
        "material_id",
        "formula",
        "formula_x",
        "formula_y",
        "doi",
        "split_group",
        "source_path",
        "poscar_path",
    ]
    cols = [c for c in preferred if c in df.columns]
    if not cols:
        cols = [resolve_id_col(df)]
    meta = df[cols].copy()
    meta.to_csv(output_path, index=False)
    return cols


def main() -> None:
    parser = argparse.ArgumentParser(description="Build inference NPZ pack for stage2 GFlowNet sampling.")
    parser.add_argument("--infer_hybrid_csv", type=str, required=True)
    parser.add_argument("--template_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--split_name", type=str, default="test")
    args = parser.parse_args()

    infer_hybrid_csv = Path(args.infer_hybrid_csv).expanduser().resolve()
    template_dir = Path(args.template_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    split_name = args.split_name

    if not infer_hybrid_csv.exists():
        raise FileNotFoundError(f"Missing infer_hybrid_csv: {infer_hybrid_csv}")
    if not template_dir.exists():
        raise FileNotFoundError(f"Missing template_dir: {template_dir}")

    ensure_dir(output_dir)

    copied_files = []
    for name in JSON_FILES_TO_COPY + NPY_FILES_TO_COPY:
        src = template_dir / name
        if src.exists():
            dst = output_dir / name
            shutil.copyfile(src, dst)
            copied_files.append(str(dst))

    feature_cols = read_json(template_dir / "feature_cols.json")
    label_cols = read_json(template_dir / "label_cols.json")
    feature_mean = np.load(template_dir / "feature_mean.npy")
    feature_std = np.load(template_dir / "feature_std.npy")

    template_npz = np.load(template_dir / f"{split_name}.npz")
    max_traj_len = int(template_npz["traj_actions"].shape[1])
    n_labels = int(len(label_cols))

    df = pd.read_csv(infer_hybrid_csv)
    if df.empty:
        raise ValueError(f"Infer hybrid CSV is empty: {infer_hybrid_csv}")

    if "formula" not in df.columns:
        if "formula_x" in df.columns:
            df["formula"] = df["formula_x"]
        elif "formula_y" in df.columns:
            df["formula"] = df["formula_y"]

    missing_feature_cols = [c for c in feature_cols if c not in df.columns]
    extra_feature_like_cols = [
        c for c in df.columns
        if c not in set(feature_cols)
        and c not in {"sample_id", "id", "material_id", "formula", "formula_x", "formula_y", "doi", "split_group", "source_path", "poscar_path"}
    ]

    aligned = df.reindex(columns=feature_cols, fill_value=0.0).copy()
    x_raw = aligned.to_numpy(dtype=np.float32)

    denom = feature_std.astype(np.float32).copy()
    denom[np.abs(denom) < 1e-12] = 1.0
    x = ((x_raw - feature_mean.astype(np.float32)) / denom).astype(np.float32)

    n = x.shape[0]
    y_multi_hot = np.zeros((n, n_labels), dtype=np.float32)
    traj_actions = np.zeros((n, max_traj_len), dtype=np.int64)
    traj_mask = np.zeros((n, max_traj_len), dtype=np.int64)
    set_len = np.zeros((n,), dtype=np.int64)

    npz_path = output_dir / f"{split_name}.npz"
    np.savez_compressed(
        npz_path,
        x_raw=x_raw,
        x=x,
        y_multi_hot=y_multi_hot,
        traj_actions=traj_actions,
        traj_mask=traj_mask,
        set_len=set_len,
    )

    meta_cols = build_meta_csv(df, output_dir / f"{split_name}_meta.csv")

    summary = {
        "infer_hybrid_csv": str(infer_hybrid_csv),
        "template_dir": str(template_dir),
        "output_dir": str(output_dir),
        "split_name": split_name,
        "n_rows": int(n),
        "x_dim": int(x.shape[1]),
        "n_labels": int(n_labels),
        "max_traj_len": int(max_traj_len),
        "meta_cols": meta_cols,
        "missing_feature_cols_count": int(len(missing_feature_cols)),
        "missing_feature_cols_preview": missing_feature_cols[:30],
        "extra_feature_like_cols_count": int(len(extra_feature_like_cols)),
        "extra_feature_like_cols_preview": extra_feature_like_cols[:30],
        "copied_files": copied_files,
        "artifacts": {
            "npz": str(npz_path),
            "meta_csv": str(output_dir / f"{split_name}_meta.csv"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
