#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def subset_pack(pack: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[indices] for key, value in pack.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build formula-group-safe OOF Stage2 candidate folds.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    pack = {key: value for key, value in np.load(input_dir / "train.npz", allow_pickle=True).items()}
    meta = pd.read_csv(input_dir / "train_meta.csv", low_memory=False)
    folds = sorted(pd.to_numeric(meta["split_fold"], errors="raise").astype(int).unique().tolist())
    static_files = (
        "action_to_id.json", "action_vocab.json", "feature_cols.json", "label_cols.json",
        "precursor_names.json", "scaler.json", "summary.json", "split_manifest.json",
    )
    manifest = {"source": str(input_dir), "rows": len(meta), "folds": {}}
    for fold in folds:
        fold_dir = output_root / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        query_indices = np.flatnonzero(meta["split_fold"].to_numpy(dtype=int) == fold)
        train_indices = np.flatnonzero(meta["split_fold"].to_numpy(dtype=int) != fold)
        np.savez_compressed(fold_dir / "train.npz", **subset_pack(pack, train_indices))
        np.savez_compressed(fold_dir / "val.npz", **subset_pack(pack, query_indices))
        meta.iloc[train_indices].reset_index(drop=True).to_csv(fold_dir / "train_meta.csv", index=False)
        meta.iloc[query_indices].reset_index(drop=True).to_csv(fold_dir / "val_meta.csv", index=False)
        np.save(fold_dir / "val_global_row_indices.npy", query_indices)
        for filename in static_files:
            shutil.copy2(input_dir / filename, fold_dir / filename)
        manifest["folds"][str(fold)] = {
            "train_rows": int(len(train_indices)),
            "query_rows": int(len(query_indices)),
            "query_global_min": int(query_indices.min()) if len(query_indices) else None,
            "query_global_max": int(query_indices.max()) if len(query_indices) else None,
        }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
