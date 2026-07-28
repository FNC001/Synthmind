#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


FAMILY_FEATURE_COUNT = 24
STATIC_JSON_FILES = (
    "action_to_id.json",
    "action_vocab.json",
    "feature_cols.json",
    "label_cols.json",
    "precursor_names.json",
)


def refit_features(
    trainval_raw: np.ndarray,
    split_raw: np.ndarray,
    family_feature_count: int = FAMILY_FEATURE_COUNT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if trainval_raw.ndim != 2 or split_raw.ndim != 2:
        raise ValueError("feature arrays must be two-dimensional")
    if trainval_raw.shape[1] != split_raw.shape[1]:
        raise ValueError("feature dimensions do not match")
    base_dim = int(trainval_raw.shape[1] - family_feature_count)
    if base_dim <= 0:
        raise ValueError("family feature count leaves no base features")
    train_base = np.asarray(trainval_raw[:, :base_dim], dtype=np.float32)
    split_base = np.asarray(split_raw[:, :base_dim], dtype=np.float32)
    mean = np.nanmean(train_base, axis=0).astype(np.float32)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    train_filled = np.where(np.isfinite(train_base), train_base, mean[None, :]).astype(np.float32)
    split_filled = np.where(np.isfinite(split_base), split_base, mean[None, :]).astype(np.float32)
    std = np.std(train_filled, axis=0).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0).astype(np.float32)
    train_scaled = np.hstack([
        (train_filled - mean[None, :]) / std[None, :],
        np.asarray(trainval_raw[:, base_dim:], dtype=np.float32),
    ]).astype(np.float32)
    split_scaled = np.hstack([
        (split_filled - mean[None, :]) / std[None, :],
        np.asarray(split_raw[:, base_dim:], dtype=np.float32),
    ]).astype(np.float32)
    full_mean = np.concatenate([mean, np.zeros(family_feature_count, dtype=np.float32)])
    full_std = np.concatenate([std, np.ones(family_feature_count, dtype=np.float32)])
    return train_scaled, split_scaled, full_mean, full_std


def merge_candidate_files(train_path: Path, val_path: Path, output_path: Path, offset: int) -> int:
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for path, row_offset in ((train_path, 0), (val_path, int(offset))):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    record["row_index"] = int(record["row_index"]) + row_offset
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a train+validation Stage2 pack for final fixed-epoch refitting."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_candidates", default="")
    parser.add_argument("--val_candidates", default="")
    parser.add_argument("--output_candidates", default="")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    packs: Dict[str, np.lib.npyio.NpzFile] = {
        split: np.load(input_dir / f"{split}.npz", allow_pickle=True)
        for split in ("train", "val", "test")
    }
    train_rows = int(packs["train"]["x_raw"].shape[0])
    val_rows = int(packs["val"]["x_raw"].shape[0])
    trainval_arrays = {
        key: np.concatenate([packs["train"][key], packs["val"][key]], axis=0)
        for key in packs["train"].files
        if key != "x"
    }
    trainval_x, test_x, mean, std = refit_features(
        np.asarray(trainval_arrays["x_raw"], dtype=np.float32),
        np.asarray(packs["test"]["x_raw"], dtype=np.float32),
    )
    _, val_x, _, _ = refit_features(
        np.asarray(trainval_arrays["x_raw"], dtype=np.float32),
        np.asarray(packs["val"]["x_raw"], dtype=np.float32),
    )
    trainval_arrays["x"] = trainval_x
    np.savez_compressed(output_dir / "train.npz", **trainval_arrays)
    val_arrays = {key: np.asarray(packs["val"][key]) for key in packs["val"].files}
    val_arrays["x"] = val_x
    np.savez_compressed(output_dir / "val.npz", **val_arrays)
    test_arrays = {key: np.asarray(packs["test"][key]) for key in packs["test"].files}
    test_arrays["x"] = test_x
    np.savez_compressed(output_dir / "test.npz", **test_arrays)

    train_meta = pd.concat([
        pd.read_csv(input_dir / "train_meta.csv", low_memory=False),
        pd.read_csv(input_dir / "val_meta.csv", low_memory=False),
    ], ignore_index=True)
    train_meta.to_csv(output_dir / "train_meta.csv", index=False)
    shutil.copy2(input_dir / "val_meta.csv", output_dir / "val_meta.csv")
    shutil.copy2(input_dir / "test_meta.csv", output_dir / "test_meta.csv")
    for filename in STATIC_JSON_FILES:
        shutil.copy2(input_dir / filename, output_dir / filename)
    for split in ("val", "test"):
        source = input_dir / f"family_assignments_{split}.csv"
        if source.exists():
            shutil.copy2(source, output_dir / source.name)
    family_train = input_dir / "family_assignments_train.csv"
    family_val = input_dir / "family_assignments_val.csv"
    if family_train.exists() and family_val.exists():
        pd.concat([
            pd.read_csv(family_train, low_memory=False),
            pd.read_csv(family_val, low_memory=False),
        ], ignore_index=True).to_csv(output_dir / "family_assignments_train.csv", index=False)

    (output_dir / "scaler.json").write_text(json.dumps({
        "mean": mean.tolist(), "std": std.tolist(), "fit_splits": ["train", "val"]
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    candidate_rows = None
    if any((args.train_candidates, args.val_candidates, args.output_candidates)):
        if not all((args.train_candidates, args.val_candidates, args.output_candidates)):
            parser.error("all three candidate-file arguments must be supplied together")
        candidate_rows = merge_candidate_files(
            Path(args.train_candidates).resolve(),
            Path(args.val_candidates).resolve(),
            Path(args.output_candidates).resolve(),
            train_rows,
        )
    report = {
        "protocol": "stage2_final_refit_train_plus_validation_test_held_out",
        "source_dir": str(input_dir),
        "rows": {"trainval": train_rows + val_rows, "monitor_val": val_rows, "test": int(len(test_x))},
        "scaler_fit_rows": train_rows + val_rows,
        "candidate_rows": candidate_rows,
        "warning": "val is a monitor-only seen subset; use fixed epochs and checkpoint_selection=last",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
