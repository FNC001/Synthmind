#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


SPLITS = ("train", "val", "test")


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a same-split Stage2 ablation dataset without cation-family features."
    )
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    source = Path(args.source_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    n_family = int(summary["schema"]["family_feature_count"])
    n_all = int(summary["schema"]["n_features"])
    n_base = n_all - n_family
    if n_family <= 0 or n_base <= 0:
        raise ValueError(f"invalid dimensions: all={n_all}, family={n_family}")

    for split in SPLITS:
        pack_file = np.load(source / f"{split}.npz", allow_pickle=True)
        pack = {key: pack_file[key] for key in pack_file.files}
        pack["x_raw"] = np.asarray(pack["x_raw"])[:, :n_base]
        pack["x"] = np.asarray(pack["x"])[:, :n_base]
        np.savez_compressed(output / f"{split}.npz", **pack)
        shutil.copy2(source / f"{split}_meta.csv", output / f"{split}_meta.csv")

    for name in (
        "action_vocab.json",
        "action_to_id.json",
        "precursor_names.json",
        "label_cols.json",
        "label_names.json",
        "split_manifest.json",
    ):
        if (source / name).exists():
            shutil.copy2(source / name, output / name)

    feature_cols = json.loads((source / "feature_cols.json").read_text(encoding="utf-8"))[:n_base]
    write_json(output / "feature_cols.json", feature_cols)
    scaler = json.loads((source / "scaler.json").read_text(encoding="utf-8"))
    scaler["mean"] = scaler["mean"][:n_base]
    scaler["std"] = scaler["std"][:n_base]
    scaler["feature_cols"] = scaler["feature_cols"][:n_base]
    write_json(output / "scaler.json", scaler)

    summary["schema"]["n_features"] = n_base
    summary["schema"]["family_feature_count"] = 0
    summary["schema"]["family_feature_names"] = []
    summary["ablation"] = {
        "name": "no_target_cation_family_features",
        "source_dataset": str(source),
        "removed_feature_count": n_family,
        "base_feature_count": n_base,
        "same_split_manifest": str((source / "split_manifest.json").resolve()),
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary["ablation"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
