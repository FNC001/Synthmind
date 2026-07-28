#!/usr/bin/env python3
"""Merge disjoint Synthmind GNoME shard outputs and rebuild global Top100."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_adapter(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "synthmind_gnome_adapter_for_merge", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def list_input_ids(input_dir: Path) -> list[str]:
    files = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and (
            path.suffix.lower() in {".cif", ".vasp", ".poscar"}
            or path.name.upper() == "POSCAR"
        )
    )
    return [path.stem for path in files]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--adapter-code", required=True)
    parser.add_argument("--shard-output", action="append", required=True)
    args = parser.parse_args()

    started = time.time()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    adapter_code = Path(args.adapter_code).expanduser().resolve()
    shard_dirs = [
        Path(value).expanduser().resolve() for value in args.shard_output
    ]
    output_dir.mkdir(parents=True, exist_ok=True)

    input_ids = list_input_ids(input_dir)
    input_index = {sample_id: index for index, sample_id in enumerate(input_ids)}
    if len(input_index) != len(input_ids):
        raise ValueError("Duplicate IDs in original input directory")

    frames = []
    shard_records = []
    for shard_index, shard_dir in enumerate(shard_dirs):
        prediction_path = shard_dir / "all_predictions.csv.gz"
        summary_path = shard_dir / "summary.json"
        manifest_path = shard_dir / "model_manifest.json"
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)
        frame = pd.read_csv(prediction_path, low_memory=False)
        frame["_shard_id"] = shard_index
        frames.append(frame)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        shard_records.append(
            {
                "shard_id": shard_index,
                "output_dir": str(shard_dir),
                "rows": int(len(frame)),
                "status_counts": summary["prediction_status_counts"],
                "elapsed_seconds": summary["elapsed_seconds"],
                "predictions_sha256": sha256_file(prediction_path),
                "summary_sha256": sha256_file(summary_path),
                "model_manifest_sha256": sha256_file(manifest_path),
            }
        )

    merged = pd.concat(frames, ignore_index=True)
    sample_ids = merged["sample_id"].astype(str)
    duplicate_ids = sample_ids[sample_ids.duplicated(keep=False)].unique().tolist()
    if duplicate_ids:
        raise ValueError(f"Duplicate shard IDs: {duplicate_ids[:20]}")
    output_set = set(sample_ids)
    input_set = set(input_ids)
    if output_set != input_set:
        raise ValueError(
            "Shard/input ID mismatch: "
            f"missing={sorted(input_set - output_set)[:20]}, "
            f"unexpected={sorted(output_set - input_set)[:20]}"
        )

    merged["input_index"] = sample_ids.map(input_index).astype(int)
    merged = merged.sort_values("input_index").reset_index(drop=True)
    if merged["input_index"].tolist() != list(range(len(input_ids))):
        raise RuntimeError("Merged input_index is not contiguous")
    merged = merged.drop(columns=["_shard_id"])

    predictions_path = output_dir / "all_predictions.csv.gz"
    merged.to_csv(predictions_path, index=False, compression="gzip")

    source_manifest = json.loads(
        (shard_dirs[0] / "model_manifest.json").read_text(encoding="utf-8")
    )
    source_manifest["parallel_execution"] = {
        "strategy": "four_disjoint_id_shards_then_exact_id_order_merge",
        "shard_count": len(shard_dirs),
        "shards": shard_records,
        "adapter_code": str(adapter_code),
        "adapter_code_sha256": sha256_file(adapter_code),
    }
    write_json(output_dir / "model_manifest.json", source_manifest)

    adapter = load_adapter(adapter_code)
    top100 = adapter.build_top100(predictions_path, output_dir)
    status_counts = {
        str(key): int(value)
        for key, value in merged["prediction_status"].value_counts(dropna=False).items()
    }
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "input_structure_count": len(input_ids),
        "merged_output_rows": int(len(merged)),
        "prediction_status_counts": status_counts,
        "top100_count": int(len(top100)),
        "top100_selection_policy": (
            str(top100["top100_selection_policy"].iloc[0])
            if len(top100)
            else ""
        ),
        "strict_quality_eligible_pool_size": (
            int(top100["strict_quality_eligible_pool_size"].iloc[0])
            if len(top100)
            else 0
        ),
        "stage2_policy": (
            "frozen_factorized3_chemistry_aware_rrf_plus_"
            "composition_training_prior"
        ),
        "stage2_s9161_gate_status": (
            "not applied online; see model_manifest.json"
        ),
        "stage3_policy": (
            "equal-sample frozen NF + CVAE + Diffusion ensemble"
        ),
        "stage3_samples_per_model": int(
            source_manifest["stage3"]["samples_per_model"]
        ),
        "ranking_score_semantics": (
            "uncalibrated ranking proxy; not experimental success probability"
        ),
        "parallel_execution": source_manifest["parallel_execution"],
        "elapsed_seconds_merge": time.time() - started,
        "outputs": {
            "all_predictions": str(predictions_path),
            "top100_csv": str(
                output_dir / "top100_most_synthesizable.csv"
            ),
            "top100_markdown": str(
                output_dir / "top100_most_synthesizable.md"
            ),
            "model_manifest": str(output_dir / "model_manifest.json"),
        },
    }
    write_json(output_dir / "summary.json", summary)

    for filename in (
        "README_synthmind_gnome_frozen_adapter_zh.md",
        "synthmind_gnome_frozen_adapter.py",
        "validate_synthmind_gnome_output.py",
        "merge_synthmind_gnome_shards.py",
    ):
        source = adapter_code.parent / filename
        if source.is_file():
            shutil.copy2(source, output_dir / filename)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
