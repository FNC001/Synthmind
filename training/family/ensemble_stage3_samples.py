#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Concatenate aligned Stage3 distribution samples from multiple model families."
    )
    parser.add_argument("--input_npz", action="append", required=True)
    parser.add_argument(
        "--sample_limit",
        action="append",
        type=int,
        default=[],
        help="Optional per-input sample limits; defaults to all samples.",
    )
    parser.add_argument("--output_npz", required=True)
    args = parser.parse_args()
    if args.sample_limit and len(args.sample_limit) != len(args.input_npz):
        parser.error("--sample_limit must be repeated once per --input_npz")

    continuous: List[np.ndarray] = []
    discrete: List[np.ndarray] = []
    sample_ids: np.ndarray | None = None
    contributions = []
    for index, value in enumerate(args.input_npz):
        path = Path(value).expanduser().resolve()
        with np.load(path, allow_pickle=False) as source:
            current_ids = np.asarray(source["sample_id"]).astype(str)
            current_continuous = np.asarray(source["continuous_samples"], dtype=np.float32)
            current_discrete = np.asarray(source["discrete_samples"], dtype=np.int16)
        if sample_ids is None:
            sample_ids = current_ids
        elif not np.array_equal(sample_ids, current_ids):
            raise ValueError(f"sample IDs are not aligned for {path}")
        if current_continuous.shape[:2] != current_discrete.shape[:2]:
            raise ValueError(f"continuous/discrete sample shapes do not align for {path}")
        limit = (
            min(int(args.sample_limit[index]), current_continuous.shape[1])
            if args.sample_limit
            else current_continuous.shape[1]
        )
        continuous.append(current_continuous[:, :limit])
        discrete.append(current_discrete[:, :limit])
        contributions.append({"path": str(path), "samples_per_row": int(limit)})
    assert sample_ids is not None
    output = Path(args.output_npz).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        continuous_samples=np.concatenate(continuous, axis=1),
        discrete_samples=np.concatenate(discrete, axis=1),
        sample_id=sample_ids,
    )
    report = {
        "protocol": "aligned_stage3_model_family_sample_ensemble",
        "output_npz": str(output),
        "rows": int(len(sample_ids)),
        "total_samples_per_row": int(sum(row["samples_per_row"] for row in contributions)),
        "contributions": contributions,
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
