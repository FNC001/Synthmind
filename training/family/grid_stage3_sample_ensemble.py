#!/usr/bin/env python3
"""Search validation-only sample contributions for a Stage 3 model ensemble."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from training.family.evaluate_stage3_sample_topk import metrics, ranked_buckets


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"expected NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    return name.strip(), Path(path).expanduser().resolve()


def parse_limit_grid(value: str) -> list[int]:
    values = sorted({int(item) for item in value.split(",")})
    if any(item < 0 for item in values):
        raise ValueError("sample limits must be non-negative")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--model", action="append", required=True, help="Repeat NAME=NPZ")
    parser.add_argument(
        "--limit_grid",
        action="append",
        required=True,
        help="Repeat once per model, e.g. 64,128,192,256",
    )
    parser.add_argument("--temperature_bin", type=float, default=100.0)
    parser.add_argument("--time_bin", type=float, default=24.0)
    parser.add_argument("--temperature_tolerance", type=float, default=200.0)
    parser.add_argument("--time_tolerance", type=float, default=48.0)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_npz", required=True)
    args = parser.parse_args()
    if len(args.model) != len(args.limit_grid):
        parser.error("--limit_grid must be repeated once per --model")

    input_dir = Path(args.input_dir).expanduser().resolve()
    with np.load(input_dir / f"{args.split}.npz", allow_pickle=True) as pack:
        continuous_true = np.asarray(pack["y_cond_continuous_raw"], dtype=np.float64)
        continuous_mask = np.asarray(pack["y_cond_continuous_mask"], dtype=np.float32) > 0.5
        discrete_true = np.asarray(pack["y_cond_discrete"], dtype=np.int64)
        discrete_mask = np.asarray(pack["y_cond_discrete_mask"], dtype=np.float32) > 0.5

    names: list[str] = []
    continuous_sources: list[np.ndarray] = []
    discrete_sources: list[np.ndarray] = []
    sample_ids: np.ndarray | None = None
    grids = [parse_limit_grid(value) for value in args.limit_grid]
    for value in args.model:
        name, path = parse_named_path(value)
        with np.load(path, allow_pickle=False) as pack:
            current_ids = np.asarray(pack["sample_id"]).astype(str)
            current_continuous = np.asarray(pack["continuous_samples"], dtype=np.float32)
            current_discrete = np.asarray(pack["discrete_samples"], dtype=np.int16)
        if sample_ids is None:
            sample_ids = current_ids
        elif not np.array_equal(sample_ids, current_ids):
            raise ValueError(f"sample IDs are not aligned for {path}")
        names.append(name)
        continuous_sources.append(current_continuous)
        discrete_sources.append(current_discrete)
    assert sample_ids is not None

    all_indices = np.arange(len(continuous_true), dtype=np.int64)
    strict_indices = np.flatnonzero(continuous_mask[:, 0] & continuous_mask[:, 1])
    trials = []
    best_key: tuple[float, float, float] | None = None
    best_limits: tuple[int, ...] | None = None
    for limits in itertools.product(*grids):
        if not any(limits):
            continue
        if any(limit > source.shape[1] for limit, source in zip(limits, continuous_sources)):
            continue
        continuous = np.concatenate(
            [source[:, :limit] for source, limit in zip(continuous_sources, limits) if limit],
            axis=1,
        )
        discrete = np.concatenate(
            [source[:, :limit] for source, limit in zip(discrete_sources, limits) if limit],
            axis=1,
        )
        ranked = [
            ranked_buckets(
                continuous[row],
                discrete[row],
                float(args.temperature_bin),
                float(args.time_bin),
            )
            for row in range(len(continuous))
        ]
        relaxed = metrics(
            ranked,
            all_indices,
            continuous_true,
            continuous_mask,
            discrete_true,
            discrete_mask,
            float(args.temperature_tolerance),
            float(args.time_tolerance),
            False,
        )
        method = metrics(
            ranked,
            all_indices,
            continuous_true,
            continuous_mask,
            discrete_true,
            discrete_mask,
            float(args.temperature_tolerance),
            float(args.time_tolerance),
            True,
        )
        strict_method = metrics(
            ranked,
            strict_indices,
            continuous_true,
            continuous_mask,
            discrete_true,
            discrete_mask,
            float(args.temperature_tolerance),
            float(args.time_tolerance),
            True,
        )
        row = {
            "limits": dict(zip(names, limits)),
            "samples_per_row": int(sum(limits)),
            "mean_unique_condition_buckets": float(np.mean([len(item) for item in ranked])),
            "missing_aware_relaxed": relaxed,
            "missing_aware_method_inclusive": method,
            "strict_comparable_method_inclusive": strict_method,
        }
        trials.append(row)
        key = (
            float(method["hit@10"]),
            float(relaxed["hit@10"]),
            float(method["hit@20"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_limits = limits
        print(json.dumps({"limits": row["limits"], "key": key}), flush=True)
    if best_limits is None:
        raise RuntimeError("grid produced no valid trial")
    best_continuous = np.concatenate(
        [source[:, :limit] for source, limit in zip(continuous_sources, best_limits) if limit],
        axis=1,
    )
    best_discrete = np.concatenate(
        [source[:, :limit] for source, limit in zip(discrete_sources, best_limits) if limit],
        axis=1,
    )
    output_npz = Path(args.output_npz).expanduser().resolve()
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        continuous_samples=best_continuous,
        discrete_samples=best_discrete,
        sample_id=sample_ids,
    )
    best = max(
        trials,
        key=lambda row: (
            row["missing_aware_method_inclusive"]["hit@10"],
            row["missing_aware_relaxed"]["hit@10"],
            row["missing_aware_method_inclusive"]["hit@20"],
        ),
    )
    report = {
        "protocol": f"{args.split}_stage3_sample_contribution_grid",
        "selection_note": "The frozen test split is not used when split=val.",
        "models": names,
        "best": best,
        "trials": sorted(
            trials,
            key=lambda row: row["missing_aware_method_inclusive"]["hit@10"],
            reverse=True,
        ),
        "output_npz": str(output_npz),
    }
    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
