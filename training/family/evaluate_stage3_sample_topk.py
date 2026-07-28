#!/usr/bin/env python3
"""Evaluate ranked Top-K condition tuples from generative Stage3 samples."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


TOP_K = (1, 3, 5, 10, 20, 50)


def ranked_buckets(
    continuous: np.ndarray,
    discrete: np.ndarray,
    temperature_bin: float,
    time_bin: float,
) -> list[tuple[float, float, int, int, int]]:
    buckets: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    for index in range(len(continuous)):
        key = (
            int(np.floor(float(continuous[index, 0]) / temperature_bin)),
            int(np.floor(float(continuous[index, 1]) / time_bin)),
            int(discrete[index, 0]),
            int(discrete[index, 1]),
        )
        buckets[key].append(index)
    candidates = []
    for key, indices in buckets.items():
        values = continuous[np.asarray(indices, dtype=np.int64)]
        candidates.append(
            (
                float(np.median(values[:, 0])),
                float(np.median(values[:, 1])),
                int(key[2]),
                int(key[3]),
                int(len(indices)),
            )
        )
    return sorted(
        candidates,
        key=lambda value: (-value[4], value[0], value[1], value[2], value[3]),
    )


def candidate_hit(
    candidate: tuple[float, float, int, int, int],
    row: int,
    continuous_true: np.ndarray,
    continuous_mask: np.ndarray,
    discrete_true: np.ndarray,
    discrete_mask: np.ndarray,
    temperature_tolerance: float,
    time_tolerance: float,
    method_inclusive: bool,
) -> bool:
    if continuous_mask[row, 0] and abs(candidate[0] - continuous_true[row, 0]) > temperature_tolerance:
        return False
    if continuous_mask[row, 1] and abs(candidate[1] - continuous_true[row, 1]) > time_tolerance:
        return False
    if discrete_mask[row, 0] and candidate[2] != int(discrete_true[row, 0]):
        return False
    if method_inclusive and discrete_mask[row, 1] and candidate[3] != int(discrete_true[row, 1]):
        return False
    return True


def metrics(
    rows: list[list[tuple[float, float, int, int, int]]],
    indices: np.ndarray,
    continuous_true: np.ndarray,
    continuous_mask: np.ndarray,
    discrete_true: np.ndarray,
    discrete_mask: np.ndarray,
    temperature_tolerance: float,
    time_tolerance: float,
    method_inclusive: bool,
) -> dict[str, float | int]:
    result: dict[str, float | int] = {"n": int(len(indices))}
    for k in TOP_K:
        hits = []
        for row in indices:
            hits.append(
                any(
                    candidate_hit(
                        candidate,
                        int(row),
                        continuous_true,
                        continuous_mask,
                        discrete_true,
                        discrete_mask,
                        temperature_tolerance,
                        time_tolerance,
                        method_inclusive,
                    )
                    for candidate in rows[int(row)][:k]
                )
            )
        result[f"hit@{k}"] = float(np.mean(hits)) if hits else float("nan")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Top-K condition tuple evaluation from Stage3 generated sample NPZ."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--predictions_npz", required=True)
    parser.add_argument("--temperature_bin", type=float, default=100.0)
    parser.add_argument("--time_bin", type=float, default=24.0)
    parser.add_argument("--temperature_tolerance", type=float, default=200.0)
    parser.add_argument("--time_tolerance", type=float, default=48.0)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", default="")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    with np.load(input_dir / f"{args.split}.npz", allow_pickle=True) as pack:
        continuous_true = np.asarray(pack["y_cond_continuous_raw"], dtype=np.float64)
        continuous_mask = np.asarray(pack["y_cond_continuous_mask"], dtype=np.float32) > 0.5
        discrete_true = np.asarray(pack["y_cond_discrete"], dtype=np.int64)
        discrete_mask = np.asarray(pack["y_cond_discrete_mask"], dtype=np.float32) > 0.5
    with np.load(Path(args.predictions_npz).resolve(), allow_pickle=False) as predictions:
        continuous = np.asarray(predictions["continuous_samples"], dtype=np.float64)
        discrete = np.asarray(predictions["discrete_samples"], dtype=np.int64)
    if continuous.shape[:2] != discrete.shape[:2] or continuous.shape[0] != len(continuous_true):
        raise ValueError("prediction sample shapes do not match the requested split")

    ranked = [
        ranked_buckets(
            continuous[row],
            discrete[row],
            float(args.temperature_bin),
            float(args.time_bin),
        )
        for row in range(len(continuous))
    ]
    all_indices = np.arange(len(ranked), dtype=np.int64)
    strict_indices = np.flatnonzero(continuous_mask[:, 0] & continuous_mask[:, 1])
    report: dict[str, Any] = {
        "protocol": f"{args.split}_generated_sample_frequency_bucket_condition_topk",
        "config": vars(args),
        "rows": int(len(ranked)),
        "samples_per_row": int(continuous.shape[1]),
        "mean_unique_condition_buckets": float(np.mean([len(row) for row in ranked])),
        "missing_aware_relaxed": metrics(
            ranked,
            all_indices,
            continuous_true,
            continuous_mask,
            discrete_true,
            discrete_mask,
            float(args.temperature_tolerance),
            float(args.time_tolerance),
            False,
        ),
        "missing_aware_method_inclusive": metrics(
            ranked,
            all_indices,
            continuous_true,
            continuous_mask,
            discrete_true,
            discrete_mask,
            float(args.temperature_tolerance),
            float(args.time_tolerance),
            True,
        ),
        "strict_comparable_relaxed": metrics(
            ranked,
            strict_indices,
            continuous_true,
            continuous_mask,
            discrete_true,
            discrete_mask,
            float(args.temperature_tolerance),
            float(args.time_tolerance),
            False,
        ),
        "strict_comparable_method_inclusive": metrics(
            ranked,
            strict_indices,
            continuous_true,
            continuous_mask,
            discrete_true,
            discrete_mask,
            float(args.temperature_tolerance),
            float(args.time_tolerance),
            True,
        ),
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_candidates_jsonl:
        candidate_path = Path(args.output_candidates_jsonl).resolve()
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        with candidate_path.open("w", encoding="utf-8") as handle:
            for row_index, candidates in enumerate(ranked):
                handle.write(
                    json.dumps(
                        {
                            "row_index": row_index,
                            "candidates": [
                                {
                                    "temperature_c": value[0],
                                    "time_h": value[1],
                                    "atmosphere_coarse": value[2],
                                    "reaction_method": value[3],
                                    "sample_count": value[4],
                                }
                                for value in candidates
                            ],
                        }
                    )
                    + "\n"
                )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
