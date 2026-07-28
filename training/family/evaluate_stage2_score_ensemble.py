#!/usr/bin/env python3
"""Ensemble precursor candidate energies from independently trained checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from training.family.evaluate_stage2_candidate_fusion import load_source
from training.family.evaluate_stage2_precursor_family_slate import precursor_family
from training.family.train_stage2_structured_energy_ranker import (
    best_trials_by_strategy,
    evaluate_grid,
    targets_from_matrix,
)
from training.family.train_stage2_within_family_variant_ranker import exact_metrics


SetKey = Tuple[int, ...]


def candidate_fingerprint(candidate: SetKey) -> np.uint64:
    payload = ",".join(str(int(value)) for value in candidate).encode("ascii")
    return np.uint64(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big"))


def candidate_fingerprints(rows: Sequence[Sequence[SetKey]]) -> np.ndarray:
    return np.asarray(
        [candidate_fingerprint(candidate) for row in rows for candidate in row],
        dtype=np.uint64,
    )


def rowwise_zscore(scores: np.ndarray, spans: Sequence[Tuple[int, int]]) -> np.ndarray:
    output = np.asarray(scores, dtype=np.float32).copy()
    for start, end in spans:
        values = output[int(start) : int(end)]
        scale = max(float(values.std(dtype=np.float64)), 1e-6)
        output[int(start) : int(end)] = (values - float(values.mean(dtype=np.float64))) / scale
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--base_val_candidates", required=True)
    parser.add_argument("--score_npz", action="append", default=[])
    parser.add_argument("--weights", default="")
    parser.add_argument("--candidate_limit", type=int, default=100)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", required=True)
    args = parser.parse_args()
    if not args.score_npz:
        parser.error("at least one --score_npz is required")

    input_dir = Path(args.input_dir).resolve()
    val_y = np.asarray(
        np.load(input_dir / "val.npz", allow_pickle=True)["y_multi_hot"], dtype=np.float32
    )
    targets = targets_from_matrix(val_y)
    base_rows = load_source(
        args.base_val_candidates, len(targets), int(args.candidate_limit)
    )
    names = [
        str(value)
        for value in json.loads((input_dir / "precursor_names.json").read_text(encoding="utf-8"))
    ]
    label_families = [precursor_family(name) for name in names]

    packs = [np.load(Path(path).resolve(), allow_pickle=False) for path in args.score_npz]
    try:
        spans = np.asarray(packs[0]["spans"], dtype=np.int64)
        expected_length = int(spans[-1, 1]) if len(spans) else 0
        vectors: List[np.ndarray] = []
        for path, pack in zip(args.score_npz, packs):
            current_spans = np.asarray(pack["spans"], dtype=np.int64)
            values = np.asarray(pack["raw_scores"], dtype=np.float32)
            if not np.array_equal(current_spans, spans):
                raise ValueError(f"candidate spans differ for {path}")
            if len(values) != expected_length:
                raise ValueError(f"score length differs for {path}")
            if "candidate_hashes" not in pack.files:
                raise ValueError(f"candidate identity hashes are missing for {path}")
            hashes = np.asarray(pack["candidate_hashes"], dtype=np.uint64)
            reference_hashes = np.asarray(packs[0]["candidate_hashes"], dtype=np.uint64)
            if not np.array_equal(hashes, reference_hashes):
                raise ValueError(f"candidate identities or order differ for {path}")
            vectors.append(
                rowwise_zscore(values, [(int(start), int(end)) for start, end in spans])
            )
    finally:
        for pack in packs:
            pack.close()

    if str(args.weights):
        weights = np.asarray([float(value) for value in str(args.weights).split(",")])
        if len(weights) != len(vectors):
            raise ValueError("--weights must match the number of --score_npz inputs")
    else:
        weights = np.ones(len(vectors), dtype=np.float64)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
        raise ValueError("ensemble weights must be finite with a positive sum")
    weights = weights / float(weights.sum())
    combined = np.zeros(expected_length, dtype=np.float32)
    for weight, values in zip(weights, vectors):
        combined += float(weight) * values

    alpha_grid = (0.0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8)
    protected_grid = (0, 1, 3, 5, 7, 9, 10)
    minimum_gain_grid = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    best, ranked, trials = evaluate_grid(
        targets,
        base_rows,
        combined,
        [(int(start), int(end)) for start, end in spans],
        label_families,
        alpha_grid,
        protected_grid,
        minimum_gain_grid,
    )
    report = {
        "protocol": "validation_score_ensemble_formula_group_disjoint_no_test",
        "sources": [str(Path(path).resolve()) for path in args.score_npz],
        "weights": weights.tolist(),
        "rows": int(len(targets)),
        "base": exact_metrics(targets, base_rows),
        "best": best,
        "best_by_strategy": best_trials_by_strategy(trials),
    }
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_candidates = Path(args.output_candidates_jsonl).resolve()
    output_candidates.parent.mkdir(parents=True, exist_ok=True)
    with output_candidates.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(ranked):
            handle.write(
                json.dumps(
                    {"row_index": row_index, "candidate_label_ids": [list(value) for value in row]}
                )
                + "\n"
            )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
