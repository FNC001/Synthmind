#!/usr/bin/env python3
"""Attach stable candidate identities to a legacy Stage-2 score archive."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from training.family.evaluate_stage2_candidate_fusion import load_source
from training.family.evaluate_stage2_score_ensemble import candidate_fingerprints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores_npz", required=True)
    parser.add_argument("--candidate_jsonl", required=True)
    parser.add_argument("--output_npz", required=True)
    args = parser.parse_args()

    source = Path(args.scores_npz).resolve()
    with np.load(source, allow_pickle=False) as pack:
        arrays = {key: np.asarray(pack[key]) for key in pack.files}
    spans = np.asarray(arrays["spans"], dtype=np.int64)
    rows = load_source(args.candidate_jsonl, len(spans), 0)
    expected_spans = []
    offset = 0
    for row in rows:
        expected_spans.append((offset, offset + len(row)))
        offset += len(row)
    if not np.array_equal(spans, np.asarray(expected_spans, dtype=np.int64)):
        raise ValueError("candidate rows do not match score spans")
    if len(arrays["raw_scores"]) != offset:
        raise ValueError("candidate count does not match score vector")
    arrays["candidate_hashes"] = candidate_fingerprints(rows)
    output = Path(args.output_npz).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    print({"pairs": int(offset), "output": str(output)})


if __name__ == "__main__":
    main()
