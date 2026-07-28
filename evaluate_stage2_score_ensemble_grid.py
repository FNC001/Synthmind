#!/usr/bin/env python3
"""Tune a label-free convex ensemble of two saved cross-encoder score files.

The component files must describe identical candidate rows. Scores are
normalized within every query before interpolation, preventing one model's
logit scale from dominating the mixture. Validation selects the interpolation
and slate policy; that configuration must be frozen before final test use.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from training.family.evaluate_stage2_candidate_fusion import load_source
from training.family.evaluate_stage2_precursor_family_slate import precursor_family
from training.family.evaluate_stage2_score_ensemble import rowwise_zscore
from training.family.train_stage2_structured_energy_ranker import (
    best_trials_by_strategy,
    evaluate_grid,
    targets_from_matrix,
)
from training.family.train_stage2_within_family_variant_ranker import exact_metrics


SetKey = Tuple[int, ...]


def parse_mix_grid(raw: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in str(raw).split(",") if value.strip())
    if not values:
        raise ValueError("--mix_grid must contain at least one value")
    if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise ValueError("--mix_grid values must be finite and between 0 and 1")
    return tuple(dict.fromkeys(values))


def trial_key(trial: dict[str, object], mix: float) -> tuple[float, ...]:
    """Use the same metric priority as the underlying slate search."""

    return (
        float(trial["exact_hit@10"]),
        float(trial["exact_hit@5"]),
        float(trial["exact_hit@1"]),
        -float(trial["lost_hits_vs_base"]),
        -abs(float(mix) - 0.5),
    )


def write_candidates(path: Path, rows: Sequence[Sequence[SetKey]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(rows):
            handle.write(
                json.dumps(
                    {
                        "row_index": int(row_index),
                        "candidate_label_ids": [list(value) for value in row],
                    }
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--base_val_candidates", required=True)
    parser.add_argument("--score_npz_a", required=True)
    parser.add_argument("--score_npz_b", required=True)
    parser.add_argument(
        "--mix_grid",
        default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1",
        help="weight of score A; score B receives 1-weight",
    )
    parser.add_argument("--candidate_limit", type=int, default=100)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    val_y = np.asarray(
        np.load(input_dir / "val.npz", allow_pickle=True)["y_multi_hot"], dtype=np.float32
    )
    targets = targets_from_matrix(val_y)
    base_rows = load_source(
        args.base_val_candidates, len(targets), int(args.candidate_limit)
    )
    names = json.loads((input_dir / "precursor_names.json").read_text(encoding="utf-8"))
    label_families = [precursor_family(str(name)) for name in names]

    score_paths = [Path(args.score_npz_a).resolve(), Path(args.score_npz_b).resolve()]
    packs = [np.load(path, allow_pickle=False) for path in score_paths]
    try:
        spans_array = np.asarray(packs[0]["spans"], dtype=np.int64)
        spans = [(int(start), int(end)) for start, end in spans_array]
        expected_length = int(spans[-1][1]) if spans else 0
        vectors: List[np.ndarray] = []
        for path, pack in zip(score_paths, packs):
            current_spans = np.asarray(pack["spans"], dtype=np.int64)
            raw = np.asarray(pack["raw_scores"], dtype=np.float32)
            if not np.array_equal(current_spans, spans_array):
                raise ValueError(f"candidate spans differ for {path}")
            if len(raw) != expected_length:
                raise ValueError(f"score length differs for {path}")
            vectors.append(rowwise_zscore(raw, spans))
    finally:
        for pack in packs:
            pack.close()

    alpha_grid = (0.0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8)
    protected_grid = (0, 1, 3, 5, 7, 9, 10)
    minimum_gain_grid = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    candidate_windows = (20, 50, 100, 200, 400)

    best_key: tuple[float, ...] | None = None
    best_trial: dict[str, object] = {}
    best_rows: List[List[SetKey]] = []
    summaries: List[dict[str, object]] = []
    for mix in parse_mix_grid(args.mix_grid):
        combined = float(mix) * vectors[0] + (1.0 - float(mix)) * vectors[1]
        current, rows, trials = evaluate_grid(
            targets,
            base_rows,
            combined,
            spans,
            label_families,
            alpha_grid,
            protected_grid,
            minimum_gain_grid,
            candidate_windows,
        )
        current = {"mix_a": float(mix), "mix_b": 1.0 - float(mix), **current}
        summaries.append(
            {
                "mix_a": float(mix),
                "mix_b": 1.0 - float(mix),
                "best": current,
                "best_by_strategy": best_trials_by_strategy(trials),
            }
        )
        key = trial_key(current, float(mix))
        if best_key is None or key > best_key:
            best_key = key
            best_trial = current
            best_rows = rows

    report = {
        "protocol": "validation_score_ensemble_grid_formula_group_disjoint_no_test",
        "selection_note": "freeze the selected mix and slate policy before test evaluation",
        "sources": [str(path) for path in score_paths],
        "rows": int(len(targets)),
        "base": exact_metrics(targets, base_rows),
        "best": best_trial,
        "mix_summaries": summaries,
    }
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_candidates(Path(args.output_candidates_jsonl).resolve(), best_rows)
    print(json.dumps({"base": report["base"], "best": best_trial}, indent=2))


if __name__ == "__main__":
    main()
