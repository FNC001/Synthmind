#!/usr/bin/env python3
"""Re-evaluate saved Stage-2 cross-encoder scores over wider candidate windows.

The expensive transformer forward pass is independent of the final protected
reranking grid.  This utility makes that separation explicit so a validation
window can be enlarged without silently rescoring or touching the frozen test
split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from training.family.evaluate_stage2_candidate_fusion import load_source
from training.family.evaluate_stage2_precursor_family_slate import precursor_family
from training.family.train_stage2_matscibert_cross_encoder import (
    evaluate_specialist_grid,
    parse_family_filter,
)
from training.family.train_stage2_structured_energy_ranker import best_trials_by_strategy
from training.family.train_stage2_within_family_variant_ranker import exact_metrics


def parse_numbers(raw: str, cast) -> tuple:
    return tuple(cast(value.strip()) for value in str(raw).split(",") if value.strip())


def write_candidates(path: Path, rows: Sequence[Sequence[tuple[int, ...]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row_index, candidates in enumerate(rows):
            handle.write(
                json.dumps(
                    {
                        "row_index": int(row_index),
                        "candidate_label_ids": [list(candidate) for candidate in candidates],
                    }
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--base_candidates", required=True)
    parser.add_argument("--scores_npz", required=True)
    parser.add_argument("--families", required=True)
    parser.add_argument("--candidate_limit", type=int, default=100)
    parser.add_argument("--alphas", default="0,0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6")
    parser.add_argument("--protected_prefixes", default="0,1,3,5,7,9,10")
    parser.add_argument("--minimum_gains", default="0,0.05,0.1,0.25,0.5,1,2,4,8")
    parser.add_argument("--candidate_windows", default="20,50,100,200,500,1000,10000")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    pack = np.load(input_dir / "val.npz", allow_pickle=True)
    targets = [tuple(np.flatnonzero(row > 0.5).tolist()) for row in pack["y_multi_hot"]]
    meta = pd.read_csv(input_dir / "val_meta.csv", low_memory=False)
    row_families = meta["family_signature_primary"].fillna("UNK").astype(str).to_numpy()
    selected = parse_family_filter(args.families)
    active_indices = np.asarray(
        [index for index, family in enumerate(row_families) if str(family) in selected],
        dtype=np.int64,
    )
    if not len(active_indices):
        raise ValueError(f"no validation rows match --families={args.families!r}")

    base_rows = load_source(args.base_candidates, len(targets), int(args.candidate_limit))
    scores_pack = np.load(args.scores_npz, allow_pickle=True)
    raw_scores = np.asarray(scores_pack["raw_scores"], dtype=np.float32)
    spans = [tuple(map(int, values)) for values in np.asarray(scores_pack["spans"])]
    expected = sum(len(row) for row in base_rows)
    if expected != len(raw_scores) or spans[-1][1] != len(raw_scores):
        raise ValueError(
            f"candidate/score mismatch: candidates={expected}, scores={len(raw_scores)}, "
            f"span_end={spans[-1][1]}"
        )

    names = json.loads((input_dir / "precursor_names.json").read_text(encoding="utf-8"))
    label_families = [precursor_family(str(name)) for name in names]
    alphas = parse_numbers(args.alphas, float)
    protected = parse_numbers(args.protected_prefixes, int)
    minimum_gains = parse_numbers(args.minimum_gains, float)
    candidate_windows = parse_numbers(args.candidate_windows, int)
    best, rows, trials = evaluate_specialist_grid(
        targets,
        base_rows,
        raw_scores,
        spans,
        label_families,
        active_indices,
        alphas,
        protected,
        minimum_gains,
        candidate_windows,
    )
    report = {
        "protocol": "saved_cross_encoder_fixed_validation_wide_window_rerank",
        "config": vars(args),
        "rows": int(len(targets)),
        "selected_rows": int(len(active_indices)),
        "selected_families": sorted(selected),
        "base": exact_metrics(targets, base_rows),
        "best": best,
        "best_by_strategy": best_trials_by_strategy(trials),
        "trials": trials,
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_candidates(Path(args.output_candidates_jsonl).resolve(), rows)
    print(
        json.dumps(
            {
                "base": report["base"],
                "best": report["best"],
                "best_by_strategy": report["best_by_strategy"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
