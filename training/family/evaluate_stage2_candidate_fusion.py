#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


SetKey = Tuple[int, ...]


def load_source(path: str, n_rows: int, source_limit: int = 0) -> List[List[SetKey]]:
    rows: List[List[SetKey]] = [[] for _ in range(n_rows)]
    with Path(path).resolve().open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            values = item["candidate_label_ids"]
            if source_limit > 0:
                values = values[:source_limit]
            rows[int(item["row_index"])] = [
                tuple(sorted({int(value) for value in candidate}))
                for candidate in values
            ]
    return rows


def fuse_row(
    rows: Sequence[Sequence[SetKey]],
    weights: Sequence[float],
    rrf_constant: float,
) -> List[SetKey]:
    scores: Dict[SetKey, float] = {}
    best_rank: Dict[SetKey, int] = {}
    for source_index, candidates in enumerate(rows):
        weight = float(weights[source_index])
        for rank, candidate in enumerate(candidates, start=1):
            scores[candidate] = scores.get(candidate, 0.0) + weight / (rrf_constant + rank)
            best_rank[candidate] = min(best_rank.get(candidate, rank), rank)
    return sorted(scores, key=lambda key: (-scores[key], best_rank[key], key))


def fuse_row_topk(
    rows: Sequence[Sequence[SetKey]],
    weights: Sequence[float],
    rrf_constant: float,
    k: int,
) -> List[SetKey]:
    """Return the exact RRF prefix without sorting the full candidate union."""
    if k <= 0:
        return []
    scores: Dict[SetKey, float] = {}
    best_rank: Dict[SetKey, int] = {}
    for source_index, candidates in enumerate(rows):
        weight = float(weights[source_index])
        for rank, candidate in enumerate(candidates, start=1):
            scores[candidate] = scores.get(candidate, 0.0) + weight / (rrf_constant + rank)
            best_rank[candidate] = min(best_rank.get(candidate, rank), rank)
    sort_key = lambda key: (-scores[key], best_rank[key], key)
    if k >= len(scores):
        return sorted(scores, key=sort_key)
    return heapq.nsmallest(k, scores, key=sort_key)


def topk(targets: Sequence[SetKey], candidates: Sequence[Sequence[SetKey]], k: int) -> float:
    return float(np.mean([target in set(row[:k]) for target, row in zip(targets, candidates)]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation grid search for reciprocal-rank fusion of Stage2 candidates.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--candidate_sources", nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", default="")
    parser.add_argument("--source_limit", type=int, default=0)
    parser.add_argument(
        "--fixed_rrf_constant",
        type=float,
        default=None,
        help="Use a fixed, label-free RRF constant instead of searching on the requested split.",
    )
    parser.add_argument(
        "--fixed_weights",
        nargs="+",
        type=float,
        default=[],
        help="Optional fixed source weights; defaults to equal weights.",
    )
    parser.add_argument("--output_limit", type=int, default=0)
    args = parser.parse_args()
    input_dir = Path(args.input_dir).resolve()
    pack = np.load(input_dir / f"{args.split}.npz", allow_pickle=True)
    targets = [tuple(np.flatnonzero(row > 0.5).tolist()) for row in pack["y_multi_hot"]]
    sources = [load_source(path, len(targets), args.source_limit) for path in args.candidate_sources]
    if args.fixed_weights and len(args.fixed_weights) != len(sources):
        parser.error("--fixed_weights must contain one value per candidate source")
    fixed_weights = list(args.fixed_weights) if args.fixed_weights else [1.0] * len(sources)
    constants = (1.0, 5.0, 10.0, 20.0, 50.0, 100.0)
    weight_values = (0.5, 1.0, 2.0)
    trials = []
    best = None
    grid = (
        [(float(args.fixed_rrf_constant), tuple(fixed_weights))]
        if args.fixed_rrf_constant is not None
        else [
            (constant, weights)
            for constant in constants
            for weights in itertools.product(weight_values, repeat=len(sources))
            if weights[0] == 1.0
        ]
    )
    for constant, weights in grid:
        fused = [
            fuse_row([source[row_index] for source in sources], weights, constant)
            for row_index in range(len(targets))
        ]
        metrics = {
            f"exact_hit@{k}": topk(targets, fused, k)
            for k in (1, 3, 5, 10, 20, 50, 100)
        }
        trial = {"rrf_constant": constant, "weights": list(weights), **metrics}
        trials.append(trial)
        if best is None or (trial["exact_hit@10"], trial["exact_hit@50"]) > (
            best["exact_hit@10"], best["exact_hit@50"]
        ):
            best = trial
    assert best is not None
    best_candidates = [
        fuse_row(
            [source[row_index] for source in sources],
            best["weights"],
            best["rrf_constant"],
        )
        for row_index in range(len(targets))
    ]
    if int(args.output_limit) > 0:
        best_candidates = [row[: int(args.output_limit)] for row in best_candidates]
    report: Dict[str, Any] = {
        "protocol": f"{args.split}_formula_disjoint_exact_precursor_set",
        "sources": args.candidate_sources,
        "selection_mode": "fixed_label_free" if args.fixed_rrf_constant is not None else "split_grid_search",
        "best": best,
        "oracle_union_recall": float(
            np.mean([target in set(row) for target, row in zip(targets, best_candidates)])
        ),
        "mean_union_candidates": float(np.mean([len(row) for row in best_candidates])),
        "n_trials": len(trials),
        "top_trials": sorted(trials, key=lambda row: (-row["exact_hit@10"], -row["exact_hit@50"]))[:20],
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_candidates_jsonl:
        candidate_output = Path(args.output_candidates_jsonl).resolve()
        with candidate_output.open("w", encoding="utf-8") as handle:
            for row_index, candidates in enumerate(best_candidates):
                handle.write(json.dumps({"row_index": row_index, "candidate_label_ids": [list(value) for value in candidates]}) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
