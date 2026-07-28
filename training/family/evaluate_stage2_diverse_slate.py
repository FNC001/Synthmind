#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


SetKey = Tuple[int, ...]


def load_candidates(path: Path, n_rows: int, limit: int) -> List[List[SetKey]]:
    rows: List[List[SetKey]] = [[] for _ in range(n_rows)]
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            rows[int(record["row_index"])] = [
                tuple(sorted({int(value) for value in item}))
                for item in record["candidate_label_ids"][:limit]
            ]
    return rows


def jaccard(left: SetKey, right: SetKey) -> float:
    a, b = set(left), set(right)
    return len(a & b) / max(1, len(a | b))


def select_slate(
    candidates: Sequence[SetKey],
    slate_size: int,
    pool_size: int,
    diversity_weight: float,
    length_weight: float,
    rank_temperature: float,
) -> List[SetKey]:
    pool = list(dict.fromkeys(candidates[:pool_size]))
    if len(pool) <= slate_size:
        return pool
    selected_indices: List[int] = []
    remaining = set(range(len(pool)))
    while remaining and len(selected_indices) < slate_size:
        best_index = -1
        best_score = -math.inf
        selected_lengths = {len(pool[index]) for index in selected_indices}
        for index in remaining:
            rank_score = -rank_temperature * math.log1p(index)
            if selected_indices:
                novelty = min(1.0 - jaccard(pool[index], pool[chosen]) for chosen in selected_indices)
            else:
                novelty = 0.0
            length_novelty = float(len(pool[index]) not in selected_lengths)
            score = rank_score + diversity_weight * novelty + length_weight * length_novelty
            if score > best_score or (score == best_score and index < best_index):
                best_score = score
                best_index = index
        selected_indices.append(best_index)
        remaining.remove(best_index)
    selected = [pool[index] for index in selected_indices]
    selected_set = set(selected)
    return selected + [candidate for candidate in candidates if candidate not in selected_set]


def metrics(targets: Sequence[SetKey], rows: Sequence[Sequence[SetKey]]) -> Dict[str, float]:
    return {
        f"exact_hit@{k}": float(np.mean([target in set(row[:k]) for target, row in zip(targets, rows)]))
        for k in (1, 3, 5, 10, 20, 50, 100)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation search for diverse exact-set Top-K slates.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--candidate_source", required=True)
    parser.add_argument("--candidate_limit", type=int, default=500)
    parser.add_argument("--slate_size", type=int, default=10)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", required=True)
    args = parser.parse_args()
    input_dir = Path(args.input_dir).resolve()
    pack = np.load(input_dir / f"{args.split}.npz", allow_pickle=True)
    targets = [tuple(np.flatnonzero(row > 0.5).tolist()) for row in pack["y_multi_hot"]]
    candidates = load_candidates(Path(args.candidate_source).resolve(), len(targets), int(args.candidate_limit))
    trials = []
    best = None
    best_rows: List[List[SetKey]] = []
    for pool_size, diversity_weight, length_weight, rank_temperature in itertools.product(
        (25, 50, 100),
        (0.0, 0.1, 0.25, 0.5, 1.0),
        (0.0, 0.1, 0.25, 0.5),
        (1.0,),
    ):
        ranked = [
            select_slate(
                row,
                int(args.slate_size),
                min(pool_size, int(args.candidate_limit)),
                diversity_weight,
                length_weight,
                rank_temperature,
            )
            for row in candidates
        ]
        current_metrics = metrics(targets, ranked)
        trial = {
            "pool_size": pool_size,
            "diversity_weight": diversity_weight,
            "length_weight": length_weight,
            "rank_temperature": rank_temperature,
            **current_metrics,
        }
        trials.append(trial)
        if best is None or (trial["exact_hit@10"], trial["exact_hit@5"], trial["exact_hit@1"]) > (
            best["exact_hit@10"], best["exact_hit@5"], best["exact_hit@1"]
        ):
            best = trial
            best_rows = ranked
    assert best is not None
    report: Dict[str, Any] = {
        "protocol": f"{args.split}_formula_disjoint_exact_precursor_set_diverse_slate",
        "config": vars(args),
        "best": best,
        "n_trials": len(trials),
        "top_trials": sorted(trials, key=lambda row: (-row["exact_hit@10"], -row["exact_hit@5"]))[:20],
    }
    Path(args.output_json).resolve().write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with Path(args.output_candidates_jsonl).resolve().open("w", encoding="utf-8") as handle:
        for row, values in enumerate(best_rows):
            handle.write(json.dumps({"row_index": row, "candidate_label_ids": [list(value) for value in values]}) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
