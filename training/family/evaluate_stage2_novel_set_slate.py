#!/usr/bin/env python3
"""Validation search for Top-K slates with train-novel precursor-set coverage."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from training.family.evaluate_stage2_candidate_fusion import load_source


SetKey = Tuple[int, ...]


def ensure_quota(
    selected: List[SetKey],
    pool: Sequence[SetKey],
    protected_prefix: int,
    quota: int,
    qualifies,
) -> List[SetKey]:
    result = list(selected)
    current = sum(bool(qualifies(value)) for value in result)
    if current >= int(quota):
        return result
    available = [value for value in pool if value not in set(result) and qualifies(value)]
    for candidate in available:
        replace = next(
            (
                index
                for index in range(len(result) - 1, int(protected_prefix) - 1, -1)
                if not qualifies(result[index])
            ),
            None,
        )
        if replace is None:
            break
        result[replace] = candidate
        current += 1
        if current >= int(quota):
            break
    return result


def novel_slate(
    candidates: Sequence[SetKey],
    train_sets: set[SetKey],
    train_seen_labels: np.ndarray,
    slate_size: int,
    pool_size: int,
    protected_prefix: int,
    novel_set_quota: int,
    unseen_label_quota: int,
) -> List[SetKey]:
    unique = list(dict.fromkeys(candidates))
    selected = unique[: int(slate_size)]
    pool = unique[: int(pool_size)]
    selected = ensure_quota(
        selected,
        pool,
        int(protected_prefix),
        int(novel_set_quota),
        lambda value: value not in train_sets,
    )
    selected = ensure_quota(
        selected,
        pool,
        int(protected_prefix),
        int(unseen_label_quota),
        lambda value: any(not bool(train_seen_labels[int(label)]) for label in value),
    )
    selected_set = set(selected)
    return selected + [value for value in unique if value not in selected_set]


def metrics(targets: Sequence[SetKey], rows: Sequence[Sequence[SetKey]]) -> Dict[str, float]:
    return {
        f"exact_hit@{k}": float(
            np.mean([target in set(row[:k]) for target, row in zip(targets, rows)])
        )
        for k in (1, 3, 5, 10, 20, 50, 100)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Search train-novel exact-set Top-K slate quotas.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--candidate_source", required=True)
    parser.add_argument("--candidate_limit", type=int, default=100)
    parser.add_argument("--slate_size", type=int, default=10)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    train_y = np.asarray(
        np.load(input_dir / "train.npz", allow_pickle=True)["y_multi_hot"], dtype=np.float32
    )
    query_y = np.asarray(
        np.load(input_dir / f"{args.split}.npz", allow_pickle=True)["y_multi_hot"],
        dtype=np.float32,
    )
    train_sets = {
        tuple(np.flatnonzero(row > 0.5).tolist()) for row in train_y
    }
    train_seen_labels = train_y.sum(axis=0) > 0.5
    targets = [tuple(np.flatnonzero(row > 0.5).tolist()) for row in query_y]
    candidates = load_source(
        args.candidate_source, len(targets), int(args.candidate_limit)
    )

    trials = []
    best = None
    best_rows: List[List[SetKey]] = []
    for pool_size, protected_prefix, novel_set_quota, unseen_label_quota in itertools.product(
        (20, 50, 100),
        (0, 1, 3, 5, 7, 9),
        tuple(range(0, int(args.slate_size) + 1)),
        (0, 1, 2),
    ):
        if int(protected_prefix) + int(novel_set_quota) > int(args.slate_size) + 4:
            continue
        ranked = [
            novel_slate(
                row,
                train_sets,
                train_seen_labels,
                int(args.slate_size),
                min(int(pool_size), int(args.candidate_limit)),
                int(protected_prefix),
                int(novel_set_quota),
                int(unseen_label_quota),
            )
            for row in candidates
        ]
        current = metrics(targets, ranked)
        trial = {
            "pool_size": int(pool_size),
            "protected_prefix": int(protected_prefix),
            "novel_set_quota": int(novel_set_quota),
            "unseen_label_quota": int(unseen_label_quota),
            **current,
        }
        trials.append(trial)
        if best is None or (
            trial["exact_hit@10"], trial["exact_hit@5"], trial["exact_hit@1"]
        ) > (best["exact_hit@10"], best["exact_hit@5"], best["exact_hit@1"]):
            best = trial
            best_rows = ranked
    assert best is not None
    report = {
        "protocol": f"{args.split}_train_observable_novel_set_quota_slate_search",
        "config": vars(args),
        "train_unique_sets": int(len(train_sets)),
        "train_seen_labels": int(train_seen_labels.sum()),
        "best": best,
        "n_trials": int(len(trials)),
        "top_trials": sorted(
            trials,
            key=lambda row: (-row["exact_hit@10"], -row["exact_hit@5"], -row["exact_hit@1"]),
        )[:30],
    }
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    candidate_output = Path(args.output_candidates_jsonl).expanduser().resolve()
    candidate_output.parent.mkdir(parents=True, exist_ok=True)
    with candidate_output.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(best_rows):
            handle.write(
                json.dumps(
                    {"row_index": row_index, "candidate_label_ids": [list(value) for value in row]}
                )
                + "\n"
            )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
