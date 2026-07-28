#!/usr/bin/env python3
"""Build a label-free candidate union while preserving a trusted base prefix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from training.family.evaluate_stage2_candidate_fusion import load_source


SetKey = Tuple[int, ...]


def protected_union_row(
    base: Sequence[SetKey],
    experts: Sequence[Sequence[SetKey]],
    protected_prefix: int,
    expert_limit: int,
    rrf_constant: float,
    output_limit: int,
) -> List[SetKey]:
    """Keep the base prefix, then rank expert outsiders before base fallbacks."""
    protected = list(dict.fromkeys(base[: int(protected_prefix)]))
    protected_set = set(protected)
    scores: Dict[SetKey, float] = {}
    best_rank: Dict[SetKey, int] = {}
    source_count: Dict[SetKey, int] = {}
    for row in experts:
        seen: set[SetKey] = set()
        for rank, candidate in enumerate(row[: int(expert_limit)], start=1):
            if not candidate or candidate in protected_set or candidate in seen:
                continue
            seen.add(candidate)
            scores[candidate] = scores.get(candidate, 0.0) + 1.0 / (
                float(rrf_constant) + rank
            )
            best_rank[candidate] = min(best_rank.get(candidate, rank), rank)
            source_count[candidate] = source_count.get(candidate, 0) + 1
    outsiders = sorted(
        scores,
        key=lambda candidate: (
            -scores[candidate],
            -source_count[candidate],
            best_rank[candidate],
            candidate,
        ),
    )
    output: List[SetKey] = []
    seen: set[SetKey] = set()
    for candidate in [*protected, *outsiders, *base]:
        if candidate and candidate not in seen:
            output.append(candidate)
            seen.add(candidate)
        if int(output_limit) > 0 and len(output) >= int(output_limit):
            break
    return output


def exact_metrics(
    targets: Sequence[SetKey], rows: Sequence[Sequence[SetKey]]
) -> Dict[str, float]:
    return {
        f"exact_hit@{cutoff}": float(
            np.mean([target in set(row[:cutoff]) for target, row in zip(targets, rows)])
        )
        for cutoff in (1, 3, 5, 10, 20, 50, 100)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--base_candidates", required=True)
    parser.add_argument("--expert", action="append", default=[], help="Repeat candidate path")
    parser.add_argument(
        "--expert_family",
        action="append",
        default=[],
        help=(
            "Optional comma-separated family filter aligned with each --expert. "
            "When supplied, an expert contributes only to matching rows."
        ),
    )
    parser.add_argument("--protected_prefix", type=int, default=10)
    parser.add_argument("--expert_limit", type=int, default=10)
    parser.add_argument("--rrf_constant", type=float, default=10.0)
    parser.add_argument("--output_limit", type=int, default=200)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", required=True)
    args = parser.parse_args()
    if not args.expert:
        parser.error("at least one --expert source is required")
    if args.expert_family and len(args.expert_family) != len(args.expert):
        parser.error("--expert_family must be omitted or repeated once per --expert")
    if int(args.output_limit) and int(args.output_limit) < int(args.protected_prefix):
        parser.error("--output_limit must be zero or at least --protected_prefix")

    input_dir = Path(args.input_dir).resolve()
    pack = np.load(input_dir / f"{args.split}.npz", allow_pickle=True)
    targets = [tuple(np.flatnonzero(row > 0.5).tolist()) for row in pack["y_multi_hot"]]
    meta = pd.read_csv(input_dir / f"{args.split}_meta.csv", low_memory=False)
    row_families = meta["family_signature_primary"].fillna("UNK").astype(str).to_numpy()
    base_rows = load_source(args.base_candidates, len(targets), int(args.output_limit))
    expert_rows = [
        load_source(path, len(targets), int(args.expert_limit)) for path in args.expert
    ]
    expert_family_sets = (
        [
            {family.strip() for family in value.split(",") if family.strip()}
            for value in args.expert_family
        ]
        if args.expert_family
        else [set() for _ in args.expert]
    )
    rows = [
        protected_union_row(
            base_rows[row_index],
            [
                source[row_index]
                for source, allowed in zip(expert_rows, expert_family_sets)
                if not allowed or str(row_families[row_index]) in allowed
            ],
            int(args.protected_prefix),
            int(args.expert_limit),
            float(args.rrf_constant),
            int(args.output_limit),
        )
        for row_index in range(len(targets))
    ]
    base_metrics = exact_metrics(targets, base_rows)
    union_metrics = exact_metrics(targets, rows)
    base_top10 = np.asarray(
        [target in set(row[:10]) for target, row in zip(targets, base_rows)], dtype=bool
    )
    union_top10 = np.asarray(
        [target in set(row[:10]) for target, row in zip(targets, rows)], dtype=bool
    )
    report = {
        "protocol": f"{args.split}_label_free_protected_candidate_union",
        "config": vars(args),
        "base": base_metrics,
        "union": union_metrics,
        "top10_preserved": bool(np.array_equal(base_top10, union_top10)),
        "oracle_union_recall": float(
            np.mean([target in set(row) for target, row in zip(targets, rows)])
        ),
        "mean_candidates": float(np.mean([len(row) for row in rows])),
    }
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_candidates = Path(args.output_candidates_jsonl).resolve()
    output_candidates.parent.mkdir(parents=True, exist_ok=True)
    with output_candidates.open("w", encoding="utf-8") as handle:
        for row_index, candidates in enumerate(rows):
            handle.write(
                json.dumps(
                    {
                        "row_index": row_index,
                        "candidate_label_ids": [list(candidate) for candidate in candidates],
                    }
                )
                + "\n"
            )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
