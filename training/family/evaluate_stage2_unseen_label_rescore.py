#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.family.evaluate_stage2_candidate_fusion import load_source  # noqa: E402


SetKey = Tuple[int, ...]


def unseen_rescore_row(
    candidates: Sequence[SetKey],
    train_seen: np.ndarray,
    unseen_bonus: float,
) -> List[SetKey]:
    """Promote candidates containing vocabulary labels absent from train.

    The prior is observable at inference: the vocabulary is frozen from the
    database, while ``train_seen`` is computed from the training partition
    only.  Original candidate rank remains the tie breaker.
    """
    scored = []
    for rank, candidate in enumerate(candidates, start=1):
        unseen_count = sum(not bool(train_seen[int(label)]) for label in candidate)
        score = -math.log1p(rank - 1) + float(unseen_bonus) * unseen_count
        scored.append((candidate, score, rank))
    return [candidate for candidate, _, _ in sorted(scored, key=lambda row: (-row[1], row[2], row[0]))]


def exact_metrics(targets: Sequence[SetKey], rows: Sequence[Sequence[SetKey]]) -> Dict[str, float]:
    return {
        f"exact_hit@{k}": float(np.mean([target in set(row[:k]) for target, row in zip(targets, rows)]))
        for k in (1, 3, 5, 10, 20, 50, 100, 500)
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rescore Stage2 candidates using train-observable unseen-label evidence."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--candidate_limit", type=int, default=500)
    parser.add_argument("--unseen_bonus", type=float, required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    train_y = np.asarray(np.load(input_dir / "train.npz", allow_pickle=True)["y_multi_hot"])
    split_y = np.asarray(np.load(input_dir / f"{args.split}.npz", allow_pickle=True)["y_multi_hot"])
    train_seen = np.asarray(train_y.sum(axis=0) > 0)
    targets = [tuple(np.flatnonzero(row > 0.5).tolist()) for row in split_y]
    source = load_source(args.candidates, len(targets), int(args.candidate_limit))
    ranked = [
        unseen_rescore_row(row, train_seen, float(args.unseen_bonus))
        for row in source
    ]
    report = {
        "protocol": f"{args.split}_formula_disjoint_train_observable_unseen_label_rescore",
        "source": args.candidates,
        "candidate_limit": int(args.candidate_limit),
        "unseen_bonus": float(args.unseen_bonus),
        "train_seen_labels": int(train_seen.sum()),
        "vocabulary_labels": int(len(train_seen)),
        "overall": exact_metrics(targets, ranked),
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with Path(args.output_candidates_jsonl).resolve().open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(ranked):
            handle.write(json.dumps({
                "row_index": row_index,
                "candidate_label_ids": [list(candidate) for candidate in row],
            }) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
