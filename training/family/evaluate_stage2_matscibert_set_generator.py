#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.family.train_stage2_oof_candidate_stacker import (  # noqa: E402
    load_matsci_multilabel_scores,
)


SetKey = Tuple[int, ...]


def generate_candidate_sets(
    logits: np.ndarray,
    top_labels: int,
    max_set_length: int,
    limit: int,
) -> tuple[List[SetKey], List[float]]:
    label_order = np.argsort(-np.asarray(logits, dtype=np.float32), kind="stable")[: int(top_labels)]
    candidates = [
        tuple(sorted(int(label_order[position]) for position in positions))
        for length in range(1, int(max_set_length) + 1)
        for positions in itertools.combinations(range(len(label_order)), length)
    ]
    scored = []
    for candidate in candidates:
        values = logits[np.asarray(candidate, dtype=np.int64)]
        # The weakest required precursor is the bottleneck for an exact set.
        # Mean logit and shorter length only break ties deterministically.
        score = float(values.min())
        scored.append((score, float(values.mean()), -len(candidate), candidate))
    scored.sort(key=lambda value: (-value[0], -value[1], -value[2], value[3]))
    selected = scored[: int(limit)]
    return [value[3] for value in selected], [value[0] for value in selected]


def exact_metrics(targets: Sequence[SetKey], rows: Sequence[Sequence[SetKey]]) -> dict[str, float]:
    return {
        f"exact_hit@{k}": float(np.mean([target in set(row[:k]) for target, row in zip(targets, rows)]))
        for k in (1, 3, 5, 10, 20, 50, 100, 500, 1000)
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate exact precursor-set candidates from train-only MatSciBERT label logits."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--score_cache", required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--top_labels", type=int, default=12)
    parser.add_argument("--max_set_length", type=int, default=4)
    parser.add_argument("--limit", type=int, default=800)
    parser.add_argument("--evaluate_labels", action="store_true")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    precursor_names = json.loads((input_dir / "precursor_names.json").read_text(encoding="utf-8"))
    logits = load_matsci_multilabel_scores(
        Path(args.score_cache).resolve(), input_dir, str(args.split), precursor_names
    )
    rows = []
    scores = []
    for row_logits in logits:
        candidates, current_scores = generate_candidate_sets(
            row_logits, int(args.top_labels), int(args.max_set_length), int(args.limit)
        )
        rows.append(candidates)
        scores.append(current_scores)
    report = {
        "protocol": f"{args.split}_train_only_matscibert_combinatorial_set_generator",
        "config": vars(args),
        "rows": len(rows),
        "mean_candidates": float(np.mean([len(row) for row in rows])),
        "ranking": "minimum member logit, then mean member logit",
        "evaluation": None,
    }
    if bool(args.evaluate_labels):
        pack = np.load(input_dir / f"{args.split}.npz", allow_pickle=True)
        targets = [tuple(np.flatnonzero(row > 0.5).tolist()) for row in pack["y_multi_hot"]]
        report["evaluation"] = exact_metrics(targets, rows)
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with Path(args.output_candidates_jsonl).resolve().open("w", encoding="utf-8") as handle:
        for row_index, (candidates, current_scores) in enumerate(zip(rows, scores)):
            handle.write(json.dumps({
                "row_index": row_index,
                "candidate_label_ids": [list(candidate) for candidate in candidates],
                "scores": current_scores,
            }) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
