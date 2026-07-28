#!/usr/bin/env python3
"""Audit Stage2 ranking quality on train-seen and zero-shot validation slices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from training.family.evaluate_stage2_candidate_fusion import load_source


SetKey = Tuple[int, ...]
TOP_K = (1, 3, 5, 10, 20, 50, 100)


def slice_metrics(
    targets: Sequence[SetKey],
    candidates: Sequence[Sequence[SetKey]],
    mask: np.ndarray,
    groups: np.ndarray,
) -> Dict[str, object]:
    indices = np.flatnonzero(mask)
    output: Dict[str, object] = {"rows": int(len(indices))}
    if not len(indices):
        return output
    local_groups = groups[indices].astype(str)
    for k in TOP_K:
        hits = np.asarray(
            [targets[index] in set(candidates[index][:k]) for index in indices],
            dtype=np.float32,
        )
        output[f"exact_hit@{k}"] = float(hits.mean())
        frame = pd.DataFrame({"group": local_groups, "hit": hits})
        output[f"formula_group_macro_exact_hit@{k}"] = float(
            frame.groupby("group", sort=False)["hit"].mean().mean()
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit train-seen and zero-shot Stage2 slices.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--ranking", required=True)
    parser.add_argument("--candidate_limit", type=int, default=100)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    train_y = np.asarray(
        np.load(input_dir / "train.npz", allow_pickle=True)["y_multi_hot"], dtype=np.float32
    )
    query_y = np.asarray(
        np.load(input_dir / f"{args.split}.npz", allow_pickle=True)["y_multi_hot"],
        dtype=np.float32,
    )
    train_targets = [tuple(np.flatnonzero(row > 0.5).tolist()) for row in train_y]
    targets = [tuple(np.flatnonzero(row > 0.5).tolist()) for row in query_y]
    candidates: List[List[SetKey]] = load_source(
        args.ranking, len(targets), int(args.candidate_limit)
    )
    groups = pd.read_csv(
        input_dir / f"{args.split}_meta.csv", usecols=["family_group_key"]
    )["family_group_key"].fillna("UNK").astype(str).to_numpy()

    seen_labels = train_y.sum(axis=0) > 0.5
    all_labels_seen = np.asarray(
        [bool(np.all(seen_labels[np.asarray(target, dtype=np.int64)])) for target in targets],
        dtype=bool,
    )
    train_set = set(train_targets)
    exact_set_seen = np.asarray([target in train_set for target in targets], dtype=bool)
    report = {
        "protocol": f"{args.split}_train_observable_seen_unseen_exact_set_ranking_audit",
        "input_dir": str(input_dir),
        "ranking": str(Path(args.ranking).expanduser().resolve()),
        "candidate_limit": int(args.candidate_limit),
        "slices": {
            "all": slice_metrics(
                targets, candidates, np.ones(len(targets), dtype=bool), groups
            ),
            "all_target_labels_seen_in_train": slice_metrics(
                targets, candidates, all_labels_seen, groups
            ),
            "any_target_label_unseen_in_train": slice_metrics(
                targets, candidates, ~all_labels_seen, groups
            ),
            "exact_target_set_seen_in_train": slice_metrics(
                targets, candidates, exact_set_seen, groups
            ),
            "exact_target_set_unseen_in_train": slice_metrics(
                targets, candidates, ~exact_set_seen, groups
            ),
        },
    }
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
