#!/usr/bin/env python3
"""Slice a global training candidate file into local OOF fold row order."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np


def load_records(path: Path) -> Dict[int, dict]:
    records: Dict[int, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            row_index = int(record["row_index"])
            if row_index in records:
                raise ValueError(f"duplicate row_index {row_index} in {path}")
            records[row_index] = record
    if not records:
        raise ValueError(f"candidate source is empty: {path}")
    expected = set(range(max(records) + 1))
    if set(records) != expected:
        missing = sorted(expected - set(records))[:10]
        raise ValueError(f"candidate source is not contiguous; missing examples: {missing}")
    return records


def write_subset(
    path: Path,
    global_indices: np.ndarray,
    records: Dict[int, dict],
    candidate_limit: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for local_index, global_index in enumerate(global_indices.tolist()):
            record = dict(records[int(global_index)])
            record["row_index"] = int(local_index)
            record["global_row_index"] = int(global_index)
            if int(candidate_limit) > 0:
                record["candidate_label_ids"] = record.get(
                    "candidate_label_ids", []
                )[: int(candidate_limit)]
                if isinstance(record.get("scores"), list):
                    record["scores"] = record["scores"][: int(candidate_limit)]
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold_root", required=True)
    parser.add_argument("--global_candidates", required=True)
    parser.add_argument("--output_stem", required=True)
    parser.add_argument(
        "--candidate_limit",
        type=int,
        default=0,
        help="Keep only the first N candidates per row; zero keeps the complete source.",
    )
    args = parser.parse_args()

    fold_root = Path(args.fold_root).resolve()
    records = load_records(Path(args.global_candidates).resolve())
    n_rows = len(records)
    all_indices = np.arange(n_rows, dtype=np.int64)
    report = {"rows": n_rows, "folds": {}}
    for fold_dir in sorted(fold_root.glob("fold_*")):
        query_indices = np.asarray(
            np.load(fold_dir / "val_global_row_indices.npy"), dtype=np.int64
        )
        query_mask = np.zeros(n_rows, dtype=bool)
        query_mask[query_indices] = True
        train_indices = all_indices[~query_mask]
        train_path = fold_dir / f"{args.output_stem}_train_candidates.jsonl"
        val_path = fold_dir / f"{args.output_stem}_val_candidates.jsonl"
        write_subset(train_path, train_indices, records, int(args.candidate_limit))
        write_subset(val_path, query_indices, records, int(args.candidate_limit))
        report["folds"][fold_dir.name] = {
            "train_rows": int(len(train_indices)),
            "val_rows": int(len(query_indices)),
            "train_output": str(train_path),
            "val_output": str(val_path),
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
