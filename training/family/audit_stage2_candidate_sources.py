#!/usr/bin/env python3
"""Audit validation candidate sources against a fixed Stage-2 base ranking.

The report is deliberately descriptive: it does not change rankings or touch
the frozen test split.  It identifies sources whose validation errors are
complementary to the current base model and reports results separately for
targets containing train-unseen labels.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


SetKey = Tuple[int, ...]
TOP_K = (1, 3, 5, 10, 20, 50, 100)


def target_keys(matrix: np.ndarray) -> List[SetKey]:
    return [tuple(np.flatnonzero(row > 0.5).astype(int).tolist()) for row in matrix]


def load_ranks(path: Path, n_rows: int, limit: int) -> tuple[List[List[SetKey]], int]:
    rows: List[List[SetKey]] = [[] for _ in range(n_rows)]
    found = np.zeros(n_rows, dtype=bool)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            row_index = int(record["row_index"])
            if row_index < 0 or row_index >= n_rows:
                continue
            values = record.get("candidate_label_ids", [])[:limit]
            deduped: List[SetKey] = []
            seen: set[SetKey] = set()
            for candidate in values:
                key = tuple(sorted({int(value) for value in candidate}))
                if key and key not in seen:
                    seen.add(key)
                    deduped.append(key)
            rows[row_index] = deduped
            found[row_index] = True
    return rows, int(found.sum())


def hit_matrix(targets: Sequence[SetKey], rows: Sequence[Sequence[SetKey]]) -> Dict[int, np.ndarray]:
    return {
        k: np.asarray([target in set(row[:k]) for target, row in zip(targets, rows)], dtype=bool)
        for k in TOP_K
    }


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    return float(values[mask].mean()) if bool(mask.any()) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--base_candidates", required=True)
    parser.add_argument("--source_globs", nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--source_limit", type=int, default=100)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    train_y = np.asarray(np.load(input_dir / "train.npz", allow_pickle=True)["y_multi_hot"])
    val_y = np.asarray(np.load(input_dir / "val.npz", allow_pickle=True)["y_multi_hot"])
    targets = target_keys(val_y)
    train_seen = np.asarray(train_y.sum(axis=0) > 0)
    unseen = np.asarray(
        [any(not bool(train_seen[label]) for label in target) for target in targets], dtype=bool
    )
    base_path = Path(args.base_candidates).resolve()
    base_rows, base_found = load_ranks(base_path, len(targets), int(args.source_limit))
    if base_found != len(targets):
        raise ValueError(f"base source has {base_found}/{len(targets)} rows: {base_path}")
    base_hits = hit_matrix(targets, base_rows)

    source_paths: List[Path] = []
    for pattern in args.source_globs:
        source_paths.extend(Path(value).resolve() for value in glob.glob(pattern, recursive=True))
    source_paths = sorted({path for path in source_paths if path.is_file() and path != base_path})

    sources = []
    for path in source_paths:
        try:
            rows, found = load_ranks(path, len(targets), int(args.source_limit))
            if found != len(targets):
                sources.append({"path": str(path), "status": "incomplete", "rows": found})
                continue
            hits = hit_matrix(targets, rows)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            sources.append({"path": str(path), "status": "invalid", "error": str(exc)})
            continue
        item = {"path": str(path), "status": "ok", "rows": found}
        for k in TOP_K:
            item[f"exact_hit@{k}"] = float(hits[k].mean())
        source_top10 = hits[10]
        base_top10 = base_hits[10]
        item.update(
            {
                "exact_hit@10_seen": masked_mean(source_top10, ~unseen),
                "exact_hit@10_unseen": masked_mean(source_top10, unseen),
                "new_top10_hits_over_base": int((source_top10 & ~base_top10).sum()),
                "lost_top10_hits_vs_base": int((base_top10 & ~source_top10).sum()),
                "oracle_union_top10": float((source_top10 | base_top10).mean()),
                "oracle_union_top10_unseen": masked_mean(source_top10 | base_top10, unseen),
            }
        )
        sources.append(item)

    valid_sources = [row for row in sources if row.get("status") == "ok"]
    valid_sources.sort(
        key=lambda row: (
            -int(row["new_top10_hits_over_base"]),
            -float(row["exact_hit@10"]),
            row["path"],
        )
    )
    report = {
        "protocol": "fixed_formula_disjoint_validation_candidate_source_audit",
        "config": vars(args),
        "validation": {
            "rows": len(targets),
            "unseen_label_rows": int(unseen.sum()),
            "base_source": str(base_path),
            "base_exact_hit@10": float(base_hits[10].mean()),
            "base_exact_hit@10_seen": masked_mean(base_hits[10], ~unseen),
            "base_exact_hit@10_unseen": masked_mean(base_hits[10], unseen),
        },
        "sources": valid_sources,
        "skipped_sources": [row for row in sources if row.get("status") != "ok"],
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**report["validation"], "top_complements": valid_sources[:20]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
