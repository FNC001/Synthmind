#!/usr/bin/env python3
"""Audit whether a reranker complements or merely replaces the base slate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from training.family.evaluate_stage2_candidate_fusion import load_source
from training.family.train_stage2_structured_energy_ranker import targets_from_matrix
from training.family.train_stage2_validation_meta_lambdarank import (
    parse_expert_source,
    protected_expert_union,
    score_sorted_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--base_candidates", required=True)
    parser.add_argument("--expert_source", action="append", default=[])
    parser.add_argument(
        "--expert_manifest",
        default="",
        help="Coverage-audit JSON with an expert_paths mapping; its base entry is ignored.",
    )
    parser.add_argument("--base_limit", type=int, default=100)
    parser.add_argument("--expert_limit", type=int, default=10)
    parser.add_argument("--scores_npz", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    targets = targets_from_matrix(
        np.asarray(
            np.load(input_dir / "val.npz", allow_pickle=True)["y_multi_hot"],
            dtype=np.float32,
        )
    )
    base_rows = load_source(args.base_candidates, len(targets), int(args.base_limit))
    expert_specs = [parse_expert_source(value) for value in args.expert_source]
    if str(args.expert_manifest).strip():
        manifest = json.loads(
            Path(args.expert_manifest).resolve().read_text(encoding="utf-8")
        )
        for name, path in manifest.get("expert_paths", {}).items():
            if str(name) != "base":
                expert_specs.append((str(name), str(Path(path).resolve())))
    expert_rows = [
        load_source(path, len(targets), int(args.expert_limit)) for _, path in expert_specs
    ]
    pool_rows = protected_expert_union(
        base_rows, expert_rows, int(args.base_limit), int(args.expert_limit)
    )
    pack = np.load(Path(args.scores_npz).resolve(), allow_pickle=False)
    scores = np.asarray(pack["raw_scores"], dtype=np.float32)
    spans = [tuple(map(int, value)) for value in np.asarray(pack["spans"], dtype=np.int64)]
    if len(spans) != len(targets) or (spans and spans[-1][1] != len(scores)):
        raise ValueError("score spans do not align with validation rows")
    ranked_rows = score_sorted_rows(pool_rows, scores, spans)
    base_hit = np.asarray(
        [target in set(row[:10]) for target, row in zip(targets, base_rows)], dtype=bool
    )
    ranked_hit = np.asarray(
        [target in set(row[:10]) for target, row in zip(targets, ranked_rows)], dtype=bool
    )
    pool_hit = np.asarray(
        [target in set(row) for target, row in zip(targets, pool_rows)], dtype=bool
    )
    truth_ranks = []
    for target, row in zip(targets, ranked_rows):
        try:
            truth_ranks.append(int(row.index(target) + 1))
        except ValueError:
            truth_ranks.append(0)
    missed_ranks = [rank for rank, hit in zip(truth_ranks, base_hit) if not hit and rank > 0]
    report = {
        "protocol": "validation_reranker_complementarity_audit",
        "rows": int(len(targets)),
        "base_hits": int(base_hit.sum()),
        "reranker_hits": int(ranked_hit.sum()),
        "new_hits_over_base": int((ranked_hit & ~base_hit).sum()),
        "lost_hits_vs_base": int((base_hit & ~ranked_hit).sum()),
        "perfect_gate_hits": int((base_hit | ranked_hit).sum()),
        "perfect_gate_hit_rate": float((base_hit | ranked_hit).mean()),
        "candidate_pool_hits": int(pool_hit.sum()),
        "candidate_pool_hit_rate": float(pool_hit.mean()),
        "base_miss_truth_rank_histogram": {
            "1-10": int(sum(1 <= value <= 10 for value in missed_ranks)),
            "11-20": int(sum(11 <= value <= 20 for value in missed_ranks)),
            "21-50": int(sum(21 <= value <= 50 for value in missed_ranks)),
            "51-100": int(sum(51 <= value <= 100 for value in missed_ranks)),
            ">100": int(sum(value > 100 for value in missed_ranks)),
            "missing": int((~base_hit & ~pool_hit).sum()),
        },
        "expert_sources": [
            {"name": name, "path": path} for name, path in expert_specs
        ],
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
