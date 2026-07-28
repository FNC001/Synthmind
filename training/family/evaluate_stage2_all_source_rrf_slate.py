#!/usr/bin/env python3
"""Audit protected Top-10 slates built from many frozen ranking sources.

This evaluator is deliberately labelled exploratory: it searches aggregation
hyperparameters on the requested split.  Its purpose is to determine whether
source agreement can recover the complementarity seen in a many-model oracle.
Any deployable configuration must subsequently be selected on train OOF data
and applied unchanged to validation/test data.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from training.family.evaluate_stage2_candidate_fusion import load_source


SetKey = Tuple[int, ...]


def parse_grid(raw: str, cast=float) -> list:
    values = [cast(value.strip()) for value in str(raw).split(",") if value.strip()]
    if not values:
        raise ValueError("grid must contain at least one value")
    return values


def exact_metrics(targets: Sequence[SetKey], rows: Sequence[Sequence[SetKey]]) -> Dict[str, float]:
    return {
        f"exact_hit@{k}": float(
            np.mean([target in set(row[:k]) for target, row in zip(targets, rows)])
        )
        for k in (1, 3, 5, 10, 20, 50, 100)
    }


def source_paths(args: argparse.Namespace) -> List[str]:
    paths = [str(value) for value in args.source]
    if str(args.source_manifest).strip():
        report = json.loads(Path(args.source_manifest).read_text(encoding="utf-8"))
        manifest_paths = report.get("sources")
        if not isinstance(manifest_paths, list):
            raise ValueError("source manifest must contain a list-valued 'sources' field")
        paths.extend(str(value) for value in manifest_paths)
    base_resolved = str(Path(args.base_candidates).resolve())
    output: List[str] = []
    seen: set[str] = set()
    for value in paths:
        path = Path(value).resolve()
        resolved = str(path)
        if resolved == base_resolved or resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        output.append(resolved)
    return output


def aggregate_features(
    base_rows: Sequence[Sequence[SetKey]],
    expert_rows: Sequence[Sequence[Sequence[SetKey]]],
    source_limit: int,
    union_limit: int,
    rrf_constants: Sequence[float],
) -> tuple[List[List[SetKey]], List[np.ndarray]]:
    candidate_rows: List[List[SetKey]] = []
    feature_rows: List[np.ndarray] = []
    n_sources = max(1, len(expert_rows))
    for row_index, base in enumerate(base_rows):
        rank_maps = []
        for source in expert_rows:
            mapping = {
                candidate: rank
                for rank, candidate in enumerate(source[row_index][: int(source_limit)], start=1)
            }
            rank_maps.append(mapping)
        base_map = {candidate: rank for rank, candidate in enumerate(base, start=1)}
        universe = set(base_map)
        for mapping in rank_maps:
            universe.update(mapping)
        ordered = sorted(
            universe,
            key=lambda candidate: (
                base_map.get(candidate, 10**9),
                min((mapping.get(candidate, 10**9) for mapping in rank_maps), default=10**9),
                -sum(candidate in mapping for mapping in rank_maps),
                candidate,
            ),
        )[: int(union_limit)]
        features = np.zeros((len(ordered), 3 + len(rrf_constants)), dtype=np.float32)
        for index, candidate in enumerate(ordered):
            base_rank = base_map.get(candidate)
            features[index, 0] = (
                1.0 / math.log2(int(base_rank) + 2.0) if base_rank is not None else 0.0
            )
            ranks = [mapping[candidate] for mapping in rank_maps if candidate in mapping]
            features[index, 1] = sum(rank <= 10 for rank in ranks) / n_sources
            features[index, 2] = sum(rank <= 50 for rank in ranks) / n_sources
            for offset, constant in enumerate(rrf_constants, start=3):
                features[index, offset] = sum(
                    1.0 / (float(constant) + float(rank)) for rank in ranks
                ) / n_sources
        candidate_rows.append(ordered)
        feature_rows.append(features)
    return candidate_rows, feature_rows


def select_rows(
    base_rows: Sequence[Sequence[SetKey]],
    candidate_rows: Sequence[Sequence[SetKey]],
    feature_rows: Sequence[np.ndarray],
    protected_prefix: int,
    base_weight: float,
    support10_weight: float,
    support50_weight: float,
    rrf_weight: float,
    rrf_index: int,
) -> List[List[SetKey]]:
    output: List[List[SetKey]] = []
    for base, candidates, features in zip(base_rows, candidate_rows, feature_rows):
        prefix = list(dict.fromkeys(base[: int(protected_prefix)]))
        prefix_set = set(prefix)
        scores = (
            float(base_weight) * features[:, 0]
            + float(support10_weight) * features[:, 1]
            + float(support50_weight) * features[:, 2]
            + float(rrf_weight) * features[:, int(rrf_index)]
        )
        order = np.argsort(-scores, kind="stable")
        selected = list(prefix)
        selected_set = set(prefix_set)
        for index in order:
            candidate = candidates[int(index)]
            if candidate not in selected_set:
                selected.append(candidate)
                selected_set.add(candidate)
            if len(selected) >= 10:
                break
        for candidate in [*base, *candidates]:
            if candidate and candidate not in selected_set:
                selected.append(candidate)
                selected_set.add(candidate)
        output.append(selected)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--base_candidates", required=True)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--source_manifest", default="")
    parser.add_argument("--source_limit", type=int, default=50)
    parser.add_argument("--base_limit", type=int, default=100)
    parser.add_argument("--union_limit", type=int, default=600)
    parser.add_argument("--protected_prefix_grid", default="5,7,8,9")
    parser.add_argument("--base_weight_grid", default="0.5,1,2,4,8")
    parser.add_argument("--support10_weight_grid", default="0,0.25,1")
    parser.add_argument("--support50_weight_grid", default="0,0.25")
    parser.add_argument("--rrf_weight_grid", default="0.5,1,2")
    parser.add_argument("--rrf_constant_grid", default="1,10,50")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    values = np.load(input_dir / f"{args.split}.npz", allow_pickle=True)["y_multi_hot"]
    targets = [tuple(np.flatnonzero(row > 0.5).tolist()) for row in values]
    base_rows = load_source(args.base_candidates, len(targets), int(args.base_limit))
    paths = source_paths(args)
    if not paths:
        parser.error("no valid complementary sources were supplied")
    experts = [load_source(path, len(targets), int(args.source_limit)) for path in paths]
    rrf_constants = parse_grid(args.rrf_constant_grid, float)
    candidates, features = aggregate_features(
        base_rows, experts, int(args.source_limit), int(args.union_limit), rrf_constants
    )

    base_metrics = exact_metrics(targets, base_rows)
    trials = []
    best = None
    best_rows: List[List[SetKey]] = [list(row) for row in base_rows]
    grid = itertools.product(
        parse_grid(args.protected_prefix_grid, int),
        parse_grid(args.base_weight_grid, float),
        parse_grid(args.support10_weight_grid, float),
        parse_grid(args.support50_weight_grid, float),
        parse_grid(args.rrf_weight_grid, float),
        range(len(rrf_constants)),
    )
    for protected, base_weight, support10, support50, rrf_weight, rrf_index in grid:
        rows = select_rows(
            base_rows,
            candidates,
            features,
            int(protected),
            float(base_weight),
            float(support10),
            float(support50),
            float(rrf_weight),
            3 + int(rrf_index),
        )
        current = {
            "protected_prefix": int(protected),
            "base_weight": float(base_weight),
            "support10_weight": float(support10),
            "support50_weight": float(support50),
            "rrf_weight": float(rrf_weight),
            "rrf_constant": float(rrf_constants[int(rrf_index)]),
            **exact_metrics(targets, rows),
        }
        trials.append(current)
        key = (current["exact_hit@10"], current["exact_hit@5"], current["exact_hit@1"])
        if best is None or key > best[0]:
            best = (key, current)
            best_rows = rows
    assert best is not None
    base_hits = np.asarray(
        [target in set(row[:10]) for target, row in zip(targets, base_rows)], dtype=bool
    )
    best_hits = np.asarray(
        [target in set(row[:10]) for target, row in zip(targets, best_rows)], dtype=bool
    )
    report = {
        "protocol": f"exploratory_{args.split}_label_selected_all_source_rrf_slate",
        "warning": (
            "Hyperparameters were selected on the evaluated split. This is an exploratory "
            "coverage audit, not a formal held-out accuracy result."
        ),
        "config": vars(args),
        "valid_sources": len(paths),
        "base": base_metrics,
        "best": {
            **best[1],
            "new_hits_over_base": int((best_hits & ~base_hits).sum()),
            "lost_hits_vs_base": int((base_hits & ~best_hits).sum()),
        },
        "n_trials": len(trials),
        "top_trials": sorted(
            trials,
            key=lambda row: (-row["exact_hit@10"], -row["exact_hit@5"], -row["exact_hit@1"]),
        )[:50],
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    candidate_output = Path(args.output_candidates_jsonl).resolve()
    candidate_output.parent.mkdir(parents=True, exist_ok=True)
    with candidate_output.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(best_rows):
            handle.write(
                json.dumps(
                    {"row_index": row_index, "candidate_label_ids": [list(value) for value in row]}
                )
                + "\n"
            )
    print(json.dumps({"base": base_metrics, "best": report["best"]}, indent=2))


if __name__ == "__main__":
    main()
