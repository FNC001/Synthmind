#!/usr/bin/env python3
"""Use precursor-family templates to allocate exact candidates in a Top-K slate."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from synthmind.chemistry.families import assign_cation_family
from training.family.evaluate_stage2_candidate_fusion import load_source


SetKey = Tuple[int, ...]
FamilyKey = Tuple[str, ...]


def precursor_family(name: str) -> str:
    try:
        return str(assign_cation_family(str(name)).family_signature_primary)
    except Exception:
        return f"TEXT::{str(name).strip().lower()}"


def family_key(candidate: SetKey, label_families: Sequence[str]) -> FamilyKey:
    return tuple(sorted(str(label_families[int(label)]) for label in candidate))


def family_slate(
    candidates: Sequence[SetKey],
    label_families: Sequence[str],
    slate_size: int,
    pool_size: int,
    protected_prefix: int,
    family_weight: float,
    within_family_weight: float,
) -> List[SetKey]:
    unique = list(dict.fromkeys(candidates))
    pool = unique[: int(pool_size)]
    protected = pool[: min(int(protected_prefix), int(slate_size))]
    protected_set = set(protected)
    first_rank: Dict[FamilyKey, int] = {}
    within_rank: Dict[FamilyKey, int] = {}
    scored = []
    for rank, candidate in enumerate(pool, start=1):
        key = family_key(candidate, label_families)
        first_rank.setdefault(key, rank)
        local_rank = within_rank.get(key, 0)
        within_rank[key] = local_rank + 1
        score = (
            float(rank)
            + float(family_weight) * float(first_rank[key])
            + float(within_family_weight) * float(local_rank)
        )
        scored.append((score, rank, candidate))
    selected = list(protected)
    for _, _, candidate in sorted(scored, key=lambda value: (value[0], value[1])):
        if candidate in protected_set:
            continue
        selected.append(candidate)
        if len(selected) >= int(slate_size):
            break
    selected_set = set(selected)
    return selected + [candidate for candidate in unique if candidate not in selected_set]


def metrics(
    targets: Sequence[SetKey],
    rows: Sequence[Sequence[SetKey]],
    label_families: Sequence[str],
) -> Dict[str, float]:
    target_family = [family_key(target, label_families) for target in targets]
    output: Dict[str, float] = {}
    for k in (1, 3, 5, 10, 20, 50, 100):
        output[f"exact_hit@{k}"] = float(
            np.mean([target in set(row[:k]) for target, row in zip(targets, rows)])
        )
        output[f"precursor_family_equivalent_hit@{k}"] = float(
            np.mean([
                target_key
                in {family_key(candidate, label_families) for candidate in row[:k]}
                for target_key, row in zip(target_family, rows)
            ])
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Search precursor-family-aware exact Top-K slates.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--candidate_source", required=True)
    parser.add_argument("--candidate_limit", type=int, default=100)
    parser.add_argument("--slate_size", type=int, default=10)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    pack = np.load(input_dir / f"{args.split}.npz", allow_pickle=True)
    targets = [
        tuple(np.flatnonzero(row > 0.5).tolist())
        for row in np.asarray(pack["y_multi_hot"], dtype=np.float32)
    ]
    names = [
        str(value)
        for value in json.loads((input_dir / "precursor_names.json").read_text(encoding="utf-8"))
    ]
    label_families = [precursor_family(name) for name in names]
    candidates = load_source(args.candidate_source, len(targets), int(args.candidate_limit))

    trials = []
    best = None
    best_rows: List[List[SetKey]] = []
    for pool_size, protected_prefix, family_weight, within_family_weight in itertools.product(
        (20, 50, 100),
        (0, 1, 3, 5, 7),
        (0.0, 0.5, 1.0, 2.0, 5.0, 10.0),
        (0.0, 0.25, 0.5, 1.0, 2.0),
    ):
        ranked = [
            family_slate(
                row,
                label_families,
                int(args.slate_size),
                min(int(pool_size), int(args.candidate_limit)),
                int(protected_prefix),
                float(family_weight),
                float(within_family_weight),
            )
            for row in candidates
        ]
        current = metrics(targets, ranked, label_families)
        trial = {
            "pool_size": int(pool_size),
            "protected_prefix": int(protected_prefix),
            "family_weight": float(family_weight),
            "within_family_weight": float(within_family_weight),
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
        "protocol": f"{args.split}_precursor_cation_family_template_slate_search",
        "config": vars(args),
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
