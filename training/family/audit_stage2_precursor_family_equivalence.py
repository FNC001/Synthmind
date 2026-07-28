#!/usr/bin/env python3
"""Report exact and cation-family-equivalent precursor-set Top-K metrics."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from synthmind.chemistry.families import assign_cation_family
from training.family.evaluate_stage2_candidate_fusion import load_source


SetKey = Tuple[int, ...]
FamilyKey = Tuple[str, ...]
TOP_K = (1, 3, 5, 10, 20, 50, 100)


def precursor_family(name: str) -> str:
    try:
        assignment = assign_cation_family(str(name))
        return str(assignment.family_signature_primary)
    except Exception:
        return f"TEXT::{str(name).strip().lower()}"


def family_key(candidate: SetKey, label_families: Sequence[str]) -> FamilyKey:
    return tuple(sorted(str(label_families[int(label)]) for label in candidate))


def metrics(
    targets: Sequence[SetKey],
    candidates: Sequence[Sequence[SetKey]],
    label_families: Sequence[str],
    groups: np.ndarray,
) -> Dict[str, float]:
    target_families = [family_key(target, label_families) for target in targets]
    output: Dict[str, float] = {}
    for k in TOP_K:
        exact_hits = np.asarray(
            [target in set(row[:k]) for target, row in zip(targets, candidates)],
            dtype=np.float32,
        )
        family_hits = np.asarray(
            [
                target_family
                in {family_key(candidate, label_families) for candidate in row[:k]}
                for target_family, row in zip(target_families, candidates)
            ],
            dtype=np.float32,
        )
        output[f"exact_hit@{k}"] = float(exact_hits.mean())
        output[f"precursor_family_equivalent_hit@{k}"] = float(family_hits.mean())
        frame = pd.DataFrame(
            {"group": groups.astype(str), "exact": exact_hits, "family": family_hits}
        )
        output[f"formula_group_macro_exact_hit@{k}"] = float(
            frame.groupby("group", sort=False)["exact"].mean().mean()
        )
        output[f"formula_group_macro_precursor_family_equivalent_hit@{k}"] = float(
            frame.groupby("group", sort=False)["family"].mean().mean()
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit precursor family-equivalent Top-K.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--ranking", required=True)
    parser.add_argument("--candidate_limit", type=int, default=100)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    pack = np.load(input_dir / f"{args.split}.npz", allow_pickle=True)
    y = np.asarray(pack["y_multi_hot"], dtype=np.float32)
    targets = [tuple(np.flatnonzero(row > 0.5).tolist()) for row in y]
    names = [
        str(value)
        for value in json.loads((input_dir / "precursor_names.json").read_text(encoding="utf-8"))
    ]
    label_families = [precursor_family(name) for name in names]
    candidates: List[List[SetKey]] = load_source(
        args.ranking, len(targets), int(args.candidate_limit)
    )
    groups = pd.read_csv(
        input_dir / f"{args.split}_meta.csv", usecols=["family_group_key"]
    )["family_group_key"].fillna("UNK").astype(str).to_numpy()
    counts = Counter(label_families)
    report = {
        "protocol": f"{args.split}_exact_and_precursor_cation_family_equivalent_set_topk",
        "family_definition": (
            "Each precursor formula is mapped by primary cation periodic-table group; "
            "stoichiometry and anion identity do not affect that precursor family. "
            "Multiplicity of precursor-family members in a set is retained."
        ),
        "input_dir": str(input_dir),
        "ranking": str(Path(args.ranking).expanduser().resolve()),
        "rows": int(len(targets)),
        "labels": int(len(names)),
        "precursor_families": int(len(counts)),
        "largest_precursor_families": [
            {"family": family, "labels": int(count)}
            for family, count in counts.most_common(20)
        ],
        "metrics": metrics(targets, candidates, label_families, groups),
    }
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
