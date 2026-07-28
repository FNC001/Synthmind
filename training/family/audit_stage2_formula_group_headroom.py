#!/usr/bin/env python3
"""Audit exact Top-K headroom within formula-disjoint evaluation groups."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.family.evaluate_stage2_candidate_fusion import load_source


SetKey = Tuple[int, ...]


def target_rank(target: SetKey, row: Sequence[SetKey]) -> int:
    for rank, candidate in enumerate(row, start=1):
        if candidate == target:
            return int(rank)
    return 0


def group_report(
    frame: pd.DataFrame,
    targets: Sequence[SetKey],
    rows: Sequence[Sequence[SetKey]],
    names: Sequence[str],
    top_k: int,
) -> list[dict[str, object]]:
    output = []
    for group, indices in frame.groupby("group", sort=False).indices.items():
        selected = [int(index) for index in indices]
        group_targets = [targets[index] for index in selected]
        counts = Counter(group_targets)
        hits = [target in set(rows[index][: int(top_k)]) for index, target in zip(selected, group_targets)]
        ranks = [target_rank(target, rows[index]) for index, target in zip(selected, group_targets)]
        oracle_hits = sum(count for _, count in counts.most_common(int(top_k)))
        top_targets = []
        for target, count in counts.most_common(min(int(top_k) + 5, len(counts))):
            top_targets.append(
                {
                    "candidate_label_ids": list(target),
                    "candidate_formulas": [str(names[int(label)]) for label in target],
                    "rows": int(count),
                }
            )
        first = frame.iloc[selected[0]]
        output.append(
            {
                "formula_group": str(group),
                "canonical_formula": str(first["formula"]),
                "family": str(first["family"]),
                "rows": int(len(selected)),
                "unique_exact_routes": int(len(counts)),
                f"current_exact_hit@{int(top_k)}": float(np.mean(hits)),
                f"within_group_label_oracle@{int(top_k)}": float(oracle_hits / len(selected)),
                "current_hits": int(sum(hits)),
                "oracle_hits": int(oracle_hits),
                "potential_extra_hits": int(oracle_hits - sum(hits)),
                "target_rank_bands": {
                    f"1-{int(top_k)}": int(sum(1 <= rank <= int(top_k) for rank in ranks)),
                    f"{int(top_k) + 1}-20": int(
                        sum(int(top_k) < rank <= 20 for rank in ranks)
                    ),
                    "21-50": int(sum(20 < rank <= 50 for rank in ranks)),
                    "51-100": int(sum(50 < rank <= 100 for rank in ranks)),
                    "outside_top100": int(sum(rank == 0 or rank > 100 for rank in ranks)),
                },
                "top_observed_routes": top_targets,
            }
        )
    return sorted(
        output,
        key=lambda item: (
            -int(item["potential_extra_hits"]),
            -int(item["rows"]),
            str(item["formula_group"]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--ranking", required=True)
    parser.add_argument("--candidate_limit", type=int, default=800)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    pack = np.load(input_dir / f"{args.split}.npz", allow_pickle=True)
    targets = [
        tuple(np.flatnonzero(row > 0.5).tolist())
        for row in np.asarray(pack["y_multi_hot"], dtype=np.float32)
    ]
    meta = pd.read_csv(input_dir / f"{args.split}_meta.csv", low_memory=False)
    group_column = "family_group_key"
    formula_column = "canonical_formula" if "canonical_formula" in meta else "formula"
    frame = pd.DataFrame(
        {
            "group": meta[group_column].fillna("UNK").astype(str),
            "formula": meta[formula_column].fillna("").astype(str),
            "family": meta["family_signature_primary"].fillna("UNK").astype(str),
        }
    )
    rows = load_source(args.ranking, len(targets), int(args.candidate_limit))
    names = json.loads((input_dir / "precursor_names.json").read_text(encoding="utf-8"))
    groups = group_report(frame, targets, rows, names, int(args.top_k))
    report = {
        "protocol": f"{args.split}_formula_group_exact_route_headroom_audit",
        "ranking": str(Path(args.ranking).resolve()),
        "rows": int(len(targets)),
        "formula_groups": int(len(groups)),
        "top_k": int(args.top_k),
        "current_hits": int(sum(item["current_hits"] for item in groups)),
        "within_group_label_oracle_hits": int(sum(item["oracle_hits"] for item in groups)),
        "potential_extra_hits": int(sum(item["potential_extra_hits"] for item in groups)),
        "groups": groups,
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "groups"},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(json.dumps({"largest_headroom_groups": groups[:20]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
