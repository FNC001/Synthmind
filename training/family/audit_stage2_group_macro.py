#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.family.evaluate_stage2_candidate_fusion import load_source  # noqa: E402


SetKey = Tuple[int, ...]


def grouped_metrics(
    targets: Sequence[SetKey],
    rows: Sequence[Sequence[SetKey]],
    groups: Sequence[str],
    families: Sequence[str],
) -> Dict[str, float]:
    output: Dict[str, float] = {}
    groups_array = np.asarray(groups, dtype=object).astype(str)
    families_array = np.asarray(families, dtype=object).astype(str)
    for k in (1, 3, 5, 10, 20, 50, 100):
        hits = np.asarray(
            [target in set(candidates[:k]) for target, candidates in zip(targets, rows)],
            dtype=np.float32,
        )
        frame = pd.DataFrame({"group": groups_array, "family": families_array, "hit": hits})
        output[f"row_exact_hit@{k}"] = float(hits.mean())
        output[f"group_macro_exact_hit@{k}"] = float(frame.groupby("group")["hit"].mean().mean())
        output[f"family_macro_exact_hit@{k}"] = float(frame.groupby("family")["hit"].mean().mean())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Stage2 rankings with row-, formula-group-, and family-macro exact Top-K."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--source_limit", type=int, default=100)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    pack = np.load(input_dir / f"{args.split}.npz", allow_pickle=True)
    targets = [tuple(np.flatnonzero(row > 0.5).tolist()) for row in pack["y_multi_hot"]]
    rows = load_source(args.candidates, len(targets), int(args.source_limit))
    meta = pd.read_csv(
        input_dir / f"{args.split}_meta.csv",
        usecols=["family_group_key", "family_signature_primary"],
    )
    group_sizes = meta["family_group_key"].fillna("UNK").astype(str).value_counts()
    report = {
        "protocol": f"{args.split}_formula_disjoint_group_macro_exact_precursor_set",
        "candidates": args.candidates,
        "rows": int(len(targets)),
        "groups": int(group_sizes.size),
        "group_size_quantiles": {
            str(q): float(group_sizes.quantile(q)) for q in (0.0, 0.5, 0.9, 0.99, 1.0)
        },
        "metrics": grouped_metrics(
            targets,
            rows,
            meta["family_group_key"].fillna("UNK").astype(str).to_numpy(),
            meta["family_signature_primary"].fillna("UNK").astype(str).to_numpy(),
        ),
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
