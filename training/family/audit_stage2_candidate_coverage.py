#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.family.evaluate_stage2_candidate_fusion import load_source  # noqa: E402


SetKey = Tuple[int, ...]
TOP_K = (1, 3, 5, 10, 20, 50, 100)


def parse_named_source(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"expert must be NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    return name.strip(), path.strip()


def exact_metrics(targets: Sequence[SetKey], rows: Sequence[Sequence[SetKey]]) -> Dict[str, float]:
    return {
        f"exact_hit@{k}": float(
            np.mean([target in set(candidates[:k]) for target, candidates in zip(targets, rows)])
        )
        for k in TOP_K
    }


def group_macro(values: np.ndarray, groups: np.ndarray) -> float:
    frame = pd.DataFrame({"group": groups.astype(str), "value": values.astype(float)})
    return float(frame.groupby("group", sort=False)["value"].mean().mean())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit exact-set coverage, overlap, and ranking bottlenecks for Stage2 candidates."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--expert", action="append", default=[], help="Repeat NAME=candidates.jsonl")
    parser.add_argument(
        "--expert_manifest",
        default="",
        help="Optional JSON report containing config.expert or sources candidate paths.",
    )
    parser.add_argument("--ranking", default="", help="Optional final ranked candidates JSONL.")
    parser.add_argument("--source_limit", type=int, default=100)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    if not args.expert and not str(args.expert_manifest).strip():
        parser.error("at least one --expert or --expert_manifest is required")

    input_dir = Path(args.input_dir).resolve()
    pack = np.load(input_dir / f"{args.split}.npz", allow_pickle=True)
    y = np.asarray(pack["y_multi_hot"], dtype=np.float32)
    targets: List[SetKey] = [tuple(np.flatnonzero(row > 0.5).tolist()) for row in y]
    meta = pd.read_csv(input_dir / f"{args.split}_meta.csv", low_memory=False)
    groups = meta["family_group_key"].fillna("UNK").astype(str).to_numpy()
    families = meta["family_signature_primary"].fillna("UNK").astype(str).to_numpy()
    named_paths = [parse_named_source(value) for value in args.expert]
    if str(args.expert_manifest).strip():
        manifest_path = Path(args.expert_manifest).resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_paths = manifest.get("sources")
        if not isinstance(manifest_paths, list):
            manifest_paths = manifest.get("config", {}).get("expert")
        if not isinstance(manifest_paths, list):
            raise ValueError("expert manifest must contain sources or config.expert")
        used_names = {name for name, _ in named_paths}
        used_paths = {str(Path(path).resolve()) for _, path in named_paths}
        for index, value in enumerate(manifest_paths):
            resolved = str(Path(str(value)).resolve())
            if resolved in used_paths:
                continue
            base_name = Path(resolved).stem
            name = base_name
            suffix = 2
            while name in used_names:
                name = f"{base_name}_{suffix}"
                suffix += 1
            named_paths.append((name, resolved))
            used_names.add(name)
            used_paths.add(resolved)
    expert_rows = {
        name: load_source(path, len(targets), int(args.source_limit)) for name, path in named_paths
    }

    expert_hits: Dict[str, np.ndarray] = {}
    expert_reports: Dict[str, object] = {}
    for name, rows in expert_rows.items():
        hits = np.asarray([target in set(row) for target, row in zip(targets, rows)], dtype=bool)
        expert_hits[name] = hits
        expert_reports[name] = {
            **exact_metrics(targets, rows),
            "oracle_at_source_limit": float(hits.mean()),
            "formula_group_macro_oracle": group_macro(hits, groups),
            "mean_candidates": float(np.mean([len(row) for row in rows])),
        }

    hit_matrix = np.column_stack([expert_hits[name] for name, _ in named_paths])
    union_hits = hit_matrix.any(axis=1)
    marginal = []
    running = np.zeros(len(targets), dtype=bool)
    for name, _ in named_paths:
        new = expert_hits[name] & ~running
        running |= expert_hits[name]
        marginal.append(
            {
                "expert": name,
                "new_rows": int(new.sum()),
                "cumulative_rows": int(running.sum()),
                "cumulative_recall": float(running.mean()),
            }
        )

    coverage_count = hit_matrix.sum(axis=1)
    coverage_histogram = {
        str(key): int(value) for key, value in sorted(Counter(coverage_count.tolist()).items())
    }
    family_frame = pd.DataFrame(
        {"family": families, "covered": union_hits.astype(float), "rows": 1}
    ).groupby("family", sort=False).agg(rows=("rows", "sum"), oracle=("covered", "mean"))
    family_frame = family_frame.sort_values(["oracle", "rows"], ascending=[True, False])

    report: Dict[str, object] = {
        "protocol": f"{args.split}_formula_disjoint_exact_set_candidate_coverage_audit",
        "rows": int(len(targets)),
        "formula_groups": int(pd.Series(groups).nunique()),
        "families": int(pd.Series(families).nunique()),
        "source_limit": int(args.source_limit),
        "expert_paths": dict(named_paths),
        "experts": expert_reports,
        "union": {
            "covered_rows": int(union_hits.sum()),
            "uncovered_rows": int((~union_hits).sum()),
            "oracle_recall": float(union_hits.mean()),
            "formula_group_macro_oracle": group_macro(union_hits, groups),
            "family_macro_oracle": group_macro(union_hits, families),
            "coverage_count_histogram": coverage_histogram,
            "marginal_gain_in_declared_order": marginal,
            "worst_families": [
                {"family": str(index), "rows": int(row["rows"]), "oracle": float(row["oracle"])}
                for index, row in family_frame.head(20).iterrows()
            ],
        },
    }

    if str(args.ranking).strip():
        ranked = load_source(args.ranking, len(targets), max(TOP_K))
        first_ranks = []
        for target, row in zip(targets, ranked):
            mapping = {candidate: rank for rank, candidate in enumerate(row, start=1)}
            first_ranks.append(mapping.get(target, 0))
        ranks = np.asarray(first_ranks, dtype=np.int32)
        report["ranking"] = {
            "path": str(Path(args.ranking).resolve()),
            **exact_metrics(targets, ranked),
            "rank_bands": {
                "1-10": int(((ranks >= 1) & (ranks <= 10)).sum()),
                "11-20": int(((ranks >= 11) & (ranks <= 20)).sum()),
                "21-50": int(((ranks >= 21) & (ranks <= 50)).sum()),
                "51-100": int(((ranks >= 51) & (ranks <= 100)).sum()),
                "not_in_ranked_top100": int((ranks == 0).sum()),
            },
            "formula_group_macro_exact_hit@10": group_macro(
                (ranks >= 1) & (ranks <= 10), groups
            ),
            "family_macro_exact_hit@10": group_macro(
                (ranks >= 1) & (ranks <= 10), families
            ),
        }

    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
