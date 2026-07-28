#!/usr/bin/env python3
"""Leakage-safe family router over already-frozen Stage2 rankers."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.family.evaluate_stage2_candidate_fusion import (  # noqa: E402
    fuse_row_topk,
    load_source,
)
from training.family.train_stage2_oof_candidate_stacker import (  # noqa: E402
    formula_group_folds,
)


SetKey = Tuple[int, ...]


def parse_named_source(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"expert must be NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    return name.strip(), path.strip()


def config_key(config: Dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def metrics(targets: Sequence[SetKey], rows: Sequence[Sequence[SetKey]]) -> Dict[str, float]:
    return {
        f"exact_hit@{k}": float(
            np.mean([target in set(row[:k]) for target, row in zip(targets, rows)])
        )
        for k in (1, 3, 5, 10, 20, 50, 100)
    }


def choose_config(
    configs: Sequence[Dict[str, Any]],
    config_rows: Dict[str, List[List[SetKey]]],
    targets: Sequence[SetKey],
    indices: np.ndarray,
    shrinkage: float,
    global_rates: Dict[str, float],
    base_key: str,
    minimum_gain: float,
) -> str:
    n_rows = int(len(indices))
    if not n_rows:
        return base_key
    best_key = base_key
    best_score = -1.0
    scores: Dict[str, float] = {}
    for config in configs:
        key = config_key(config)
        hits = sum(
            targets[int(index)] in set(config_rows[key][int(index)][:10])
            for index in indices
        )
        score = (float(hits) + float(shrinkage) * global_rates[key]) / (
            n_rows + float(shrinkage)
        )
        scores[key] = score
        if score > best_score:
            best_score = score
            best_key = key
    if scores[best_key] < scores[base_key] + float(minimum_gain):
        return base_key
    return best_key


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Formula-group OOF family router with shrinkage toward global ranker accuracy."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--base_candidates", required=True)
    parser.add_argument("--expert", action="append", default=[])
    parser.add_argument("--source_limit", type=int, default=200)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min_family_rows", type=int, default=15)
    parser.add_argument("--shrinkage", type=float, default=25.0)
    parser.add_argument("--minimum_gain", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--routing_json", default="")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    pack = np.load(input_dir / f"{args.split}.npz", allow_pickle=True)
    targets = [
        tuple(np.flatnonzero(row > 0.5).tolist())
        for row in np.asarray(pack["y_multi_hot"], dtype=np.float32)
    ]
    meta = pd.read_csv(
        input_dir / f"{args.split}_meta.csv",
        usecols=["family_signature_primary", "family_group_key"],
        low_memory=False,
    )
    families = meta["family_signature_primary"].fillna("UNK").astype(str).to_numpy()
    formula_groups = meta["family_group_key"].fillna("UNK").astype(str).to_numpy()
    base = load_source(args.base_candidates, len(targets), int(args.source_limit))
    expert_paths = dict(parse_named_source(value) for value in args.expert)
    experts = {
        name: load_source(path, len(targets), int(args.source_limit))
        for name, path in expert_paths.items()
    }
    configs: List[Dict[str, Any]] = [{"kind": "base"}]
    for name in experts:
        configs.append({"kind": "expert", "source": name})
        for constant in (1.0, 5.0, 10.0, 20.0, 50.0):
            for weight in (0.5, 1.0, 2.0):
                configs.append(
                    {
                        "kind": "rrf",
                        "source": name,
                        "rrf_constant": constant,
                        "expert_weight": weight,
                    }
                )
    config_rows: Dict[str, List[List[SetKey]]] = {}
    for config in configs:
        key = config_key(config)
        if config["kind"] == "base":
            rows = [list(row[:100]) for row in base]
        elif config["kind"] == "expert":
            rows = [list(row[:100]) for row in experts[str(config["source"])]]
        else:
            current_expert = experts[str(config["source"])]
            rows = [
                fuse_row_topk(
                    [base[row_index], current_expert[row_index]],
                    [1.0, float(config["expert_weight"])],
                    float(config["rrf_constant"]),
                    100,
                )
                for row_index in range(len(targets))
            ]
        config_rows[key] = rows
    base_key = config_key({"kind": "base"})

    if args.routing_json:
        if args.split != "test":
            raise ValueError("--routing_json may only be applied to test")
        frozen = json.loads(Path(args.routing_json).resolve().read_text(encoding="utf-8"))[
            "full_routing"
        ]
        routed = [
            config_rows.get(str(frozen.get(str(family), base_key)), config_rows[base_key])[index]
            for index, family in enumerate(families)
        ]
        fold_reports: List[Dict[str, Any]] = []
        full_routing = frozen
    else:
        splits = formula_group_folds(formula_groups, int(args.folds), int(args.seed))
        routed: List[List[SetKey]] = [[] for _ in targets]
        fold_reports = []
        all_indices = np.arange(len(targets), dtype=np.int32)
        for fold, (train_indices, query_indices) in enumerate(splits):
            global_rates = {
                key: float(
                    np.mean(
                        [
                            targets[int(index)] in set(rows[int(index)][:10])
                            for index in train_indices
                        ]
                    )
                )
                for key, rows in config_rows.items()
            }
            global_key = max(global_rates, key=global_rates.get)
            fold_route: Dict[str, str] = {}
            for family in np.unique(families[query_indices]):
                family_train = train_indices[families[train_indices] == family]
                if len(family_train) < int(args.min_family_rows):
                    fold_route[str(family)] = global_key
                else:
                    fold_route[str(family)] = choose_config(
                        configs,
                        config_rows,
                        targets,
                        family_train,
                        float(args.shrinkage),
                        global_rates,
                        base_key,
                        float(args.minimum_gain),
                    )
            for index in query_indices:
                selected_key = fold_route.get(str(families[int(index)]), global_key)
                routed[int(index)] = config_rows[selected_key][int(index)]
            fold_reports.append(
                {
                    "fold": int(fold),
                    "train_rows": int(len(train_indices)),
                    "query_rows": int(len(query_indices)),
                    "global_config": json.loads(global_key),
                    "routed_families": int(len(fold_route)),
                    **metrics(
                        [targets[int(index)] for index in query_indices],
                        [routed[int(index)] for index in query_indices],
                    ),
                }
            )
        global_rates = {
            key: float(
                np.mean(
                    [target in set(row[:10]) for target, row in zip(targets, rows)]
                )
            )
            for key, rows in config_rows.items()
        }
        global_key = max(global_rates, key=global_rates.get)
        full_routing: Dict[str, str] = {}
        for family in np.unique(families):
            indices = all_indices[families == family]
            if len(indices) < int(args.min_family_rows):
                full_routing[str(family)] = global_key
            else:
                full_routing[str(family)] = choose_config(
                    configs,
                    config_rows,
                    targets,
                    indices,
                    float(args.shrinkage),
                    global_rates,
                    base_key,
                    float(args.minimum_gain),
                )

    report = {
        "protocol": f"{args.split}_formula_group_oof_shrunk_family_router",
        "config": vars(args),
        "expert_paths": expert_paths,
        "candidate_configs": configs,
        "oof": metrics(targets, routed),
        "folds": fold_reports,
        "full_routing": full_routing,
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with Path(args.output_candidates_jsonl).resolve().open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(routed):
            handle.write(
                json.dumps(
                    {
                        "row_index": row_index,
                        "candidate_label_ids": [list(value) for value in row],
                    }
                )
                + "\n"
            )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
