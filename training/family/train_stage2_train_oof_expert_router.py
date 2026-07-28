#!/usr/bin/env python3
"""Select a frozen expert using train-OOF subgroup rules and apply it to validation.

The routing configuration and subgroup switches are learned exclusively from
training rows whose expert predictions were generated out of fold.  Validation
labels are loaded only after the router is frozen and are used for reporting.
This permits train and validation candidate files to use different precursor
vocabularies as long as each file is aligned to its own data pack.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from pymatgen.core import Element

from training.family.evaluate_stage2_candidate_fusion import load_source
from training.family.evaluate_stage2_oof_chemistry_rescore import json_set


SetKey = Tuple[int, ...]


def targets_from_pack(path: Path, split: str) -> List[SetKey]:
    values = np.asarray(
        np.load(path / f"{split}.npz", allow_pickle=True)["y_multi_hot"],
        dtype=np.float32,
    )
    return [tuple(np.flatnonzero(row > 0.5).tolist()) for row in values]


def exact_metrics(targets: Sequence[SetKey], rows: Sequence[Sequence[SetKey]]) -> Dict[str, float]:
    return {
        f"exact_hit@{k}": float(
            np.mean([target in set(row[:k]) for target, row in zip(targets, rows)])
        )
        for k in (1, 3, 5, 10, 20, 50, 100)
    }


def periodic_groups(symbols: set[str]) -> str:
    groups = []
    for symbol in symbols:
        try:
            group = Element(str(symbol)).group
        except (ValueError, AttributeError, TypeError):
            group = None
        if group is not None:
            groups.append(int(group))
    return "+".join(f"G{value:02d}" for value in sorted(set(groups))) or "NONE"


def count_bin(value: int) -> str:
    if int(value) <= 2:
        return "LE2"
    if int(value) == 3:
        return "EQ3"
    if int(value) == 4:
        return "EQ4"
    return "GE5"


def route_key(row: pd.Series, kind: str) -> str:
    anions = periodic_groups(json_set(row.get("target_anion_elements", "[]")))
    cations = json_set(row.get("target_cation_elements", "[]"))
    elements = json_set(row.get("target_elements", "[]"))
    synthesis = str(row.get("synthesis_type", "UNK") or "UNK").strip().lower()
    values = [anions]
    if "cationbin" in kind:
        values.append(f"C:{count_bin(len(cations))}")
    if "elementbin" in kind:
        values.append(f"E:{count_bin(len(elements))}")
    if "synthesis" in kind:
        values.append(f"S:{synthesis}")
    return "|".join(values)


def learn_rules(
    keys: np.ndarray,
    base_hits: np.ndarray,
    expert_hits: np.ndarray,
    indices: np.ndarray,
    min_rows: int,
    min_gain_hits: int,
    min_gain_rate: float,
) -> Dict[str, dict]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for index in indices.tolist():
        grouped[str(keys[int(index)])].append(int(index))
    rules: Dict[str, dict] = {}
    for key, values in grouped.items():
        local = np.asarray(values, dtype=np.int64)
        n_rows = int(len(local))
        base_count = int(base_hits[local].sum())
        expert_count = int(expert_hits[local].sum())
        gain = expert_count - base_count
        if (
            n_rows >= int(min_rows)
            and gain >= int(min_gain_hits)
            and gain / max(1, n_rows) >= float(min_gain_rate)
        ):
            rules[key] = {
                "rows": n_rows,
                "base_hits": base_count,
                "expert_hits": expert_count,
                "gain_hits": gain,
                "gain_rate": gain / n_rows,
            }
    return rules


def routed_rows(
    base_rows: Sequence[Sequence[SetKey]],
    expert_rows: Sequence[Sequence[SetKey]],
    keys: np.ndarray,
    family_mask: np.ndarray,
    rules: Dict[str, dict],
) -> tuple[List[List[SetKey]], np.ndarray]:
    switched = np.asarray(
        [bool(family_mask[index]) and str(keys[index]) in rules for index in range(len(keys))],
        dtype=bool,
    )
    rows = [
        list(expert_rows[index] if switched[index] else base_rows[index])
        for index in range(len(keys))
    ]
    return rows, switched


def hit_vector(targets: Sequence[SetKey], rows: Sequence[Sequence[SetKey]]) -> np.ndarray:
    return np.asarray(
        [target in set(row[:10]) for target, row in zip(targets, rows)], dtype=bool
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_input_dir", required=True)
    parser.add_argument("--val_input_dir", required=True)
    parser.add_argument("--train_base_candidates", required=True)
    parser.add_argument("--train_expert_candidates", required=True)
    parser.add_argument("--val_base_candidates", required=True)
    parser.add_argument("--val_expert_candidates", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--candidate_limit", type=int, default=100)
    parser.add_argument(
        "--key_grid",
        default="anion,anion_elementbin,anion_elementbin_synthesis,anion_cationbin_elementbin,anion_cationbin_elementbin_synthesis",
    )
    parser.add_argument("--min_rows_grid", default="5,10,20,50,100")
    parser.add_argument("--min_gain_hits_grid", default="1,2,3,5,10")
    parser.add_argument("--min_gain_rate_grid", default="0,0.01,0.02,0.05,0.1")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", required=True)
    args = parser.parse_args()

    train_dir = Path(args.train_input_dir).resolve()
    val_dir = Path(args.val_input_dir).resolve()
    train_targets = targets_from_pack(train_dir, "train")
    val_targets = targets_from_pack(val_dir, "val")
    train_meta = pd.read_csv(train_dir / "train_meta.csv", low_memory=False)
    val_meta = pd.read_csv(val_dir / "val_meta.csv", low_memory=False)
    train_base = load_source(args.train_base_candidates, len(train_targets), int(args.candidate_limit))
    train_expert = load_source(
        args.train_expert_candidates, len(train_targets), int(args.candidate_limit)
    )
    val_base = load_source(args.val_base_candidates, len(val_targets), int(args.candidate_limit))
    val_expert = load_source(args.val_expert_candidates, len(val_targets), int(args.candidate_limit))

    family = str(args.family)
    train_family = (
        train_meta["family_signature_primary"].fillna("UNK").astype(str).to_numpy() == family
    )
    val_family = (
        val_meta["family_signature_primary"].fillna("UNK").astype(str).to_numpy() == family
    )
    train_base_hits = hit_vector(train_targets, train_base)
    train_expert_hits = hit_vector(train_targets, train_expert)
    train_folds = pd.to_numeric(train_meta["split_fold"], errors="raise").astype(int).to_numpy()
    family_indices = np.flatnonzero(train_family)
    if not len(family_indices):
        raise RuntimeError(f"family {family!r} has no training rows")

    key_kinds = [value.strip() for value in str(args.key_grid).split(",") if value.strip()]
    min_rows_grid = [int(value) for value in str(args.min_rows_grid).split(",")]
    min_gain_hits_grid = [int(value) for value in str(args.min_gain_hits_grid).split(",")]
    min_gain_rate_grid = [float(value) for value in str(args.min_gain_rate_grid).split(",")]
    key_cache = {
        kind: np.asarray([route_key(row, kind) for _, row in train_meta.iterrows()], dtype=object)
        for kind in key_kinds
    }
    baseline_trial = {
        "key_kind": "no_change",
        "min_rows": 0,
        "min_gain_hits": 0,
        "min_gain_rate": 0.0,
        "family_rows": int(train_family.sum()),
        "switched_rows": 0,
        "new_hits_over_base": 0,
        "lost_hits_vs_base": 0,
        "exact_hit@10": float(train_base_hits[train_family].mean()),
        "folds": [],
    }
    trials = [baseline_trial]
    best = (
        (baseline_trial["exact_hit@10"], 0, 0),
        baseline_trial,
    )
    unique_folds = sorted(np.unique(train_folds[train_family]).tolist())
    for kind in key_kinds:
        keys = key_cache[kind]
        for min_rows in min_rows_grid:
            for min_gain_hits in min_gain_hits_grid:
                for min_gain_rate in min_gain_rate_grid:
                    selected = np.zeros(len(train_targets), dtype=bool)
                    fold_reports = []
                    for fold in unique_folds:
                        fit_indices = family_indices[train_folds[family_indices] != int(fold)]
                        query_indices = family_indices[train_folds[family_indices] == int(fold)]
                        rules = learn_rules(
                            keys,
                            train_base_hits,
                            train_expert_hits,
                            fit_indices,
                            min_rows,
                            min_gain_hits,
                            min_gain_rate,
                        )
                        local_switch = np.asarray(
                            [str(keys[index]) in rules for index in query_indices], dtype=bool
                        )
                        selected[query_indices] = local_switch
                        local_hits = np.where(
                            local_switch,
                            train_expert_hits[query_indices],
                            train_base_hits[query_indices],
                        )
                        fold_reports.append(
                            {
                                "fold": int(fold),
                                "rows": int(len(query_indices)),
                                "rules": int(len(rules)),
                                "switched": int(local_switch.sum()),
                                "exact_hit@10": float(local_hits.mean()) if len(local_hits) else None,
                            }
                        )
                    routed_hits = np.where(selected, train_expert_hits, train_base_hits)
                    family_routed = routed_hits[train_family]
                    new_hits = int((routed_hits & ~train_base_hits & train_family).sum())
                    lost_hits = int((~routed_hits & train_base_hits & train_family).sum())
                    trial = {
                        "key_kind": kind,
                        "min_rows": int(min_rows),
                        "min_gain_hits": int(min_gain_hits),
                        "min_gain_rate": float(min_gain_rate),
                        "family_rows": int(train_family.sum()),
                        "switched_rows": int(selected[train_family].sum()),
                        "new_hits_over_base": new_hits,
                        "lost_hits_vs_base": lost_hits,
                        "exact_hit@10": float(family_routed.mean()),
                        "folds": fold_reports,
                    }
                    trials.append(trial)
                    key = (
                        trial["exact_hit@10"],
                        -trial["lost_hits_vs_base"],
                        -trial["switched_rows"],
                    )
                    if best is None or key > best[0]:
                        best = (key, trial)
    assert best is not None
    selected_config = best[1]
    selected_kind = str(selected_config["key_kind"])
    if selected_kind == "no_change":
        final_rules: Dict[str, dict] = {}
        val_key_kind = key_kinds[0]
    else:
        full_train_keys = key_cache[selected_kind]
        final_rules = learn_rules(
            full_train_keys,
            train_base_hits,
            train_expert_hits,
            family_indices,
            int(selected_config["min_rows"]),
            int(selected_config["min_gain_hits"]),
            float(selected_config["min_gain_rate"]),
        )
        val_key_kind = selected_kind
    val_keys = np.asarray(
        [route_key(row, val_key_kind) for _, row in val_meta.iterrows()], dtype=object
    )
    val_rows, val_switched = routed_rows(
        val_base, val_expert, val_keys, val_family, final_rules
    )
    val_base_hits = hit_vector(val_targets, val_base)
    val_hits = hit_vector(val_targets, val_rows)
    report = {
        "protocol": "train_formula_group_oof_subgroup_router_val_formula_group_disjoint",
        "selection_policy": "router configuration and subgroup rules use train OOF labels only",
        "config": vars(args),
        "train_oof": {
            "family": family,
            "base_exact_hit@10": float(train_base_hits[train_family].mean()),
            "expert_exact_hit@10": float(train_expert_hits[train_family].mean()),
            "best_router": selected_config,
            "trials": sorted(
                trials,
                key=lambda row: (
                    -row["exact_hit@10"], row["lost_hits_vs_base"], row["switched_rows"]
                ),
            )[:50],
        },
        "frozen_rules": final_rules,
        "validation": {
            "base": exact_metrics(val_targets, val_base),
            "routed": exact_metrics(val_targets, val_rows),
            "family_rows": int(val_family.sum()),
            "switched_rows": int(val_switched.sum()),
            "new_hits_over_base": int((val_hits & ~val_base_hits).sum()),
            "lost_hits_vs_base": int((~val_hits & val_base_hits).sum()),
            "final_hit_rows": int(val_hits.sum()),
        },
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    candidate_output = Path(args.output_candidates_jsonl).resolve()
    candidate_output.parent.mkdir(parents=True, exist_ok=True)
    with candidate_output.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(val_rows):
            handle.write(
                json.dumps(
                    {
                        "row_index": row_index,
                        "routed_to_expert": bool(val_switched[row_index]),
                        "route_key": str(val_keys[row_index]),
                        "candidate_label_ids": [list(value) for value in row],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(json.dumps({"train_oof": report["train_oof"]["best_router"], "validation": report["validation"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
