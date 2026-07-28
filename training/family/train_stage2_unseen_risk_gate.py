#!/usr/bin/env python3
"""Train an OOF unseen-label risk gate from query and candidate observables."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from pymatgen.core import Element
from sklearn.metrics import average_precision_score, roc_auc_score

from training.family.evaluate_stage2_candidate_fusion import load_source
from training.family.train_stage2_unseen_candidate_ranker import (
    build_oof_reference_statistics,
    build_query_features,
    exact_metrics,
    json_set,
    key_from_row,
    merge_top10,
    slice_metrics,
)


SetKey = Tuple[int, ...]
TOP_K = (1, 3, 5, 10, 20, 50, 100)


def label_cation_touch(names: Sequence[str]) -> List[set[str]]:
    output: List[set[str]] = []
    import re

    pattern = re.compile(r"[A-Z][a-z]?")
    for name in names:
        metals = set()
        for symbol in pattern.findall(str(name)):
            try:
                element = Element(symbol)
                if bool(element.is_metal) or bool(getattr(element, "is_metalloid", False)):
                    metals.add(symbol)
            except ValueError:
                continue
        output.append(metals)
    return output


def candidate_observables(
    path: Path,
    n_rows: int,
    meta: pd.DataFrame,
    seen_masks: Sequence[np.ndarray],
    row_mask_index: np.ndarray,
    label_metals: Sequence[set[str]],
    limit: int,
) -> tuple[np.ndarray, List[str]]:
    cutoffs = (10, 20, 50, 100, 500)
    names = []
    for cutoff in cutoffs:
        names.extend(
            [
                f"candidate_unseen_fraction_at_{cutoff}",
                f"candidate_unseen_count_mean_at_{cutoff}",
                f"candidate_unique_unseen_labels_at_{cutoff}",
                f"candidate_target_touching_unseen_labels_at_{cutoff}",
            ]
        )
    names.extend(
        [
            "candidate_first_unseen_log_rank",
            "candidate_first_unseen_score_delta",
            "candidate_total_count_log",
        ]
    )
    output = np.zeros((n_rows, len(names)), dtype=np.float32)
    found = np.zeros(n_rows, dtype=bool)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            row = int(record["row_index"])
            if row < 0 or row >= n_rows:
                continue
            candidates = list(record.get("candidate_label_ids", []))[: int(limit)]
            scores = list(record.get("scores", []))[: len(candidates)]
            seen = seen_masks[int(row_mask_index[row])]
            target_cations = json_set(meta.iloc[row]["target_cation_elements"])
            unseen_counts = []
            unique_prefix: set[int] = set()
            touching_prefix: set[int] = set()
            first_rank = 0
            first_score_delta = -50.0
            top_score = float(scores[0]) if scores else 0.0
            offset = 0
            for cutoff in cutoffs:
                prefix = candidates[: min(int(cutoff), len(candidates))]
                while len(unseen_counts) < len(prefix):
                    index = len(unseen_counts)
                    candidate = prefix[index]
                    unseen_labels = [int(label) for label in candidate if not bool(seen[int(label)])]
                    unseen_counts.append(len(unseen_labels))
                    unique_prefix.update(unseen_labels)
                    touching_prefix.update(
                        label for label in unseen_labels if label_metals[label] & target_cations
                    )
                    if unseen_labels and first_rank == 0:
                        first_rank = index + 1
                        if index < len(scores):
                            first_score_delta = float(np.clip(float(scores[index]) - top_score, -50.0, 0.0))
                values = np.asarray(unseen_counts, dtype=np.float32)
                output[row, offset : offset + 4] = [
                    float(np.mean(values > 0)) if len(values) else 0.0,
                    float(values.mean()) if len(values) else 0.0,
                    min(len(unique_prefix), 100) / 100.0,
                    min(len(touching_prefix), 100) / 100.0,
                ]
                offset += 4
            output[row, offset:] = [
                math.log1p(first_rank) if first_rank else math.log1p(max(1, len(candidates)) + 1),
                first_score_delta,
                math.log1p(len(candidates)),
            ]
            found[row] = True
    if not found.all():
        raise ValueError(f"candidate signal source is missing {int((~found).sum())} rows: {path}")
    return output, names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--oof_fold_dir", required=True)
    parser.add_argument("--train_signal_candidates", required=True)
    parser.add_argument("--val_signal_candidates", required=True)
    parser.add_argument("--base_val_candidates", required=True)
    parser.add_argument("--specialist_val_candidates", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", required=True)
    parser.add_argument("--signal_limit", type=int, default=500)
    parser.add_argument("--candidate_limit", type=int, default=100)
    parser.add_argument("--n_jobs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    train_pack = np.load(input_dir / "train.npz", allow_pickle=True)
    val_pack = np.load(input_dir / "val.npz", allow_pickle=True)
    train_y = np.asarray(train_pack["y_multi_hot"], dtype=np.float32)
    val_y = np.asarray(val_pack["y_multi_hot"], dtype=np.float32)
    train_targets = [key_from_row(row) for row in train_y]
    val_targets = [key_from_row(row) for row in val_y]
    train_meta = pd.read_csv(input_dir / "train_meta.csv", low_memory=False)
    val_meta = pd.read_csv(input_dir / "val_meta.csv", low_memory=False)
    train_query, val_query, query_names = build_query_features(
        np.asarray(train_pack["x"], dtype=np.float32),
        np.asarray(val_pack["x"], dtype=np.float32),
        train_meta,
        val_meta,
    )
    precursor_names = json.loads((input_dir / "precursor_names.json").read_text(encoding="utf-8"))
    label_metals = label_cation_touch([str(value) for value in precursor_names])
    row_fold, fold_seen, _, fold_names = build_oof_reference_statistics(
        train_y, Path(args.oof_fold_dir).resolve()
    )
    pseudo_unseen = np.asarray(
        [
            any(not bool(fold_seen[int(row_fold[row])][label]) for label in target)
            for row, target in enumerate(train_targets)
        ],
        dtype=bool,
    )
    full_seen = np.asarray(train_y.sum(axis=0) > 0)
    val_unseen = np.asarray(
        [any(not bool(full_seen[label]) for label in target) for target in val_targets], dtype=bool
    )
    train_signal, signal_names = candidate_observables(
        Path(args.train_signal_candidates).resolve(),
        len(train_targets),
        train_meta,
        fold_seen,
        row_fold,
        label_metals,
        int(args.signal_limit),
    )
    val_signal, _ = candidate_observables(
        Path(args.val_signal_candidates).resolve(),
        len(val_targets),
        val_meta,
        [full_seen],
        np.zeros(len(val_targets), dtype=np.int16),
        label_metals,
        int(args.signal_limit),
    )
    train_features = np.hstack([train_query, train_signal]).astype(np.float32)
    val_features = np.hstack([val_query, val_signal]).astype(np.float32)
    base_rows = load_source(args.base_val_candidates, len(val_targets), int(args.candidate_limit))
    specialist_rows = load_source(
        args.specialist_val_candidates, len(val_targets), int(args.candidate_limit)
    )

    trials = []
    best = None
    best_rows: List[List[SetKey]] = []
    best_model = None
    best_probability = None
    for num_leaves in (31, 63, 127, 255):
        for min_child_samples in (20, 50):
            for estimators in (400, 900, 1600):
                model = lgb.LGBMClassifier(
                    objective="binary",
                    n_estimators=estimators,
                    learning_rate=0.025,
                    num_leaves=num_leaves,
                    min_child_samples=min_child_samples,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    class_weight="balanced",
                    random_state=int(args.seed),
                    n_jobs=int(args.n_jobs),
                    verbosity=-1,
                )
                model.fit(train_features, pseudo_unseen.astype(np.int32))
                probability = model.predict_proba(val_features)[:, 1]
                thresholds = sorted(
                    {
                        0.0,
                        1.0,
                        *np.linspace(0.02, 0.98, 49).tolist(),
                        *np.quantile(probability, np.linspace(0.01, 0.99, 49)).tolist(),
                    }
                )
                for threshold in thresholds:
                    gate_mask = probability >= float(threshold)
                    for slots in range(1, 11):
                        rows = [
                            merge_top10(base, specialist, slots) if gated else list(base)
                            for base, specialist, gated in zip(base_rows, specialist_rows, gate_mask)
                        ]
                        metrics = exact_metrics(val_targets, rows)
                        trial = {
                            "num_leaves": num_leaves,
                            "min_child_samples": min_child_samples,
                            "estimators": estimators,
                            "threshold": float(threshold),
                            "specialist_slots": slots,
                            "gated_rows": int(gate_mask.sum()),
                            "gate_roc_auc": float(roc_auc_score(val_unseen, probability)),
                            "gate_average_precision": float(average_precision_score(val_unseen, probability)),
                            **metrics,
                        }
                        key = (metrics["exact_hit@10"], metrics["exact_hit@5"], metrics["exact_hit@1"])
                        if best is None or key > best[0]:
                            best = (key, trial)
                            best_rows = rows
                            best_model = model
                            best_probability = probability
                        trials.append(trial)
                print(json.dumps({
                    "stage": "gate_model_complete",
                    "num_leaves": num_leaves,
                    "min_child_samples": min_child_samples,
                    "estimators": estimators,
                    "roc_auc": float(roc_auc_score(val_unseen, probability)),
                }), flush=True)
    assert best is not None and best_model is not None and best_probability is not None
    report = {
        "protocol": "val_formula_disjoint_oof_candidate_observable_unseen_risk_gate",
        "config": vars(args),
        "training": {
            "rows": len(train_targets),
            "pseudo_unseen_rows": int(pseudo_unseen.sum()),
            "oof_folds": fold_names,
        },
        "validation": {
            "rows": len(val_targets),
            "unseen_rows": int(val_unseen.sum()),
            "base": exact_metrics(val_targets, base_rows),
            "specialist": {
                "all": exact_metrics(val_targets, specialist_rows),
                "seen": slice_metrics(val_targets, specialist_rows, ~val_unseen),
                "unseen": slice_metrics(val_targets, specialist_rows, val_unseen),
            },
            "best": best[1],
            "best_slices": {
                "seen": slice_metrics(val_targets, best_rows, ~val_unseen),
                "unseen": slice_metrics(val_targets, best_rows, val_unseen),
            },
            "top_trials": sorted(
                trials,
                key=lambda row: (row["exact_hit@10"], row["exact_hit@5"], row["exact_hit@1"]),
                reverse=True,
            )[:100],
        },
        "feature_names": [*query_names, *signal_names],
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    candidate_output = Path(args.output_candidates_jsonl).resolve()
    candidate_output.parent.mkdir(parents=True, exist_ok=True)
    with candidate_output.open("w", encoding="utf-8") as handle:
        for row_index, (row, probability) in enumerate(zip(best_rows, best_probability)):
            handle.write(json.dumps({
                "row_index": row_index,
                "candidate_label_ids": [list(value) for value in row],
                "gate_probability": float(probability),
            }) + "\n")
    best_model.booster_.save_model(str(output.with_suffix(".model.txt")))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
