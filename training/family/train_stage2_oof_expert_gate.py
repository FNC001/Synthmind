#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.family.evaluate_stage2_candidate_fusion import load_source  # noqa: E402
from training.family.train_stage2_listwise_ranker import precursor_formula_features  # noqa: E402
from training.family.train_stage2_oof_candidate_stacker import formula_group_folds  # noqa: E402


SetKey = Tuple[int, ...]


def parse_named_source(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"expert must be NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    return name.strip(), path.strip()


def overlap_features(rows: Sequence[Sequence[Sequence[SetKey]]]) -> np.ndarray:
    """Prediction-only agreement features; no ground-truth information is used."""
    n_experts = len(rows)
    n_rows = len(rows[0])
    output = np.zeros((n_rows, n_experts * 7), dtype=np.float32)
    for row_index in range(n_rows):
        top_sets = {
            k: [set(expert_rows[row_index][:k]) for expert_rows in rows]
            for k in (1, 3, 10, 50)
        }
        for expert_index, expert_rows in enumerate(rows):
            current = expert_rows[row_index]
            top10 = top_sets[10][expert_index]
            base = expert_index * 7
            for offset, k in enumerate((1, 3, 10, 50)):
                reference = top_sets[k][0]
                output[row_index, base + offset] = len(top_sets[k][expert_index] & reference) / max(
                    1, len(top_sets[k][expert_index] | reference)
                )
            if current:
                top1 = current[0]
                output[row_index, base + 4] = sum(
                    top1 in top_sets[10][other] for other in range(n_experts)
                ) / max(1, n_experts)
                output[row_index, base + 5] = min(len(top1), 10) / 10.0
            output[row_index, base + 6] = float(np.mean([
                len(top10 & top_sets[10][other]) / max(1, len(top10 | top_sets[10][other]))
                for other in range(n_experts)
            ]))
    return output


def gate_features(
    formulas: Sequence[str],
    families: Sequence[str],
    expert_rows: Sequence[Sequence[Sequence[SetKey]]],
    family_vocab: Sequence[str],
) -> np.ndarray:
    chemistry = precursor_formula_features([str(value) for value in formulas])
    family_to_index = {str(value): index for index, value in enumerate(family_vocab)}
    family_one_hot = np.zeros((len(families), len(family_vocab)), dtype=np.float32)
    for row, family in enumerate(families):
        index = family_to_index.get(str(family))
        if index is not None:
            family_one_hot[row, index] = 1.0
    return np.hstack([chemistry, family_one_hot, overlap_features(expert_rows)]).astype(np.float32)


def exact_metrics(targets: Sequence[SetKey], rows: Sequence[Sequence[SetKey]]) -> Dict[str, float]:
    return {
        f"exact_hit@{k}": float(np.mean([target in set(row[:k]) for target, row in zip(targets, rows)]))
        for k in (1, 3, 5, 10, 20, 50, 100)
    }


def make_classifier(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        random_state=int(seed),
        n_jobs=-1,
        verbosity=-1,
    )


def make_regressor(seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression_l2",
        n_estimators=600,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_lambda=3.0,
        random_state=int(seed),
        n_jobs=-1,
        verbosity=-1,
    )


def fit_predict_target(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    seed: int,
    objective: str,
) -> tuple[np.ndarray, object]:
    if len(np.unique(train_y)) < 2:
        value = float(np.mean(train_y))
        return np.full(len(query_x), value, dtype=np.float32), {"constant": value}
    model = make_classifier(seed) if objective == "hit_probability" else make_regressor(seed)
    model.fit(train_x, train_y)
    if len(query_x) == 0:
        return np.zeros(0, dtype=np.float32), model
    if objective == "hit_probability":
        prediction = model.predict_proba(query_x)[:, 1]
    else:
        prediction = model.predict(query_x)
    return np.asarray(prediction, dtype=np.float32), model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OOF meta-gate that chooses among frozen Stage2 ranking experts."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--expert", action="append", default=[], help="Repeat NAME=candidates.jsonl; first is fallback/base")
    parser.add_argument("--source_limit", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--fold_strategy", choices=("formula_group", "row"), default="formula_group"
    )
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument(
        "--gate_objective", choices=("hit_probability", "switch_utility"),
        default="switch_utility",
    )
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", required=True)
    parser.add_argument("--output_model", required=True)
    args = parser.parse_args()
    if len(args.expert) < 2:
        parser.error("at least two --expert sources are required")

    input_dir = Path(args.input_dir).resolve()
    pack = np.load(input_dir / "val.npz", allow_pickle=True)
    y = np.asarray(pack["y_multi_hot"], dtype=np.float32)
    targets = [tuple(np.flatnonzero(row > 0.5).tolist()) for row in y]
    meta = pd.read_csv(input_dir / "val_meta.csv", low_memory=False)
    expert_paths = dict(parse_named_source(value) for value in args.expert)
    expert_names = list(expert_paths)
    experts = [
        load_source(path, len(targets), int(args.source_limit))
        for path in expert_paths.values()
    ]
    family_vocab = sorted(meta["family_signature_primary"].fillna("UNK").astype(str).unique().tolist())
    features = gate_features(
        meta["formula"].fillna("").astype(str).tolist(),
        meta["family_signature_primary"].fillna("UNK").astype(str).tolist(),
        experts,
        family_vocab,
    )
    hit_labels = np.asarray([
        [target in set(expert_rows[row_index][:10]) for expert_rows in experts]
        for row_index, target in enumerate(targets)
    ], dtype=np.int8)
    if args.gate_objective == "switch_utility":
        gate_labels = hit_labels.astype(np.float32) - hit_labels[:, [0]].astype(np.float32)
    else:
        gate_labels = hit_labels

    if args.fold_strategy == "formula_group":
        formula_groups = meta["family_group_key"].fillna("UNK").astype(str).to_numpy()
        splits = formula_group_folds(formula_groups, int(args.folds), int(args.seed))
    else:
        splitter = KFold(n_splits=int(args.folds), shuffle=True, random_state=int(args.seed))
        splits = list(splitter.split(features))
    probabilities = np.zeros_like(hit_labels, dtype=np.float32)
    fold_reports = []
    for fold, (train_indices, query_indices) in enumerate(splits):
        for expert_index in range(len(expert_names)):
            predicted, _ = fit_predict_target(
                features[train_indices], gate_labels[train_indices, expert_index],
                features[query_indices], int(args.seed) + fold * 1009 + expert_index,
                str(args.gate_objective),
            )
            probabilities[query_indices, expert_index] = predicted
        selected = probabilities[query_indices].argmax(axis=1)
        fold_rows = [experts[int(choice)][int(row)] for row, choice in zip(query_indices, selected)]
        fold_targets = [targets[int(row)] for row in query_indices]
        fold_reports.append({"fold": int(fold), "n_rows": int(len(query_indices)), **exact_metrics(fold_targets, fold_rows)})

    margin_trials = []
    best_margin = None
    best_selected = None
    best_rows = None
    for margin in (
        0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3,
        0.4, 0.5, 0.6, 0.75, 1.0, 2.0,
    ):
        preferred = probabilities.argmax(axis=1)
        preferred_probability = probabilities[np.arange(len(targets)), preferred]
        selected_margin = preferred.copy()
        selected_margin[preferred_probability < probabilities[:, 0] + float(margin)] = 0
        rows_margin = [experts[int(choice)][row] for row, choice in enumerate(selected_margin)]
        trial = {"margin": float(margin), **exact_metrics(targets, rows_margin)}
        margin_trials.append(trial)
        if best_margin is None or (trial["exact_hit@10"], trial["exact_hit@50"]) > (
            best_margin["exact_hit@10"], best_margin["exact_hit@50"]
        ):
            best_margin = trial
            best_selected = selected_margin
            best_rows = rows_margin
    assert best_margin is not None and best_selected is not None and best_rows is not None
    selected = best_selected
    oof_rows = best_rows
    full_models: List[object] = []
    for expert_index in range(len(expert_names)):
        _, model = fit_predict_target(
            features, gate_labels[:, expert_index], features[:0],
            int(args.seed) + 100000 + expert_index,
            str(args.gate_objective),
        )
        full_models.append(model)
    oracle = np.any(hit_labels > 0, axis=1)
    selection_counts = {
        name: int(np.count_nonzero(selected == index))
        for index, name in enumerate(expert_names)
    }
    report = {
        "protocol": f"val_{args.fold_strategy}_disjoint_oof_frozen_expert_gate",
        "config": vars(args),
        "expert_paths": expert_paths,
        "feature_dim": int(features.shape[1]),
        "family_vocab_size": int(len(family_vocab)),
        "oracle_exact_hit@10": float(oracle.mean()),
        "oof": exact_metrics(targets, oof_rows),
        "best_base_margin": best_margin,
        "base_margin_trials": margin_trials,
        "folds": fold_reports,
        "selection_counts": selection_counts,
        "per_expert_exact_hit@10": {
            name: float(hit_labels[:, index].mean())
            for index, name in enumerate(expert_names)
        },
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with Path(args.output_candidates_jsonl).resolve().open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(oof_rows):
            handle.write(json.dumps({
                "row_index": row_index,
                "selected_expert": expert_names[int(selected[row_index])],
                "candidate_label_ids": [list(candidate) for candidate in row],
            }) + "\n")
    model_output = Path(args.output_model).resolve()
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "models": full_models,
        "expert_names": expert_names,
        "family_vocab": family_vocab,
        "source_limit": int(args.source_limit),
        "seed": int(args.seed),
        "base_margin": float(best_margin["margin"]),
        "gate_objective": str(args.gate_objective),
    }, model_output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
