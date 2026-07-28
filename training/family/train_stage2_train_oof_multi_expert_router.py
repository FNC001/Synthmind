#!/usr/bin/env python3
"""Fit a multi-expert Stage-2 router on honest train-OOF predictions.

Every routing decision, threshold, risk penalty, and merge policy is selected
from formula-group-disjoint OOF predictions on the training split.  Validation
labels are evaluated only after the full router has been frozen.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from pymatgen.core import Element


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.family.evaluate_stage2_candidate_fusion import load_source  # noqa: E402
from training.family.evaluate_stage2_oof_chemistry_rescore import json_set  # noqa: E402
from training.family.train_stage2_listwise_ranker import precursor_formula_features  # noqa: E402
from training.family.train_stage2_oof_candidate_stacker import formula_group_folds  # noqa: E402
from training.family.train_stage2_oof_expert_gate import overlap_features  # noqa: E402


SetKey = Tuple[int, ...]


def parse_expert_pair(value: str) -> tuple[str, str, str]:
    """Parse NAME=TRAIN_PATH::VAL_PATH without constraining path characters."""
    if "=" not in value or "::" not in value:
        raise ValueError(
            f"expert_pair must be NAME=TRAIN_PATH::VAL_PATH, got {value!r}"
        )
    name, paths = value.split("=", 1)
    train_path, val_path = paths.split("::", 1)
    if not name.strip() or not train_path.strip() or not val_path.strip():
        raise ValueError(f"incomplete expert_pair {value!r}")
    return name.strip(), train_path.strip(), val_path.strip()


def targets_from_pack(path: Path, split: str) -> List[SetKey]:
    values = np.asarray(
        np.load(path / f"{split}.npz", allow_pickle=True)["y_multi_hot"],
        dtype=np.float32,
    )
    return [tuple(np.flatnonzero(row > 0.5).tolist()) for row in values]


def exact_metrics(
    targets: Sequence[SetKey], rows: Sequence[Sequence[SetKey]]
) -> Dict[str, float]:
    return {
        f"exact_hit@{k}": float(
            np.mean([target in set(row[:k]) for target, row in zip(targets, rows)])
        )
        for k in (1, 3, 5, 10, 20, 50, 100)
    }


def hit_vector(
    targets: Sequence[SetKey], rows: Sequence[Sequence[SetKey]], k: int = 10
) -> np.ndarray:
    return np.asarray(
        [target in set(row[: int(k)]) for target, row in zip(targets, rows)],
        dtype=bool,
    )


def periodic_groups(symbols: set[str]) -> str:
    values = []
    for symbol in symbols:
        try:
            group = Element(str(symbol)).group
        except (ValueError, AttributeError, TypeError):
            group = None
        if group is not None:
            values.append(int(group))
    return "+".join(f"G{group:02d}" for group in sorted(set(values))) or "NONE"


def count_bin(value: int) -> str:
    if int(value) <= 2:
        return "LE2"
    if int(value) == 3:
        return "EQ3"
    if int(value) == 4:
        return "EQ4"
    return "GE5"


def observable_categories(meta: pd.DataFrame) -> List[List[str]]:
    output: List[List[str]] = []
    for _, row in meta.iterrows():
        cations = json_set(row.get("target_cation_elements", "[]"))
        anions = json_set(row.get("target_anion_elements", "[]"))
        elements = json_set(row.get("target_elements", "[]"))
        output.append(
            [
                f"F:{str(row.get('family_signature_primary', 'UNK') or 'UNK')}",
                f"A:{periodic_groups(anions)}",
                f"C:{count_bin(len(cations))}",
                f"E:{count_bin(len(elements))}",
                f"S:{str(row.get('synthesis_type', 'UNK') or 'UNK').strip().lower()}",
                f"D:{str(row.get('source_dataset', 'UNK') or 'UNK').strip().lower()}",
            ]
        )
    return output


def fit_category_vocab(rows: Sequence[Sequence[str]], minimum_count: int) -> List[str]:
    counts = Counter(value for row in rows for value in row)
    return sorted(value for value, count in counts.items() if count >= int(minimum_count))


def encode_categories(rows: Sequence[Sequence[str]], vocab: Sequence[str]) -> np.ndarray:
    lookup = {str(value): index for index, value in enumerate(vocab)}
    output = np.zeros((len(rows), len(vocab)), dtype=np.float32)
    for row_index, row in enumerate(rows):
        for value in row:
            index = lookup.get(str(value))
            if index is not None:
                output[row_index, index] = 1.0
    return output


def expert_profile_features(
    experts: Sequence[Sequence[Sequence[SetKey]]],
) -> np.ndarray:
    n_rows = len(experts[0])
    output = np.zeros((n_rows, len(experts) * 5), dtype=np.float32)
    for expert_index, expert in enumerate(experts):
        base = expert_index * 5
        for row_index, candidates in enumerate(expert):
            top10 = list(candidates[:10])
            top50 = list(candidates[:50])
            lengths = [len(value) for value in top10]
            output[row_index, base] = min(len(candidates), 100) / 100.0
            output[row_index, base + 1] = len(top10[0]) / 10.0 if top10 else 0.0
            output[row_index, base + 2] = float(np.mean(lengths)) / 10.0 if lengths else 0.0
            output[row_index, base + 3] = len(set(top10)) / 10.0
            output[row_index, base + 4] = len(set(top50)) / 50.0
    return output


def gate_features(
    meta: pd.DataFrame,
    experts: Sequence[Sequence[Sequence[SetKey]]],
    category_vocab: Sequence[str],
) -> np.ndarray:
    formulas = meta["formula"].fillna("").astype(str).tolist()
    chemistry = precursor_formula_features(formulas)
    categories = encode_categories(observable_categories(meta), category_vocab)
    return np.hstack(
        [chemistry, categories, overlap_features(experts), expert_profile_features(experts)]
    ).astype(np.float32)


def make_classifier(seed: int, positive_weight: float = 1.0) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=700,
        learning_rate=0.025,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=25,
        subsample=0.85,
        colsample_bytree=0.82,
        reg_alpha=0.2,
        reg_lambda=3.0,
        scale_pos_weight=float(positive_weight),
        random_state=int(seed),
        n_jobs=-1,
        verbosity=-1,
    )


def fit_predict_binary(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, object]:
    positives = int(np.count_nonzero(train_y > 0.5))
    negatives = int(len(train_y) - positives)
    if positives == 0 or negatives == 0:
        value = float(positives > 0)
        return np.full(len(query_x), value, dtype=np.float32), {"constant": value}
    positive_weight = min(20.0, max(1.0, negatives / max(1, positives)))
    model = make_classifier(seed, positive_weight)
    model.fit(train_x, train_y)
    prediction = (
        model.predict_proba(query_x)[:, 1]
        if len(query_x)
        else np.zeros(0, dtype=np.float32)
    )
    return np.asarray(prediction, dtype=np.float32), model


def merge_expert_row(
    base: Sequence[SetKey], expert: Sequence[SetKey], base_keep: int
) -> List[SetKey]:
    """Preserve a base prefix, fill the Top-10 from the expert, then retain depth."""
    if int(base_keep) >= 10:
        return list(expert)
    prefix = list(base[: max(0, int(base_keep))])
    output = list(prefix)
    seen = set(output)
    for candidate in expert:
        if candidate not in seen:
            output.append(candidate)
            seen.add(candidate)
        if len(output) >= 10:
            break
    for source in (base, expert):
        for candidate in source:
            if candidate not in seen:
                output.append(candidate)
                seen.add(candidate)
    return output


def select_experts(
    gain_probability: np.ndarray,
    loss_probability: np.ndarray,
    risk_weight: float,
    threshold: float,
) -> np.ndarray:
    """Return 0 for the base or 1..N for a safely preferred expert."""
    utility = gain_probability - float(risk_weight) * loss_probability
    preferred = utility.argmax(axis=1)
    best = utility[np.arange(len(utility)), preferred]
    return np.where(best >= float(threshold), preferred + 1, 0).astype(np.int32)


def apply_selection(
    experts: Sequence[Sequence[Sequence[SetKey]]],
    selected: np.ndarray,
    base_keep: int,
) -> List[List[SetKey]]:
    rows: List[List[SetKey]] = []
    for row_index, choice in enumerate(selected.tolist()):
        if int(choice) == 0:
            rows.append(list(experts[0][row_index]))
        else:
            rows.append(
                merge_expert_row(
                    experts[0][row_index], experts[int(choice)][row_index], int(base_keep)
                )
            )
    return rows


def selection_report(
    names: Sequence[str], selected: np.ndarray
) -> Dict[str, int]:
    return {
        str(name): int(np.count_nonzero(selected == index))
        for index, name in enumerate(names)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_input_dir", required=True)
    parser.add_argument("--val_input_dir", required=True)
    parser.add_argument(
        "--expert_pair",
        action="append",
        default=[],
        help="Repeat NAME=TRAIN_OOF_CANDIDATES::VAL_CANDIDATES; first pair is base.",
    )
    parser.add_argument("--source_limit", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--category_min_count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", required=True)
    parser.add_argument("--output_model", required=True)
    args = parser.parse_args()
    if len(args.expert_pair) < 2:
        parser.error("at least two --expert_pair values are required")

    pairs = [parse_expert_pair(value) for value in args.expert_pair]
    names = [value[0] for value in pairs]
    if len(names) != len(set(names)):
        parser.error("expert names must be unique")
    train_dir = Path(args.train_input_dir).resolve()
    val_dir = Path(args.val_input_dir).resolve()
    train_targets = targets_from_pack(train_dir, "train")
    train_meta = pd.read_csv(train_dir / "train_meta.csv", low_memory=False)
    train_experts = [
        load_source(train_path, len(train_targets), int(args.source_limit))
        for _, train_path, _ in pairs
    ]
    category_vocab = fit_category_vocab(
        observable_categories(train_meta), int(args.category_min_count)
    )
    train_x = gate_features(train_meta, train_experts, category_vocab)
    train_hits = np.stack(
        [hit_vector(train_targets, expert) for expert in train_experts], axis=1
    )
    gain_labels = (train_hits[:, 1:] & ~train_hits[:, [0]]).astype(np.int8)
    loss_labels = (~train_hits[:, 1:] & train_hits[:, [0]]).astype(np.int8)

    if "split_fold" in train_meta:
        fold_ids = pd.to_numeric(train_meta["split_fold"], errors="raise").astype(int).to_numpy()
        unique = sorted(np.unique(fold_ids).tolist())
        splits = [
            (np.flatnonzero(fold_ids != fold), np.flatnonzero(fold_ids == fold))
            for fold in unique
        ]
    else:
        groups = train_meta["family_group_key"].fillna("UNK").astype(str).to_numpy()
        splits = formula_group_folds(groups, int(args.folds), int(args.seed))

    oof_gain = np.zeros_like(gain_labels, dtype=np.float32)
    oof_loss = np.zeros_like(loss_labels, dtype=np.float32)
    fold_reports = []
    for fold_index, (fit_indices, query_indices) in enumerate(splits):
        for expert_index in range(len(names) - 1):
            oof_gain[query_indices, expert_index], _ = fit_predict_binary(
                train_x[fit_indices],
                gain_labels[fit_indices, expert_index],
                train_x[query_indices],
                int(args.seed) + fold_index * 1009 + expert_index,
            )
            oof_loss[query_indices, expert_index], _ = fit_predict_binary(
                train_x[fit_indices],
                loss_labels[fit_indices, expert_index],
                train_x[query_indices],
                int(args.seed) + 50000 + fold_index * 1009 + expert_index,
            )
        fold_reports.append(
            {
                "fold": int(fold_index),
                "fit_rows": int(len(fit_indices)),
                "query_rows": int(len(query_indices)),
            }
        )

    trials = []
    best_key = None
    best_trial = None
    best_selected = None
    best_rows = None
    base_hits = train_hits[:, 0]
    for risk_weight in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
        for threshold in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3):
            selected = select_experts(oof_gain, oof_loss, risk_weight, threshold)
            for base_keep in (3, 5, 6, 7, 8, 9, 10):
                rows = apply_selection(train_experts, selected, base_keep)
                hits = hit_vector(train_targets, rows)
                metrics = exact_metrics(train_targets, rows)
                trial = {
                    "risk_weight": float(risk_weight),
                    "threshold": float(threshold),
                    "base_keep": int(base_keep),
                    "new_hits_over_base": int((hits & ~base_hits).sum()),
                    "lost_hits_vs_base": int((~hits & base_hits).sum()),
                    "switched_rows": int(np.count_nonzero(selected)),
                    **metrics,
                }
                trials.append(trial)
                key = (
                    trial["exact_hit@10"],
                    trial["exact_hit@5"],
                    trial["exact_hit@1"],
                    -trial["lost_hits_vs_base"],
                    -trial["switched_rows"],
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_trial = trial
                    best_selected = selected.copy()
                    best_rows = rows
    assert best_trial is not None and best_selected is not None and best_rows is not None

    gain_models: List[object] = []
    loss_models: List[object] = []
    for expert_index in range(len(names) - 1):
        _, gain_model = fit_predict_binary(
            train_x,
            gain_labels[:, expert_index],
            train_x[:0],
            int(args.seed) + 100000 + expert_index,
        )
        _, loss_model = fit_predict_binary(
            train_x,
            loss_labels[:, expert_index],
            train_x[:0],
            int(args.seed) + 150000 + expert_index,
        )
        gain_models.append(gain_model)
        loss_models.append(loss_model)

    # The router is fully frozen before validation targets are loaded.
    val_meta = pd.read_csv(val_dir / "val_meta.csv", low_memory=False)
    val_experts = [
        load_source(val_path, len(val_meta), int(args.source_limit))
        for _, _, val_path in pairs
    ]
    val_x = gate_features(val_meta, val_experts, category_vocab)
    val_gain = np.column_stack(
        [
            np.full(len(val_x), float(model["constant"]), dtype=np.float32)
            if isinstance(model, dict)
            else model.predict_proba(val_x)[:, 1]
            for model in gain_models
        ]
    ).astype(np.float32)
    val_loss = np.column_stack(
        [
            np.full(len(val_x), float(model["constant"]), dtype=np.float32)
            if isinstance(model, dict)
            else model.predict_proba(val_x)[:, 1]
            for model in loss_models
        ]
    ).astype(np.float32)
    val_selected = select_experts(
        val_gain,
        val_loss,
        float(best_trial["risk_weight"]),
        float(best_trial["threshold"]),
    )
    val_rows = apply_selection(val_experts, val_selected, int(best_trial["base_keep"]))
    val_targets = targets_from_pack(val_dir, "val")
    val_hits = np.stack(
        [hit_vector(val_targets, expert) for expert in val_experts], axis=1
    )
    val_final_hits = hit_vector(val_targets, val_rows)

    report = {
        "protocol": "train_formula_group_oof_multi_expert_router_val_formula_group_disjoint",
        "selection_policy": (
            "all model fitting, risk penalties, thresholds, and merge policies use train OOF labels only; "
            "validation labels are report-only"
        ),
        "config": vars(args),
        "expert_pairs": {
            name: {"train_oof": train_path, "validation": val_path}
            for name, train_path, val_path in pairs
        },
        "feature_dim": int(train_x.shape[1]),
        "category_vocab_size": int(len(category_vocab)),
        "train_oof": {
            "rows": int(len(train_targets)),
            "folds": fold_reports,
            "per_expert_exact_hit@10": {
                name: float(train_hits[:, index].mean()) for index, name in enumerate(names)
            },
            "oracle_exact_hit@10": float(np.any(train_hits, axis=1).mean()),
            "base": exact_metrics(train_targets, train_experts[0]),
            "best_router": best_trial,
            "selection_counts": selection_report(names, best_selected),
            "top_trials": sorted(
                trials,
                key=lambda row: (
                    -row["exact_hit@10"],
                    -row["exact_hit@5"],
                    row["lost_hits_vs_base"],
                    row["switched_rows"],
                ),
            )[:50],
        },
        "validation": {
            "rows": int(len(val_targets)),
            "per_expert_exact_hit@10": {
                name: float(val_hits[:, index].mean()) for index, name in enumerate(names)
            },
            "oracle_exact_hit@10": float(np.any(val_hits, axis=1).mean()),
            "base": exact_metrics(val_targets, val_experts[0]),
            "routed": exact_metrics(val_targets, val_rows),
            "selection_counts": selection_report(names, val_selected),
            "new_hits_over_base": int((val_final_hits & ~val_hits[:, 0]).sum()),
            "lost_hits_vs_base": int((~val_final_hits & val_hits[:, 0]).sum()),
            "final_hit_rows": int(val_final_hits.sum()),
        },
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    candidates_output = Path(args.output_candidates_jsonl).resolve()
    candidates_output.parent.mkdir(parents=True, exist_ok=True)
    with candidates_output.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(val_rows):
            handle.write(
                json.dumps(
                    {
                        "row_index": int(row_index),
                        "selected_expert": names[int(val_selected[row_index])],
                        "candidate_label_ids": [list(candidate) for candidate in row],
                    }
                )
                + "\n"
            )
    model_output = Path(args.output_model).resolve()
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "schema": "stage2_train_oof_multi_expert_router_v1",
            "expert_names": names,
            "category_vocab": category_vocab,
            "gain_models": gain_models,
            "loss_models": loss_models,
            "frozen_policy": {
                "risk_weight": float(best_trial["risk_weight"]),
                "threshold": float(best_trial["threshold"]),
                "base_keep": int(best_trial["base_keep"]),
            },
            "source_limit": int(args.source_limit),
        },
        model_output,
    )
    print(
        json.dumps(
            {
                "train_oof": report["train_oof"]["best_router"],
                "validation": report["validation"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
