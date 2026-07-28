#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance


def stable_seed(value: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def subsample(values: np.ndarray, limit: int, seed: int) -> np.ndarray:
    values = np.asarray(values)
    if limit <= 0 or len(values) <= limit:
        return values
    rng = np.random.default_rng(seed)
    return values[rng.choice(len(values), size=limit, replace=False)]


def coverage_1d(predicted: np.ndarray, truth: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    if not len(predicted) or not len(truth):
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
    distance = np.abs(predicted[:, None] - truth[None, :])
    precision = float(np.mean(distance.min(axis=1) <= float(threshold)))
    recall = float(np.mean(distance.min(axis=0) <= float(threshold)))
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1}


def categorical_coverage(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    predicted = np.asarray(predicted, dtype=np.int64).reshape(-1)
    truth = np.asarray(truth, dtype=np.int64).reshape(-1)
    if not len(predicted) or not len(truth):
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
    predicted_support = set(predicted.tolist())
    truth_support = set(truth.tolist())
    precision = float(np.mean([value in truth_support for value in predicted]))
    recall = float(np.mean([value in predicted_support for value in truth]))
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1}


def finite_mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else float("nan")


def categorical_js(predicted: np.ndarray, truth: np.ndarray, classes: int) -> float:
    pred_hist = np.bincount(predicted.astype(np.int64), minlength=classes).astype(np.float64)
    true_hist = np.bincount(truth.astype(np.int64), minlength=classes).astype(np.float64)
    pred_hist = (pred_hist + 1e-8) / (pred_hist.sum() + 1e-8 * classes)
    true_hist = (true_hist + 1e-8) / (true_hist.sum() + 1e-8 * classes)
    return float(jensenshannon(pred_hist, true_hist, base=2.0) ** 2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DiffSyn-style distribution coverage evaluation for Stage3 samples."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument(
        "--predictions_npz",
        required=True,
        help=(
            "NPZ containing continuous_samples [rows,samples,continuous_fields] and "
            "discrete_samples [rows,samples,discrete_fields]."
        ),
    )
    parser.add_argument("--group_column", default="family_group_key")
    parser.add_argument("--temperature_threshold", type=float, default=5.0)
    parser.add_argument("--time_threshold", type=float, default=12.0)
    parser.add_argument("--coverage_sample_limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    schema = json.loads((input_dir / "schema.json").read_text(encoding="utf-8"))
    true_pack = np.load(input_dir / f"{args.split}.npz", allow_pickle=True)
    predictions = np.load(Path(args.predictions_npz).resolve(), allow_pickle=False)
    continuous_true = np.asarray(true_pack["y_cond_continuous_raw"], dtype=np.float64)
    continuous_true_mask = np.asarray(true_pack["y_cond_continuous_mask"], dtype=np.float32) > 0.5
    discrete_true = np.asarray(true_pack["y_cond_discrete"], dtype=np.int64)
    discrete_true_mask = np.asarray(true_pack["y_cond_discrete_mask"], dtype=np.float32) > 0.5
    continuous_samples = np.asarray(predictions["continuous_samples"], dtype=np.float64)
    discrete_samples = np.asarray(predictions["discrete_samples"], dtype=np.int64)
    expected_continuous = (len(continuous_true), None, continuous_true.shape[1])
    expected_discrete = (len(discrete_true), None, discrete_true.shape[1])
    if (
        continuous_samples.ndim != 3
        or continuous_samples.shape[0] != expected_continuous[0]
        or continuous_samples.shape[2] != expected_continuous[2]
    ):
        raise ValueError(
            f"continuous_samples shape {continuous_samples.shape} does not match {expected_continuous}"
        )
    if (
        discrete_samples.ndim != 3
        or discrete_samples.shape[0] != expected_discrete[0]
        or discrete_samples.shape[2] != expected_discrete[2]
    ):
        raise ValueError(
            f"discrete_samples shape {discrete_samples.shape} does not match {expected_discrete}"
        )

    meta = pd.read_csv(input_dir / f"{args.split}_meta.csv", low_memory=False)
    if args.group_column not in meta:
        raise ValueError(f"missing group column {args.group_column!r}")
    groups = meta[args.group_column].fillna("UNK").astype(str).to_numpy()
    continuous_names = [str(value) for value in schema["continuous_cols"]]
    discrete_names = [str(value) for value in schema["discrete_cols"]]
    thresholds = {
        "temperature_c": float(args.temperature_threshold),
        "time_h": float(args.time_threshold),
    }
    continuous_reports: dict[str, Any] = {}
    discrete_reports: dict[str, Any] = {}

    for field_index, name in enumerate(continuous_names):
        rows = []
        for group in np.unique(groups):
            indices = np.flatnonzero(groups == group)
            truth = continuous_true[indices, field_index][continuous_true_mask[indices, field_index]]
            predicted = continuous_samples[indices, :, field_index].reshape(-1)
            predicted = predicted[np.isfinite(predicted)]
            if not len(truth) or not len(predicted):
                continue
            predicted = subsample(
                predicted,
                int(args.coverage_sample_limit),
                stable_seed(f"continuous:{name}:{group}", int(args.seed)),
            )
            coverage = coverage_1d(predicted, truth, thresholds[name])
            rows.append({
                "group": str(group),
                "truth_rows": int(len(truth)),
                "generated_rows": int(len(predicted)),
                "wasserstein": float(wasserstein_distance(truth, predicted)),
                "mean_absolute_error": float(abs(truth.mean() - predicted.mean())),
                **coverage,
            })
        continuous_reports[name] = {
            "threshold": thresholds[name],
            "systems": int(len(rows)),
            "wasserstein_macro": finite_mean([row["wasserstein"] for row in rows]),
            "mean_absolute_error_macro": finite_mean([row["mean_absolute_error"] for row in rows]),
            "coverage_precision_macro": finite_mean([row["precision"] for row in rows]),
            "coverage_recall_macro": finite_mean([row["recall"] for row in rows]),
            "coverage_f1_macro": finite_mean([row["f1"] for row in rows]),
            "per_system": rows,
        }

    for field_index, name in enumerate(discrete_names):
        classes = len(schema["discrete_schema"][name]["vocab"])
        rows = []
        for group in np.unique(groups):
            indices = np.flatnonzero(groups == group)
            truth = discrete_true[indices, field_index][discrete_true_mask[indices, field_index]]
            predicted = discrete_samples[indices, :, field_index].reshape(-1)
            predicted = predicted[(predicted >= 0) & (predicted < classes)]
            if not len(truth) or not len(predicted):
                continue
            predicted = subsample(
                predicted,
                int(args.coverage_sample_limit),
                stable_seed(f"discrete:{name}:{group}", int(args.seed)),
            )
            coverage = categorical_coverage(predicted, truth)
            rows.append({
                "group": str(group),
                "truth_rows": int(len(truth)),
                "generated_rows": int(len(predicted)),
                "jensen_shannon": categorical_js(predicted, truth, classes),
                **coverage,
            })
        discrete_reports[name] = {
            "classes": int(classes),
            "systems": int(len(rows)),
            "jensen_shannon_macro": finite_mean([row["jensen_shannon"] for row in rows]),
            "coverage_precision_macro": finite_mean([row["precision"] for row in rows]),
            "coverage_recall_macro": finite_mean([row["recall"] for row in rows]),
            "coverage_f1_macro": finite_mean([row["f1"] for row in rows]),
            "per_system": rows,
        }

    field_f1 = [
        report["coverage_f1_macro"]
        for report in [*continuous_reports.values(), *discrete_reports.values()]
    ]
    report = {
        "protocol": f"{args.split}_formula_group_macro_diffsyn_style_distribution_coverage",
        "config": vars(args),
        "rows": int(len(groups)),
        "systems": int(pd.Series(groups).nunique()),
        "samples_per_row": {
            "continuous": int(continuous_samples.shape[1]),
            "discrete": int(discrete_samples.shape[1]),
        },
        "continuous": continuous_reports,
        "discrete": discrete_reports,
        "coverage_f1_field_macro": finite_mean(field_f1),
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
