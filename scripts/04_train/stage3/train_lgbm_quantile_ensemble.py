#!/usr/bin/env python3
"""
Train LightGBM Quantile Ensemble + Discrete Classifiers for Stage3.

Trains:
  1) Temperature quantile regression (9 quantiles × 1 model)
  2) Time quantile regression (9 quantiles × 1 model)
  3) Atmosphere binary classifier (oxidizing vs non-oxidizing)
  4) Time bucket multiclass classifier (short/medium/long)

Usage:
    python train_lgbm_quantile_ensemble.py
    python train_lgbm_quantile_ensemble.py --project_root /path/to/SynPred
    python train_lgbm_quantile_ensemble.py --skip_classifiers
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    print("ERROR: lightgbm required. Install: pip install lightgbm")
    sys.exit(1)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    r2_score,
)

DEFAULT_PROJECT_ROOT = Path("/Users/wyc/SynPred")
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

QUANTILE_PARAMS = {
    "objective": "quantile",
    "metric": "quantile",
    "boosting_type": "gbdt",
    "num_leaves": 255,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 20,
    "verbose": -1,
    "seed": 42,
    "n_jobs": -1,
}

ATM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 10,
    "verbose": -1,
    "seed": 42,
    "n_jobs": -1,
}

TIME_BUCKET_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 10,
    "verbose": -1,
    "seed": 42,
    "n_jobs": -1,
}


def load_npz_data(input_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """Load train/val/test NPZ files."""
    data = {}
    for split in ["train", "val", "test"]:
        npz = np.load(input_dir / f"{split}.npz", allow_pickle=True)
        data[split] = {k: npz[k] for k in npz.files}
    return data


def prepare_quantile_features(data: Dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate x and y_set as features (matching inference behavior)."""
    return np.hstack([data["x"], data["y_set"]])


def train_quantile_models(
    data: Dict[str, Dict[str, np.ndarray]],
    target_idx: int,
    target_name: str,
    output_dir: Path,
    num_boost_round: int = 200,
    early_stopping_rounds: int = 20,
) -> Dict[str, Any]:
    """Train quantile models for a single target variable."""
    X_train = prepare_quantile_features(data["train"])
    X_val = prepare_quantile_features(data["val"])
    X_test = prepare_quantile_features(data["test"])

    y_train = data["train"]["y_cond_continuous"][:, target_idx]
    y_val = data["val"]["y_cond_continuous"][:, target_idx]
    y_test = data["test"]["y_cond_continuous"][:, target_idx]

    mask_train = data["train"]["y_cond_continuous_mask"][:, target_idx] > 0.5
    mask_val = data["val"]["y_cond_continuous_mask"][:, target_idx] > 0.5
    mask_test = data["test"]["y_cond_continuous_mask"][:, target_idx] > 0.5

    X_train_m, y_train_m = X_train[mask_train], y_train[mask_train]
    X_val_m, y_val_m = X_val[mask_val], y_val[mask_val]
    X_test_m, y_test_m = X_test[mask_test], y_test[mask_test]

    print(f"\n  [{target_name}] train={mask_train.sum()}, val={mask_val.sum()}, test={mask_test.sum()}")

    test_preds = {}
    for q in QUANTILES:
        params = {**QUANTILE_PARAMS, "alpha": q}
        dtrain = lgb.Dataset(X_train_m, label=y_train_m)
        dval = lgb.Dataset(X_val_m, label=y_val_m, reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(0)],
        )

        model_path = output_dir / f"{target_name}_q{q:.1f}.txt"
        model.save_model(str(model_path))
        test_preds[q] = model.predict(X_test_m)
        print(f"    q={q:.1f}: {model.best_iteration} iters")

    # Evaluate: top1 = median (q=0.5)
    top1_pred = test_preds[0.5]
    mae = float(np.abs(y_test_m - top1_pred).mean())
    median_ae = float(np.median(np.abs(y_test_m - top1_pred)))
    r2 = float(r2_score(y_test_m, top1_pred))

    # Oracle: best quantile per sample
    all_preds = np.stack([test_preds[q] for q in QUANTILES], axis=1)
    errors = np.abs(all_preds - y_test_m[:, None])
    oracle_pred = all_preds[np.arange(len(y_test_m)), errors.argmin(axis=1)]
    oracle_mae = float(np.abs(y_test_m - oracle_pred).mean())

    metrics = {
        "top1_mae": mae,
        "top1_median_ae": median_ae,
        "top1_r2": r2,
        "oracle_mae": oracle_mae,
        "n_test": int(mask_test.sum()),
    }

    # Within-threshold metrics
    abs_err = np.abs(y_test_m - top1_pred)
    if target_name == "temp":
        for thresh in [25, 50, 100, 200]:
            metrics[f"top1_within_{thresh}"] = float((abs_err <= thresh).mean())
    else:
        for thresh in [2, 5, 10]:
            metrics[f"top1_within_{thresh}h"] = float((abs_err <= thresh).mean())

    return metrics


def train_atmosphere_classifier(
    task_views_dir: Path,
    output_dir: Path,
    num_boost_round: int = 200,
    early_stopping_rounds: int = 20,
) -> Dict[str, Any]:
    """Train binary atmosphere classifier (oxidizing vs non-oxidizing)."""
    print("\n  [atmosphere] Loading task views...")

    df_train = pd.read_csv(task_views_dir / "atmosphere_coarse_train.csv")
    df_val = pd.read_csv(task_views_dir / "atmosphere_coarse_val.csv")
    df_test = pd.read_csv(task_views_dir / "atmosphere_coarse_test.csv")

    meta_cols = [
        "id", "material_id", "formula", "doi", "split_group",
        "source_dataset", "synthesis_type", "target_atmosphere_coarse",
        "target_atmosphere", "has_target",
    ]
    feat_cols = [c for c in df_train.columns if c not in meta_cols]

    # Binary: oxidizing (air_or_oxidizing) vs non-oxidizing (inert/reducing/vacuum)
    def to_binary(df: pd.DataFrame) -> np.ndarray:
        return (df["target_atmosphere_coarse"] == "air_or_oxidizing").astype(int).values

    X_train, y_train = df_train[feat_cols].values, to_binary(df_train)
    X_val, y_val = df_val[feat_cols].values, to_binary(df_val)
    X_test, y_test = df_test[feat_cols].values, to_binary(df_test)

    # Filter rows with valid targets
    valid_train = df_train["has_target"].values > 0.5
    valid_val = df_val["has_target"].values > 0.5
    valid_test = df_test["has_target"].values > 0.5

    X_train, y_train = X_train[valid_train], y_train[valid_train]
    X_val, y_val = X_val[valid_val], y_val[valid_val]
    X_test, y_test = X_test[valid_test], y_test[valid_test]

    print(f"    train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        ATM_PARAMS,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(0)],
    )

    model_path = output_dir / "model_atmosphere_binary_final.txt"
    model.save_model(str(model_path))

    proba = model.predict(X_test)
    threshold = 0.6
    pred = (proba >= threshold).astype(int)
    acc = float(accuracy_score(y_test, pred))
    macro_f1 = float(f1_score(y_test, pred, average="macro"))

    metrics = {
        "binary": {
            "threshold": threshold,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "classes": ["oxidizing (air_or_oxidizing)", "non-oxidizing (inert/reducing/vacuum)"],
        },
        "n_features": len(feat_cols),
        "train_size": len(y_train),
        "test_size": len(y_test),
        "best_iteration": model.best_iteration,
    }

    print(f"    accuracy={acc:.4f}, macro_f1={macro_f1:.4f}")
    return metrics


def train_time_bucket_classifier(
    task_views_dir: Path,
    output_dir: Path,
    num_boost_round: int = 200,
    early_stopping_rounds: int = 20,
) -> Dict[str, Any]:
    """Train multiclass time bucket classifier (short/medium/long)."""
    print("\n  [time_bucket] Loading task views...")

    df_train = pd.read_csv(task_views_dir / "time_bucket_train.csv")
    df_val = pd.read_csv(task_views_dir / "time_bucket_val.csv")
    df_test = pd.read_csv(task_views_dir / "time_bucket_test.csv")

    meta_cols = [
        "id", "material_id", "formula", "doi", "split_group",
        "source_dataset", "synthesis_type", "target_time_bucket",
        "target_time_h", "has_target",
    ]
    leak_cols = {"target_time_h_log1p", "target_time_h_clean"}
    feat_cols = [c for c in df_train.columns if c not in meta_cols and c not in leak_cols]

    classes = ["short", "medium", "long"]
    class_map = {c: i for i, c in enumerate(classes)}

    def to_label(df: pd.DataFrame) -> np.ndarray:
        return df["target_time_bucket"].map(class_map).values

    X_train, y_train = df_train[feat_cols].values, to_label(df_train)
    X_val, y_val = df_val[feat_cols].values, to_label(df_val)
    X_test, y_test = df_test[feat_cols].values, to_label(df_test)

    # Filter valid
    valid_train = ~np.isnan(y_train)
    valid_val = ~np.isnan(y_val)
    valid_test = ~np.isnan(y_test)

    X_train, y_train = X_train[valid_train], y_train[valid_train].astype(int)
    X_val, y_val = X_val[valid_val], y_val[valid_val].astype(int)
    X_test, y_test = X_test[valid_test], y_test[valid_test].astype(int)

    print(f"    train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        TIME_BUCKET_PARAMS,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(0)],
    )

    model_path = output_dir / "model_time_bucket.txt"
    model.save_model(str(model_path))

    proba = model.predict(X_test)
    pred = proba.argmax(axis=1)
    acc = float(accuracy_score(y_test, pred))
    macro_f1 = float(f1_score(y_test, pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, pred, average="weighted"))

    metrics = {
        "model": "lgbm_time_bucket_classifier",
        "task": "time_bucket",
        "classes": classes,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "train_size": len(y_train),
        "test_size": len(y_test),
        "best_iteration": model.best_iteration,
    }

    print(f"    accuracy={acc:.4f}, macro_f1={macro_f1:.4f}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train LightGBM quantile ensemble + classifiers for Stage3."
    )
    parser.add_argument("--project_root", type=str, default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--input_dir", type=str, default="")
    parser.add_argument("--task_views_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--atm_output_dir", type=str, default="")
    parser.add_argument("--time_bucket_output_dir", type=str, default="")
    parser.add_argument("--num_boost_round", type=int, default=200)
    parser.add_argument("--early_stopping_rounds", type=int, default=20)
    parser.add_argument("--skip_quantile", action="store_true")
    parser.add_argument("--skip_classifiers", action="store_true")
    args = parser.parse_args()

    root = Path(args.project_root)
    input_dir = Path(args.input_dir) if args.input_dir else (
        root / "data/interim/generative/stage3_condition_dataset/hybrid_mixed_v1"
    )
    task_views_dir = Path(args.task_views_dir) if args.task_views_dir else (
        root / "data/interim/features/stage3_task_views"
    )
    output_dir = Path(args.output_dir) if args.output_dir else (
        root / "runs/stage3/lgbm_quantile_ensemble_v2_fulldata"
    )
    atm_output_dir = Path(args.atm_output_dir) if args.atm_output_dir else (
        root / "runs/stage3/lgbm_atmosphere_classifier_v1"
    )
    time_bucket_output_dir = Path(args.time_bucket_output_dir) if args.time_bucket_output_dir else (
        root / "runs/stage3/lgbm_time_bucket_classifier_v1"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    atm_output_dir.mkdir(parents=True, exist_ok=True)
    time_bucket_output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Stage3 LightGBM Training")
    print("=" * 60)
    print(f"  input_dir:    {input_dir}")
    print(f"  task_views:   {task_views_dir}")
    print(f"  output_dir:   {output_dir}")
    print(f"  atm_dir:      {atm_output_dir}")
    print(f"  time_bucket:  {time_bucket_output_dir}")

    all_metrics: Dict[str, Any] = {
        "model": "lgbm_quantile_ensemble_v2_fulldata",
    }

    # --- Quantile ensemble ---
    if not args.skip_quantile:
        print("\n[1/3] Training quantile ensemble...")
        data = load_npz_data(input_dir)
        all_metrics["data"] = f"hybrid_mixed_v1 (full, {len(data['train']['x'])})"
        all_metrics["n_quantiles"] = len(QUANTILES)

        schema = json.loads((input_dir / "schema.json").read_text())
        cont_names = list(schema["continuous_schema"].keys())

        for idx, name in enumerate(cont_names):
            short_name = "temp" if "temperature" in name else "time"
            metrics = train_quantile_models(
                data, idx, short_name, output_dir,
                num_boost_round=args.num_boost_round,
                early_stopping_rounds=args.early_stopping_rounds,
            )
            all_metrics[name.replace("_c", "").replace("_h", "")] = metrics

        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False))
        print(f"\n  [SAVED] {metrics_path}")

    # --- Atmosphere classifier ---
    if not args.skip_classifiers:
        print("\n[2/3] Training atmosphere classifier...")
        atm_metrics = train_atmosphere_classifier(
            task_views_dir, atm_output_dir,
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        atm_metrics_path = atm_output_dir / "metrics_final.json"
        atm_metrics_path.write_text(json.dumps(atm_metrics, indent=2, ensure_ascii=False))
        print(f"  [SAVED] {atm_metrics_path}")

        print("\n[3/3] Training time bucket classifier...")
        tb_metrics = train_time_bucket_classifier(
            task_views_dir, time_bucket_output_dir,
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        tb_metrics_path = time_bucket_output_dir / "metrics.json"
        tb_metrics_path.write_text(json.dumps(tb_metrics, indent=2, ensure_ascii=False))
        print(f"  [SAVED] {tb_metrics_path}")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
