#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score


SPLITS = ("train", "val", "test")


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(to_builtin(value), ensure_ascii=False, indent=2), encoding="utf-8")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else math.nan,
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def train_regressor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    rounds: int,
    patience: int,
    num_threads: int,
) -> lgb.Booster:
    params = {
        "objective": "regression_l1",
        "metric": "l1",
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "learning_rate": 0.035,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 20,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": int(seed),
        "num_threads": int(num_threads),
    }
    train = lgb.Dataset(x_train, label=y_train)
    val = lgb.Dataset(x_val, label=y_val, reference=train)
    return lgb.train(
        params,
        train,
        num_boost_round=int(rounds),
        valid_sets=[val],
        callbacks=[lgb.early_stopping(int(patience)), lgb.log_evaluation(100)],
    )


def train_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    seed: int,
    rounds: int,
    patience: int,
    num_threads: int,
) -> lgb.Booster:
    params = {
        "objective": "multiclass",
        "num_class": int(n_classes),
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.04,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 10,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": int(seed),
        "num_threads": int(num_threads),
    }
    train = lgb.Dataset(x_train, label=y_train)
    val = lgb.Dataset(x_val, label=y_val, reference=train)
    return lgb.train(
        params,
        train,
        num_boost_round=int(rounds),
        valid_sets=[val],
        callbacks=[lgb.early_stopping(int(patience)), lgb.log_evaluation(100)],
    )


def predict_regression(model: lgb.Booster, x: np.ndarray, log_target: bool) -> np.ndarray:
    prediction = np.asarray(model.predict(x, num_iteration=model.best_iteration), dtype=np.float32)
    return np.expm1(prediction) if log_target else prediction


def evaluate(
    pack: Mapping[str, np.ndarray],
    predictions: Mapping[str, np.ndarray],
    meta: pd.DataFrame,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"continuous": {}, "discrete": {}, "by_family": {}}
    for index, name in enumerate(("temperature_c", "time_h")):
        mask = pack["y_cond_continuous_mask"][:, index] > 0.5
        result["continuous"][name] = {
            **regression_metrics(pack["y_cond_continuous_raw"][mask, index], predictions[name][mask]),
            "n": int(mask.sum()),
        }
    for index, name in enumerate(("atmosphere_coarse", "reaction_method")):
        mask = pack["y_cond_discrete_mask"][:, index] > 0.5
        result["discrete"][name] = {
            **classification_metrics(pack["y_cond_discrete"][mask, index], predictions[name][mask]),
            "n": int(mask.sum()),
        }
    family_values = meta["family_signature_primary"].fillna("UNK").astype(str).to_numpy()
    for family in sorted(set(family_values)):
        family_mask = family_values == family
        family_result: Dict[str, Any] = {"n": int(family_mask.sum())}
        for index, name in enumerate(("temperature_c", "time_h")):
            mask = family_mask & (pack["y_cond_continuous_mask"][:, index] > 0.5)
            if mask.sum() >= 2:
                family_result[name] = {
                    **regression_metrics(pack["y_cond_continuous_raw"][mask, index], predictions[name][mask]),
                    "n": int(mask.sum()),
                }
        for index, name in enumerate(("atmosphere_coarse", "reaction_method")):
            mask = family_mask & (pack["y_cond_discrete_mask"][:, index] > 0.5)
            if mask.sum() >= 2:
                family_result[name] = {
                    **classification_metrics(pack["y_cond_discrete"][mask, index], predictions[name][mask]),
                    "n": int(mask.sum()),
                }
        result["by_family"][family] = family_result
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train full-database Stage3 family-conditioned LightGBM models.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--drop_family_features", action="store_true")
    parser.add_argument("--num_boost_round", type=int, default=500)
    parser.add_argument("--early_stopping_rounds", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--num_threads", type=int, default=12)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    schema = json.loads((input_dir / "schema.json").read_text(encoding="utf-8"))
    packs = {
        split: {key: value for key, value in np.load(input_dir / f"{split}.npz", allow_pickle=True).items()}
        for split in SPLITS
    }
    metas = {split: pd.read_csv(input_dir / f"{split}_meta.csv", low_memory=False) for split in SPLITS}
    base_count = int(schema["base_feature_count"])
    x = {}
    for split, pack in packs.items():
        structure = np.asarray(pack["x"], dtype=np.float32)
        if args.drop_family_features:
            structure = structure[:, :base_count]
        x[split] = np.hstack([structure, np.asarray(pack["y_set"], dtype=np.float32)]).astype(np.float32)

    models: Dict[str, Any] = {}
    class_maps: Dict[str, Any] = {}
    predictions: Dict[str, Dict[str, np.ndarray]] = {split: {} for split in ("val", "test")}
    for target_index, name in enumerate(("temperature_c", "time_h")):
        train_mask = packs["train"]["y_cond_continuous_mask"][:, target_index] > 0.5
        val_mask = packs["val"]["y_cond_continuous_mask"][:, target_index] > 0.5
        y_train = packs["train"]["y_cond_continuous_raw"][train_mask, target_index]
        y_val = packs["val"]["y_cond_continuous_raw"][val_mask, target_index]
        log_target = name == "time_h"
        if log_target:
            y_train = np.log1p(np.clip(y_train, 0.0, None))
            y_val = np.log1p(np.clip(y_val, 0.0, None))
        model = train_regressor(
            x["train"][train_mask], y_train, x["val"][val_mask], y_val,
            args.seed + target_index, args.num_boost_round, args.early_stopping_rounds, args.num_threads,
        )
        models[name] = model
        for split in predictions:
            predictions[split][name] = predict_regression(model, x[split], log_target)

    for target_index, name in enumerate(("atmosphere_coarse", "reaction_method")):
        train_mask = packs["train"]["y_cond_discrete_mask"][:, target_index] > 0.5
        val_mask = packs["val"]["y_cond_discrete_mask"][:, target_index] > 0.5
        original_train = packs["train"]["y_cond_discrete"][train_mask, target_index].astype(int)
        original_val = packs["val"]["y_cond_discrete"][val_mask, target_index].astype(int)
        classes = np.asarray(sorted(set(original_train.tolist())), dtype=int)
        class_to_local = {int(value): index for index, value in enumerate(classes.tolist())}
        keep_val = np.asarray([int(value) in class_to_local for value in original_val], dtype=bool)
        val_indices = np.flatnonzero(val_mask)[keep_val]
        y_train = np.asarray([class_to_local[int(value)] for value in original_train], dtype=int)
        y_val = np.asarray([class_to_local[int(value)] for value in original_val[keep_val]], dtype=int)
        model = train_classifier(
            x["train"][train_mask], y_train, x["val"][val_indices], y_val, len(classes),
            args.seed + 10 + target_index, args.num_boost_round, args.early_stopping_rounds, args.num_threads,
        )
        models[name] = model
        class_maps[name] = classes
        for split in predictions:
            probability = model.predict(x[split], num_iteration=model.best_iteration)
            predictions[split][name] = classes[np.argmax(probability, axis=1)]

    metrics = {
        split: evaluate(packs[split], predictions[split], metas[split])
        for split in ("val", "test")
    }
    for split in ("val", "test"):
        frame = metas[split][
            [column for column in ("sample_id", "formula", "family_signature_primary", "family_id_primary", "source_dataset", "reaction_method") if column in metas[split]]
        ].copy()
        for index, name in enumerate(("temperature_c", "time_h")):
            frame[f"true_{name}"] = packs[split]["y_cond_continuous_raw"][:, index]
            frame[f"has_{name}"] = packs[split]["y_cond_continuous_mask"][:, index]
            frame[f"pred_{name}"] = predictions[split][name]
        for index, name in enumerate(("atmosphere_coarse", "reaction_method")):
            frame[f"true_{name}_id"] = packs[split]["y_cond_discrete"][:, index]
            frame[f"has_{name}"] = packs[split]["y_cond_discrete_mask"][:, index]
            frame[f"pred_{name}_id"] = predictions[split][name]
        frame.to_csv(run_dir / f"pred_{split}.csv", index=False)

    artifact = {
        "models": models,
        "class_maps": class_maps,
        "schema": schema,
        "drop_family_features": bool(args.drop_family_features),
        "feature_mode": "structure_plus_predicted_precursor_set",
    }
    joblib.dump(artifact, run_dir / "stage3_family_lgbm.joblib")
    summary = {
        "model": "stage3_family_conditioned_lgbm",
        "config": vars(args),
        "data": {"rows": {split: int(len(metas[split])) for split in SPLITS}, "input_dim": int(x["train"].shape[1])},
        "best_iterations": {name: int(model.best_iteration) for name, model in models.items()},
        "metrics": metrics,
        "artifacts": {
            "model": str(run_dir / "stage3_family_lgbm.joblib"),
            "pred_val": str(run_dir / "pred_val.csv"),
            "pred_test": str(run_dir / "pred_test.csv"),
        },
    }
    write_json(run_dir / "metrics.json", summary)
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
