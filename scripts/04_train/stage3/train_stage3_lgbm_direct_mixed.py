#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    return {k: arr[k] for k in arr.files}


def make_x(pack: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.hstack([
        np.asarray(pack["x"], dtype=np.float32),
        np.asarray(pack["y_set"], dtype=np.float32),
    ]).astype(np.float32)


def raw_cont(values_norm: np.ndarray, schema: Mapping[str, Any], names: Sequence[str]) -> np.ndarray:
    out = np.asarray(values_norm, dtype=np.float32).copy()
    stats = schema.get("continuous_schema", {}) or {}
    for j, name in enumerate(names):
        st = stats.get(name, {}) or {}
        out[:, j] = out[:, j] * float(st.get("std", 1.0)) + float(st.get("mean", 0.0))
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else math.nan,
    }


def discrete_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def train_lgb_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    objective: str,
    seed: int,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> lgb.Booster:
    params = {
        "objective": objective,
        "metric": "l1",
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "learning_rate": 0.035,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 20,
        "lambda_l2": 1.0,
        "verbose": -1,
        "seed": int(seed),
        "num_threads": -1,
    }
    dtrain = lgb.Dataset(x_train, label=y_train)
    dval = lgb.Dataset(x_val, label=y_val, reference=dtrain)
    return lgb.train(
        params,
        dtrain,
        num_boost_round=int(num_boost_round),
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(int(early_stopping_rounds)), lgb.log_evaluation(25)],
    )


def train_lgb_multiclass(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    seed: int,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> lgb.Booster:
    params = {
        "objective": "multiclass",
        "num_class": int(n_classes),
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.04,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 10,
        "lambda_l2": 1.0,
        "verbose": -1,
        "seed": int(seed),
        "num_threads": -1,
    }
    dtrain = lgb.Dataset(x_train, label=y_train)
    dval = lgb.Dataset(x_val, label=y_val, reference=dtrain)
    return lgb.train(
        params,
        dtrain,
        num_boost_round=int(num_boost_round),
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(int(early_stopping_rounds)), lgb.log_evaluation(25)],
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Train direct Stage3 LightGBM models on mixed condition dataset.")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--num_boost_round", type=int, default=500)
    ap.add_argument("--early_stopping_rounds", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    schema = load_json(input_dir / "schema.json")
    cont_names = [str(x) for x in schema["continuous_cols"]]
    disc_names = [str(x) for x in schema["discrete_cols"]]
    disc_schema = schema.get("discrete_schema", {}) or {}

    packs = {split: load_npz(input_dir / f"{split}.npz") for split in ["train", "val", "test"]}
    X = {split: make_x(pack) for split, pack in packs.items()}
    y_cont_raw = {
        split: raw_cont(np.asarray(pack["y_cond_continuous"], dtype=np.float32), schema, cont_names)
        for split, pack in packs.items()
    }
    cont_mask = {split: np.asarray(pack["y_cond_continuous_mask"], dtype=np.float32) for split, pack in packs.items()}
    y_disc = {split: np.asarray(pack["y_cond_discrete"]) for split, pack in packs.items()}
    disc_mask = {split: np.asarray(pack["y_cond_discrete_mask"], dtype=np.float32) for split, pack in packs.items()}

    models: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {"continuous": {}, "discrete": {}}

    # Temperature in raw Celsius.
    j_temp = cont_names.index("target_temperature_c")
    masks = {s: cont_mask[s][:, j_temp] > 0.5 for s in packs}
    temp_model = train_lgb_regression(
        X["train"][masks["train"]], y_cont_raw["train"][masks["train"], j_temp],
        X["val"][masks["val"]], y_cont_raw["val"][masks["val"], j_temp],
        objective="regression_l1",
        seed=args.seed,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    models["target_temperature_c"] = temp_model
    for split in ["val", "test"]:
        pred = temp_model.predict(X[split][masks[split]], num_iteration=temp_model.best_iteration)
        metrics["continuous"].setdefault(split, {})["target_temperature_c"] = regression_metrics(
            y_cont_raw[split][masks[split], j_temp], pred
        )

    # Time in log1p(hours), evaluated back in hours.
    j_time = cont_names.index("target_time_h")
    masks = {s: cont_mask[s][:, j_time] > 0.5 for s in packs}
    y_time_log = {s: np.log1p(np.clip(y_cont_raw[s][:, j_time], 0.0, None)) for s in packs}
    time_model = train_lgb_regression(
        X["train"][masks["train"]], y_time_log["train"][masks["train"]],
        X["val"][masks["val"]], y_time_log["val"][masks["val"]],
        objective="regression_l1",
        seed=args.seed + 1,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    models["target_time_h_log1p"] = time_model
    for split in ["val", "test"]:
        pred_log = time_model.predict(X[split][masks[split]], num_iteration=time_model.best_iteration)
        pred = np.expm1(pred_log)
        metrics["continuous"].setdefault(split, {})["target_time_h"] = regression_metrics(
            y_cont_raw[split][masks[split], j_time], pred
        )

    for split in ["val", "test"]:
        vals = [
            metrics["continuous"][split]["target_temperature_c"]["mae"],
            metrics["continuous"][split]["target_time_h"]["mae"],
        ]
        metrics["continuous"][split]["mean_mae"] = float(np.mean(vals))

    for j, name in enumerate(disc_names):
        n_classes = int((disc_schema.get(name, {}) or {}).get("n_classes", int(np.max(y_disc["train"][:, j]) + 1)))
        masks = {s: disc_mask[s][:, j] > 0.5 for s in packs}
        if masks["train"].sum() == 0 or len(np.unique(y_disc["train"][masks["train"], j])) <= 1:
            continue
        model = train_lgb_multiclass(
            X["train"][masks["train"]], y_disc["train"][masks["train"], j],
            X["val"][masks["val"]], y_disc["val"][masks["val"], j],
            n_classes=n_classes,
            seed=args.seed + 10 + j,
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        models[name] = model
        for split in ["val", "test"]:
            prob = model.predict(X[split][masks[split]], num_iteration=model.best_iteration)
            pred = np.argmax(prob, axis=1)
            metrics["discrete"].setdefault(split, {})[name] = discrete_metrics(y_disc[split][masks[split], j], pred)

    for split in ["val", "test"]:
        accs = [v["accuracy"] for v in metrics["discrete"].get(split, {}).values()]
        metrics["discrete"].setdefault(split, {})["mean_accuracy"] = float(np.mean(accs)) if accs else math.nan

    pack = {
        "models": models,
        "schema": schema,
        "cont_names": cont_names,
        "disc_names": disc_names,
        "config": vars(args),
        "feature_mode": "x_plus_y_set",
    }
    joblib.dump(pack, run_dir / "stage3_lgbm_direct_mixed.joblib")
    summary = {
        "model": "stage3_lgbm_direct_mixed",
        "config": vars(args),
        "data": {
            "n_train": int(X["train"].shape[0]),
            "n_val": int(X["val"].shape[0]),
            "n_test": int(X["test"].shape[0]),
            "input_dim": int(X["train"].shape[1]),
        },
        "metrics": metrics,
        "artifacts": {
            "model": str(run_dir / "stage3_lgbm_direct_mixed.joblib"),
            "metrics": str(run_dir / "metrics.json"),
        },
    }
    write_json(run_dir / "metrics.json", summary)
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
