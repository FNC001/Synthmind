#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder


CORE_METHODS = {"solid_state", "solution", "melt_arc"}
MISSING = "<UNK_OR_MISSING>"


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_builtin(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def read_table(path_base: Path) -> pd.DataFrame:
    csv_path = path_base.with_suffix(".csv")
    parquet_path = path_base.with_suffix(".parquet")
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception:
            pass
    return pd.read_csv(csv_path)


def feature_columns(df: pd.DataFrame) -> List[str]:
    prefixes = ("feat_", "precursor_family_count__", "precursor_family_frac__")
    cols = [c for c in df.columns if c.startswith(prefixes)]
    cols += [
        "precursor_set_size",
        "precursor_confidence_score",
        "contains_open_generated_precursor",
        "contains_repair_precursor",
        "contains_raw_model_precursor",
        "temperature_iqr",
        "time_iqr",
        "multimodal_group_size",
        "is_multimodal_group",
        "precursor_f1_to_true",
        "precursor_jaccard_to_true",
    ]
    return [c for c in cols if c in df.columns]


def make_x(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, numeric_cols: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    for df in [train, val, test]:
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cat_cols = [c for c in ["reaction_method", "precursor_input_mode", "precursor_source_mix"] if c in train.columns]
    combined = pd.concat([train[cat_cols], val[cat_cols], test[cat_cols]], ignore_index=True).astype(str)
    cat = pd.get_dummies(combined, columns=cat_cols, dtype=np.float32)
    n_train, n_val = len(train), len(val)
    train_cat = cat.iloc[:n_train].reset_index(drop=True)
    val_cat = cat.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_cat = cat.iloc[n_train + n_val:].reset_index(drop=True)
    train_x = pd.concat([train[list(numeric_cols)].reset_index(drop=True), train_cat], axis=1)
    val_x = pd.concat([val[list(numeric_cols)].reset_index(drop=True), val_cat], axis=1)
    test_x = pd.concat([test[list(numeric_cols)].reset_index(drop=True), test_cat], axis=1)
    return train_x.to_numpy(np.float32), val_x.to_numpy(np.float32), test_x.to_numpy(np.float32), train_x.columns.tolist()


def reg_model(objective: str = "regression_l1", alpha: float | None = None, n_estimators: int = 450, seed: int = 42) -> lgb.LGBMRegressor:
    params: Dict[str, Any] = {
        "objective": objective,
        "n_estimators": int(n_estimators),
        "learning_rate": 0.035,
        "num_leaves": 127,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "min_child_samples": 25,
        "reg_lambda": 1.0,
        "random_state": int(seed),
        "n_jobs": 8,
        "verbose": -1,
    }
    if alpha is not None:
        params["alpha"] = float(alpha)
    return lgb.LGBMRegressor(**params)


def clf_model(n_estimators: int = 350, seed: int = 42) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=int(n_estimators),
        learning_rate=0.04,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.80,
        min_child_samples=15,
        reg_lambda=1.0,
        random_state=int(seed),
        n_jobs=8,
        verbose=-1,
    )


def topk_acc(proba: np.ndarray, y: np.ndarray, k: int) -> float:
    if len(y) == 0:
        return math.nan
    top = np.argsort(proba, axis=1)[:, -k:]
    return float(np.mean([int(y[i] in top[i]) for i in range(len(y))]))


def condition_hits(df: pd.DataFrame, pred_temp: np.ndarray, pred_time: np.ndarray, pred_atm: np.ndarray) -> Dict[str, float]:
    true_temp = pd.to_numeric(df["temperature_c_raw"], errors="coerce").to_numpy(float)
    true_time = pd.to_numeric(df["time_h_raw"], errors="coerce").to_numpy(float)
    true_atm = df["atmosphere_normalized"].astype(str).to_numpy()
    known_atm = pd.to_numeric(df["atmosphere_known_mask"], errors="coerce").fillna(0).to_numpy(int) == 1
    temp_err = np.abs(pred_temp - true_temp)
    time_err = np.abs(pred_time - true_time)
    atm_ok = (pred_atm.astype(str) == true_atm) | (~known_atm)
    strict = (temp_err <= 100) & (time_err <= 24) & atm_ok
    relaxed = (temp_err <= 200) & (time_err <= 48) & atm_ok
    return {
        "strict_condition": float(np.mean(strict)),
        "relaxed_condition": float(np.mean(relaxed)),
        "temp_mae": float(mean_absolute_error(true_temp, pred_temp)),
        "temp_median_ae": float(np.median(temp_err)),
        "temp_within50": float(np.mean(temp_err <= 50)),
        "temp_within100": float(np.mean(temp_err <= 100)),
        "temp_within200": float(np.mean(temp_err <= 200)),
        "time_mae": float(mean_absolute_error(true_time, pred_time)),
        "time_median_ae": float(np.median(time_err)),
        "time_within12": float(np.mean(time_err <= 12)),
        "time_within24": float(np.mean(time_err <= 24)),
        "time_within48": float(np.mean(time_err <= 48)),
        "atm_acc_known": float(np.mean(pred_atm[known_atm].astype(str) == true_atm[known_atm])) if known_atm.any() else math.nan,
    }


def metrics_by_group(df: pd.DataFrame, pred_temp: np.ndarray, pred_time: np.ndarray, pred_atm: np.ndarray) -> Dict[str, Any]:
    out = {"all": condition_hits(df, pred_temp, pred_time, pred_atm)}
    core_mask = df["reaction_method"].isin(CORE_METHODS).to_numpy()
    if core_mask.any():
        out["core"] = condition_hits(df.iloc[core_mask], pred_temp[core_mask], pred_time[core_mask], pred_atm[core_mask])
    if (~core_mask).any():
        out["non_core"] = condition_hits(df.iloc[~core_mask], pred_temp[~core_mask], pred_time[~core_mask], pred_atm[~core_mask])
    for method, idx in df.groupby("reaction_method", sort=False).groups.items():
        arr = np.asarray(list(idx), dtype=int)
        out[f"method::{method}"] = condition_hits(df.iloc[arr], pred_temp[arr], pred_time[arr], pred_atm[arr])
    return out


def fit_label_classifier(x_train: np.ndarray, y_train_raw: Sequence[str], x_val: np.ndarray, y_val_raw: Sequence[str], n_estimators: int, seed: int) -> tuple[lgb.LGBMClassifier, LabelEncoder]:
    enc = LabelEncoder()
    y_train = enc.fit_transform(np.asarray(y_train_raw, dtype=str))
    y_val = np.asarray(y_val_raw, dtype=str)
    model = clf_model(n_estimators=n_estimators, seed=seed)
    model.fit(x_train, y_train, eval_set=[(x_val, enc.transform(np.where(np.isin(y_val, enc.classes_), y_val, enc.classes_[0])))], callbacks=[lgb.early_stopping(35, verbose=False)])
    return model, enc


def safe_predict_label(model: lgb.LGBMClassifier, enc: LabelEncoder, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    proba = model.predict_proba(x)
    idx = np.argmax(proba, axis=1)
    return enc.inverse_transform(idx), proba


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Stage3 v3 distributional condition models with missing-aware labels.")
    ap.add_argument("--input_dir", default="data/interim/generative/stage3_condition_targets_v3_20260610")
    ap.add_argument("--run_dir", default="runs/stage3/distributional_condition_v3_20260610")
    ap.add_argument("--n_estimators_reg", type=int, default=450)
    ap.add_argument("--n_estimators_clf", type=int, default=350)
    ap.add_argument("--seed", type=int, default=20260612)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = (read_table(input_dir / s) for s in ["train", "val", "test"])
    num_cols = feature_columns(train)
    x_train, x_val, x_test, feat_cols = make_x(train, val, test, num_cols)

    y_temp = pd.to_numeric(train["temperature_clipped"], errors="coerce").to_numpy(float)
    y_temp_val = pd.to_numeric(val["temperature_clipped"], errors="coerce").to_numpy(float)
    y_time = pd.to_numeric(train["log_time_clipped"], errors="coerce").to_numpy(float)
    y_time_val = pd.to_numeric(val["log_time_clipped"], errors="coerce").to_numpy(float)

    models: Dict[str, Any] = {}
    models["temperature_point"] = reg_model("regression_l1", n_estimators=args.n_estimators_reg, seed=args.seed)
    models["temperature_point"].fit(x_train, y_temp, eval_set=[(x_val, y_temp_val)], callbacks=[lgb.early_stopping(40, verbose=False)])
    for q in [10, 50, 90]:
        m = reg_model("quantile", alpha=q / 100, n_estimators=args.n_estimators_reg, seed=args.seed + q)
        m.fit(x_train, y_temp, eval_set=[(x_val, y_temp_val)], callbacks=[lgb.early_stopping(40, verbose=False)])
        models[f"temperature_quantile_p{q}"] = m

    models["time_point"] = reg_model("regression_l1", n_estimators=args.n_estimators_reg, seed=args.seed + 100)
    models["time_point"].fit(x_train, y_time, eval_set=[(x_val, y_time_val)], callbacks=[lgb.early_stopping(40, verbose=False)])
    for q in [10, 50, 90]:
        m = reg_model("quantile", alpha=q / 100, n_estimators=args.n_estimators_reg, seed=args.seed + 100 + q)
        m.fit(x_train, y_time, eval_set=[(x_val, y_time_val)], callbacks=[lgb.early_stopping(40, verbose=False)])
        models[f"time_quantile_p{q}"] = m

    encoders: Dict[str, LabelEncoder] = {}
    models["temperature_bin"], encoders["temperature_bin"] = fit_label_classifier(x_train, train["temperature_bin"], x_val, val["temperature_bin"], args.n_estimators_clf, args.seed + 200)
    models["time_bin"], encoders["time_bin"] = fit_label_classifier(x_train, train["time_bin"], x_val, val["time_bin"], args.n_estimators_clf, args.seed + 201)

    atm_train_mask = train["atmosphere_known_mask"].astype(int).to_numpy() == 1
    atm_val_mask = val["atmosphere_known_mask"].astype(int).to_numpy() == 1
    models["atmosphere"], encoders["atmosphere"] = fit_label_classifier(
        x_train[atm_train_mask], train.loc[atm_train_mask, "atmosphere_target_class"], x_val[atm_val_mask], val.loc[atm_val_mask, "atmosphere_target_class"], args.n_estimators_clf, args.seed + 300
    )
    solv_train_mask = train["solvent_known_mask"].astype(int).to_numpy() == 1
    solv_val_mask = val["solvent_known_mask"].astype(int).to_numpy() == 1
    models["solvent"], encoders["solvent"] = fit_label_classifier(
        x_train[solv_train_mask], train.loc[solv_train_mask, "solvent_target_class"], x_val[solv_val_mask], val.loc[solv_val_mask, "solvent_target_class"], args.n_estimators_clf, args.seed + 301
    )

    def evaluate(split: str, df: pd.DataFrame, x: np.ndarray) -> Dict[str, Any]:
        pred_temp = np.clip(models["temperature_point"].predict(x), 20, 1800)
        pred_time = np.expm1(models["time_point"].predict(x)).clip(0.05, 500)
        pred_atm, atm_proba = safe_predict_label(models["atmosphere"], encoders["atmosphere"], x)
        pred_temp_bin, temp_bin_proba = safe_predict_label(models["temperature_bin"], encoders["temperature_bin"], x)
        pred_time_bin, time_bin_proba = safe_predict_label(models["time_bin"], encoders["time_bin"], x)
        tb_true = encoders["temperature_bin"].transform(np.where(np.isin(df["temperature_bin"].astype(str), encoders["temperature_bin"].classes_), df["temperature_bin"].astype(str), encoders["temperature_bin"].classes_[0]))
        timeb_true = encoders["time_bin"].transform(np.where(np.isin(df["time_bin"].astype(str), encoders["time_bin"].classes_), df["time_bin"].astype(str), encoders["time_bin"].classes_[0]))
        known_atm = df["atmosphere_known_mask"].astype(int).to_numpy() == 1
        atm_true = encoders["atmosphere"].transform(np.where(np.isin(df.loc[known_atm, "atmosphere_target_class"].astype(str), encoders["atmosphere"].classes_), df.loc[known_atm, "atmosphere_target_class"].astype(str), encoders["atmosphere"].classes_[0]))
        metrics = metrics_by_group(df, pred_temp, pred_time, pred_atm.astype(str))
        metrics["bin_metrics"] = {
            "temperature_bin_top1": float(accuracy_score(tb_true, np.argmax(temp_bin_proba, axis=1))),
            "temperature_bin_top3": topk_acc(temp_bin_proba, tb_true, 3),
            "time_bin_top1": float(accuracy_score(timeb_true, np.argmax(time_bin_proba, axis=1))),
            "time_bin_top3": topk_acc(time_bin_proba, timeb_true, 3),
            "atmosphere_top1_known": float(accuracy_score(atm_true, np.argmax(atm_proba[known_atm], axis=1))) if known_atm.any() else math.nan,
            "atmosphere_top3_known": topk_acc(atm_proba[known_atm], atm_true, 3) if known_atm.any() else math.nan,
        }
        pred_df = pd.DataFrame({
            "sample_id": df["sample_id"],
            "reaction_method": df["reaction_method"],
            "pred_temperature_c": pred_temp,
            "pred_time_h": pred_time,
            "pred_atmosphere": pred_atm,
            "pred_temperature_bin": pred_temp_bin,
            "pred_time_bin": pred_time_bin,
        })
        pred_df.to_csv(run_dir / f"{split}_predictions.csv", index=False)
        return metrics

    metrics = {"val": evaluate("val", val, x_val), "test": evaluate("test", test, x_test)}
    pack = {
        "models": models,
        "encoders": encoders,
        "feature_cols": feat_cols,
        "numeric_feature_cols": num_cols,
        "config": vars(args),
        "metrics": metrics,
    }
    joblib.dump(pack, run_dir / "stage3_distributional_condition_v3.joblib")
    for name, model in models.items():
        joblib.dump(model, run_dir / f"model_{name}.pkl")
    write_json(run_dir / "feature_schema.json", {"feature_cols": feat_cols, "numeric_feature_cols": num_cols, "encoders": {k: v.classes_.tolist() for k, v in encoders.items()}})
    write_json(run_dir / "metrics.json", metrics)
    report = ["# Stage3 Distributional Condition v3 Training Report", "", json.dumps(to_builtin(metrics), ensure_ascii=False, indent=2)]
    (run_dir / "training_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(to_builtin(metrics), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
