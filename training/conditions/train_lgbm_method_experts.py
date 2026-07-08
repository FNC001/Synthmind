#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(obj), ensure_ascii=False, indent=2), encoding="utf-8")


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


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def classify_reaction_method(row: Mapping[str, Any]) -> str:
    text = " ".join([
        norm_text(row.get("synthesis_type")),
        norm_text(row.get("synthesis_text")),
        norm_text(row.get("reaction_string")),
    ])
    solvent = norm_text(row.get("solvent"))

    def has(*patterns: str) -> bool:
        return any(p in text for p in patterns)

    # More specific solution routes must be checked before generic heat words.
    if has("hydrothermal", "solvothermal", "teflon-lined autoclave", "teflon lined autoclave", "autoclave"):
        return "hydro_solvothermal"
    if has("sol-gel", "sol gel", "pechini", "citrate gel", "gel combustion"):
        return "sol_gel"
    if has("co-precip", "coprecip", "precipitat"):
        return "precipitation"
    if has("combustion"):
        return "combustion"
    if has("molten salt", "flux"):
        return "flux_molten_salt"
    if has("arc-melting", "arc melting", "arc-melt", "melted", "melting"):
        return "melt_arc"
    if has("mechanochemical", "ball mill", "ball-mill", "milling"):
        return "mechanochemical"
    if has("thermal decomposition", "decomposed", "decomposition"):
        return "thermal_decomposition"
    if has(
        "solid-state",
        "solid state",
        "sinter",
        "calcined",
        "calcination",
        "anneal",
        "fired",
        "pellet",
        "ground and heated",
        "heated at",
    ):
        return "solid_state"
    if solvent or has("aqueous", "solution", "dissolved", "stirred", "reflux"):
        return "solution"
    return "other"


def build_method_map(refined_dir: Path) -> Dict[str, str]:
    id_to_method: Dict[str, str] = {}
    for name in ["stage3_train_relaxed.jsonl", "stage3_gold.jsonl"]:
        path = refined_dir / name
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                sid = str(row.get("id") or "")
                if sid:
                    id_to_method[sid] = classify_reaction_method(row)
    return id_to_method


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
        callbacks=[lgb.early_stopping(int(early_stopping_rounds)), lgb.log_evaluation(100)],
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
        callbacks=[lgb.early_stopping(int(early_stopping_rounds)), lgb.log_evaluation(100)],
    )


def train_model_set(
    X: Mapping[str, np.ndarray],
    y_cont_raw: Mapping[str, np.ndarray],
    cont_mask: Mapping[str, np.ndarray],
    y_disc: Mapping[str, np.ndarray],
    disc_mask: Mapping[str, np.ndarray],
    cont_names: Sequence[str],
    disc_names: Sequence[str],
    disc_schema: Mapping[str, Any],
    train_sel: np.ndarray,
    val_sel: np.ndarray,
    seed: int,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> Dict[str, Any]:
    models: Dict[str, Any] = {}

    j_temp = list(cont_names).index("target_temperature_c")
    train_mask = train_sel & (cont_mask["train"][:, j_temp] > 0.5)
    val_mask = val_sel & (cont_mask["val"][:, j_temp] > 0.5)
    if train_mask.sum() >= 50 and val_mask.sum() >= 10:
        models["target_temperature_c"] = train_lgb_regression(
            X["train"][train_mask],
            y_cont_raw["train"][train_mask, j_temp],
            X["val"][val_mask],
            y_cont_raw["val"][val_mask, j_temp],
            objective="regression_l1",
            seed=seed,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
        )

    j_time = list(cont_names).index("target_time_h")
    train_mask = train_sel & (cont_mask["train"][:, j_time] > 0.5)
    val_mask = val_sel & (cont_mask["val"][:, j_time] > 0.5)
    if train_mask.sum() >= 50 and val_mask.sum() >= 10:
        y_time_log = {s: np.log1p(np.clip(y_cont_raw[s][:, j_time], 0.0, None)) for s in X}
        models["target_time_h_log1p"] = train_lgb_regression(
            X["train"][train_mask],
            y_time_log["train"][train_mask],
            X["val"][val_mask],
            y_time_log["val"][val_mask],
            objective="regression_l1",
            seed=seed + 1,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
        )

    for j, name in enumerate(disc_names):
        n_classes = int((disc_schema.get(name, {}) or {}).get("n_classes", int(np.max(y_disc["train"][:, j]) + 1)))
        train_mask = train_sel & (disc_mask["train"][:, j] > 0.5)
        val_mask = val_sel & (disc_mask["val"][:, j] > 0.5)
        if train_mask.sum() < 50 or val_mask.sum() < 10:
            continue
        if len(np.unique(y_disc["train"][train_mask, j])) <= 1:
            continue
        models[name] = train_lgb_multiclass(
            X["train"][train_mask],
            y_disc["train"][train_mask, j],
            X["val"][val_mask],
            y_disc["val"][val_mask, j],
            n_classes=n_classes,
            seed=seed + 10 + j,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
        )
    return models


def predict_routed(
    X_split: np.ndarray,
    methods: np.ndarray,
    global_models: Mapping[str, Any],
    experts: Mapping[str, Mapping[str, Any]],
    cont_names: Sequence[str],
    disc_names: Sequence[str],
    disc_schema: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    y_cont = np.zeros((X_split.shape[0], len(cont_names)), dtype=np.float32)
    y_disc = np.zeros((X_split.shape[0], len(disc_names)), dtype=np.int64)

    groups = sorted(set(str(x) for x in methods.tolist()))
    for method in groups:
        idx = np.where(methods == method)[0]
        if idx.size == 0:
            continue
        model_set = experts.get(method, {})
        for j, name in enumerate(cont_names):
            key = "target_time_h_log1p" if name == "target_time_h" else name
            model = model_set.get(key) or global_models.get(key)
            if model is None:
                continue
            pred = np.asarray(model.predict(X_split[idx], num_iteration=model.best_iteration), dtype=np.float32)
            y_cont[idx, j] = np.expm1(pred) if key.endswith("_log1p") else pred
        for j, name in enumerate(disc_names):
            model = model_set.get(name) or global_models.get(name)
            if model is None:
                missing = int((disc_schema.get(name, {}) or {}).get("missing_index", 0))
                y_disc[idx, j] = missing
                continue
            prob = model.predict(X_split[idx], num_iteration=model.best_iteration)
            y_disc[idx, j] = np.asarray(np.argmax(prob, axis=1), dtype=np.int64)
    return y_cont, y_disc


def evaluate_split(
    split: str,
    X: Mapping[str, np.ndarray],
    methods: Mapping[str, np.ndarray],
    y_cont_raw: Mapping[str, np.ndarray],
    cont_mask: Mapping[str, np.ndarray],
    y_disc: Mapping[str, np.ndarray],
    disc_mask: Mapping[str, np.ndarray],
    global_models: Mapping[str, Any],
    experts: Mapping[str, Mapping[str, Any]],
    cont_names: Sequence[str],
    disc_names: Sequence[str],
    disc_schema: Mapping[str, Any],
) -> Dict[str, Any]:
    pred_cont, pred_disc = predict_routed(
        X[split], methods[split], global_models, experts, cont_names, disc_names, disc_schema
    )
    out: Dict[str, Any] = {"continuous": {}, "discrete": {}, "by_method": {}}
    for j, name in enumerate(cont_names):
        mask = cont_mask[split][:, j] > 0.5
        out["continuous"][name] = regression_metrics(y_cont_raw[split][mask, j], pred_cont[mask, j])
    for j, name in enumerate(disc_names):
        mask = disc_mask[split][:, j] > 0.5
        if mask.sum() == 0:
            continue
        out["discrete"][name] = discrete_metrics(y_disc[split][mask, j], pred_disc[mask, j])
    for method in sorted(set(methods[split].tolist())):
        sel = methods[split] == method
        if sel.sum() == 0:
            continue
        sub: Dict[str, Any] = {"n": int(sel.sum())}
        for j, name in enumerate(cont_names):
            mask = sel & (cont_mask[split][:, j] > 0.5)
            if mask.sum() >= 2:
                sub[name] = regression_metrics(y_cont_raw[split][mask, j], pred_cont[mask, j])
        out["by_method"][method] = sub
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Stage3 LightGBM reaction-method experts on combined data.")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--refined_dir", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--num_boost_round", type=int, default=500)
    ap.add_argument("--early_stopping_rounds", type=int, default=40)
    ap.add_argument("--min_expert_train", type=int, default=300)
    ap.add_argument("--min_expert_val", type=int, default=20)
    ap.add_argument(
        "--global_val_fallback",
        action="store_true",
        help="Use the full validation split for expert early stopping when a method has too few validation rows.",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    schema = load_json(input_dir / "schema.json")
    cont_names = [str(x) for x in schema["continuous_cols"]]
    disc_names = [str(x) for x in schema["discrete_cols"]]
    disc_schema = schema.get("discrete_schema", {}) or {}

    id_to_method = build_method_map(Path(args.refined_dir))
    packs = {split: load_npz(input_dir / f"{split}.npz") for split in ["train", "val", "test"]}
    X = {split: make_x(pack) for split, pack in packs.items()}
    sample_ids = {split: np.asarray([str(x) for x in pack["sample_id"]]) for split, pack in packs.items()}
    methods = {
        split: np.asarray([id_to_method.get(str(sid), "other") for sid in sample_ids[split]])
        for split in packs
    }
    y_cont_raw = {
        split: raw_cont(np.asarray(pack["y_cond_continuous"], dtype=np.float32), schema, cont_names)
        for split, pack in packs.items()
    }
    cont_mask = {split: np.asarray(pack["y_cond_continuous_mask"], dtype=np.float32) for split, pack in packs.items()}
    y_disc = {split: np.asarray(pack["y_cond_discrete"]) for split, pack in packs.items()}
    disc_mask = {split: np.asarray(pack["y_cond_discrete_mask"], dtype=np.float32) for split, pack in packs.items()}

    all_train = np.ones(X["train"].shape[0], dtype=bool)
    all_val = np.ones(X["val"].shape[0], dtype=bool)
    global_models = train_model_set(
        X, y_cont_raw, cont_mask, y_disc, disc_mask, cont_names, disc_names, disc_schema,
        all_train, all_val, args.seed, args.num_boost_round, args.early_stopping_rounds,
    )

    method_counts = {split: Counter(methods[split].tolist()) for split in methods}
    experts: Dict[str, Dict[str, Any]] = {}
    expert_training: Dict[str, Any] = {}
    for method, n_train in method_counts["train"].most_common():
        n_val = int(method_counts["val"].get(method, 0))
        if n_train < args.min_expert_train:
            expert_training[method] = {"trained": False, "n_train": int(n_train), "n_val": n_val}
            continue
        print(f"\n=== Training expert: {method} train={n_train} val={n_val} ===")
        train_sel = methods["train"] == method
        val_source = "method"
        if n_val >= args.min_expert_val:
            val_sel = methods["val"] == method
        elif args.global_val_fallback:
            val_sel = all_val
            val_source = "global"
        else:
            expert_training[method] = {"trained": False, "n_train": int(n_train), "n_val": n_val}
            continue
        models = train_model_set(
            X, y_cont_raw, cont_mask, y_disc, disc_mask, cont_names, disc_names, disc_schema,
            train_sel, val_sel, args.seed + 1000 + len(experts), args.num_boost_round, args.early_stopping_rounds,
        )
        if models:
            experts[method] = models
        expert_training[method] = {
            "trained": bool(models),
            "n_train": int(n_train),
            "n_val": n_val,
            "val_source": val_source,
            "heads": sorted(models.keys()),
        }

    metrics = {
        split: evaluate_split(
            split, X, methods, y_cont_raw, cont_mask, y_disc, disc_mask,
            global_models, experts, cont_names, disc_names, disc_schema,
        )
        for split in ["val", "test"]
    }

    pack = {
        "model_type": "stage3_lgbm_method_experts",
        "global_models": global_models,
        "experts": experts,
        "id_to_method": id_to_method,
        "method_counts": {k: dict(v) for k, v in method_counts.items()},
        "expert_training": expert_training,
        "schema": schema,
        "cont_names": cont_names,
        "disc_names": disc_names,
        "config": vars(args),
        "feature_mode": "x_plus_y_set",
    }
    joblib.dump(pack, run_dir / "stage3_lgbm_method_experts.joblib")
    summary = {
        "model": "stage3_lgbm_method_experts",
        "config": vars(args),
        "data": {
            "n_train": int(X["train"].shape[0]),
            "n_val": int(X["val"].shape[0]),
            "n_test": int(X["test"].shape[0]),
            "input_dim": int(X["train"].shape[1]),
            "method_counts": {k: dict(v) for k, v in method_counts.items()},
            "expert_training": expert_training,
        },
        "metrics": metrics,
        "artifacts": {
            "model": str(run_dir / "stage3_lgbm_method_experts.joblib"),
            "metrics": str(run_dir / "metrics.json"),
        },
    }
    write_json(run_dir / "metrics.json", summary)
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
