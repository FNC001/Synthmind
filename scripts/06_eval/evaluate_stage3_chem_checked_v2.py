#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score


CORE_METHODS = {"solid_state", "solution", "melt_arc"}
DEFAULT_METHODS = [
    "solid_state", "solution", "melt_arc", "other", "hydro_solvothermal",
    "precipitation", "flux_molten_salt", "thermal_decomposition",
    "mechanochemical", "sol_gel", "combustion",
]


def load_stage3_module():
    path = Path(__file__).resolve().parents[1] / "04_train" / "stage3" / "train_stage3_lgbm_method_experts.py"
    spec = importlib.util.spec_from_file_location("stage3_lgbm_method_experts_mod", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helper module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, thresholds: Sequence[float]) -> Dict[str, float]:
    err = np.abs(y_pred - y_true)
    out = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "median_ae": float(np.median(err)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
    }
    for th in thresholds:
        out[f"within_{th:g}"] = float(np.mean(err <= float(th)))
    return out


def discrete_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def evaluate_subset(rec: Mapping[str, Any], cont_names: Sequence[str], disc_names: Sequence[str], subset: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n": int(subset.sum()), "continuous": {}, "discrete": {}, "condition_success": {}}
    for j, name in enumerate(cont_names):
        mask = subset & (rec["cont_mask"][:, j] > 0.5)
        if mask.sum() >= 2:
            th = [50, 100, 200] if name == "target_temperature_c" else [12, 24, 48]
            out["continuous"][name] = regression_metrics(rec["y_cont"][mask, j], rec["pred_cont"][mask, j], th)
    for j, name in enumerate(disc_names):
        mask = subset & (rec["disc_mask"][:, j] > 0.5)
        if mask.sum() >= 2:
            out["discrete"][name] = discrete_metrics(rec["y_disc"][mask, j], rec["pred_disc"][mask, j])
    try:
        jt = list(cont_names).index("target_temperature_c")
        jh = list(cont_names).index("target_time_h")
        ja = list(disc_names).index("target_atmosphere")
    except ValueError:
        return out
    temp_err = np.abs(rec["pred_cont"][:, jt] - rec["y_cont"][:, jt])
    time_err = np.abs(rec["pred_cont"][:, jh] - rec["y_cont"][:, jh])
    atm_ok = rec["pred_disc"][:, ja] == rec["y_disc"][:, ja]
    evaluable = subset & (rec["cont_mask"][:, jt] > 0.5) & (rec["cont_mask"][:, jh] > 0.5) & (rec["disc_mask"][:, ja] > 0.5)
    strict = evaluable & (temp_err <= 100.0) & (time_err <= 24.0) & atm_ok
    relaxed = evaluable & (temp_err <= 200.0) & (time_err <= 48.0) & atm_ok
    denom_all = max(int(subset.sum()), 1)
    denom_eval = max(int(evaluable.sum()), 1)
    out["condition_success"] = {
        "n_evaluable": int(evaluable.sum()),
        "strict_all_rows": float(strict.sum() / denom_all),
        "relaxed_all_rows": float(relaxed.sum() / denom_all),
        "strict_evaluable": float(strict.sum() / denom_eval),
        "relaxed_evaluable": float(relaxed.sum() / denom_eval),
        "temperature_within_100_all_rows": float((evaluable & (temp_err <= 100.0)).sum() / denom_all),
        "temperature_within_200_all_rows": float((evaluable & (temp_err <= 200.0)).sum() / denom_all),
        "time_within_24_all_rows": float((evaluable & (time_err <= 24.0)).sum() / denom_all),
        "time_within_48_all_rows": float((evaluable & (time_err <= 48.0)).sum() / denom_all),
        "atmosphere_correct_all_rows": float((evaluable & atm_ok).sum() / denom_all),
    }
    return out


def apply_core_val_calibration(pred_data: Dict[str, Any]) -> Dict[str, Any]:
    cont_names = pred_data["cont_names"]
    disc_names = pred_data["disc_names"]
    val = pred_data["splits"]["val"]
    calibration: Dict[str, Any] = {"continuous_bias": {}, "discrete_override": {}}
    for method in sorted(CORE_METHODS):
        msel = (val["methods"] == method)
        calibration["continuous_bias"][method] = {}
        calibration["discrete_override"][method] = {}
        for j, name in enumerate(cont_names):
            mask = msel & (val["cont_mask"][:, j] > 0.5)
            bias = float(np.median(val["y_cont"][mask, j] - val["pred_cont"][mask, j])) if mask.sum() else 0.0
            calibration["continuous_bias"][method][name] = bias
        for j, name in enumerate(disc_names):
            mask = msel & (val["disc_mask"][:, j] > 0.5)
            if mask.sum() == 0:
                continue
            true = val["y_disc"][mask, j].astype(int)
            pred = val["pred_disc"][mask, j].astype(int)
            vals, counts = np.unique(true, return_counts=True)
            majority = int(vals[int(np.argmax(counts))])
            majority_pred = np.full_like(true, majority)
            if accuracy_score(true, majority_pred) > accuracy_score(true, pred):
                calibration["discrete_override"][method][name] = majority
    for split in pred_data["splits"]:
        rec = pred_data["splits"][split]
        for method in sorted(CORE_METHODS):
            idx = np.where(rec["methods"] == method)[0]
            for j, name in enumerate(cont_names):
                rec["pred_cont"][idx, j] += float(calibration["continuous_bias"][method].get(name, 0.0))
            for j, name in enumerate(disc_names):
                if name in calibration["discrete_override"][method]:
                    rec["pred_disc"][idx, j] = int(calibration["discrete_override"][method][name])
    pred_data["calibration"] = calibration
    return pred_data


def predict_model(model_pack: Mapping[str, Any], input_dir: Path, refined_dir: Path) -> Dict[str, Any]:
    mod = load_stage3_module()
    schema = mod.load_json(input_dir / "schema.json")
    cont_names = [str(x) for x in schema["continuous_cols"]]
    disc_names = [str(x) for x in schema["discrete_cols"]]
    disc_schema = schema.get("discrete_schema", {}) or {}
    id_to_method = model_pack.get("id_to_method") or mod.build_method_map(refined_dir)
    out: Dict[str, Any] = {"schema": schema, "cont_names": cont_names, "disc_names": disc_names, "splits": {}}
    for split in ["val", "test"]:
        pack = mod.load_npz(input_dir / f"{split}.npz")
        x = mod.make_x(pack)
        sample_ids = np.asarray([str(x) for x in pack["sample_id"]])
        methods = np.asarray([id_to_method.get(str(sid), "other") for sid in sample_ids])
        y_cont = mod.raw_cont(np.asarray(pack["y_cond_continuous"], dtype=np.float32), schema, cont_names)
        cont_mask = np.asarray(pack["y_cond_continuous_mask"], dtype=np.float32)
        y_disc = np.asarray(pack["y_cond_discrete"])
        disc_mask = np.asarray(pack["y_cond_discrete_mask"], dtype=np.float32)
        pred_cont, pred_disc = mod.predict_routed(
            x, methods, model_pack["global_models"], model_pack["experts"], cont_names, disc_names, disc_schema
        )
        out["splits"][split] = {
            "sample_ids": sample_ids,
            "methods": methods,
            "core_mask": np.asarray([m in CORE_METHODS for m in methods]),
            "y_cont": y_cont,
            "cont_mask": cont_mask,
            "y_disc": y_disc,
            "disc_mask": disc_mask,
            "pred_cont": pred_cont,
            "pred_disc": pred_disc,
        }
    return out


def evaluate(pred_data: Dict[str, Any], methods_to_report: Sequence[str]) -> Dict[str, Any]:
    cont_names = pred_data["cont_names"]
    disc_names = pred_data["disc_names"]
    out: Dict[str, Any] = {}
    for split, rec in pred_data["splits"].items():
        split_out: Dict[str, Any] = {}
        all_mask = np.ones(len(rec["methods"]), dtype=bool)
        split_out["all"] = evaluate_subset(rec, cont_names, disc_names, all_mask)
        split_out["core"] = evaluate_subset(rec, cont_names, disc_names, rec["core_mask"])
        split_out["by_method"] = {}
        for method in methods_to_report:
            split_out["by_method"][method] = evaluate_subset(rec, cont_names, disc_names, rec["methods"] == method)
        out[split] = split_out
    if "calibration" in pred_data:
        out["calibration"] = pred_data["calibration"]
    return out


def save_predictions(pred_data: Dict[str, Any], output_dir: Path) -> None:
    cont_names = pred_data["cont_names"]
    disc_names = pred_data["disc_names"]
    schema = pred_data["schema"]
    disc_schema = schema.get("discrete_schema", {}) or {}
    for split, rec in pred_data["splits"].items():
        rows = []
        for i, sid in enumerate(rec["sample_ids"]):
            row = {"sample_id": sid, "reaction_method": rec["methods"][i]}
            for j, name in enumerate(cont_names):
                short = name.replace("target_", "")
                row[f"true_{short}"] = rec["y_cont"][i, j]
                row[f"pred_{short}"] = rec["pred_cont"][i, j]
                row[f"has_{short}"] = int(rec["cont_mask"][i, j] > 0.5)
            for j, name in enumerate(disc_names):
                vocab = (disc_schema.get(name, {}) or {}).get("vocab", [])
                true_idx = int(rec["y_disc"][i, j])
                pred_idx = int(rec["pred_disc"][i, j])
                row[f"true_{name.replace('target_', '')}"] = vocab[true_idx] if 0 <= true_idx < len(vocab) else str(true_idx)
                row[f"pred_{name.replace('target_', '')}"] = vocab[pred_idx] if 0 <= pred_idx < len(vocab) else str(pred_idx)
                row[f"has_{name.replace('target_', '')}"] = int(rec["disc_mask"][i, j] > 0.5)
            rows.append(row)
        pd.DataFrame(rows).to_csv(output_dir / f"{split}_stage3_predictions.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate Stage3 method experts on chemistry-checked Stage3 input.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--input_dir", default="data/interim/generative/stage3_condition_dataset_chem_checked/method_stratified_v5_20260610")
    ap.add_argument("--refined_dir", default="data/interim/refined/structdesc_refined_route_unified_20260609_units_normalized")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--calibrate_core_val", action="store_true")
    ap.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_pack = joblib.load(args.model)
    pred_data = predict_model(model_pack, Path(args.input_dir), Path(args.refined_dir))
    if args.calibrate_core_val:
        pred_data = apply_core_val_calibration(pred_data)
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    metrics = evaluate(pred_data, methods)
    save_predictions(pred_data, output_dir)
    write_json(output_dir / "stage3_eval_metrics.json", {"config": vars(args), "metrics": metrics})
    print(json.dumps(to_builtin(metrics), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
