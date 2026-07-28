#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd


def load_helper() -> Any:
    path = Path(__file__).resolve().with_name("generate_stage3_condition_candidates_v3.py")
    spec = importlib.util.spec_from_file_location("stage3_cond_v3_helper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helper {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def top_labels(proba: np.ndarray, classes: np.ndarray, k: int) -> List[List[Tuple[str, float]]]:
    idx = np.argsort(proba, axis=1)[:, ::-1][:, :k]
    return [[(str(classes[j]), float(proba[i, j])) for j in idx[i]] for i in range(proba.shape[0])]


def main() -> None:
    ap = argparse.ArgumentParser(description="Fast train-only Stage3 v3 condition candidate generation without nearest-neighbor retrieval.")
    ap.add_argument("--input_dir", default="data/interim/generative/stage3_condition_targets_v3_20260610")
    ap.add_argument("--model", default="runs/stage3/distributional_condition_v3_20260610/stage3_distributional_condition_v3.joblib")
    ap.add_argument("--output_dir", default="outputs/evaluation/stage3_condition_candidates_v3_20260610")
    args = ap.parse_args()
    helper = load_helper()
    input_dir = Path(args.input_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    train = helper.read_table(input_dir / "train")
    pack = joblib.load(args.model)
    x = helper.make_x(train, pack["numeric_feature_cols"], pack["feature_cols"])
    models, enc = pack["models"], pack["encoders"]
    pred_temp = np.clip(models["temperature_point"].predict(x), 20, 1800)
    pred_time = np.expm1(models["time_point"].predict(x)).clip(0.05, 500)
    tq = {q: np.clip(models[f"temperature_quantile_p{q}"].predict(x), 20, 1800) for q in [10, 50, 90]}
    hq = {q: np.expm1(models[f"time_quantile_p{q}"].predict(x)).clip(0.05, 500) for q in [10, 50, 90]}
    atm_proba = models["atmosphere"].predict_proba(x)
    solv_proba = models["solvent"].predict_proba(x)
    tb_proba = models["temperature_bin"].predict_proba(x)
    hb_proba = models["time_bin"].predict_proba(x)
    atm_top = top_labels(atm_proba, enc["atmosphere"].classes_, 1)
    solv_top = top_labels(solv_proba, enc["solvent"].classes_, 1)
    tb_top = top_labels(tb_proba, enc["temperature_bin"].classes_, 3)
    hb_top = top_labels(hb_proba, enc["time_bin"].classes_, 3)
    templates = helper.method_templates(train)
    sample_ids = train["sample_id"].astype(str).to_numpy()
    methods = train["reaction_method"].astype(str).to_numpy()
    true_temp = pd.to_numeric(train["temperature_c_raw"], errors="coerce").to_numpy(float)
    true_time = pd.to_numeric(train["time_h_raw"], errors="coerce").to_numpy(float)
    true_atm = train["atmosphere_normalized"].astype(str).to_numpy()
    known_atm = pd.to_numeric(train["atmosphere_known_mask"], errors="coerce").fillna(0).to_numpy(int)
    open_flags = pd.to_numeric(train.get("contains_open_generated_precursor", 0), errors="coerce").fillna(0).to_numpy(float)
    repair_flags = pd.to_numeric(train.get("contains_repair_precursor", 0), errors="coerce").fillna(0).to_numpy(float)
    precursor_f1 = pd.to_numeric(train.get("precursor_f1_to_true", 0), errors="coerce").fillna(0).to_numpy(float)
    precursor_conf = pd.to_numeric(train.get("precursor_confidence_score", 0), errors="coerce").fillna(0).to_numpy(float)
    temp_percentiles = {q: pd.to_numeric(train.get(f"temperature_p{q}", pd.Series(pred_temp)), errors="coerce").fillna(pd.Series(pred_temp)).to_numpy(float) for q in [10, 25, 50, 75, 90]}
    time_percentiles = {q: pd.to_numeric(train.get(f"time_p{q}", pd.Series(pred_time)), errors="coerce").fillna(pd.Series(pred_time)).to_numpy(float) for q in [10, 25, 50, 75, 90]}
    rows: List[Dict[str, Any]] = []
    seen: set[tuple[str, int, int, str]] = set()

    def add(i: int, temp: float, time_h: float, atm: str, solv: str, source: str, scores: Dict[str, float]) -> None:
        sid = sample_ids[i]
        temp = float(np.clip(temp, 20, 1800))
        time_h = float(np.clip(time_h, 0.05, 500))
        key = (sid, int(round(temp / 10) * 10), int(round(time_h)), str(atm))
        if key in seen:
            return
        seen.add(key)
        strict, relaxed, te, he, atm_ok = helper.hit_flags(
            true_temp[i],
            true_time[i],
            true_atm[i],
            int(known_atm[i]),
            temp,
            time_h,
            atm,
        )
        open_pen = float(open_flags[i]) * 0.15
        repair_pen = float(repair_flags[i]) * 0.10
        total = (
            scores.get("temperature_point_score", 0.0)
            + scores.get("time_point_score", 0.0)
            + 0.8 * scores.get("temperature_bin_score", 0.0)
            + 0.8 * scores.get("time_bin_score", 0.0)
            + 0.7 * scores.get("atmosphere_probability", 0.0)
            + 0.2 * scores.get("solvent_probability", 0.0)
            + 0.6 * scores.get("multimodal_template_score", 0.0)
            + 0.5 * scores.get("method_template_score", 0.0)
            + 0.4 * scores.get("condition_prior_score", 0.0)
            - open_pen
            - repair_pen
        )
        rows.append({
            "sample_id": sid,
            "condition_candidate_id": f"{sid}::trainfast_c{len(rows)}",
            "reaction_method": methods[i],
            "temperature_c": temp,
            "temperature_low_c": max(20.0, temp - 100.0),
            "temperature_high_c": min(1800.0, temp + 100.0),
            "time_h": time_h,
            "time_low_h": max(0.05, time_h * 0.5),
            "time_high_h": min(500.0, time_h * 1.5 + 1e-6),
            "atmosphere": atm,
            "solvent": solv,
            "condition_source": source,
            "retrieval_score": 0.0,
            **{k: float(v) for k, v in scores.items()},
            "open_generated_penalty": open_pen,
            "repair_penalty": repair_pen,
            "total_score_raw": float(total),
            "true_temperature_c": float(true_temp[i]),
            "true_time_h": float(true_time[i]),
            "true_atmosphere": true_atm[i],
            "atmosphere_known_mask": int(known_atm[i]),
            "strict_hit_if_eval": strict,
            "relaxed_hit_if_eval": relaxed,
            "temp_error": te,
            "time_error": he,
            "atmosphere_correct": atm_ok,
            "precursor_f1_to_true": float(precursor_f1[i]),
            "precursor_confidence_score": float(precursor_conf[i]),
        })

    for i in range(len(train)):
        atm, ap = atm_top[i][0]
        solv, sp = solv_top[i][0]
        add(i, pred_temp[i], pred_time[i], atm, solv, "point_model", {
            "temperature_point_score": 1.0, "temperature_bin_score": float(np.max(tb_proba[i])),
            "time_point_score": 1.0, "time_bin_score": float(np.max(hb_proba[i])),
            "atmosphere_probability": ap, "solvent_probability": sp, "condition_prior_score": 0.4,
        })
        for qname, tv, hv in [("p10", tq[10][i], hq[10][i]), ("p50", tq[50][i], hq[50][i]), ("p90", tq[90][i], hq[90][i]), ("p10_p90", tq[10][i], hq[90][i]), ("p90_p10", tq[90][i], hq[10][i])]:
            add(i, tv, hv, atm, solv, f"quantile_model_{qname}", {
                "temperature_point_score": 0.6, "temperature_bin_score": 0.5,
                "time_point_score": 0.6, "time_bin_score": 0.5,
                "atmosphere_probability": ap, "solvent_probability": sp, "condition_prior_score": 0.35,
            })
        for tb, tbp in tb_top[i]:
            for hb, hbp in hb_top[i]:
                add(i, helper.temp_bin_center(tb), helper.time_bin_center(hb), atm, solv, "bin_center", {
                    "temperature_point_score": 0.3, "temperature_bin_score": tbp,
                    "time_point_score": 0.3, "time_bin_score": hbp,
                    "atmosphere_probability": ap, "solvent_probability": sp, "condition_prior_score": 0.3,
                })
        for q in [10, 25, 50, 75, 90]:
            add(i, float(temp_percentiles[q][i]), float(time_percentiles[q][i]), atm, solv, f"multimodal_group_template_p{q}", {
                "temperature_point_score": 0.2, "temperature_bin_score": 0.4,
                "time_point_score": 0.2, "time_bin_score": 0.4,
                "atmosphere_probability": ap, "solvent_probability": sp, "multimodal_template_score": 1.0,
                "condition_prior_score": 0.3,
            })
        tpl = templates.get(methods[i])
        if tpl:
            add(i, tpl["temperature"], tpl["time"], tpl["atmosphere"], tpl["solvent"], "method_condition_template", {
                "method_template_score": 1.0, "condition_prior_score": 0.8,
                "atmosphere_probability": 0.5, "solvent_probability": 0.5,
            })
    out = pd.DataFrame(rows).sort_values(["sample_id", "total_score_raw"], ascending=[True, False], kind="mergesort")
    out["condition_rank_raw"] = out.groupby("sample_id", sort=False).cumcount() + 1
    out.to_csv(outdir / "train_condition_candidates.csv", index=False)
    metrics = helper.metric_by_k(out)
    write_json(outdir / "train_condition_candidate_metrics.json", {"metrics": metrics, "n_candidates": int(len(out))})
    print(json.dumps(to_builtin({"rows": int(train["sample_id"].nunique()), "candidates": int(len(out)), "metrics": metrics}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
