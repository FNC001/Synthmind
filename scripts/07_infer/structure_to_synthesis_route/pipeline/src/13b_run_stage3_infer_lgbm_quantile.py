#!/usr/bin/env python3
"""
Stage 3 LightGBM inference — drop-in replacement for 13_run_stage3_infer_mixture_flow_conditioned.py

Uses LightGBM quantile ensemble for continuous conditions (temperature, time)
and LightGBM classifiers for discrete conditions (atmosphere, time_bucket).

Output format is identical to the flow-based sampler: test_candidates_flat.csv
with columns: sample_id, material_id, parent_precursor_rank, parent_precursor_set_key,
parent_precursor_set, condition_rank, stage3_score, condition_source,
temperature_c, time_h, pred_atmosphere, pred_time_bucket
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError("lightgbm is required: pip install lightgbm")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_list_cell(v: Any) -> List[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    s = str(v).strip()
    if not s:
        return []
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass
    if ";" in s:
        return [x.strip() for x in s.split(";") if x.strip()]
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    return [s]


def strip_label_prefix(x: str) -> str:
    s = str(x).strip()
    if s.startswith("label_prec__"):
        return s[len("label_prec__"):]
    return s


def encode_y_set(labels: Sequence[str], vocab: Sequence[str]) -> np.ndarray:
    index = {strip_label_prefix(v): i for i, v in enumerate(vocab)}
    y = np.zeros((len(vocab),), dtype=np.float32)
    for raw in labels:
        s = strip_label_prefix(str(raw).strip())
        if s in index:
            y[index[s]] = 1.0
    return y


def safe_float_array(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    tmp = df.reindex(columns=list(cols), fill_value=0.0)
    for c in cols:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(0.0)
    return tmp[list(cols)].to_numpy(dtype=np.float32)


QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ATM_CLASSES = ["oxidizing", "non-oxidizing"]
ATM_THRESHOLD = 0.60
TIME_BUCKET_CLASSES = ["short", "medium", "long"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Stage3 LightGBM inference (drop-in for flow model).")
    ap.add_argument("--conditioned_x_csv", required=True)
    ap.add_argument("--schema_json", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--top_k_conditions", type=int, default=5)
    ap.add_argument("--temp_model_dir", type=str,
                    default="/Users/wyc/SynPred/runs/stage3/lgbm_quantile_ensemble_v2_fulldata")
    ap.add_argument("--time_model_dir", type=str,
                    default="/Users/wyc/SynPred/runs/stage3/lgbm_quantile_ensemble_v2_fulldata")
    ap.add_argument("--atm_model", type=str,
                    default="/Users/wyc/SynPred/runs/stage3/lgbm_atmosphere_classifier_v1/model_atmosphere_binary_final.txt")
    ap.add_argument("--time_bucket_model", type=str,
                    default="/Users/wyc/SynPred/runs/stage3/lgbm_time_bucket_classifier_v1/model_time_bucket.txt")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="cpu", help="Unused, kept for CLI compatibility with flow script")
    args = ap.parse_args()

    conditioned_x_csv = Path(args.conditioned_x_csv).expanduser().resolve()
    schema_json = Path(args.schema_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    schema = json.load(open(schema_json, "r", encoding="utf-8"))
    feature_cols = list(schema["feature_cols"])
    precursor_vocab = list(schema["precursor_vocab"])

    df = pd.read_csv(conditioned_x_csv)
    print(f"[INFO] Input: {len(df)} rows from {conditioned_x_csv}")

    x_raw = safe_float_array(df, feature_cols)

    y_rows = []
    for _, row in df.iterrows():
        labels = parse_list_cell(row.get("parent_precursor_set", "[]"))
        y_rows.append(encode_y_set(labels, precursor_vocab))
    y_set = np.stack(y_rows, axis=0).astype(np.float32)

    X = np.concatenate([x_raw, y_set], axis=1).astype(np.float32)
    print(f"[INFO] Feature matrix: {X.shape}")

    # Load quantile models
    temp_model_dir = Path(args.temp_model_dir)
    time_model_dir = Path(args.time_model_dir)

    temp_models = {}
    time_models = {}
    for q in QUANTILES:
        tp = temp_model_dir / f"temp_q{q:.1f}.txt"
        if tp.exists():
            temp_models[q] = lgb.Booster(model_file=str(tp))
        tp2 = time_model_dir / f"time_q{q:.1f}.txt"
        if tp2.exists():
            time_models[q] = lgb.Booster(model_file=str(tp2))

    print(f"[INFO] Loaded {len(temp_models)} temp models, {len(time_models)} time models")

    # Load discrete classifiers
    atm_model = None
    atm_path = Path(args.atm_model)
    if atm_path.exists():
        atm_model = lgb.Booster(model_file=str(atm_path))
        print(f"[INFO] Atmosphere model loaded")

    tb_model = None
    tb_path = Path(args.time_bucket_model)
    if tb_path.exists():
        tb_model = lgb.Booster(model_file=str(tb_path))
        print(f"[INFO] Time bucket model loaded")

    # Predict continuous conditions
    temp_preds = {q: model.predict(X) for q, model in temp_models.items()}
    time_preds = {q: model.predict(X) for q, model in time_models.items()}

    # Denormalize predictions if schema has normalization stats
    cont_schema = schema.get("continuous_schema", {})
    cont_keys = list(cont_schema.keys())
    temp_stats = cont_schema.get(cont_keys[0], {}) if len(cont_keys) > 0 else {}
    time_stats = cont_schema.get(cont_keys[1], {}) if len(cont_keys) > 1 else {}
    if temp_stats.get("mean") is not None and temp_stats.get("std") is not None:
        t_mean, t_std = temp_stats["mean"], temp_stats["std"]
        temp_preds = {q: v * t_std + t_mean for q, v in temp_preds.items()}
    if time_stats.get("mean") is not None and time_stats.get("std") is not None:
        h_mean, h_std = time_stats["mean"], time_stats["std"]
        time_preds = {q: v * h_std + h_mean for q, v in time_preds.items()}

    # Predict discrete conditions (atmosphere uses only material features = first len(feature_cols) dims)
    atm_pred = None
    atm_proba = None
    if atm_model is not None:
        feat_cols_atm = [c for c in feature_cols if c.startswith("feat_")]
        X_atm = safe_float_array(df, feat_cols_atm)
        atm_proba_raw = atm_model.predict(X_atm)
        atm_pred = (atm_proba_raw > ATM_THRESHOLD).astype(int)
        atm_proba = atm_proba_raw

    tb_pred = None
    if tb_model is not None:
        feat_cols_tb = [c for c in feature_cols if c.startswith("feat_")]
        X_tb = safe_float_array(df, feat_cols_tb)
        tb_proba = tb_model.predict(X_tb)
        tb_pred = tb_proba.argmax(axis=1)

    # Select top-k candidates per sample
    # Strategy: use diverse quantiles, prioritize median, then spread outward
    # Order: Q0.5, Q0.4, Q0.6, Q0.3, Q0.7, Q0.2, Q0.8, Q0.1, Q0.9
    quantile_priority = [0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.1, 0.9]
    k = min(int(args.top_k_conditions), len(quantile_priority))

    rows = []
    debug_rows = []

    for i in range(len(df)):
        parent_set = parse_list_cell(df.iloc[i].get("parent_precursor_set", "[]"))
        parent_rank = int(df.iloc[i].get("parent_precursor_rank", i))
        sample_id = str(df.iloc[i].get("sample_id", ""))
        material_id = str(df.iloc[i].get("material_id", sample_id))
        parent_key = str(df.iloc[i].get("parent_precursor_set_key", ""))

        debug_rows.append({
            "sample_id": sample_id,
            "material_id": material_id,
            "parent_precursor_rank": parent_rank,
            "parent_precursor_set_key": parent_key,
            "parent_precursor_set": json.dumps(parent_set, ensure_ascii=False),
            "n_conditions_exported": k,
            "pred_atmosphere": ATM_CLASSES[int(atm_pred[i])] if atm_pred is not None else "",
            "pred_time_bucket": TIME_BUCKET_CLASSES[int(tb_pred[i])] if tb_pred is not None else "",
        })

        # Deduplicate by rounded temperature (within 10°C)
        seen_temps = set()
        exported = 0

        for q in quantile_priority:
            if q not in temp_preds:
                continue

            temp_val = float(temp_preds[q][i])
            time_val = float(time_preds.get(q, temp_preds[q])[i]) if q in time_preds else 12.0

            temp_val = max(0.0, min(2000.0, temp_val))
            time_val = max(0.0, min(5000.0, time_val))

            temp_rounded = round(temp_val / 10) * 10
            if temp_rounded in seen_temps:
                continue
            seen_temps.add(temp_rounded)

            # Score: median gets highest, decreasing outward
            score = 1.0 - abs(q - 0.5) * 2.0

            cont_conditions = {
                "temperature_c": temp_val,
                "time_h": time_val,
            }

            row_out = {
                "sample_id": sample_id,
                "material_id": material_id,
                "parent_precursor_rank": parent_rank,
                "parent_precursor_set_key": parent_key,
                "parent_precursor_set": parent_set,
                "condition_rank": exported,
                "mixture_index": 0,
                "stage3_score": score,
                "condition_source": "lgbm_top1" if q == 0.5 else f"lgbm_q{q:.1f}",
                "cont_conditions": cont_conditions,
                "stage3_model": "lgbm_quantile_ensemble_v2",
            }

            if atm_pred is not None:
                row_out["pred_atmosphere"] = ATM_CLASSES[int(atm_pred[i])]
                row_out["pred_atmosphere_proba"] = float(atm_proba[i])
            if tb_pred is not None:
                row_out["pred_time_bucket"] = TIME_BUCKET_CLASSES[int(tb_pred[i])]

            rows.append(row_out)
            exported += 1
            if exported >= k:
                break

    # Write outputs (same format as flow-based sampler)
    out_jsonl = output_dir / "test_candidates.jsonl"
    out_csv = output_dir / "test_candidates_flat.csv"
    debug_csv = output_dir / "debug_parent_candidates.csv"
    summary_json = output_dir / "candidate_summary.json"

    write_jsonl(out_jsonl, rows)
    pd.DataFrame(debug_rows).to_csv(debug_csv, index=False)

    flat_rows = []
    for r_item in rows:
        rr = {k2: v2 for k2, v2 in r_item.items() if k2 not in ["cont_conditions", "parent_precursor_set"]}
        rr["parent_precursor_set"] = json.dumps(r_item["parent_precursor_set"], ensure_ascii=False)
        for k2, v2 in r_item["cont_conditions"].items():
            rr[k2] = v2
        flat_rows.append(rr)
    pd.DataFrame(flat_rows).to_csv(out_csv, index=False)

    summary = {
        "mode": "stage3_lgbm_quantile_infer",
        "conditioned_x_csv": str(conditioned_x_csv),
        "schema_json": str(schema_json),
        "output_dir": str(output_dir),
        "n_input_parent_candidates": int(len(df)),
        "n_output_rows": int(len(rows)),
        "x_dim": int(X.shape[1]),
        "n_temp_quantile_models": len(temp_models),
        "n_time_quantile_models": len(time_models),
        "has_atmosphere_model": atm_model is not None,
        "has_time_bucket_model": tb_model is not None,
        "top_k_conditions": k,
        "quantile_priority": quantile_priority[:k],
        "artifacts": {
            "test_candidates_jsonl": str(out_jsonl),
            "test_candidates_flat_csv": str(out_csv),
            "debug_parent_candidates_csv": str(debug_csv),
        },
    }
    write_json(summary_json, summary)

    print(f"[DONE] summary -> {summary_json}")
    print(f"[DONE] jsonl   -> {out_jsonl}")
    print(f"[DONE] flat    -> {out_csv}")
    print(f"[DONE] debug   -> {debug_csv}")
    print(f"[DONE] {len(rows)} condition candidates for {len(df)} parent precursor sets")


if __name__ == "__main__":
    main()
