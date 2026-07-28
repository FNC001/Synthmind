#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


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


def decode_disc(schema: Mapping[str, Any], name: str, idx: int) -> str:
    vocab = ((schema.get("discrete_schema", {}) or {}).get(name, {}) or {}).get("vocab", [])
    return str(vocab[idx]) if 0 <= int(idx) < len(vocab) else str(idx)


def encode_disc(schema: Mapping[str, Any], name: str, value: str) -> int:
    vocab = ((schema.get("discrete_schema", {}) or {}).get(name, {}) or {}).get("vocab", [])
    try:
        return int(vocab.index(str(value)))
    except ValueError:
        return int(((schema.get("discrete_schema", {}) or {}).get(name, {}) or {}).get("missing_index", 0))


def plausibility(method: str, temp: float, time_h: float) -> tuple[float, float]:
    ranges = {
        "solid_state": ((300, 1400), (1, 240)),
        "solution": ((20, 350), (0.1, 168)),
        "hydro_solvothermal": ((80, 300), (1, 240)),
        "precipitation": ((20, 250), (0.05, 72)),
        "flux_molten_salt": ((300, 1200), (1, 240)),
        "melt_arc": ((500, 2200), (0.01, 240)),
    }
    tr, hr = ranges.get(method, ((20, 1800), (0.1, 1000)))
    tscore = 1.0 if tr[0] <= temp <= tr[1] else max(0.0, 1.0 - min(abs(temp - tr[0]), abs(temp - tr[1])) / 500.0)
    hscore = 1.0 if hr[0] <= time_h <= hr[1] else max(0.0, 1.0 - min(abs(time_h - hr[0]), abs(time_h - hr[1])) / 200.0)
    return float(tscore), float(hscore)


def hit_flags(row: Mapping[str, Any]) -> tuple[int, int]:
    if not (row["has_temperature_c"] and row["has_time_h"] and row["has_atmosphere"]):
        return 0, 0
    temp_err = abs(float(row["temperature_c"]) - float(row["true_temperature_c"]))
    time_err = abs(float(row["time_h"]) - float(row["true_time_h"]))
    atm_ok = str(row["atmosphere"]) == str(row["true_atmosphere"])
    strict = int(temp_err <= 100 and time_err <= 24 and atm_ok)
    relaxed = int(temp_err <= 200 and time_err <= 48 and atm_ok)
    return strict, relaxed


def metric_by_k(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in [1, 3, 5, 10]:
        sub = df[df["condition_rank_raw"] <= k]
        g = sub.groupby("sample_id", sort=False)
        out[f"top{k}_strict_condition"] = float(g["strict_hit_if_eval"].max().mean()) if len(g) else 0.0
        out[f"top{k}_relaxed_condition"] = float(g["relaxed_hit_if_eval"].max().mean()) if len(g) else 0.0
    g = df.groupby("sample_id", sort=False)
    out["oracle_strict_condition"] = float(g["strict_hit_if_eval"].max().mean()) if len(g) else 0.0
    out["oracle_relaxed_condition"] = float(g["relaxed_hit_if_eval"].max().mean()) if len(g) else 0.0
    return out


def make_train_index(input_dir: Path, mod: Any, schema: Mapping[str, Any], id_to_method: Mapping[str, str]) -> Dict[str, Any]:
    train = mod.load_npz(input_dir / "train.npz")
    x_struct = np.asarray(train["x"], dtype=np.float32)
    cont_names = list(schema["continuous_cols"])
    disc_names = list(schema["discrete_cols"])
    y_cont = mod.raw_cont(np.asarray(train["y_cond_continuous"], dtype=np.float32), schema, cont_names)
    y_disc = np.asarray(train["y_cond_discrete"])
    cont_mask = np.asarray(train["y_cond_continuous_mask"], dtype=np.float32)
    disc_mask = np.asarray(train["y_cond_discrete_mask"], dtype=np.float32)
    ids = np.asarray([str(x) for x in train["sample_id"]])
    methods = np.asarray([id_to_method.get(str(x), "other") for x in ids])
    scaler = StandardScaler().fit(x_struct)
    x_scaled = scaler.transform(x_struct)
    by_method: Dict[str, Any] = {}
    for method in sorted(set(methods.tolist())):
        idx = np.where(methods == method)[0]
        if len(idx) < 2:
            continue
        nn = NearestNeighbors(n_neighbors=min(6, len(idx)), metric="euclidean")
        nn.fit(x_scaled[idx])
        by_method[method] = {"idx": idx, "nn": nn}
    return {
        "x": x_struct,
        "x_scaled": x_scaled,
        "scaler": scaler,
        "y_cont": y_cont,
        "y_disc": y_disc,
        "cont_mask": cont_mask,
        "disc_mask": disc_mask,
        "methods": methods,
        "by_method": by_method,
    }


def method_templates(train_index: Mapping[str, Any], schema: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    cont_names = list(schema["continuous_cols"])
    disc_names = list(schema["discrete_cols"])
    jt = cont_names.index("target_temperature_c")
    jh = cont_names.index("target_time_h")
    for method in sorted(set(train_index["methods"].tolist())):
        idx = np.where(train_index["methods"] == method)[0]
        temp = train_index["y_cont"][idx, jt]
        time_h = train_index["y_cont"][idx, jh]
        rec = {
            "temperature_median": float(np.median(temp)),
            "temperature_p25": float(np.quantile(temp, 0.25)),
            "temperature_p75": float(np.quantile(temp, 0.75)),
            "time_median": float(np.median(time_h)),
            "time_p25": float(np.quantile(time_h, 0.25)),
            "time_p75": float(np.quantile(time_h, 0.75)),
        }
        for j, name in enumerate(disc_names):
            vals, counts = np.unique(train_index["y_disc"][idx, j].astype(int), return_counts=True)
            rec[name] = decode_disc(schema, name, int(vals[int(np.argmax(counts))])) if len(vals) else "<UNK_OR_MISSING>"
        out[method] = rec
    return out


def generate_split(split: str, input_dir: Path, output_dir: Path, model_pack: Mapping[str, Any], mod: Any, schema: Mapping[str, Any], id_to_method: Mapping[str, str], train_index: Mapping[str, Any], templates: Mapping[str, Mapping[str, Any]], top_neighbors: int) -> pd.DataFrame:
    pack = mod.load_npz(input_dir / f"{split}.npz")
    x_struct = np.asarray(pack["x"], dtype=np.float32)
    x_full = mod.make_x(pack)
    ids = np.asarray([str(x) for x in pack["sample_id"]])
    methods = np.asarray([id_to_method.get(str(x), "other") for x in ids])
    cont_names = list(schema["continuous_cols"])
    disc_names = list(schema["discrete_cols"])
    y_cont = mod.raw_cont(np.asarray(pack["y_cond_continuous"], dtype=np.float32), schema, cont_names)
    y_disc = np.asarray(pack["y_cond_discrete"])
    cont_mask = np.asarray(pack["y_cond_continuous_mask"], dtype=np.float32)
    disc_mask = np.asarray(pack["y_cond_discrete_mask"], dtype=np.float32)
    pred_cont, pred_disc = mod.predict_routed(
        x_full, methods, model_pack["global_models"], model_pack["experts"], cont_names, disc_names, schema.get("discrete_schema", {}) or {}
    )
    jt = cont_names.index("target_temperature_c")
    jh = cont_names.index("target_time_h")
    ja = disc_names.index("target_atmosphere")
    js = disc_names.index("target_solvent") if "target_solvent" in disc_names else -1

    rows: List[Dict[str, Any]] = []
    x_scaled = train_index["scaler"].transform(x_struct)
    for i, sid in enumerate(ids):
        method = str(methods[i])
        true_atm = decode_disc(schema, "target_atmosphere", int(y_disc[i, ja]))
        true_solvent = decode_disc(schema, "target_solvent", int(y_disc[i, js])) if js >= 0 else "<UNK_OR_MISSING>"

        def add(temp: float, time_h: float, atm: str, solvent: str, source: str, model_score: float, retrieval_score: float, template_score: float, prior_score: float) -> None:
            tps, hps = plausibility(method, temp, time_h)
            total = model_score + retrieval_score + template_score + prior_score + 0.35 * tps + 0.35 * hps
            rec = {
                "sample_id": sid,
                "precursor_candidate_id": f"{sid}::p0",
                "reaction_method": method,
                "temperature_c": float(temp),
                "temperature_low_c": float(min(temp, temp - 50)),
                "temperature_high_c": float(max(temp, temp + 50)),
                "time_h": float(max(time_h, 0.0)),
                "time_low_h": float(max(time_h * 0.5, 0.0)),
                "time_high_h": float(time_h * 1.5 + 1e-6),
                "atmosphere": atm,
                "solvent": solvent,
                "condition_source": source,
                "model_score": float(model_score),
                "retrieval_score": float(retrieval_score),
                "method_template_score": float(template_score),
                "condition_prior_score": float(prior_score),
                "temperature_plausibility_score": tps,
                "time_plausibility_score": hps,
                "atmosphere_probability": 1.0 if atm == decode_disc(schema, "target_atmosphere", int(pred_disc[i, ja])) else 0.45,
                "condition_total_score": float(total),
                "true_temperature_c": float(y_cont[i, jt]),
                "true_time_h": float(y_cont[i, jh]),
                "true_atmosphere": true_atm,
                "true_solvent": true_solvent,
                "has_temperature_c": int(cont_mask[i, jt] > 0.5),
                "has_time_h": int(cont_mask[i, jh] > 0.5),
                "has_atmosphere": int(disc_mask[i, ja] > 0.5),
            }
            strict, relaxed = hit_flags(rec)
            rec["strict_hit_if_eval"] = strict
            rec["relaxed_hit_if_eval"] = relaxed
            rows.append(rec)

        pred_atm = decode_disc(schema, "target_atmosphere", int(pred_disc[i, ja]))
        pred_solvent = decode_disc(schema, "target_solvent", int(pred_disc[i, js])) if js >= 0 else "<UNK_OR_MISSING>"
        add(pred_cont[i, jt], pred_cont[i, jh], pred_atm, pred_solvent, "model_point", 1.0, 0.0, 0.0, 0.2)

        tpl = templates.get(method) or templates.get("other") or {}
        if tpl:
            tpl_atm = str(tpl.get("target_atmosphere", "<UNK_OR_MISSING>"))
            tpl_solvent = str(tpl.get("target_solvent", "<UNK_OR_MISSING>"))
            add(tpl["temperature_median"], tpl["time_median"], tpl_atm, tpl_solvent, "method_template", 0.2, 0.0, 1.0, 0.6)
            add(tpl["temperature_p25"], tpl["time_p25"], tpl_atm, tpl_solvent, "model_quantile_low", 0.35, 0.0, 0.7, 0.4)
            add(tpl["temperature_p75"], tpl["time_p75"], tpl_atm, tpl_solvent, "model_quantile_high", 0.35, 0.0, 0.7, 0.4)
            add(0.5 * pred_cont[i, jt] + 0.5 * tpl["temperature_median"], 0.5 * pred_cont[i, jh] + 0.5 * tpl["time_median"], pred_atm, pred_solvent, "calibrated_blend", 0.7, 0.0, 0.5, 0.5)

        ref = train_index["by_method"].get(method) or train_index["by_method"].get("other")
        if ref is not None:
            dist, nbr_local = ref["nn"].kneighbors(x_scaled[i:i + 1], n_neighbors=min(top_neighbors, len(ref["idx"])))
            for d, loc in zip(dist[0], nbr_local[0]):
                j = int(ref["idx"][int(loc)])
                atm = decode_disc(schema, "target_atmosphere", int(train_index["y_disc"][j, ja]))
                solvent = decode_disc(schema, "target_solvent", int(train_index["y_disc"][j, js])) if js >= 0 else "<UNK_OR_MISSING>"
                add(train_index["y_cont"][j, jt], train_index["y_cont"][j, jh], atm, solvent, "retrieval_template", 0.15, 1.0 / (1.0 + float(d)), 0.25, 0.35)

    df = pd.DataFrame(rows)
    df = df.sort_values(["sample_id", "condition_total_score"], ascending=[True, False], kind="mergesort")
    df["condition_rank_raw"] = df.groupby("sample_id", sort=False).cumcount() + 1
    df.to_csv(output_dir / f"{split}_condition_candidates.csv", index=False)
    by_method = {m: metric_by_k(g) for m, g in df.groupby("reaction_method")}
    summary = {"split": split, "n_rows": int(len(ids)), "n_candidates": int(len(df)), "metrics": metric_by_k(df), "by_method": by_method}
    write_json(output_dir / f"{split}_condition_candidate_metrics.json", summary)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Stage3 condition candidate pool v2 from model, train-only retrieval, and method templates.")
    ap.add_argument("--input_dir", default="data/interim/generative/stage3_condition_dataset_chem_checked/method_stratified_v5_20260610")
    ap.add_argument("--model", default="runs/stage3/lgbm_method_experts_alltrain_coreval_chem_checked_20260610/stage3_lgbm_method_experts.joblib")
    ap.add_argument("--refined_dir", default="data/interim/refined/structdesc_refined_route_unified_20260609_units_normalized")
    ap.add_argument("--output_dir", default="outputs/evaluation/stage3_condition_candidates_v2_20260610")
    ap.add_argument("--top_neighbors", type=int, default=5)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mod = load_stage3_module()
    schema = mod.load_json(Path(args.input_dir) / "schema.json")
    model_pack = joblib.load(args.model)
    id_to_method = model_pack.get("id_to_method") or mod.build_method_map(Path(args.refined_dir))
    train_index = make_train_index(Path(args.input_dir), mod, schema, id_to_method)
    templates = method_templates(train_index, schema)
    summaries = {}
    for split in ["train", "val", "test"]:
        df = generate_split(split, Path(args.input_dir), output_dir, model_pack, mod, schema, id_to_method, train_index, templates, args.top_neighbors)
        summaries[split] = {
            "rows": int(df["sample_id"].nunique()),
            "candidates": int(len(df)),
            "metrics": metric_by_k(df),
        }
    write_json(output_dir / "condition_candidate_pool_v2_summary.json", {"config": vars(args), "splits": summaries})
    print(json.dumps(to_builtin({"config": vars(args), "splits": summaries}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
