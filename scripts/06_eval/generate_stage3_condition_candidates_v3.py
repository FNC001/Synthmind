#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


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
    parquet = path_base.with_suffix(".parquet")
    if parquet.exists():
        try:
            return pd.read_parquet(parquet)
        except Exception:
            pass
    return pd.read_csv(path_base.with_suffix(".csv"))


def make_x(df: pd.DataFrame, numeric_cols: Sequence[str], feature_cols: Sequence[str]) -> np.ndarray:
    numeric_data = {}
    for c in numeric_cols:
        if c in df.columns:
            numeric_data[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
        else:
            numeric_data[c] = np.zeros(len(df), dtype=np.float32)
    out = pd.DataFrame(numeric_data, index=df.index)
    cat_cols = [c for c in ["reaction_method", "precursor_input_mode", "precursor_source_mix"] if c in df.columns]
    dummies = pd.get_dummies(df[cat_cols].astype(str), columns=cat_cols, dtype=np.float32) if cat_cols else pd.DataFrame(index=df.index)
    out = pd.concat([out, dummies], axis=1)
    for c in feature_cols:
        if c not in out.columns:
            out[c] = 0.0
    return out[list(feature_cols)].to_numpy(np.float32)


def top_labels(proba: np.ndarray, classes: Sequence[str], k: int) -> List[List[Tuple[str, float]]]:
    idx = np.argsort(proba, axis=1)[:, ::-1][:, :k]
    out = []
    for i in range(proba.shape[0]):
        out.append([(str(classes[j]), float(proba[i, j])) for j in idx[i]])
    return out


def temp_bin_center(label: str) -> float:
    return float({
        "<400": 300, "400-600": 500, "600-800": 700, "800-1000": 900, "1000-1200": 1100, "1200-1400": 1300, ">1400": 1500,
        "<80": 60, "80-150": 115, "150-250": 200, "250-400": 325, "400-700": 550, ">700": 800,
        "not_applicable": 25, "post_anneal_low": 700, "post_anneal_medium": 1050, "post_anneal_high": 1350,
        "<100": 60, "100-300": 200, "300-600": 450, "600-900": 750, "900-1200": 1050, ">1200": 1350,
    }.get(str(label), 600))


def time_bin_center(label: str) -> float:
    return float({"<1h": 0.5, "1-6h": 3, "6-12h": 9, "12-24h": 18, "24-48h": 36, "48-96h": 72, ">96h": 144}.get(str(label), 24))


def hit_flags(true_temp: float, true_time: float, true_atm: str, known_atm: int, temp: float, time_h: float, atm: str) -> Tuple[int, int, float, float, int]:
    temp_err = abs(float(temp) - float(true_temp))
    time_err = abs(float(time_h) - float(true_time))
    atm_ok = int((not bool(known_atm)) or str(atm) == str(true_atm))
    strict = int(temp_err <= 100 and time_err <= 24 and atm_ok)
    relaxed = int(temp_err <= 200 and time_err <= 48 and atm_ok)
    return strict, relaxed, temp_err, time_err, atm_ok


def method_templates(train: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for method, g in train.groupby("reaction_method", sort=False):
        temp = pd.to_numeric(g["temperature_clipped"], errors="coerce")
        time_h = pd.to_numeric(g["time_h_clipped"], errors="coerce")
        atm = g.loc[g["atmosphere_known_mask"].astype(int) == 1, "atmosphere_normalized"].mode()
        solv = g.loc[g["solvent_known_mask"].astype(int) == 1, "solvent_normalized"].mode()
        out[str(method)] = {
            "temperature": float(temp.median()),
            "time": float(time_h.median()),
            "atmosphere": str(atm.iloc[0]) if len(atm) else MISSING,
            "solvent": str(solv.iloc[0]) if len(solv) else MISSING,
        }
    return out


def build_retrieval(train: pd.DataFrame, numeric_cols: Sequence[str]) -> Dict[str, Any]:
    x = train[list(numeric_cols)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    scaler = StandardScaler().fit(x)
    xs = scaler.transform(x)
    by_method: Dict[str, Any] = {}
    for method, idx_values in train.groupby("reaction_method", sort=False).groups.items():
        idx = np.asarray(list(idx_values), dtype=int)
        if len(idx) < 2:
            continue
        nn = NearestNeighbors(n_neighbors=min(5, len(idx)), metric="euclidean")
        nn.fit(xs[idx])
        by_method[str(method)] = {"idx": idx, "nn": nn}
    return {"scaler": scaler, "x": xs, "by_method": by_method}


def metric_by_k(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in [1, 3, 5, 10, 20]:
        sub = df[df["condition_rank_raw"] <= k]
        g = sub.groupby("sample_id", sort=False)
        out[f"top{k}_strict_condition"] = float(g["strict_hit_if_eval"].max().mean()) if len(g) else 0.0
        out[f"top{k}_relaxed_condition"] = float(g["relaxed_hit_if_eval"].max().mean()) if len(g) else 0.0
    g = df.groupby("sample_id", sort=False)
    out["oracle_strict_condition"] = float(g["strict_hit_if_eval"].max().mean()) if len(g) else 0.0
    out["oracle_relaxed_condition"] = float(g["relaxed_hit_if_eval"].max().mean()) if len(g) else 0.0
    core = df[df["reaction_method"].isin(CORE_METHODS)]
    if len(core):
        cg = core[core["condition_rank_raw"] <= 10].groupby("sample_id", sort=False)
        out["core_top10_relaxed_condition"] = float(cg["relaxed_hit_if_eval"].max().mean()) if len(cg) else 0.0
    return out


def generate_split(split: str, df: pd.DataFrame, train: pd.DataFrame, pack: Mapping[str, Any], templates: Mapping[str, Mapping[str, Any]], retrieval: Mapping[str, Any], output_dir: Path, top_atm: int, top_bins: int) -> pd.DataFrame:
    models = pack["models"]
    enc = pack["encoders"]
    x = make_x(df, pack["numeric_feature_cols"], pack["feature_cols"])
    pred_temp = np.clip(models["temperature_point"].predict(x), 20, 1800)
    tq = {q: np.clip(models[f"temperature_quantile_p{q}"].predict(x), 20, 1800) for q in [10, 50, 90]}
    pred_time_log = models["time_point"].predict(x)
    pred_time = np.expm1(pred_time_log).clip(0.05, 500)
    timeq = {q: np.expm1(models[f"time_quantile_p{q}"].predict(x)).clip(0.05, 500) for q in [10, 50, 90]}
    atm_proba = models["atmosphere"].predict_proba(x)
    solvent_proba = models["solvent"].predict_proba(x)
    temp_bin_proba = models["temperature_bin"].predict_proba(x)
    time_bin_proba = models["time_bin"].predict_proba(x)
    atm_top = top_labels(atm_proba, enc["atmosphere"].classes_, top_atm)
    solv_top = top_labels(solvent_proba, enc["solvent"].classes_, 1)
    temp_bin_top = top_labels(temp_bin_proba, enc["temperature_bin"].classes_, top_bins)
    time_bin_top = top_labels(time_bin_proba, enc["time_bin"].classes_, top_bins)
    retr_x = retrieval["scaler"].transform(df[list(pack["numeric_feature_cols"])].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32))

    rows: List[Dict[str, Any]] = []
    seen_keys: set[Tuple[str, int, int, str]] = set()

    def add(i: int, temp: float, time_h: float, atm: str, solv: str, source: str, scores: Dict[str, float]) -> None:
        sid = str(df.iloc[i]["sample_id"])
        temp = float(np.clip(temp, 20, 1800))
        time_h = float(np.clip(time_h, 0.05, 500))
        key = (sid, int(round(temp / 10) * 10), int(round(time_h)), str(atm))
        if key in seen_keys:
            return
        seen_keys.add(key)
        true_temp = float(df.iloc[i]["temperature_c_raw"])
        true_time = float(df.iloc[i]["time_h_raw"])
        true_atm = str(df.iloc[i]["atmosphere_normalized"])
        known_atm = int(df.iloc[i]["atmosphere_known_mask"])
        strict, relaxed, te, he, atm_ok = hit_flags(true_temp, true_time, true_atm, known_atm, temp, time_h, atm)
        open_pen = float(df.iloc[i].get("contains_open_generated_precursor", 0)) * 0.15
        rep_pen = float(df.iloc[i].get("contains_repair_precursor", 0)) * 0.10
        total = (
            1.0 * scores.get("temperature_point_score", 0)
            + 0.8 * scores.get("temperature_bin_score", 0)
            + 1.0 * scores.get("time_point_score", 0)
            + 0.8 * scores.get("time_bin_score", 0)
            + 0.7 * scores.get("atmosphere_probability", 0)
            + 0.2 * scores.get("solvent_probability", 0)
            + 0.8 * scores.get("retrieval_score", 0)
            + 0.5 * scores.get("method_template_score", 0)
            + 0.6 * scores.get("multimodal_template_score", 0)
            + 0.4 * scores.get("condition_prior_score", 0)
            + 0.3 * scores.get("precursor_confidence_score", 0)
            - open_pen
            - rep_pen
        )
        rows.append({
            "sample_id": sid,
            "condition_candidate_id": f"{sid}::c{len(rows)}",
            "reaction_method": str(df.iloc[i]["reaction_method"]),
            "temperature_c": temp,
            "temperature_low_c": max(20.0, temp - 100.0),
            "temperature_high_c": min(1800.0, temp + 100.0),
            "time_h": time_h,
            "time_low_h": max(0.05, time_h * 0.5),
            "time_high_h": min(500.0, time_h * 1.5 + 1e-6),
            "atmosphere": atm,
            "solvent": solv,
            "condition_source": source,
            **{k: float(v) for k, v in scores.items()},
            "open_generated_penalty": open_pen,
            "repair_penalty": rep_pen,
            "total_score_raw": float(total),
            "true_temperature_c": true_temp,
            "true_time_h": true_time,
            "true_atmosphere": true_atm,
            "atmosphere_known_mask": known_atm,
            "strict_hit_if_eval": strict,
            "relaxed_hit_if_eval": relaxed,
            "temp_error": te,
            "time_error": he,
            "atmosphere_correct": atm_ok,
            "precursor_confidence_score": float(df.iloc[i].get("precursor_confidence_score", 0.0)),
            "precursor_f1_to_true": float(df.iloc[i].get("precursor_f1_to_true", 0.0)),
        })

    for i in range(len(df)):
        atm, atm_p = atm_top[i][0]
        solv, solv_p = solv_top[i][0]
        add(i, pred_temp[i], pred_time[i], atm, solv, "point_model", {
            "temperature_point_score": 1.0, "temperature_bin_score": float(np.max(temp_bin_proba[i])),
            "time_point_score": 1.0, "time_bin_score": float(np.max(time_bin_proba[i])),
            "atmosphere_probability": atm_p, "solvent_probability": solv_p,
            "condition_prior_score": 0.4, "precursor_confidence_score": float(df.iloc[i].get("precursor_confidence_score", 0.0)),
        })
        for tqv, hqv, qname in [(tq[10][i], timeq[10][i], "p10"), (tq[50][i], timeq[50][i], "p50"), (tq[90][i], timeq[90][i], "p90"), (tq[10][i], timeq[90][i], "p10_p90"), (tq[90][i], timeq[10][i], "p90_p10")]:
            add(i, tqv, hqv, atm, solv, f"quantile_model_{qname}", {
                "temperature_point_score": 0.6, "temperature_bin_score": 0.5, "time_point_score": 0.6, "time_bin_score": 0.5,
                "atmosphere_probability": atm_p, "solvent_probability": solv_p, "condition_prior_score": 0.35,
                "precursor_confidence_score": float(df.iloc[i].get("precursor_confidence_score", 0.0)),
            })
        for tb, tbp in temp_bin_top[i]:
            for hb, hbp in time_bin_top[i]:
                add(i, temp_bin_center(tb), time_bin_center(hb), atm, solv, "bin_center", {
                    "temperature_point_score": 0.3, "temperature_bin_score": tbp, "time_point_score": 0.3, "time_bin_score": hbp,
                    "atmosphere_probability": atm_p, "solvent_probability": solv_p, "condition_prior_score": 0.3,
                    "precursor_confidence_score": float(df.iloc[i].get("precursor_confidence_score", 0.0)),
                })
        for q in [10, 50, 90]:
            add(i, float(df.iloc[i].get(f"temperature_p{q}", pred_temp[i])), float(df.iloc[i].get(f"time_p{q}", pred_time[i])), atm, solv, f"multimodal_group_template_p{q}", {
                "temperature_point_score": 0.2, "temperature_bin_score": 0.4, "time_point_score": 0.2, "time_bin_score": 0.4,
                "atmosphere_probability": atm_p, "solvent_probability": solv_p, "multimodal_template_score": 1.0,
                "condition_prior_score": 0.3, "precursor_confidence_score": float(df.iloc[i].get("precursor_confidence_score", 0.0)),
            })
        tpl = templates.get(str(df.iloc[i]["reaction_method"]))
        if tpl:
            add(i, tpl["temperature"], tpl["time"], tpl["atmosphere"], tpl["solvent"], "method_condition_template", {
                "method_template_score": 1.0, "condition_prior_score": 0.8, "atmosphere_probability": 0.5, "solvent_probability": 0.5,
                "precursor_confidence_score": float(df.iloc[i].get("precursor_confidence_score", 0.0)),
            })
        ref = retrieval["by_method"].get(str(df.iloc[i]["reaction_method"]))
        if ref is not None:
            dist, nbr = ref["nn"].kneighbors(retr_x[i:i + 1], n_neighbors=min(3, len(ref["idx"])))
            for d, loc in zip(dist[0], nbr[0]):
                j = int(ref["idx"][int(loc)])
                add(i, train.iloc[j]["temperature_clipped"], train.iloc[j]["time_h_clipped"], str(train.iloc[j]["atmosphere_normalized"]), str(train.iloc[j]["solvent_normalized"]), "nearest_neighbor_condition", {
                    "retrieval_score": 1.0 / (1.0 + float(d)), "condition_prior_score": 0.4,
                    "atmosphere_probability": 0.5, "solvent_probability": 0.5,
                    "precursor_confidence_score": float(df.iloc[i].get("precursor_confidence_score", 0.0)),
                })

    out = pd.DataFrame(rows)
    out = out.sort_values(["sample_id", "total_score_raw"], ascending=[True, False], kind="mergesort")
    out["condition_rank_raw"] = out.groupby("sample_id", sort=False).cumcount() + 1
    out.to_csv(output_dir / f"{split}_condition_candidates.csv", index=False)
    metrics = metric_by_k(out)
    by_method = {str(m): metric_by_k(g) for m, g in out.groupby("reaction_method", sort=False)}
    write_json(output_dir / f"{split}_condition_candidate_metrics.json", {"metrics": metrics, "by_method": by_method, "n_candidates": int(len(out))})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Stage3 v3 distributional condition candidates.")
    ap.add_argument("--input_dir", default="data/interim/generative/stage3_condition_targets_v3_20260610")
    ap.add_argument("--model", default="runs/stage3/distributional_condition_v3_20260610/stage3_distributional_condition_v3.joblib")
    ap.add_argument("--output_dir", default="outputs/evaluation/stage3_condition_candidates_v3_20260610")
    ap.add_argument("--top_atmospheres", type=int, default=2)
    ap.add_argument("--top_bins", type=int, default=3)
    ap.add_argument("--splits", default="val,test", help="Comma-separated splits to generate. Default skips train for speed.")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = (read_table(input_dir / s) for s in ["train", "val", "test"])
    pack = joblib.load(args.model)
    templates = method_templates(train)
    retrieval = build_retrieval(train, pack["numeric_feature_cols"])
    split_map = {"train": train, "val": val, "test": test}
    summaries = {}
    for split in [s.strip() for s in str(args.splits).split(",") if s.strip()]:
        df = split_map[split]
        out = generate_split(split, df, train, pack, templates, retrieval, output_dir, args.top_atmospheres, args.top_bins)
        summaries[split] = {"rows": int(df["sample_id"].nunique()), "candidates": int(len(out)), "metrics": metric_by_k(out)}
    write_json(output_dir / "condition_candidate_pool_v3_summary.json", {"config": vars(args), "splits": summaries})
    print(json.dumps(to_builtin(summaries), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
