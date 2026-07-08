#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import numpy as np
import pandas as pd


CORE_METHODS = {"solid_state", "solution", "melt_arc"}
SCORE_COLS = [
    "temperature_point_score",
    "temperature_bin_score",
    "time_point_score",
    "time_bin_score",
    "atmosphere_probability",
    "solvent_probability",
    "retrieval_score",
    "method_template_score",
    "multimodal_template_score",
    "condition_prior_score",
    "precursor_confidence_score",
    "open_generated_penalty",
    "repair_penalty",
]
SOURCES = [
    "point_model",
    "quantile_model_p10",
    "quantile_model_p50",
    "quantile_model_p90",
    "quantile_model_p10_p90",
    "quantile_model_p90_p10",
    "bin_center",
    "multimodal_group_template_p10",
    "multimodal_group_template_p50",
    "multimodal_group_template_p90",
    "method_condition_template",
    "nearest_neighbor_condition",
]


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


def feature_stats(df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    out = {}
    for col in SCORE_COLS + ["total_score_raw"]:
        vals = pd.to_numeric(df.get(col, 0), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
        lo, hi = float(np.quantile(vals, 0.01)), float(np.quantile(vals, 0.99))
        if hi <= lo:
            hi = lo + 1.0
        out[col] = (lo, hi)
    return out


def mats(df: pd.DataFrame, stats: Mapping[str, Tuple[float, float]]) -> Dict[str, np.ndarray]:
    out = {}
    for col in SCORE_COLS + ["total_score_raw"]:
        lo, hi = stats[col]
        vals = pd.to_numeric(df.get(col, 0), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
        out[col] = np.clip((vals - lo) / (hi - lo), 0, 1)
    for src in SOURCES:
        out[f"source::{src}"] = (df["condition_source"].astype(str).to_numpy() == src).astype(float)
    return out


def score(m: Mapping[str, np.ndarray], weights: Mapping[str, float]) -> np.ndarray:
    s = np.zeros_like(next(iter(m.values())), dtype=float)
    for col in SCORE_COLS + ["total_score_raw"]:
        s += float(weights.get(col, 0.0)) * m[col]
    for src in SOURCES:
        s += float(weights.get(f"source::{src}", 0.0)) * m[f"source::{src}"]
    return s


def rank(df: pd.DataFrame, m: Mapping[str, np.ndarray], weights: Mapping[str, float]) -> pd.DataFrame:
    out = df.copy()
    out["condition_calibrated_score_v3"] = score(m, weights)
    out = out.sort_values(["sample_id", "condition_calibrated_score_v3", "total_score_raw"], ascending=[True, False, False], kind="mergesort")
    out["condition_rank_calibrated_v3"] = out.groupby("sample_id", sort=False).cumcount() + 1
    return out


def ndcg10(df: pd.DataFrame, rank_col: str) -> float:
    vals = []
    for _, g in df[df[rank_col] <= 10].groupby("sample_id", sort=False):
        rel = (3 * g["relaxed_hit_if_eval"].astype(float) + 2 * g["strict_hit_if_eval"].astype(float)).to_numpy(float)
        order = np.arange(1, len(rel) + 1)
        dcg = float(np.sum(rel / np.log2(order + 1)))
        ideal = np.sort(rel)[::-1]
        idcg = float(np.sum(ideal / np.log2(order + 1)))
        vals.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def metrics(df: pd.DataFrame, rank_col: str = "condition_rank_calibrated_v3") -> Dict[str, float]:
    out = {}
    for k in [1, 3, 5, 10, 20]:
        sub = df[df[rank_col] <= k]
        g = sub.groupby("sample_id", sort=False)
        out[f"top{k}_strict_condition"] = float(g["strict_hit_if_eval"].max().mean()) if len(g) else 0.0
        out[f"top{k}_relaxed_condition"] = float(g["relaxed_hit_if_eval"].max().mean()) if len(g) else 0.0
    core = df[df["reaction_method"].isin(CORE_METHODS)]
    cg = core[core[rank_col] <= 10].groupby("sample_id", sort=False)
    out["core_top10_relaxed_condition"] = float(cg["relaxed_hit_if_eval"].max().mean()) if len(cg) else 0.0
    g = df.groupby("sample_id", sort=False)
    out["oracle_strict_condition"] = float(g["strict_hit_if_eval"].max().mean()) if len(g) else 0.0
    out["oracle_relaxed_condition"] = float(g["relaxed_hit_if_eval"].max().mean()) if len(g) else 0.0
    out["ndcg10_condition"] = ndcg10(df, rank_col)
    return out


def objective(mm: Mapping[str, float]) -> float:
    return (
        0.25 * mm["top1_relaxed_condition"]
        + 0.20 * mm["top1_strict_condition"]
        + 0.25 * mm["top10_relaxed_condition"]
        + 0.15 * mm["top10_strict_condition"]
        + 0.10 * mm["core_top10_relaxed_condition"]
        + 0.05 * mm["ndcg10_condition"]
    )


def weight_sets(n_random: int, seed: int) -> Iterable[Dict[str, float]]:
    presets = [
        {"total_score_raw": 1.0},
        {"temperature_point_score": 1, "time_point_score": 1, "atmosphere_probability": 0.7, "source::point_model": 2.0},
        {"temperature_bin_score": 1, "time_bin_score": 1, "atmosphere_probability": 0.6, "source::bin_center": 0.5, "source::point_model": 1.5},
        {"temperature_point_score": 1, "time_point_score": 1, "retrieval_score": 0.4, "multimodal_template_score": 0.5, "source::point_model": 1.5},
    ]
    yield from presets
    rng = np.random.default_rng(seed)
    for _ in range(n_random):
        w = {col: float(rng.uniform(0.0, 2.0)) for col in SCORE_COLS + ["total_score_raw"]}
        w["open_generated_penalty"] = float(rng.uniform(-1.0, -0.05))
        w["repair_penalty"] = float(rng.uniform(-1.0, -0.05))
        for src in SOURCES:
            lo, hi = (-1.0, 1.5)
            if src == "point_model":
                lo, hi = (0.0, 3.0)
            if src in {"method_condition_template", "nearest_neighbor_condition"}:
                lo, hi = (-1.2, 1.0)
            w[f"source::{src}"] = float(rng.uniform(lo, hi))
        yield w


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate Stage3 v3 distributional condition candidate scores.")
    ap.add_argument("--candidate_dir", default="outputs/evaluation/stage3_condition_candidates_v3_20260610")
    ap.add_argument("--output_dir", default="outputs/evaluation/stage3_condition_calibration_v3_20260610")
    ap.add_argument("--n_random", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=20260612)
    args = ap.parse_args()
    cand = Path(args.candidate_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    train_path = cand / "train_condition_candidates.csv"
    train = pd.read_csv(train_path) if train_path.exists() else None
    val = pd.read_csv(cand / "val_condition_candidates.csv")
    test = pd.read_csv(cand / "test_condition_candidates.csv")
    stats = feature_stats(val)
    vm = mats(val, stats)
    rows = []
    best = None
    for i, w in enumerate(weight_sets(args.n_random, args.seed)):
        rr = rank(val, vm, w)
        mm = metrics(rr)
        obj = objective(mm)
        rec = {"search_id": i, "objective": obj, **mm, "weights": json.dumps(to_builtin(w), sort_keys=True)}
        rows.append(rec)
        if best is None or obj > best["objective"]:
            best = {"search_id": i, "objective": obj, "weights": w, "metrics": mm}
    assert best is not None
    pd.DataFrame(rows).sort_values("objective", ascending=False).to_csv(outdir / "val_weight_search.csv", index=False)
    write_json(outdir / "best_weights.json", {"feature_stats": stats, **best})
    outputs = {}
    split_items = [("val", val), ("test", test)]
    if train is not None:
        split_items.insert(0, ("train", train))
    for split, df in split_items:
        rr = rank(df, mats(df, stats), best["weights"])
        rr.to_csv(outdir / f"{split}_condition_candidates_calibrated.csv", index=False)
        outputs[split] = metrics(rr)
    write_json(outdir / "test_condition_metrics_calibrated.json", outputs["test"])
    write_json(outdir / "condition_calibration_metrics_v3.json", outputs)
    report = ["# Stage3 Condition Calibration v3", "", json.dumps(to_builtin({"best": best, "metrics": outputs}), ensure_ascii=False, indent=2)]
    (outdir / "condition_calibration_v3_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(to_builtin({"best": best, "metrics": outputs}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
