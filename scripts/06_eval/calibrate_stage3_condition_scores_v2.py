#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd


FEATURE_COLS = [
    "model_score",
    "retrieval_score",
    "method_template_score",
    "condition_prior_score",
    "temperature_plausibility_score",
    "time_plausibility_score",
    "atmosphere_probability",
]

SOURCE_COL = "condition_source"
SOURCES = [
    "model_point",
    "calibrated_blend",
    "retrieval_template",
    "method_template",
    "model_quantile_low",
    "model_quantile_high",
]


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
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def metric_by_k(df: pd.DataFrame, rank_col: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in [1, 3, 5, 10]:
        sub = df[df[rank_col] <= k]
        g = sub.groupby("sample_id", sort=False)
        out[f"top{k}_strict_condition"] = float(g["strict_hit_if_eval"].max().mean()) if len(g) else 0.0
        out[f"top{k}_relaxed_condition"] = float(g["relaxed_hit_if_eval"].max().mean()) if len(g) else 0.0
    g = df.groupby("sample_id", sort=False)
    out["oracle_strict_condition"] = float(g["strict_hit_if_eval"].max().mean()) if len(g) else 0.0
    out["oracle_relaxed_condition"] = float(g["relaxed_hit_if_eval"].max().mean()) if len(g) else 0.0
    return out


def objective(metrics: Mapping[str, float]) -> float:
    return (
        0.35 * metrics.get("top1_strict_condition", 0.0)
        + 0.30 * metrics.get("top1_relaxed_condition", 0.0)
        + 0.15 * metrics.get("top5_relaxed_condition", 0.0)
        + 0.10 * metrics.get("top10_relaxed_condition", 0.0)
        + 0.05 * metrics.get("oracle_strict_condition", 0.0)
        + 0.05 * metrics.get("oracle_relaxed_condition", 0.0)
    )


def feature_stats(df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    stats: Dict[str, Tuple[float, float]] = {}
    for col in FEATURE_COLS + ["condition_total_score"]:
        vals = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
        lo = float(np.quantile(vals, 0.01))
        hi = float(np.quantile(vals, 0.99))
        if hi <= lo:
            hi = lo + 1.0
        stats[col] = (lo, hi)
    return stats


def normalized_matrix(df: pd.DataFrame, stats: Mapping[str, Tuple[float, float]]) -> Dict[str, np.ndarray]:
    mats: Dict[str, np.ndarray] = {}
    for col in FEATURE_COLS + ["condition_total_score"]:
        lo, hi = stats[col]
        vals = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
        mats[col] = np.clip((vals - lo) / (hi - lo), 0.0, 1.0)
    for source in SOURCES:
        mats[f"source::{source}"] = (df[SOURCE_COL].astype(str).to_numpy() == source).astype(float)
    return mats


def score_from_weights(mats: Mapping[str, np.ndarray], weights: Mapping[str, float]) -> np.ndarray:
    score = np.zeros_like(next(iter(mats.values())), dtype=float)
    for col in FEATURE_COLS + ["condition_total_score"]:
        score += float(weights.get(col, 0.0)) * mats[col]
    for source in SOURCES:
        score += float(weights.get(f"source::{source}", 0.0)) * mats[f"source::{source}"]
    return score


def apply_weights(df: pd.DataFrame, mats: Mapping[str, np.ndarray], weights: Mapping[str, float]) -> pd.DataFrame:
    out = df.copy()
    out["condition_calibrated_score"] = score_from_weights(mats, weights)
    out = out.sort_values(["sample_id", "condition_calibrated_score", "condition_total_score"], ascending=[True, False, False], kind="mergesort")
    out["condition_rank_calibrated"] = out.groupby("sample_id", sort=False).cumcount() + 1
    return out


def candidate_weight_sets(n_random: int, seed: int) -> Iterable[Dict[str, float]]:
    presets: List[Dict[str, float]] = [
        {"condition_total_score": 1.0},
        {"model_score": 1.0, "source::model_point": 1.0},
        {"model_score": 1.0, "atmosphere_probability": 0.4, "temperature_plausibility_score": 0.2, "time_plausibility_score": 0.2, "source::model_point": 1.0},
        {"model_score": 1.0, "source::model_point": 1.0, "source::calibrated_blend": 0.6, "source::retrieval_template": 0.2},
        {"model_score": 0.8, "retrieval_score": 0.3, "condition_prior_score": 0.2, "atmosphere_probability": 0.5, "source::model_point": 0.8, "source::calibrated_blend": 0.4},
        {"model_score": 1.0, "method_template_score": -0.8, "retrieval_score": 0.2, "source::model_point": 1.2, "source::calibrated_blend": 0.5},
    ]
    for weights in presets:
        yield weights

    rng = np.random.default_rng(seed)
    names = FEATURE_COLS + ["condition_total_score"]
    source_names = [f"source::{s}" for s in SOURCES]
    for _ in range(n_random):
        feat = rng.gamma(shape=1.2, scale=1.0, size=len(names))
        feat = feat / max(float(feat.sum()), 1e-9)
        weights = {name: float(val * rng.uniform(0.5, 4.0)) for name, val in zip(names, feat)}
        weights.update(
            {
                "source::model_point": float(rng.uniform(0.0, 3.0)),
                "source::calibrated_blend": float(rng.uniform(-0.2, 2.0)),
                "source::retrieval_template": float(rng.uniform(-0.8, 1.2)),
                "source::method_template": float(rng.uniform(-1.5, 0.6)),
                "source::model_quantile_low": float(rng.uniform(-1.5, 0.5)),
                "source::model_quantile_high": float(rng.uniform(-1.5, 0.5)),
            }
        )
        yield weights


def summarize_by_source(df: pd.DataFrame, rank_col: str) -> pd.DataFrame:
    top1 = df[df[rank_col] == 1]
    rows = []
    for source, g in top1.groupby(SOURCE_COL):
        rows.append(
            {
                "condition_source": source,
                "top1_count": int(len(g)),
                "top1_share": float(len(g) / max(len(top1), 1)),
                "top1_strict": float(g["strict_hit_if_eval"].mean()) if len(g) else 0.0,
                "top1_relaxed": float(g["relaxed_hit_if_eval"].mean()) if len(g) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("top1_count", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate Stage3 condition candidate scores v2 using validation top-K condition success.")
    ap.add_argument("--candidate_dir", default="outputs/evaluation/stage3_condition_candidates_v2_20260610")
    ap.add_argument("--output_dir", default="outputs/evaluation/stage3_condition_calibration_v2_20260610")
    ap.add_argument("--n_random", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=20260611)
    args = ap.parse_args()

    candidate_dir = Path(args.candidate_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    val = pd.read_csv(candidate_dir / "val_condition_candidates.csv")
    test = pd.read_csv(candidate_dir / "test_condition_candidates.csv")
    train = pd.read_csv(candidate_dir / "train_condition_candidates.csv")
    stats = feature_stats(val)
    val_mats = normalized_matrix(val, stats)
    records: List[Dict[str, Any]] = []
    best: Dict[str, Any] | None = None
    for idx, weights in enumerate(candidate_weight_sets(args.n_random, args.seed)):
        ranked = apply_weights(val, val_mats, weights)
        metrics = metric_by_k(ranked, "condition_rank_calibrated")
        obj = objective(metrics)
        rec = {"search_id": idx, "objective": obj, **metrics, "weights": json.dumps(to_builtin(weights), sort_keys=True)}
        records.append(rec)
        if best is None or obj > best["objective"]:
            best = {"search_id": idx, "objective": obj, "metrics": metrics, "weights": weights}

    assert best is not None
    weight_search = pd.DataFrame(records).sort_values("objective", ascending=False)
    weight_search.to_csv(output_dir / "val_weight_search.csv", index=False)
    write_json(output_dir / "best_weights.json", {"feature_stats": stats, **best})

    outputs = {}
    for split, df in [("train", train), ("val", val), ("test", test)]:
        mats = normalized_matrix(df, stats)
        ranked = apply_weights(df, mats, best["weights"])
        metrics = metric_by_k(ranked, "condition_rank_calibrated")
        ranked.to_csv(output_dir / f"{split}_condition_candidates_calibrated_v2.csv", index=False)
        summarize_by_source(ranked, "condition_rank_calibrated").to_csv(output_dir / f"{split}_top1_source_breakdown.csv", index=False)
        outputs[split] = metrics

    write_json(output_dir / "condition_calibration_metrics.json", outputs)
    report = [
        "# Stage3 Condition Score Calibration v2",
        "",
        f"- candidate_dir: `{candidate_dir}`",
        f"- searched weight sets: {len(records)}",
        f"- best validation objective: {best['objective']:.6f}",
        "",
        "## Validation Metrics",
        "",
        json.dumps(to_builtin(outputs["val"]), ensure_ascii=False, indent=2),
        "",
        "## Test Metrics",
        "",
        json.dumps(to_builtin(outputs["test"]), ensure_ascii=False, indent=2),
    ]
    (output_dir / "condition_calibration_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(to_builtin({"best": best, "metrics": outputs}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
