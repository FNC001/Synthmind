#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


CORE_METHODS = {"solid_state", "solution", "melt_arc"}
NUMERIC_FEATURES = [
    "precursor_rank", "condition_rank", "precursor_score", "condition_score", "precursor_score_norm",
    "condition_score_norm", "precursor_confidence", "condition_confidence", "precursor_condition_compatibility_score",
    "reaction_method_prior_score", "atmosphere_probability", "solvent_probability", "temperature_point_score",
    "temperature_bin_score", "time_point_score", "time_bin_score", "retrieval_score", "method_template_score",
    "multimodal_template_score", "open_generated_penalty", "repair_penalty", "route_total_score_raw",
]
CAT_FEATURES = ["reaction_method", "candidate_source", "candidate_source_mix", "condition_source", "atmosphere", "solvent"]


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


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["precursor_rank_score"] = 1.0 / pd.to_numeric(out["precursor_rank"], errors="coerce").clip(lower=1).fillna(999)
    out["condition_rank_score"] = 1.0 / pd.to_numeric(out["condition_rank"], errors="coerce").clip(lower=1).fillna(999)
    out["rank_product_score"] = out["precursor_rank_score"] * out["condition_rank_score"]
    out["condition_is_point"] = (out["condition_source"].astype(str) == "point_model").astype(int)
    out["condition_is_quantile"] = out["condition_source"].astype(str).str.contains("quantile", regex=False).astype(int)
    out["condition_is_bin"] = (out["condition_source"].astype(str) == "bin_center").astype(int)
    out["condition_is_template"] = out["condition_source"].astype(str).str.contains("template", regex=False).astype(int)
    return out


def relevance(df: pd.DataFrame) -> np.ndarray:
    strict = df["strict_route_hit_if_eval"].astype(bool).to_numpy()
    relaxed = df["relaxed_route_hit_if_eval"].astype(bool).to_numpy()
    usable = df["usable_relaxed_route_hit_if_eval"].astype(bool).to_numpy()
    good_prec = df["precursor_exact_if_eval"].astype(bool).to_numpy()
    good_cond = df["relaxed_condition_hit_if_eval"].astype(bool).to_numpy()
    rel = np.zeros(len(df), dtype=np.int32)
    rel[good_prec | good_cond] = 1
    rel[usable] = 2
    rel[relaxed] = 3
    rel[strict] = 4
    return rel


def make_x(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    train = add_features(train)
    test = add_features(test)
    nums = [c for c in NUMERIC_FEATURES + ["precursor_rank_score", "condition_rank_score", "rank_product_score", "condition_is_point", "condition_is_quantile", "condition_is_bin", "condition_is_template"] if c in train.columns and c in test.columns]
    for df in [train, test]:
        for c in nums:
            df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        for c in CAT_FEATURES:
            if c not in df.columns:
                df[c] = "missing"
    cat = pd.get_dummies(pd.concat([train[CAT_FEATURES], test[CAT_FEATURES]], ignore_index=True).astype(str), dtype=np.float32)
    tx = pd.concat([train[nums].reset_index(drop=True), cat.iloc[:len(train)].reset_index(drop=True)], axis=1)
    vx = pd.concat([test[nums].reset_index(drop=True), cat.iloc[len(train):].reset_index(drop=True)], axis=1)
    return tx, vx, tx.columns.tolist()


def metrics(df: pd.DataFrame, rank_col: str) -> Dict[str, float]:
    out = {"n_samples": int(df["sample_id"].nunique()), "n_candidates": int(len(df))}
    for k in [1, 3, 5, 10, 50, 100, 200, 400]:
        sub = df[df[rank_col] <= k]
        g = sub.groupby("sample_id", sort=False)
        if len(g):
            out[f"top{k}_strict_route"] = float(g["strict_route_hit_if_eval"].max().mean())
            out[f"top{k}_relaxed_route"] = float(g["relaxed_route_hit_if_eval"].max().mean())
            out[f"top{k}_usable_relaxed_route"] = float(g["usable_relaxed_route_hit_if_eval"].max().mean())
    return out


def rank(df: pd.DataFrame, score_col: str, rank_col: str) -> pd.DataFrame:
    out = df.sort_values(["sample_id", score_col, "route_total_score_raw"], ascending=[True, False, False], kind="mergesort").copy()
    out[rank_col] = out.groupby("sample_id", sort=False).cumcount() + 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train diagnostic Stage35 route reranker v3. Uses val candidates as training if no train route candidates exist.")
    ap.add_argument("--candidate_dir", default="outputs/evaluation/stage35_route_score_calibration_v3_20260610")
    ap.add_argument("--run_dir", default="runs/stage35/route_reranker_v3_distributional_20260610")
    ap.add_argument("--max_train_negatives", type=int, default=250000)
    ap.add_argument("--seed", type=int, default=20260612)
    args = ap.parse_args()
    cand = Path(args.candidate_dir)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(cand / "val_route_candidates_calibrated.csv")
    test = pd.read_csv(cand / "test_route_candidates_calibrated.csv")
    y_full = relevance(train)
    pos = np.where(y_full > 0)[0]
    neg = np.where(y_full == 0)[0]
    rng = np.random.default_rng(args.seed)
    if len(neg) > args.max_train_negatives:
        neg = rng.choice(neg, size=args.max_train_negatives, replace=False)
    sel = np.sort(np.concatenate([pos, neg]))
    train_small = train.iloc[sel].reset_index(drop=True)
    x_train, x_test, feat_cols = make_x(train_small, test)
    y_train = relevance(train_small)
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=5,
        n_estimators=350,
        learning_rate=0.04,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=25,
        reg_lambda=1.0,
        random_state=args.seed,
        n_jobs=8,
        verbose=-1,
    )
    model.fit(x_train.to_numpy(np.float32), y_train)
    proba = model.predict_proba(x_test.to_numpy(np.float32))
    gains = np.asarray([0, 1, 2, 4, 7], dtype=float)
    test["stage35_reranker_v3_score"] = proba @ gains
    test_ranked = rank(test, "stage35_reranker_v3_score", "stage35_reranker_v3_rank")
    test_ranked.to_csv(run_dir / "test_route_candidates_stage35_reranked_v3.csv", index=False)
    joblib.dump({"model": model, "feature_cols": feat_cols, "config": vars(args)}, run_dir / "stage35_route_reranker_v3.joblib")
    summary = {
        "config": vars(args),
        "training_note": "Diagnostic model trained on validation route candidates because full train route candidates v3 were not generated.",
        "n_train_rows": int(len(train_small)),
        "train_relevance_counts": {str(k): int(v) for k, v in pd.Series(y_train).value_counts().sort_index().items()},
        "baseline_raw_test": metrics(test, "route_rank_calibrated_v3"),
        "reranker_test": metrics(test_ranked, "stage35_reranker_v3_rank"),
    }
    write_json(run_dir / "metrics.json", summary)
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
