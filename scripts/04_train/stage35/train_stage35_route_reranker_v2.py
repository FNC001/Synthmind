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


NUMERIC_FEATURES = [
    "precursor_rank",
    "precursor_score",
    "precursor_score_norm",
    "element_coverage",
    "missing_element_count",
    "extra_element_count",
    "candidate_size",
    "retrieval_score_precursor",
    "mlp_score",
    "family_score",
    "open_vocab_score",
    "condition_rank",
    "condition_score",
    "condition_score_norm",
    "model_score",
    "retrieval_score_condition",
    "method_template_score",
    "method_template_score_condition",
    "condition_prior_score",
    "temperature_plausibility_score",
    "time_plausibility_score",
    "atmosphere_probability",
    "route_total_score",
]

CATEGORICAL_FEATURES = ["reaction_method", "candidate_source", "candidate_source_mix", "condition_source", "atmosphere", "solvent"]


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


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["precursor_rank_inv"] = 1.0 / pd.to_numeric(out["precursor_rank"], errors="coerce").clip(lower=1).fillna(999)
    out["condition_rank_inv"] = 1.0 / pd.to_numeric(out["condition_rank"], errors="coerce").clip(lower=1).fillna(999)
    out["rank_product_inv"] = 1.0 / (
        pd.to_numeric(out["precursor_rank"], errors="coerce").clip(lower=1).fillna(999)
        * pd.to_numeric(out["condition_rank"], errors="coerce").clip(lower=1).fillna(999)
    )
    out["missing_or_extra"] = pd.to_numeric(out["missing_element_count"], errors="coerce").fillna(0.0) + pd.to_numeric(out["extra_element_count"], errors="coerce").fillna(0.0)
    out["condition_is_model_point"] = (out["condition_source"].astype(str) == "model_point").astype(int)
    out["condition_is_blend"] = (out["condition_source"].astype(str) == "calibrated_blend").astype(int)
    out["condition_is_retrieval"] = (out["condition_source"].astype(str) == "retrieval_template").astype(int)
    out["precursor_is_method_template"] = out["candidate_source"].astype(str).str.contains("method_template", regex=False).astype(int)
    out["precursor_is_open_or_generated"] = out["candidate_source_mix"].astype(str).str.contains("generated|open", case=False, regex=True).astype(int)
    return out


def make_relevance(df: pd.DataFrame) -> np.ndarray:
    precursor_exact = df["precursor_exact"].astype(bool).to_numpy()
    precursor_jaccard = pd.to_numeric(df["jaccard"], errors="coerce").fillna(0.0).to_numpy(float)
    cond_strict = pd.to_numeric(df["strict_hit_if_eval"], errors="coerce").fillna(0).to_numpy(float) > 0.5
    cond_relaxed = pd.to_numeric(df["relaxed_hit_if_eval"], errors="coerce").fillna(0).to_numpy(float) > 0.5
    route_strict = df["route_exact_strict"].astype(bool).to_numpy()
    route_relaxed = df["route_exact_relaxed"].astype(bool).to_numpy()
    route_usable = df["route_usable_relaxed"].astype(bool).to_numpy()
    rel = np.zeros(len(df), dtype=np.int32)
    rel[precursor_jaccard >= 0.5] = np.maximum(rel[precursor_jaccard >= 0.5], 1)
    rel[precursor_exact] = np.maximum(rel[precursor_exact], 2)
    rel[cond_relaxed] = np.maximum(rel[cond_relaxed], 3)
    rel[cond_strict] = np.maximum(rel[cond_strict], 4)
    rel[route_usable] = np.maximum(rel[route_usable], 5)
    rel[route_relaxed] = np.maximum(rel[route_relaxed], 6)
    rel[route_strict] = np.maximum(rel[route_strict], 7)
    return rel


def make_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    train = add_derived_features(train)
    test = add_derived_features(test)
    numeric = [c for c in NUMERIC_FEATURES if c in train.columns and c in test.columns]
    numeric += [
        "precursor_rank_inv",
        "condition_rank_inv",
        "rank_product_inv",
        "missing_or_extra",
        "condition_is_model_point",
        "condition_is_blend",
        "condition_is_retrieval",
        "precursor_is_method_template",
        "precursor_is_open_or_generated",
    ]
    for c in numeric:
        train[c] = pd.to_numeric(train[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        test[c] = pd.to_numeric(test[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    for c in CATEGORICAL_FEATURES:
        if c not in train.columns:
            train[c] = "missing"
        if c not in test.columns:
            test[c] = "missing"
    combined = pd.concat([train[CATEGORICAL_FEATURES], test[CATEGORICAL_FEATURES]], axis=0, ignore_index=True)
    dummies = pd.get_dummies(combined.astype(str), columns=CATEGORICAL_FEATURES, dummy_na=False, dtype=np.float32)
    train_cat = dummies.iloc[: len(train)].reset_index(drop=True)
    test_cat = dummies.iloc[len(train) :].reset_index(drop=True)
    train_x = pd.concat([train[numeric].reset_index(drop=True), train_cat], axis=1)
    test_x = pd.concat([test[numeric].reset_index(drop=True), test_cat], axis=1)
    return train_x, test_x, train_x.columns.tolist()


def route_metrics(df: pd.DataFrame, rank_col: str) -> Dict[str, float]:
    out: Dict[str, float] = {"n_samples": int(df["sample_id"].nunique()), "n_candidates": int(len(df))}
    for k in [1, 3, 5, 10, 20, 50, 100, 200]:
        sub = df[df[rank_col] <= k]
        g = sub.groupby("sample_id", sort=False)
        if len(g) == 0:
            continue
        out[f"top{k}_route_exact_strict"] = float(g["route_exact_strict"].max().mean())
        out[f"top{k}_route_exact_relaxed"] = float(g["route_exact_relaxed"].max().mean())
        out[f"top{k}_route_usable_relaxed"] = float(g["route_usable_relaxed"].max().mean())
        out[f"top{k}_precursor_exact"] = float(g["precursor_exact"].max().mean())
        out[f"top{k}_condition_strict"] = float(g["strict_hit_if_eval"].max().mean())
        out[f"top{k}_condition_relaxed"] = float(g["relaxed_hit_if_eval"].max().mean())
    return out


def rank_by_score(df: pd.DataFrame, score_col: str, rank_col: str) -> pd.DataFrame:
    out = df.sort_values(["sample_id", score_col, "route_total_score"], ascending=[True, False, False], kind="mergesort").copy()
    out[rank_col] = out.groupby("sample_id", sort=False).cumcount() + 1
    return out


def blend_search(train: pd.DataFrame, model_col: str, base_col: str, steps: int) -> tuple[float, Dict[str, float]]:
    best_alpha = 1.0
    best_metrics: Dict[str, float] | None = None
    for alpha in np.linspace(0.0, 1.0, max(2, int(steps))):
        tmp = train.copy()
        tmp["_blend"] = (1 - alpha) * group_z(tmp, base_col) + alpha * group_z(tmp, model_col)
        tmp = rank_by_score(tmp, "_blend", "_rank")
        metrics = route_metrics(tmp, "_rank")
        key = (metrics["top1_route_exact_strict"], metrics["top1_route_exact_relaxed"], metrics["top1_route_usable_relaxed"])
        if best_metrics is None or key > (
            best_metrics["top1_route_exact_strict"],
            best_metrics["top1_route_exact_relaxed"],
            best_metrics["top1_route_usable_relaxed"],
        ):
            best_alpha = float(alpha)
            best_metrics = metrics
    return best_alpha, best_metrics or {}


def group_z(df: pd.DataFrame, col: str) -> pd.Series:
    vals = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    g = vals.groupby(df["sample_id"], sort=False)
    mean = g.transform("mean")
    std = g.transform("std").replace(0.0, np.nan)
    return ((vals - mean) / std).fillna(0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Stage35 route reranker v2 on chemistry-checked precursor and calibrated condition candidates.")
    ap.add_argument("--candidate_dir", default="outputs/evaluation/stage35_route_candidates_v2_20260610")
    ap.add_argument("--run_dir", default="runs/stage35/route_reranker_v2_chem_checked_20260610")
    ap.add_argument("--n_estimators", type=int, default=450)
    ap.add_argument("--learning_rate", type=float, default=0.035)
    ap.add_argument("--num_leaves", type=int, default=63)
    ap.add_argument("--min_child_samples", type=int, default=30)
    ap.add_argument("--blend_steps", type=int, default=41)
    ap.add_argument("--seed", type=int, default=20260611)
    args = ap.parse_args()

    candidate_dir = Path(args.candidate_dir)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(candidate_dir / "val_route_candidates.csv")
    test = pd.read_csv(candidate_dir / "test_route_candidates.csv")
    train = train.sort_values(["sample_id", "route_total_score"], ascending=[True, False], kind="mergesort").copy()
    test = test.sort_values(["sample_id", "route_total_score"], ascending=[True, False], kind="mergesort").copy()
    x_train, x_test, feature_cols = make_features(train, test)
    y_train = make_relevance(train)
    y_test = make_relevance(test)
    group_train = train.groupby("sample_id", sort=False).size().astype(int).tolist()

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        num_leaves=int(args.num_leaves),
        min_child_samples=int(args.min_child_samples),
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        label_gain=[0, 1, 2, 4, 7, 10, 15, 25],
        random_state=int(args.seed),
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(x_train.to_numpy(np.float32), y_train, group=group_train)
    train["stage35_ranker_score"] = model.predict(x_train.to_numpy(np.float32))
    test["stage35_ranker_score"] = model.predict(x_test.to_numpy(np.float32))
    train_ranked = rank_by_score(train, "stage35_ranker_score", "stage35_ranker_rank")
    test_ranked = rank_by_score(test, "stage35_ranker_score", "stage35_ranker_rank")

    alpha, train_blend_metrics = blend_search(train_ranked, "stage35_ranker_score", "route_total_score", args.blend_steps)
    train_ranked["stage35_blend_score"] = (1 - alpha) * group_z(train_ranked, "route_total_score") + alpha * group_z(train_ranked, "stage35_ranker_score")
    test_ranked["stage35_blend_score"] = (1 - alpha) * group_z(test_ranked, "route_total_score") + alpha * group_z(test_ranked, "stage35_ranker_score")
    train_ranked = rank_by_score(train_ranked, "stage35_blend_score", "stage35_blend_rank")
    test_ranked = rank_by_score(test_ranked, "stage35_blend_score", "stage35_blend_rank")

    train_ranked.to_csv(run_dir / "val_route_candidates_stage35_reranked_v2.csv", index=False)
    test_ranked.to_csv(run_dir / "test_route_candidates_stage35_reranked_v2.csv", index=False)
    artifact = {"model": model, "feature_cols": feature_cols, "blend_alpha": alpha, "config": vars(args)}
    joblib.dump(artifact, run_dir / "stage35_route_reranker_v2.joblib")
    summary = {
        "config": vars(args),
        "n_features": len(feature_cols),
        "train_relevance_counts": {str(k): int(v) for k, v in pd.Series(y_train).value_counts().sort_index().items()},
        "test_relevance_counts": {str(k): int(v) for k, v in pd.Series(y_test).value_counts().sort_index().items()},
        "baseline_test_raw": route_metrics(test, "route_rank_raw"),
        "ranker_test": route_metrics(test_ranked, "stage35_ranker_rank"),
        "blend_alpha": alpha,
        "blend_train_metrics": train_blend_metrics,
        "blend_test": route_metrics(test_ranked, "stage35_blend_rank"),
        "artifacts": {
            "model": str(run_dir / "stage35_route_reranker_v2.joblib"),
            "test_reranked": str(run_dir / "test_route_candidates_stage35_reranked_v2.csv"),
            "metrics": str(run_dir / "metrics.json"),
        },
    }
    write_json(run_dir / "metrics.json", summary)
    write_json(run_dir / "feature_cols.json", feature_cols)
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
