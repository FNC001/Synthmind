#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score


FAMILY_PATTERNS = {
    "carbonate": re.compile(r"CO3", re.I),
    "nitrate": re.compile(r"NO3", re.I),
    "oxide": re.compile(r"O(?![a-z])|O2|O3", re.I),
    "hydroxide": re.compile(r"OH", re.I),
    "sulfate_sulfide": re.compile(r"SO4|S(?![a-z])", re.I),
    "phosphate": re.compile(r"PO4", re.I),
    "halide": re.compile(r"Cl|Br|I|F"),
    "acetate": re.compile(r"CH3COO|C2H3O2", re.I),
    "organic": re.compile(r"C[0-9]*H[0-9]*", re.I),
    "hydrate": re.compile(r"H2O|·", re.I),
}


BASE_FEATURES = [
    "rank",
    "score",
    "prob_score",
    "coverage",
    "extra",
    "predicted_size",
    "candidate_size",
]


def parse_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    s = str(value).strip()
    if not s:
        return []
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(x) for x in obj]
    except Exception:
        pass
    return [x.strip() for x in s.split(",") if x.strip()]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["size_abs_delta"] = (out["candidate_size"] - out["predicted_size"]).abs()
    out["score_per_size"] = out["score"] / out["candidate_size"].clip(lower=1)
    out["prob_per_size"] = out["prob_score"] / out["candidate_size"].clip(lower=1)
    out["rank_inv"] = 1.0 / out["rank"].clip(lower=1)
    if "precursor_rank" in out.columns:
        out["precursor_rank_inv"] = 1.0 / out["precursor_rank"].clip(lower=1)
    if "condition_rank" in out.columns:
        out["condition_rank_inv"] = 1.0 / out["condition_rank"].clip(lower=1)
    if "condition_source" in out.columns:
        out["condition_is_model_point"] = (out["condition_source"].astype(str) == "model_point").astype(int)
        out["condition_is_retrieval"] = (out["condition_source"].astype(str) == "retrieval_template").astype(int)

    lists = out["pred_precursors"].apply(parse_list)
    for name, pat in FAMILY_PATTERNS.items():
        out[f"family_{name}_count"] = lists.apply(lambda xs, p=pat: sum(1 for x in xs if p.search(x)))
        out[f"family_{name}_any"] = (out[f"family_{name}_count"] > 0).astype(int)
    out["family_n_unique_types"] = 0
    family_any_cols = [f"family_{name}_any" for name in FAMILY_PATTERNS]
    if family_any_cols:
        out["family_n_unique_types"] = out[family_any_cols].sum(axis=1)
    return out


def make_target(df: pd.DataFrame, mode: str) -> np.ndarray:
    relaxed = df["relaxed_route_success"].astype(float).to_numpy()
    strict = df["strict_route_success"].astype(float).to_numpy()
    exact = df["exact"].astype(float).to_numpy()
    f1 = df["f1"].astype(float).to_numpy()
    jaccard = df["jaccard"].astype(float).to_numpy()
    cond_relaxed = df["condition_200c_48h_atm"].astype(float).to_numpy()
    cond_strict = df["condition_100c_24h_atm"].astype(float).to_numpy()

    if mode == "strict":
        return strict.astype(np.float32)
    if mode == "relaxed":
        return relaxed.astype(np.float32)
    if mode == "route_dense_strict":
        return (
            8.0 * strict
            + 4.0 * relaxed
            + 1.0 * exact
            + 0.5 * jaccard
            + 1.0 * cond_strict
            + 0.3 * cond_relaxed
        ).astype(np.float32)
    if mode == "precursor_condition":
        return (2.0 * exact + f1 + 0.3 * cond_relaxed).astype(np.float32)

    # Smooth default target: prioritize actual route success, but keep dense signal for ranking.
    return (
        5.0 * strict
        + 3.0 * relaxed
        + 1.2 * exact
        + 0.8 * jaccard
        + 0.6 * cond_relaxed
        + 0.3 * cond_strict
    ).astype(np.float32)


def route_metrics(df: pd.DataFrame, rank_col: str) -> Dict[str, float]:
    ordered = df.sort_values(["sample_index", rank_col], ascending=[True, False])
    top1 = ordered.groupby("sample_index", sort=False).head(1)
    top10 = ordered.groupby("sample_index", sort=False).head(10)
    grouped = top10.groupby("sample_index", sort=False)
    all_grouped = df.groupby("sample_index", sort=False)
    return {
        "top1_precursor_exact": float(top1["exact"].mean()),
        "top1_precursor_mean_f1": float(top1["f1"].mean()),
        "top1_precursor_mean_jaccard": float(top1["jaccard"].mean()),
        "top1_strict_route_success": float(top1["strict_route_success"].mean()),
        "top1_relaxed_route_success": float(top1["relaxed_route_success"].mean()),
        "top10_precursor_exact_recall": float(grouped["exact"].any().mean()),
        "top10_relaxed_route_oracle_success": float(grouped["relaxed_route_success"].any().mean()),
        "top10_strict_route_oracle_success": float(grouped["strict_route_success"].any().mean()),
        "pool_relaxed_route_oracle_success": float(all_grouped["relaxed_route_success"].any().mean()),
        "pool_strict_route_oracle_success": float(all_grouped["strict_route_success"].any().mean()),
        "n": int(top1.shape[0]),
    }


def group_zscore(df: pd.DataFrame, col: str) -> np.ndarray:
    grouped = df.groupby("sample_index", sort=False)[col]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0.0, np.nan)
    return ((df[col] - mean) / std).fillna(0.0).to_numpy(dtype=np.float32)


def add_blend_score(df: pd.DataFrame, model_col: str, alpha: float, out_col: str) -> None:
    base = group_zscore(df, "score")
    model = group_zscore(df, model_col)
    df[out_col] = (1.0 - alpha) * base + alpha * model


def metric_key(metrics: Dict[str, float]) -> tuple:
    return (
        metrics["top1_strict_route_success"],
        metrics["top1_relaxed_route_success"],
        metrics["top1_precursor_exact"],
        metrics["top1_precursor_mean_f1"],
    )


def choose_blend_alpha(train: pd.DataFrame, model_col: str, n_steps: int) -> tuple[float, Dict[str, float]]:
    n_steps = max(2, int(n_steps))
    best_alpha = 1.0
    best_metrics: Dict[str, float] | None = None
    for alpha in np.linspace(0.0, 1.0, n_steps):
        add_blend_score(train, model_col, float(alpha), "_blend_tmp")
        metrics = route_metrics(train, "_blend_tmp")
        if best_metrics is None or metric_key(metrics) > metric_key(best_metrics):
            best_alpha = float(alpha)
            best_metrics = metrics
    train.drop(columns=["_blend_tmp"], inplace=True, errors="ignore")
    return best_alpha, best_metrics or {}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train/apply a Stage35 top-10 route reranker.")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--n_estimators", type=int, default=500)
    ap.add_argument("--max_depth", type=int, default=8)
    ap.add_argument("--min_samples_leaf", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--target_mode",
        choices=["route_dense", "route_dense_strict", "precursor_condition", "strict", "relaxed"],
        default="route_dense",
        help="Supervision target used by the top-10 reranker.",
    )
    ap.add_argument("--blend_steps", type=int, default=41, help="Grid size for score/stage35 blend tuning on train_csv.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    train = add_features(pd.read_csv(args.train_csv))
    test = add_features(pd.read_csv(args.test_csv))

    feature_cols = [c for c in BASE_FEATURES if c in train.columns] + [
        "size_abs_delta",
        "score_per_size",
        "prob_per_size",
        "rank_inv",
        "family_n_unique_types",
    ]
    for optional in [
        "precursor_rank",
        "precursor_rank_inv",
        "condition_rank",
        "condition_rank_inv",
        "condition_score",
        "condition_is_model_point",
        "condition_is_retrieval",
        "route_rank",
        "route_score",
    ]:
        if optional in train.columns:
            feature_cols.append(optional)
    for name in FAMILY_PATTERNS:
        feature_cols.extend([f"family_{name}_count", f"family_{name}_any"])

    x_train = train[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    y_train = make_target(train, args.target_mode)
    x_test = test[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    y_test = make_target(test, args.target_mode)

    model = ExtraTreesRegressor(
        n_estimators=int(args.n_estimators),
        max_depth=int(args.max_depth) if args.max_depth > 0 else None,
        min_samples_leaf=int(args.min_samples_leaf),
        random_state=int(args.seed),
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    train["stage35_score"] = model.predict(x_train)
    test["stage35_score"] = model.predict(x_test)
    blend_alpha, blend_train_metrics = choose_blend_alpha(train, "stage35_score", args.blend_steps)
    add_blend_score(train, "stage35_score", blend_alpha, "stage35_blend_score")
    add_blend_score(test, "stage35_score", blend_alpha, "stage35_blend_score")
    train["stage35_rank"] = train.groupby("sample_index")["stage35_score"].rank(method="first", ascending=False).astype(int)
    test["stage35_rank"] = test.groupby("sample_index")["stage35_score"].rank(method="first", ascending=False).astype(int)
    train["stage35_blend_rank"] = train.groupby("sample_index")["stage35_blend_score"].rank(method="first", ascending=False).astype(int)
    test["stage35_blend_rank"] = test.groupby("sample_index")["stage35_blend_score"].rank(method="first", ascending=False).astype(int)

    joblib.dump(
        {"model": model, "feature_cols": feature_cols, "blend_alpha": blend_alpha, "config": vars(args)},
        run_dir / "stage35_top10_reranker.joblib",
    )
    write_json(run_dir / "feature_cols.json", feature_cols)
    train.to_csv(run_dir / "val_candidates_stage35_reranked.csv", index=False)
    test.to_csv(run_dir / "test_candidates_stage35_reranked.csv", index=False)

    summary = {
        "model": "stage35_top10_extratrees_regressor",
        "config": vars(args),
        "data": {
            "n_train_candidates": int(len(train)),
            "n_test_candidates": int(len(test)),
            "n_features": int(len(feature_cols)),
        },
        "regression": {
            "train_mae": float(mean_absolute_error(y_train, train["stage35_score"])),
            "test_mae": float(mean_absolute_error(y_test, test["stage35_score"])),
            "train_r2": float(r2_score(y_train, train["stage35_score"])),
            "test_r2": float(r2_score(y_test, test["stage35_score"])),
        },
        "baseline_test_metrics_original_rank": route_metrics(test, "score"),
        "stage35_test_metrics": route_metrics(test, "stage35_score"),
        "stage35_blend": {
            "alpha": blend_alpha,
            "train_metrics": blend_train_metrics,
            "test_metrics": route_metrics(test, "stage35_blend_score"),
        },
        "artifacts": {
            "model": str(run_dir / "stage35_top10_reranker.joblib"),
            "test_reranked_csv": str(run_dir / "test_candidates_stage35_reranked.csv"),
            "metrics": str(run_dir / "metrics.json"),
        },
    }
    write_json(run_dir / "metrics.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
