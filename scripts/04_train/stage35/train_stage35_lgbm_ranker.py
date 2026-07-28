#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


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
    if "rank" not in out.columns and "precursor_rank" in out.columns:
        out["rank"] = out["precursor_rank"]
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
    out["family_n_unique_types"] = out[[f"family_{n}_any" for n in FAMILY_PATTERNS]].sum(axis=1)
    return out


def make_relevance(df: pd.DataFrame) -> np.ndarray:
    strict = df["strict_route_success"].astype(bool).to_numpy()
    relaxed = df["relaxed_route_success"].astype(bool).to_numpy()
    exact = df["exact"].astype(bool).to_numpy()
    j50 = df["jaccard"].astype(float).to_numpy() >= 0.5
    cond100 = df["condition_100c_24h_atm"].astype(bool).to_numpy()
    cond200 = df["condition_200c_48h_atm"].astype(bool).to_numpy()
    f1 = df["f1"].astype(float).to_numpy()
    rel = np.zeros(len(df), dtype=np.int32)
    rel[f1 >= 0.5] = np.maximum(rel[f1 >= 0.5], 1)
    rel[exact] = np.maximum(rel[exact], 2)
    rel[j50 & cond200] = np.maximum(rel[j50 & cond200], 3)
    rel[exact & cond100] = np.maximum(rel[exact & cond100], 4)
    rel[relaxed] = np.maximum(rel[relaxed], 5)
    rel[strict] = np.maximum(rel[strict], 6)
    return rel


def route_metrics(df: pd.DataFrame, score_col: str) -> Dict[str, float]:
    ordered = df.sort_values(["sample_index", score_col], ascending=[True, False]).copy()
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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def group_sizes(df: pd.DataFrame) -> List[int]:
    return df.groupby("sample_index", sort=False).size().astype(int).tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Stage35 LightGBM LambdaRank reranker.")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--n_estimators", type=int, default=1000)
    ap.add_argument("--learning_rate", type=float, default=0.035)
    ap.add_argument("--num_leaves", type=int, default=63)
    ap.add_argument("--min_child_samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    train = add_features(pd.read_csv(args.train_csv)).sort_values(["sample_index", "score"], ascending=[True, False]).copy()
    test = add_features(pd.read_csv(args.test_csv)).sort_values(["sample_index", "score"], ascending=[True, False]).copy()

    base = [
        "rank", "score", "prob_score", "coverage", "extra", "predicted_size", "candidate_size",
        "size_abs_delta", "score_per_size", "prob_per_size", "rank_inv", "family_n_unique_types",
    ]
    optional = [
        "precursor_rank", "precursor_rank_inv", "condition_rank", "condition_rank_inv",
        "condition_score", "condition_is_model_point", "condition_is_retrieval", "route_rank", "route_score",
    ]
    feature_cols = [c for c in base + optional if c in train.columns]
    for name in FAMILY_PATTERNS:
        feature_cols.extend([f"family_{name}_count", f"family_{name}_any"])
    feature_cols = [c for c in feature_cols if c in train.columns]

    x_train = train[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)
    x_test = test[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)
    y_train = make_relevance(train)
    y_test = make_relevance(test)
    train_group = group_sizes(train)

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        num_leaves=int(args.num_leaves),
        min_child_samples=int(args.min_child_samples),
        subsample=0.85,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=int(args.seed),
        n_jobs=-1,
        verbose=-1,
        label_gain=[0, 1, 3, 6, 10, 15, 25],
    )
    model.fit(x_train, y_train, group=train_group)
    train["stage35_lgbm_rank_score"] = model.predict(x_train)
    test["stage35_lgbm_rank_score"] = model.predict(x_test)
    train["stage35_lgbm_rank"] = train.groupby("sample_index")["stage35_lgbm_rank_score"].rank(method="first", ascending=False).astype(int)
    test["stage35_lgbm_rank"] = test.groupby("sample_index")["stage35_lgbm_rank_score"].rank(method="first", ascending=False).astype(int)

    joblib.dump({"model": model, "feature_cols": feature_cols, "config": vars(args)}, run_dir / "stage35_lgbm_ranker.joblib")
    train.to_csv(run_dir / "val_candidates_stage35_lgbm_reranked.csv", index=False)
    test.to_csv(run_dir / "test_candidates_stage35_lgbm_reranked.csv", index=False)
    summary = {
        "model": "stage35_lgbm_lambdarank",
        "config": vars(args),
        "data": {
            "n_train_candidates": int(len(train)),
            "n_test_candidates": int(len(test)),
            "n_features": int(len(feature_cols)),
            "train_relevance_counts": {str(k): int(v) for k, v in pd.Series(y_train).value_counts().sort_index().items()},
            "test_relevance_counts": {str(k): int(v) for k, v in pd.Series(y_test).value_counts().sort_index().items()},
        },
        "baseline_test_metrics_original_rank": route_metrics(test, "score"),
        "stage35_lgbm_test_metrics": route_metrics(test, "stage35_lgbm_rank_score"),
        "artifacts": {
            "model": str(run_dir / "stage35_lgbm_ranker.joblib"),
            "test_reranked_csv": str(run_dir / "test_candidates_stage35_lgbm_reranked.csv"),
            "metrics": str(run_dir / "metrics.json"),
        },
    }
    write_json(run_dir / "metrics.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
