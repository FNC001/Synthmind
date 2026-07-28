#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


TARGET_COLS = {
    "exact",
    "f1",
    "jaccard",
    "precision",
    "recall",
    "any_overlap",
}

META_COLS = {
    "sample_index",
    "rank",
    "id",
    "material_id",
    "formula",
    "reaction_method",
    "true_precursors",
    "pred_precursors",
}


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


def load_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "sample_index" not in df.columns:
        raise ValueError(f"{path} missing sample_index")
    if "exact" not in df.columns or "jaccard" not in df.columns or "f1" not in df.columns:
        raise ValueError(f"{path} missing exact/jaccard/f1 labels")
    df = df.sort_values(["sample_index", "rank"], kind="mergesort").reset_index(drop=True)
    return df


def add_source_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "sources" in out.columns:
        all_sources = sorted({s for x in out["sources"].fillna("").astype(str) for s in x.split("+") if s})
        for src in all_sources:
            out[f"src_{src}"] = out["sources"].fillna("").astype(str).str.split("+").apply(lambda xs, s=src: int(s in xs))
    if "reaction_method" in out.columns:
        # Keep a compact, deterministic one-hot. Unknown categories at eval time
        # simply get all-zero for missing train-time columns.
        dummies = pd.get_dummies(out["reaction_method"].fillna("other").astype(str), prefix="method", dtype=np.int8)
        out = pd.concat([out, dummies], axis=1)
    return out


def select_feature_columns(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> List[str]:
    cols = []
    blocked = TARGET_COLS | META_COLS | {"sources"}
    for col in train_df.columns:
        if col in blocked:
            continue
        if pd.api.types.is_numeric_dtype(train_df[col]):
            cols.append(col)
    # Ensure feature columns exist in all splits after one-hot expansion.
    for df in [val_df, test_df]:
        for col in cols:
            if col not in df.columns:
                df[col] = 0
    return cols


def make_targets(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    exact = df["exact"].astype(float).to_numpy()
    jaccard = df["jaccard"].astype(float).to_numpy()
    f1 = df["f1"].astype(float).to_numpy()
    extra = df["extra"].astype(float).to_numpy() if "extra" in df.columns else np.zeros(len(df))
    dense = 3.0 * exact + 1.5 * jaccard + 1.0 * f1 - 0.15 * extra
    relevance = np.floor(5.0 * jaccard + 4.0 * exact + 1.0 * f1).astype(int)
    relevance = np.clip(relevance, 0, 10)
    return {"dense": dense.astype(np.float32), "relevance": relevance.astype(np.int32)}


def group_sizes(df: pd.DataFrame) -> List[int]:
    return df.groupby("sample_index", sort=False).size().astype(int).tolist()


def score_df(df: pd.DataFrame, feature_cols: Sequence[str], ranker: Any, regressor: Any, blend: float) -> pd.DataFrame:
    out = df.copy()
    X = out[list(feature_cols)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rank_score = np.asarray(ranker.predict(X), dtype=np.float32)
    reg_score = np.asarray(regressor.predict(X), dtype=np.float32)
    out["ranker_score"] = rank_score
    out["regressor_score"] = reg_score
    out["rerank_score"] = float(blend) * rank_score + (1.0 - float(blend)) * reg_score
    out = out.sort_values(["sample_index", "rerank_score"], ascending=[True, False]).copy()
    out["rerank_rank"] = out.groupby("sample_index").cumcount() + 1
    return out


def eval_ranked(df: pd.DataFrame, ks: Sequence[int]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    top1 = df[df["rerank_rank"] == 1]
    metrics["top1_exact"] = float(top1["exact"].mean())
    metrics["top1_f1"] = float(top1["f1"].mean())
    metrics["top1_jaccard"] = float(top1["jaccard"].mean())
    for k in ks:
        sub = df[df["rerank_rank"] <= int(k)]
        grouped = sub.groupby("sample_index", sort=False)
        metrics[f"top{k}_exact"] = float(grouped["exact"].any().mean())
        metrics[f"top{k}_best_f1"] = float(grouped["f1"].max().mean())
        metrics[f"top{k}_best_jaccard"] = float(grouped["jaccard"].max().mean())
    pool_grouped = df.groupby("sample_index", sort=False)
    metrics["pool_exact_oracle"] = float(pool_grouped["exact"].any().mean())
    metrics["pool_best_f1"] = float(pool_grouped["f1"].max().mean())
    metrics["pool_best_jaccard"] = float(pool_grouped["jaccard"].max().mean())
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a Stage2 candidate-set LightGBM reranker.")
    ap.add_argument("--train_candidates", required=True)
    ap.add_argument("--val_candidates", required=True)
    ap.add_argument("--test_candidates", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--num_leaves", type=int, default=63)
    ap.add_argument("--learning_rate", type=float, default=0.04)
    ap.add_argument("--n_estimators", type=int, default=800)
    ap.add_argument("--min_child_samples", type=int, default=25)
    ap.add_argument("--subsample", type=float, default=0.85)
    ap.add_argument("--colsample_bytree", type=float, default=0.85)
    ap.add_argument("--blend", type=float, default=0.65)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    train_df = add_source_features(load_candidates(Path(args.train_candidates)))
    val_df = add_source_features(load_candidates(Path(args.val_candidates)))
    test_df = add_source_features(load_candidates(Path(args.test_candidates)))
    feature_cols = select_feature_columns(train_df, val_df, test_df)
    for df in [train_df, val_df, test_df]:
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_val = val_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    targets_train = make_targets(train_df)
    targets_val = make_targets(val_df)

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        num_leaves=int(args.num_leaves),
        min_child_samples=int(args.min_child_samples),
        subsample=float(args.subsample),
        colsample_bytree=float(args.colsample_bytree),
        random_state=int(args.seed),
        n_jobs=-1,
        verbose=-1,
    )
    ranker.fit(
        X_train,
        targets_train["relevance"],
        group=group_sizes(train_df),
        eval_set=[(X_val, targets_val["relevance"])],
        eval_group=[group_sizes(val_df)],
        eval_at=[1, 3, 5, 10],
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(50)],
    )

    regressor = lgb.LGBMRegressor(
        objective="regression",
        metric="l2",
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        num_leaves=int(args.num_leaves),
        min_child_samples=int(args.min_child_samples),
        subsample=float(args.subsample),
        colsample_bytree=float(args.colsample_bytree),
        random_state=int(args.seed) + 7,
        n_jobs=-1,
        verbose=-1,
    )
    regressor.fit(
        X_train,
        targets_train["dense"],
        eval_set=[(X_val, targets_val["dense"])],
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(50)],
    )

    ks = [3, 5, 10, 20, 50, 100]
    ranked_val = score_df(val_df, feature_cols, ranker, regressor, float(args.blend))
    ranked_test = score_df(test_df, feature_cols, ranker, regressor, float(args.blend))
    ranked_val.to_csv(run_dir / "val_reranked_candidates.csv", index=False)
    ranked_test.to_csv(run_dir / "test_reranked_candidates.csv", index=False)

    summary = {
        "config": vars(args),
        "data": {
            "n_train_candidates": int(len(train_df)),
            "n_val_candidates": int(len(val_df)),
            "n_test_candidates": int(len(test_df)),
            "n_features": int(len(feature_cols)),
            "feature_cols": list(feature_cols),
        },
        "metrics": {
            "val": eval_ranked(ranked_val, ks),
            "test": eval_ranked(ranked_test, ks),
        },
        "artifacts": {
            "model": str((run_dir / "stage2_candidate_set_lgbm_reranker.joblib").resolve()),
            "summary_json": str((run_dir / "metrics.json").resolve()),
            "test_reranked_candidates": str((run_dir / "test_reranked_candidates.csv").resolve()),
        },
    }
    joblib.dump({"ranker": ranker, "regressor": regressor, "feature_cols": feature_cols, "config": vars(args)}, run_dir / "stage2_candidate_set_lgbm_reranker.joblib")
    write_json(run_dir / "metrics.json", summary)
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
