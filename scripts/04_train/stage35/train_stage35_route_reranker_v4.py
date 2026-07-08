#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


CORE_METHODS = {"solid_state", "solution", "melt_arc"}
STRICT_ATM_UNKNOWN = {"<UNK_OR_MISSING>", "", "nan", "none", "null"}
FORBIDDEN = {
    "precursor_exact_if_eval", "precursor_jaccard_if_eval", "precursor_f1_if_eval",
    "strict_condition_hit_if_eval", "relaxed_condition_hit_if_eval",
    "strict_comparable_condition_strict_hit_if_eval", "strict_comparable_condition_relaxed_hit_if_eval",
    "strict_route_hit_if_eval", "relaxed_route_hit_if_eval", "usable_relaxed_route_hit_if_eval",
    "strict_comparable_strict_route_hit_if_eval", "strict_comparable_relaxed_route_hit_if_eval",
    "strict_comparable_usable_relaxed_route_hit_if_eval", "exact", "f1", "jaccard",
    "precision", "recall", "temp_error", "time_error", "atmosphere_correct",
}
NUMERIC = [
    "precursor_rank", "condition_rank", "route_rank_raw", "precursor_score", "condition_score",
    "precursor_score_norm", "condition_score_norm", "precursor_confidence", "condition_confidence",
    "precursor_condition_compatibility_score", "reaction_method_prior_score", "element_coverage",
    "missing_element_count", "extra_element_count", "candidate_size", "contains_open_generated_precursor",
    "contains_repair_precursor", "oof_exact_probability", "oof_f1_prediction", "method_template_score",
    "family_score", "cooccurrence_score", "mlp_score", "retrieval_score", "set_size_score",
    "temperature_c", "temperature_low_c", "temperature_high_c", "time_h", "time_low_h", "time_high_h",
    "temperature_point_score", "temperature_bin_score", "time_point_score", "time_bin_score",
    "atmosphere_probability", "solvent_probability", "atmosphere_known_probability",
    "solvent_known_probability", "atmosphere_class_probability", "solvent_class_probability",
    "condition_prior_score", "precursor_confidence_score", "open_generated_penalty", "repair_penalty",
    "total_score_raw", "method_expert_score", "template_score", "multimodal_score",
    "multimodal_template_score", "missing_aware_score", "strict_comparable_score",
    "precursor_uncertainty_score", "condition_calibrated_score_v4", "condition_rank_calibrated_v4",
    "condition_rank_missing_aware_v4", "condition_rank_strict_comparable_v4", "route_total_score_raw",
]
CAT = ["reaction_method", "candidate_source", "condition_source", "atmosphere", "solvent"]


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


def read_filtered(path: Path, rank_cutoff: int | None = None, keep_hits: bool = False, chunksize: int = 250_000) -> pd.DataFrame:
    chunks: List[pd.DataFrame] = []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        if rank_cutoff is not None and "route_rank_raw" in chunk.columns:
            mask = pd.to_numeric(chunk["route_rank_raw"], errors="coerce").fillna(10**9) <= rank_cutoff
            if keep_hits:
                for col in [
                    "strict_route_hit_if_eval", "relaxed_route_hit_if_eval",
                    "strict_comparable_strict_route_hit_if_eval", "strict_comparable_relaxed_route_hit_if_eval",
                ]:
                    if col in chunk.columns:
                        mask = mask | (pd.to_numeric(chunk[col], errors="coerce").fillna(0) > 0)
            chunk = chunk[mask].copy()
        chunks.append(chunk)
    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, ignore_index=True)
    out = out.sort_values(["sample_id", "route_rank_raw"], ascending=[True, True], kind="mergesort").reset_index(drop=True)
    return out


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["precursor_rank_score"] = 1.0 / pd.to_numeric(out.get("precursor_rank", 999), errors="coerce").clip(lower=1).fillna(999)
    out["condition_rank_score"] = 1.0 / pd.to_numeric(out.get("condition_rank", 999), errors="coerce").clip(lower=1).fillna(999)
    out["route_rank_score"] = 1.0 / pd.to_numeric(out.get("route_rank_raw", 999), errors="coerce").clip(lower=1).fillna(999)
    out["rank_product_score"] = out["precursor_rank_score"] * out["condition_rank_score"]
    out["temperature_interval_width"] = pd.to_numeric(out.get("temperature_high_c", 0), errors="coerce").fillna(0) - pd.to_numeric(out.get("temperature_low_c", 0), errors="coerce").fillna(0)
    out["time_interval_width"] = pd.to_numeric(out.get("time_high_h", 0), errors="coerce").fillna(0) - pd.to_numeric(out.get("time_low_h", 0), errors="coerce").fillna(0)
    out["condition_is_point"] = out.get("condition_source", "").astype(str).eq("point_model").astype(int)
    out["condition_is_quantile"] = out.get("condition_source", "").astype(str).str.contains("quantile", regex=False).astype(int)
    out["condition_is_bin"] = out.get("condition_source", "").astype(str).eq("bin_center").astype(int)
    out["condition_is_template"] = out.get("condition_source", "").astype(str).str.contains("template", regex=False).astype(int)
    out["is_core_method"] = out.get("reaction_method", "").astype(str).isin(CORE_METHODS).astype(int)
    out["open_or_repair_count"] = (
        pd.to_numeric(out.get("contains_open_generated_precursor", 0), errors="coerce").fillna(0)
        + pd.to_numeric(out.get("contains_repair_precursor", 0), errors="coerce").fillna(0)
    )
    out["condition_missing_awareness_gap"] = pd.to_numeric(out.get("missing_aware_score", 0), errors="coerce").fillna(0) - pd.to_numeric(out.get("strict_comparable_score", 0), errors="coerce").fillna(0)
    out["precursor_condition_score_product"] = pd.to_numeric(out.get("precursor_score_norm", 0), errors="coerce").fillna(0) * pd.to_numeric(out.get("condition_score_norm", 0), errors="coerce").fillna(0)
    return out


def fit_category_levels(df: pd.DataFrame) -> Dict[str, List[str]]:
    levels: Dict[str, List[str]] = {}
    for col in CAT:
        if col in df.columns:
            vals = df[col].fillna("missing").astype(str)
            counts = vals.value_counts()
            keep = counts[counts >= 20].index.tolist()
            levels[col] = keep[:80] + ["__OTHER__"]
        else:
            levels[col] = ["__OTHER__"]
    return levels


def make_features(df: pd.DataFrame, cat_levels: Mapping[str, List[str]] | None = None) -> tuple[pd.DataFrame, Dict[str, List[str]], List[str]]:
    work = add_derived(df)
    extra_nums = [
        "precursor_rank_score", "condition_rank_score", "route_rank_score", "rank_product_score",
        "temperature_interval_width", "time_interval_width", "condition_is_point", "condition_is_quantile",
        "condition_is_bin", "condition_is_template", "is_core_method", "open_or_repair_count",
        "condition_missing_awareness_gap", "precursor_condition_score_product",
    ]
    nums = [c for c in NUMERIC + extra_nums if c not in FORBIDDEN]
    x_parts: List[pd.DataFrame] = []
    num_df = pd.DataFrame(index=work.index)
    for col in nums:
        num_df[col] = pd.to_numeric(work[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0) if col in work.columns else 0.0
    x_parts.append(num_df.astype(np.float32))
    if cat_levels is None:
        cat_levels = fit_category_levels(work)
    for col, levels in cat_levels.items():
        vals = work[col].fillna("missing").astype(str) if col in work.columns else pd.Series("missing", index=work.index)
        allowed = set(levels)
        vals = vals.where(vals.isin(allowed), "__OTHER__")
        dummies = pd.get_dummies(vals, prefix=col, dtype=np.float32)
        expected = [f"{col}_{v}" for v in levels]
        for c in expected:
            if c not in dummies.columns:
                dummies[c] = np.float32(0.0)
        x_parts.append(dummies[expected])
    x = pd.concat(x_parts, axis=1)
    return x, dict(cat_levels), x.columns.tolist()


def strict_protocol_flags(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    known = pd.to_numeric(df.get("atmosphere_known_mask", 0), errors="coerce").fillna(0).astype(int) == 1
    pred_atm = df.get("atmosphere", "").astype(str).str.lower()
    true_atm = df.get("true_atmosphere", "").astype(str).str.lower()
    true_known = ~true_atm.isin(STRICT_ATM_UNKNOWN)
    atm_ok = known & true_known & pred_atm.eq(true_atm)
    temp_err = pd.to_numeric(df.get("temp_error", np.inf), errors="coerce").fillna(np.inf)
    time_err = pd.to_numeric(df.get("time_error", np.inf), errors="coerce").fillna(np.inf)
    strict = ((temp_err <= 100) & (time_err <= 24) & atm_ok).to_numpy()
    relaxed = ((temp_err <= 200) & (time_err <= 48) & atm_ok).to_numpy()
    return strict, relaxed


def relevance(df: pd.DataFrame, strict_protocol: bool = False) -> np.ndarray:
    if strict_protocol:
        cond_strict, cond_relaxed = strict_protocol_flags(df)
    else:
        cond_strict = pd.to_numeric(df.get("strict_condition_hit_if_eval", 0), errors="coerce").fillna(0).to_numpy() > 0.5
        cond_relaxed = pd.to_numeric(df.get("relaxed_condition_hit_if_eval", 0), errors="coerce").fillna(0).to_numpy() > 0.5
    exact = df["precursor_exact_if_eval"].astype(bool).to_numpy()
    jac = pd.to_numeric(df.get("precursor_jaccard_if_eval", 0), errors="coerce").fillna(0.0).to_numpy(float)
    rel = np.zeros(len(df), dtype=np.int32)
    rel[exact | cond_relaxed] = 1
    rel[(jac >= 0.5) & cond_relaxed] = 2
    rel[exact & cond_relaxed] = 3
    rel[exact & cond_strict] = 4
    return rel


def rank(df: pd.DataFrame, score_col: str, rank_col: str) -> pd.DataFrame:
    out = df.sort_values(["sample_id", score_col, "route_total_score_raw"], ascending=[True, False, False], kind="mergesort").copy()
    out[rank_col] = out.groupby("sample_id", sort=False).cumcount() + 1
    return out


def route_flags(df: pd.DataFrame, protocol: str) -> pd.DataFrame:
    out = df.copy()
    if protocol == "strict_comparable":
        cond_strict, cond_relaxed = strict_protocol_flags(out)
    else:
        cond_strict = pd.to_numeric(out.get("strict_condition_hit_if_eval", 0), errors="coerce").fillna(0).to_numpy() > 0.5
        cond_relaxed = pd.to_numeric(out.get("relaxed_condition_hit_if_eval", 0), errors="coerce").fillna(0).to_numpy() > 0.5
    exact = out["precursor_exact_if_eval"].astype(bool).to_numpy()
    jac = pd.to_numeric(out.get("precursor_jaccard_if_eval", 0), errors="coerce").fillna(0.0).to_numpy(float)
    out["_strict_route"] = (exact & cond_strict).astype(int)
    out["_relaxed_route"] = (exact & cond_relaxed).astype(int)
    out["_usable_relaxed_route"] = ((jac >= 0.5) & cond_relaxed).astype(int)
    return out


def metrics(df: pd.DataFrame, rank_col: str, protocol: str = "missing_aware") -> Dict[str, float]:
    d = route_flags(df, protocol)
    out: Dict[str, float] = {"n_samples": int(d["sample_id"].nunique()), "n_candidates": int(len(d))}
    for k in [1, 3, 5, 10, 20, 50, 100, 200, 400]:
        sub = d[d[rank_col] <= k]
        g = sub.groupby("sample_id", sort=False)
        if len(g):
            out[f"top{k}_strict_route"] = float(g["_strict_route"].max().mean())
            out[f"top{k}_relaxed_route"] = float(g["_relaxed_route"].max().mean())
            out[f"top{k}_usable_relaxed_route"] = float(g["_usable_relaxed_route"].max().mean())
    core = d[d["reaction_method"].isin(CORE_METHODS)]
    cg = core[core[rank_col] <= 10].groupby("sample_id", sort=False)
    out["core_top10_relaxed_route"] = float(cg["_relaxed_route"].max().mean()) if len(cg) else 0.0
    return out


def z_by_sample(df: pd.DataFrame, col: str) -> pd.Series:
    vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    mean = vals.groupby(df["sample_id"], sort=False).transform("mean")
    std = vals.groupby(df["sample_id"], sort=False).transform("std").replace(0, np.nan)
    return ((vals - mean) / std).fillna(0.0)


def blend(df: pd.DataFrame, raw_weight: float, ranker_weight: float, binary_weight: float) -> pd.Series:
    return (
        raw_weight * z_by_sample(df, "route_total_score_raw")
        + ranker_weight * z_by_sample(df, "lambdarank_score")
        + binary_weight * z_by_sample(df, "binary_relaxed_score")
    )


def choose_blend(val: pd.DataFrame) -> tuple[Dict[str, float], Dict[str, float]]:
    best_weights = {"raw": 1.0, "ranker": 0.0, "binary": 0.0}
    best_metrics: Dict[str, float] | None = None
    grid = np.linspace(0.0, 1.0, 11)
    for wr in grid:
        for wk in grid:
            wb = 1.0 - wr - wk
            if wb < -1e-9:
                continue
            tmp = val.copy()
            tmp["_blend"] = blend(tmp, float(wr), float(wk), float(wb))
            tmp = rank(tmp, "_blend", "_blend_rank")
            mm = metrics(tmp, "_blend_rank", "missing_aware")
            key = (mm.get("top1_relaxed_route", 0), mm.get("top10_relaxed_route", 0), mm.get("top1_strict_route", 0))
            if best_metrics is None or key > (
                best_metrics.get("top1_relaxed_route", 0),
                best_metrics.get("top10_relaxed_route", 0),
                best_metrics.get("top1_strict_route", 0),
            ):
                best_metrics = mm
                best_weights = {"raw": float(wr), "ranker": float(wk), "binary": float(max(wb, 0.0))}
    return best_weights, best_metrics or {}


def predict_split(df: pd.DataFrame, model_pack: Mapping[str, Any], cat_levels: Mapping[str, List[str]], feature_cols: List[str]) -> pd.DataFrame:
    x, _, cols = make_features(df, cat_levels)
    for c in feature_cols:
        if c not in x.columns:
            x[c] = np.float32(0.0)
    x = x[feature_cols]
    arr = x.to_numpy(np.float32)
    out = df.copy()
    out["lambdarank_score"] = model_pack["ranker"].predict(arr)
    out["binary_relaxed_score"] = model_pack["binary_classifier"].predict_proba(arr)[:, 1]
    out = rank(out, "route_total_score_raw", "raw_rank")
    out = rank(out, "lambdarank_score", "lambdarank_rank")
    out = rank(out, "binary_relaxed_score", "binary_rank")
    return out


def evaluate_and_write(split: str, df: pd.DataFrame, run: Path, weights: Mapping[str, float], save_topk: int) -> Dict[str, Any]:
    out = df.copy()
    out["blend_score"] = blend(out, float(weights["raw"]), float(weights["ranker"]), float(weights["binary"]))
    out = rank(out, "blend_score", "blend_rank")
    summary = {
        "raw_missing_aware": metrics(out, "raw_rank", "missing_aware"),
        "lambdarank_missing_aware": metrics(out, "lambdarank_rank", "missing_aware"),
        "binary_missing_aware": metrics(out, "binary_rank", "missing_aware"),
        "blend_missing_aware": metrics(out, "blend_rank", "missing_aware"),
        "raw_strict_comparable": metrics(out, "raw_rank", "strict_comparable"),
        "lambdarank_strict_comparable": metrics(out, "lambdarank_rank", "strict_comparable"),
        "binary_strict_comparable": metrics(out, "binary_rank", "strict_comparable"),
        "blend_strict_comparable": metrics(out, "blend_rank", "strict_comparable"),
    }
    write_json(run / f"{split}_metrics.json", summary)
    if save_topk > 0:
        keep = out[out["blend_rank"] <= save_topk].copy()
        keep.to_csv(run / f"{split}_top{save_topk}_route_candidates_reranked_v4.csv", index=False)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Stage35 route reranker v4 on OOF-aligned route candidates with hard negatives.")
    ap.add_argument("--candidate_dir", default="outputs/evaluation/stage35_route_candidates_v4_20260612")
    ap.add_argument("--run_dir", default="runs/stage35/route_reranker_v4_20260612")
    ap.add_argument("--train_rank_cutoff", type=int, default=80, help="Keep raw top-N route candidates per train sample before adding hit rows.")
    ap.add_argument("--keep_train_hits", action="store_true", default=True, help="Also keep all train rows with strict/relaxed route hits as hard positives.")
    ap.add_argument("--val_train_rank_cutoff", type=int, default=200, help="Validation cutoff used only for binary early stopping.")
    ap.add_argument("--save_topk_csv", type=int, default=10, help="Save only top-K reranked rows per split; 0 disables CSV output.")
    ap.add_argument("--seed", type=int, default=20260612)
    args = ap.parse_args()

    cand = Path(args.candidate_dir)
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)

    train_df = read_filtered(cand / "train_route_candidates.csv", rank_cutoff=args.train_rank_cutoff, keep_hits=bool(args.keep_train_hits))
    val_fit = read_filtered(cand / "val_route_candidates.csv", rank_cutoff=args.val_train_rank_cutoff, keep_hits=True)
    cat_levels = fit_category_levels(train_df)
    x_train, cat_levels, feature_cols = make_features(train_df, cat_levels)
    x_val_fit, _, _ = make_features(val_fit, cat_levels)
    for c in feature_cols:
        if c not in x_val_fit.columns:
            x_val_fit[c] = np.float32(0.0)
    x_val_fit = x_val_fit[feature_cols]
    y_rank = relevance(train_df, strict_protocol=False)
    y_bin = (y_rank >= 3).astype(int)
    y_val_bin = (relevance(val_fit, strict_protocol=False) >= 3).astype(int)
    group = train_df.groupby("sample_id", sort=False).size().astype(int).tolist()

    ranker = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", n_estimators=220, learning_rate=0.04,
        num_leaves=63, min_child_samples=30, subsample=0.85, colsample_bytree=0.85,
        reg_lambda=1.2, label_gain=[0, 1, 2, 4, 7], random_state=args.seed, n_jobs=8, verbose=-1,
    )
    ranker.fit(x_train.to_numpy(np.float32), y_rank, group=group)
    clf = lgb.LGBMClassifier(
        objective="binary", n_estimators=260, learning_rate=0.04, num_leaves=63,
        min_child_samples=40, subsample=0.85, colsample_bytree=0.85, reg_lambda=1.2,
        random_state=args.seed + 1, n_jobs=8, verbose=-1,
    )
    clf.fit(
        x_train.to_numpy(np.float32),
        y_bin,
        eval_set=[(x_val_fit.to_numpy(np.float32), y_val_bin)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    model_pack = {
        "ranker": ranker,
        "binary_classifier": clf,
        "feature_cols": feature_cols,
        "cat_levels": cat_levels,
        "config": vars(args),
        "train_rows_used": int(len(train_df)),
        "train_samples_used": int(train_df["sample_id"].nunique()),
    }
    joblib.dump(model_pack, run / "stage35_route_reranker_v4.joblib")
    write_json(run / "feature_schema.json", {"feature_cols": feature_cols, "cat_levels": cat_levels, "forbidden_eval_features": sorted(FORBIDDEN)})

    val_full = read_filtered(cand / "val_route_candidates.csv")
    val_pred = predict_split(val_full, model_pack, cat_levels, feature_cols)
    weights, val_blend_metrics = choose_blend(val_pred)
    summaries: Dict[str, Any] = {
        "train_rows_used": int(len(train_df)),
        "train_samples_used": int(train_df["sample_id"].nunique()),
        "blend_weights_selected_on_val": weights,
        "val_blend_selection_metrics": val_blend_metrics,
        "splits": {},
    }
    summaries["splits"]["val"] = evaluate_and_write("val", val_pred, run, weights, int(args.save_topk_csv))
    del val_full, val_pred, x_val_fit, val_fit

    test_full = read_filtered(cand / "test_route_candidates.csv")
    test_pred = predict_split(test_full, model_pack, cat_levels, feature_cols)
    summaries["splits"]["test"] = evaluate_and_write("test", test_pred, run, weights, int(args.save_topk_csv))
    del test_full, test_pred

    train_eval = predict_split(train_df, model_pack, cat_levels, feature_cols)
    summaries["splits"]["train_sampled"] = evaluate_and_write("train_sampled", train_eval, run, weights, 0)

    write_json(run / "metrics.json", {"config": vars(args), **summaries})
    report = [
        "# Stage35 Route Reranker v4 Training Report",
        "",
        json.dumps(to_builtin({"blend_weights": weights, "test": summaries["splits"]["test"], "train_rows_used": len(train_df)}), ensure_ascii=False, indent=2),
    ]
    (run / "reranker_v4_training_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(to_builtin({"blend_weights": weights, "test": summaries["splits"]["test"], "train_rows_used": len(train_df)}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
