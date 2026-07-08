#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


CORE_METHODS = {"solid_state", "solution", "melt_arc"}
FORBIDDEN = {
    "precursor_exact_if_eval", "precursor_jaccard_if_eval", "precursor_f1_if_eval",
    "strict_condition_hit_if_eval", "relaxed_condition_hit_if_eval",
    "strict_route_hit_if_eval", "relaxed_route_hit_if_eval", "usable_relaxed_route_hit_if_eval",
    "exact", "f1", "jaccard",
}
NUMERIC = [
    "precursor_rank", "condition_rank", "precursor_score", "condition_score", "precursor_score_norm",
    "condition_score_norm", "precursor_confidence", "condition_confidence", "precursor_condition_compatibility_score",
    "reaction_method_prior_score", "atmosphere_probability", "solvent_probability", "temperature_point_score",
    "temperature_bin_score", "time_point_score", "time_bin_score", "retrieval_score", "method_template_score",
    "multimodal_template_score", "open_generated_penalty", "repair_penalty", "contains_open_generated_precursor",
    "contains_repair_precursor", "route_total_score_raw", "temperature_low_c", "temperature_high_c", "time_low_h", "time_high_h",
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


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["precursor_rank_score"] = 1.0 / pd.to_numeric(out["precursor_rank"], errors="coerce").clip(lower=1).fillna(999)
    out["condition_rank_score"] = 1.0 / pd.to_numeric(out["condition_rank"], errors="coerce").clip(lower=1).fillna(999)
    out["rank_product_score"] = out["precursor_rank_score"] * out["condition_rank_score"]
    out["temperature_interval_width"] = pd.to_numeric(out["temperature_high_c"], errors="coerce").fillna(0) - pd.to_numeric(out["temperature_low_c"], errors="coerce").fillna(0)
    out["time_interval_width"] = pd.to_numeric(out["time_high_h"], errors="coerce").fillna(0) - pd.to_numeric(out["time_low_h"], errors="coerce").fillna(0)
    out["condition_is_point"] = (out["condition_source"].astype(str) == "point_model").astype(int)
    out["condition_is_quantile"] = out["condition_source"].astype(str).str.contains("quantile", regex=False).astype(int)
    out["condition_is_bin"] = (out["condition_source"].astype(str) == "bin_center").astype(int)
    out["condition_is_template"] = out["condition_source"].astype(str).str.contains("template", regex=False).astype(int)
    out["is_core_method"] = out["reaction_method"].isin(CORE_METHODS).astype(int)
    return out


def relevance(df: pd.DataFrame, strict_protocol: bool = False) -> np.ndarray:
    if strict_protocol:
        known = pd.to_numeric(df.get("atmosphere_known_mask", 0), errors="coerce").fillna(0).astype(int) == 1
        atm_ok = known & (df["atmosphere"].astype(str) == df["true_atmosphere"].astype(str))
        temp_err = pd.to_numeric(df["temp_error"], errors="coerce").fillna(np.inf)
        time_err = pd.to_numeric(df["time_error"], errors="coerce").fillna(np.inf)
        cond_strict = ((temp_err <= 100) & (time_err <= 24) & atm_ok).to_numpy()
        cond_relaxed = ((temp_err <= 200) & (time_err <= 48) & atm_ok).to_numpy()
    else:
        cond_strict = pd.to_numeric(df["strict_condition_hit_if_eval"], errors="coerce").fillna(0).to_numpy() > 0.5
        cond_relaxed = pd.to_numeric(df["relaxed_condition_hit_if_eval"], errors="coerce").fillna(0).to_numpy() > 0.5
    exact = df["precursor_exact_if_eval"].astype(bool).to_numpy()
    jac = pd.to_numeric(df["precursor_jaccard_if_eval"], errors="coerce").fillna(0.0).to_numpy(float)
    rel = np.zeros(len(df), dtype=np.int32)
    rel[exact | cond_relaxed] = 1
    rel[(jac >= 0.5) & cond_relaxed] = 2
    rel[exact & cond_relaxed] = 3
    rel[exact & cond_strict] = 4
    return rel


def make_features(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    frames = [add_derived(x) for x in [train, val, test]]
    nums = [c for c in NUMERIC + ["precursor_rank_score", "condition_rank_score", "rank_product_score", "temperature_interval_width", "time_interval_width", "condition_is_point", "condition_is_quantile", "condition_is_bin", "condition_is_template", "is_core_method"] if c not in FORBIDDEN]
    for df in frames:
        for c in nums:
            df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0) if c in df.columns else 0.0
        for c in CAT:
            if c not in df.columns:
                df[c] = "missing"
    cat_cols = pd.get_dummies(pd.concat([frames[0][CAT], frames[1][CAT], frames[2][CAT]], ignore_index=True).astype(str), dtype=np.float32)
    n0, n1 = len(frames[0]), len(frames[1])
    out = []
    start = 0
    for df, n in zip(frames, [n0, n1, len(frames[2])]):
        out.append(pd.concat([df[nums].reset_index(drop=True), cat_cols.iloc[start:start+n].reset_index(drop=True)], axis=1))
        start += n
    return out[0], out[1], out[2], out[0].columns.tolist()


def route_flags(df: pd.DataFrame, protocol: str, rank_col: str) -> pd.DataFrame:
    out = df.copy()
    if protocol == "strict":
        known = pd.to_numeric(out.get("atmosphere_known_mask", 0), errors="coerce").fillna(0).astype(int) == 1
        atm_ok = known & (out["atmosphere"].astype(str) == out["true_atmosphere"].astype(str))
        temp_err = pd.to_numeric(out["temp_error"], errors="coerce").fillna(np.inf)
        time_err = pd.to_numeric(out["time_error"], errors="coerce").fillna(np.inf)
        cond_strict = ((temp_err <= 100) & (time_err <= 24) & atm_ok)
        cond_relaxed = ((temp_err <= 200) & (time_err <= 48) & atm_ok)
    else:
        cond_strict = pd.to_numeric(out["strict_condition_hit_if_eval"], errors="coerce").fillna(0) > 0.5
        cond_relaxed = pd.to_numeric(out["relaxed_condition_hit_if_eval"], errors="coerce").fillna(0) > 0.5
    exact = out["precursor_exact_if_eval"].astype(bool)
    jac = pd.to_numeric(out["precursor_jaccard_if_eval"], errors="coerce").fillna(0.0)
    out["_strict_route"] = (exact & cond_strict).astype(int)
    out["_relaxed_route"] = (exact & cond_relaxed).astype(int)
    out["_usable_relaxed_route"] = ((jac >= 0.5) & cond_relaxed).astype(int)
    return out


def metrics(df: pd.DataFrame, rank_col: str, protocol: str = "missing_aware") -> Dict[str, float]:
    d = route_flags(df, protocol, rank_col)
    out: Dict[str, float] = {"n_samples": int(d["sample_id"].nunique()), "n_candidates": int(len(d))}
    for k in [1, 3, 5, 10, 50, 200, 400]:
        g = d[d[rank_col] <= k].groupby("sample_id", sort=False)
        if len(g):
            out[f"top{k}_strict_route"] = float(g["_strict_route"].max().mean())
            out[f"top{k}_relaxed_route"] = float(g["_relaxed_route"].max().mean())
            out[f"top{k}_usable_relaxed_route"] = float(g["_usable_relaxed_route"].max().mean())
    core = d[d["reaction_method"].isin(CORE_METHODS)]
    cg = core[core[rank_col] <= 10].groupby("sample_id", sort=False)
    out["core_top10_relaxed_route"] = float(cg["_relaxed_route"].max().mean()) if len(cg) else 0.0
    return out


def rank(df: pd.DataFrame, score_col: str, rank_col: str) -> pd.DataFrame:
    out = df.sort_values(["sample_id", score_col, "route_total_score_raw"], ascending=[True, False, False], kind="mergesort").copy()
    out[rank_col] = out.groupby("sample_id", sort=False).cumcount() + 1
    return out


def blend_scores(df: pd.DataFrame, score_a: str, score_b: str, alpha: float) -> pd.Series:
    def z(col: str) -> pd.Series:
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        mean = vals.groupby(df["sample_id"], sort=False).transform("mean")
        std = vals.groupby(df["sample_id"], sort=False).transform("std").replace(0, np.nan)
        return ((vals - mean) / std).fillna(0.0)
    return (1 - alpha) * z(score_a) + alpha * z(score_b)


def choose_blend(val: pd.DataFrame, model_col: str) -> tuple[float, Dict[str, float]]:
    best_alpha = 0.0
    best = None
    for alpha in np.linspace(0, 1, 21):
        tmp = val.copy()
        tmp["_blend"] = blend_scores(tmp, "route_total_score_raw", model_col, float(alpha))
        tmp = rank(tmp, "_blend", "_blend_rank")
        m = metrics(tmp, "_blend_rank", "missing_aware")
        key = (m["top1_relaxed_route"], m["top10_relaxed_route"], m["top1_strict_route"])
        if best is None or key > (best["top1_relaxed_route"], best["top10_relaxed_route"], best["top1_strict_route"]):
            best_alpha, best = float(alpha), m
    return best_alpha, best or {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train final Stage35 reranker v3 with train route candidates, val selection, and test-only final evaluation.")
    ap.add_argument("--candidate_dir", default="outputs/evaluation/stage35_route_candidates_v3_final_20260612")
    ap.add_argument("--run_dir", default="runs/stage35/route_reranker_v3_final_20260612")
    ap.add_argument("--save_topk_csv", type=int, default=10, help="Save only top-K reranked rows per sample for inspection; 0 disables CSV output.")
    ap.add_argument("--seed", type=int, default=20260612)
    args = ap.parse_args()
    cand = Path(args.candidate_dir)
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(cand / "train_route_candidates.csv")
    val_df = pd.read_csv(cand / "val_route_candidates.csv")
    test_df = pd.read_csv(cand / "test_route_candidates.csv")
    x_train, x_val, x_test, feat_cols = make_features(train_df, val_df, test_df)
    y_rank = relevance(train_df)
    y_bin = (y_rank >= 3).astype(int)
    group = train_df.groupby("sample_id", sort=False).size().astype(int).tolist()

    ranker = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", n_estimators=280, learning_rate=0.04,
        num_leaves=63, min_child_samples=20, subsample=0.85, colsample_bytree=0.85,
        reg_lambda=1.0, label_gain=[0, 1, 2, 4, 7], random_state=args.seed, n_jobs=8, verbose=-1,
    )
    ranker.fit(x_train.to_numpy(np.float32), y_rank, group=group)
    clf = lgb.LGBMClassifier(
        objective="binary", n_estimators=350, learning_rate=0.04, num_leaves=63,
        min_child_samples=25, subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
        random_state=args.seed + 1, n_jobs=8, verbose=-1,
    )
    clf.fit(x_train.to_numpy(np.float32), y_bin, eval_set=[(x_val.to_numpy(np.float32), (relevance(val_df) >= 3).astype(int))], callbacks=[lgb.early_stopping(35, verbose=False)])

    artifacts = {"ranker": ranker, "binary_classifier": clf, "feature_cols": feat_cols, "config": vars(args)}
    joblib.dump(artifacts, run / "stage35_route_reranker_v3_final.joblib")
    write_json(run / "feature_schema.json", {"feature_cols": feat_cols, "forbidden_eval_features": sorted(FORBIDDEN)})

    summaries: Dict[str, Any] = {}
    for split, df, x in [("train", train_df, x_train), ("val", val_df, x_val), ("test", test_df, x_test)]:
        out = df.copy()
        out["lambdarank_score"] = ranker.predict(x.to_numpy(np.float32))
        out["binary_relaxed_score"] = clf.predict_proba(x.to_numpy(np.float32))[:, 1]
        out = rank(out, "route_total_score_raw", "raw_rank")
        out = rank(out, "lambdarank_score", "lambdarank_rank")
        out = rank(out, "binary_relaxed_score", "binary_rank")
        alpha_lgbm, _ = choose_blend(out if split == "val" else val_df.assign(lambdarank_score=ranker.predict(x_val.to_numpy(np.float32))), "lambdarank_score") if split == "val" else (0.0, {})
        # Use the alpha selected on val for all final split outputs. It is computed after val predictions below.
        if split == "val":
            selected_alpha = alpha_lgbm
        alpha = locals().get("selected_alpha", 0.0)
        out["blend_score"] = blend_scores(out, "route_total_score_raw", "lambdarank_score", alpha)
        out = rank(out, "blend_score", "blend_rank")
        if args.save_topk_csv > 0:
            keep_rank = "blend_rank" if split != "train" else "raw_rank"
            out[out[keep_rank] <= int(args.save_topk_csv)].to_csv(run / f"{split}_top{args.save_topk_csv}_route_candidates_reranked_v3_final.csv", index=False)
        summaries[split] = {
            "raw_missing_aware": metrics(out, "raw_rank", "missing_aware"),
            "lambdarank_missing_aware": metrics(out, "lambdarank_rank", "missing_aware"),
            "binary_missing_aware": metrics(out, "binary_rank", "missing_aware"),
            "blend_missing_aware": metrics(out, "blend_rank", "missing_aware"),
            "raw_strict_comparable": metrics(out, "raw_rank", "strict"),
            "lambdarank_strict_comparable": metrics(out, "lambdarank_rank", "strict"),
            "binary_strict_comparable": metrics(out, "binary_rank", "strict"),
            "blend_strict_comparable": metrics(out, "blend_rank", "strict"),
        }
        write_json(run / f"{split}_metrics.json", summaries[split])
    write_json(run / "metrics.json", {"config": vars(args), "blend_alpha": locals().get("selected_alpha", 0.0), "splits": summaries})
    report = ["# Stage35 Reranker v3 Final Training Report", "", json.dumps(to_builtin({"blend_alpha": locals().get("selected_alpha", 0.0), "test": summaries["test"]}), ensure_ascii=False, indent=2)]
    (run / "reranker_v3_final_training_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(to_builtin({"blend_alpha": locals().get("selected_alpha", 0.0), "test": summaries["test"]}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
