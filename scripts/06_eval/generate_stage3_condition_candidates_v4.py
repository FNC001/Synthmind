#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Sequence

import joblib
import numpy as np
import pandas as pd


CORE_METHODS = {"solid_state", "solution", "melt_arc"}


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_builtin(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def read_split(data_dir: Path, split: str) -> pd.DataFrame:
    pq = data_dir / f"{split}.parquet"
    if pq.exists():
        try:
            return pd.read_parquet(pq)
        except Exception:
            pass
    return pd.read_csv(data_dir / f"{split}.csv")


def make_x(df: pd.DataFrame, numeric_cols: Sequence[str], feature_cols: Sequence[str]) -> pd.DataFrame:
    nums = pd.DataFrame(index=df.index)
    for c in numeric_cols:
        nums[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0) if c in df.columns else 0.0
    cat_cols = [c for c in ["reaction_method", "synthesis_type", "precursor_input_mode"] if c in df.columns]
    cats = pd.get_dummies(df[cat_cols].astype(str), dtype=np.float32) if cat_cols else pd.DataFrame(index=df.index)
    x = pd.concat([nums.reset_index(drop=True), cats.reset_index(drop=True)], axis=1)
    for c in feature_cols:
        if c not in x.columns:
            x[c] = 0.0
    return x[list(feature_cols)]


def add_missing_model_fields(cand: pd.DataFrame, split_df: pd.DataFrame, atm_pack: Dict[str, Any]) -> pd.DataFrame:
    out = cand.copy()
    x = make_x(split_df, atm_pack["numeric_cols"], atm_pack["feature_cols"])
    atm_known = atm_pack["atmosphere_known"].predict_proba(x)[:, 1] if hasattr(atm_pack["atmosphere_known"], "predict_proba") and len(getattr(atm_pack["atmosphere_known"], "classes_", [0, 1])) > 1 else np.full(len(split_df), 0.5)
    solv_known = atm_pack["solvent_known"].predict_proba(x)[:, 1] if hasattr(atm_pack["solvent_known"], "predict_proba") and len(getattr(atm_pack["solvent_known"], "classes_", [0, 1])) > 1 else np.full(len(split_df), 0.5)
    atm_class = np.max(atm_pack["atmosphere_class"].predict_proba(x), axis=1) if hasattr(atm_pack["atmosphere_class"], "predict_proba") else np.full(len(split_df), 0.5)
    solv_class = np.max(atm_pack["solvent_class"].predict_proba(x), axis=1) if hasattr(atm_pack["solvent_class"], "predict_proba") else np.full(len(split_df), 0.5)
    sid = split_df["sample_id"].astype(str)
    maps = {
        "atmosphere_known_probability": dict(zip(sid, atm_known)),
        "solvent_known_probability": dict(zip(sid, solv_known)),
        "atmosphere_class_probability": dict(zip(sid, atm_class)),
        "solvent_class_probability": dict(zip(sid, solv_class)),
        "precursor_uncertainty_score": dict(zip(sid, pd.to_numeric(split_df.get("precursor_top20_uncertainty", 0), errors="coerce").fillna(0.0))),
    }
    for col, mp in maps.items():
        out[col] = out["sample_id"].astype(str).map(mp).fillna(0.0)
    out["method_expert_score"] = pd.to_numeric(out.get("method_template_score", 0), errors="coerce").fillna(0.0)
    out["template_score"] = out["method_expert_score"]
    out["multimodal_score"] = pd.to_numeric(out.get("multimodal_template_score", 0), errors="coerce").fillna(0.0)
    out["missing_aware_score"] = pd.to_numeric(out["total_score_raw"], errors="coerce").fillna(0.0) + 0.25 * out["atmosphere_known_probability"] + 0.15 * out["solvent_known_probability"]
    out["strict_comparable_score"] = pd.to_numeric(out["total_score_raw"], errors="coerce").fillna(0.0) + 0.30 * out["atmosphere_class_probability"] + 0.15 * out["atmosphere_known_probability"]
    return out


def condition_flags(df: pd.DataFrame, protocol: str) -> pd.DataFrame:
    out = df.copy()
    temp_err = pd.to_numeric(out["temp_error"], errors="coerce").fillna(np.inf)
    time_err = pd.to_numeric(out["time_error"], errors="coerce").fillna(np.inf)
    if protocol == "strict_comparable":
        known = pd.to_numeric(out.get("atmosphere_known_mask", 0), errors="coerce").fillna(0).astype(int) == 1
        atm_ok = known & (out["atmosphere"].astype(str) == out["true_atmosphere"].astype(str))
        out["_strict_condition"] = ((temp_err <= 100) & (time_err <= 24) & atm_ok).astype(int)
        out["_relaxed_condition"] = ((temp_err <= 200) & (time_err <= 48) & atm_ok).astype(int)
    else:
        out["_strict_condition"] = pd.to_numeric(out["strict_hit_if_eval"], errors="coerce").fillna(0).astype(int)
        out["_relaxed_condition"] = pd.to_numeric(out["relaxed_hit_if_eval"], errors="coerce").fillna(0).astype(int)
    return out


def metrics(df: pd.DataFrame, protocol: str, rank_col: str = "condition_rank_raw") -> Dict[str, Any]:
    d = condition_flags(df, protocol)
    out: Dict[str, Any] = {"n_samples": int(d["sample_id"].nunique()), "n_candidates": int(len(d))}
    for k in [1, 5, 10, 20]:
        sub = d[d[rank_col] <= k]
        g = sub.groupby("sample_id", sort=False)
        out[f"top{k}_strict_condition"] = float(g["_strict_condition"].max().mean()) if len(g) else 0.0
        out[f"top{k}_relaxed_condition"] = float(g["_relaxed_condition"].max().mean()) if len(g) else 0.0
    g = d.groupby("sample_id", sort=False)
    out["oracle_strict_condition"] = float(g["_strict_condition"].max().mean()) if len(g) else 0.0
    out["oracle_relaxed_condition"] = float(g["_relaxed_condition"].max().mean()) if len(g) else 0.0
    core = d[d["reaction_method"].isin(CORE_METHODS)]
    cg = core[core[rank_col] <= 10].groupby("sample_id", sort=False)
    out["core_top10_relaxed_condition"] = float(cg["_relaxed_condition"].max().mean()) if len(cg) else 0.0
    return out


def rerank(df: pd.DataFrame, score_col: str, rank_col: str) -> pd.DataFrame:
    out = df.sort_values(["sample_id", score_col, "total_score_raw"], ascending=[True, False, False], kind="mergesort").copy()
    out[rank_col] = out.groupby("sample_id", sort=False).cumcount() + 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Stage3 condition candidates v4 from v4 distributional experts and missing-aware atmosphere/solvent models.")
    ap.add_argument("--dataset_dir", default="data/interim/generative/stage3_condition_dataset_predprec_oof_v4_20260612")
    ap.add_argument("--method_model", default="runs/stage3/method_distributional_experts_v4_20260612/stage3_method_distributional_experts_v4.joblib")
    ap.add_argument("--atmosphere_solvent_model", default="runs/stage3/atmosphere_solvent_missing_v4_20260612/model_pack.joblib")
    ap.add_argument("--output_dir", default="outputs/evaluation/stage3_condition_candidates_v4_20260612")
    ap.add_argument("--python", default="/Users/lihonglin/miniconda3/envs/py311/bin/python")
    ap.add_argument("--splits", default="train,val,test")
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.python, "scripts/06_eval/generate_stage3_condition_candidates_v3.py",
        "--input_dir", args.dataset_dir,
        "--model", args.method_model,
        "--output_dir", args.output_dir,
        "--splits", args.splits,
    ]
    subprocess.run(cmd, check=True)
    atm_pack = joblib.load(args.atmosphere_solvent_model)
    data_dir = Path(args.dataset_dir)
    summaries = {"config": vars(args), "splits": {}}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        cand_path = outdir / f"{split}_condition_candidates.csv"
        cand = pd.read_csv(cand_path)
        split_df = read_split(data_dir, split)
        cand = add_missing_model_fields(cand, split_df, atm_pack)
        cand = rerank(cand, "missing_aware_score", "condition_rank_missing_aware_v4")
        cand = rerank(cand, "strict_comparable_score", "condition_rank_strict_comparable_v4")
        cand.to_csv(cand_path, index=False)
        m_missing = metrics(cand, "missing_aware", "condition_rank_missing_aware_v4")
        m_strict = metrics(cand, "strict_comparable", "condition_rank_strict_comparable_v4")
        by_method = {
            str(method): {
                "missing_aware": metrics(g, "missing_aware", "condition_rank_missing_aware_v4"),
                "strict_comparable": metrics(g, "strict_comparable", "condition_rank_strict_comparable_v4"),
            }
            for method, g in cand.groupby("reaction_method", sort=False)
        }
        write_json(outdir / f"{split}_condition_candidate_metrics_v4.json", {"missing_aware": m_missing, "strict_comparable": m_strict, "by_method": by_method})
        summaries["splits"][split] = {"missing_aware": m_missing, "strict_comparable": m_strict}
    write_json(outdir / "condition_candidate_pool_v4_summary.json", summaries)
    print(json.dumps(to_builtin(summaries), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
