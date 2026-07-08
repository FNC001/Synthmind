#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


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


def read_csv_topk(path: Path, rank_cols: List[str], top_k: int, usecols: List[str] | None = None) -> pd.DataFrame:
    chunks: List[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=lambda c: usecols is None or c in usecols, chunksize=250_000):
        rank_col = next((c for c in rank_cols if c in chunk.columns), None)
        if rank_col is None:
            raise ValueError(f"No rank column among {rank_cols} in {path}")
        chunks.append(chunk[pd.to_numeric(chunk[rank_col], errors="coerce").fillna(10**9) <= top_k].copy())
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def load_precursors(path: Path, top_k: int) -> pd.DataFrame:
    probe = pd.read_csv(path, nrows=1)
    wanted = [
        "sample_index",
        "id",
        "formula",
        "reaction_method",
        "true_precursors",
        "pred_precursors",
        "candidate_set",
        "candidate_source",
        "candidate_source_mix",
        "rank",
        "calibrated_rank_v5",
        "total_score_v5",
        "calibrated_score_v5",
        "method_template_score",
        "family_score",
        "open_vocab_score",
        "retrieval_score",
        "mlp_score",
        "element_coverage",
        "missing_element_count",
        "extra_element_count",
        "candidate_size",
        "exact",
        "f1",
        "jaccard",
    ]
    usecols = [c for c in wanted if c in probe.columns]
    df = read_csv_topk(path, ["calibrated_rank_v5", "rank"], top_k, usecols)
    df = df.rename(columns={"id": "sample_id"})
    if "pred_precursors" not in df.columns and "candidate_set" in df.columns:
        df["pred_precursors"] = df["candidate_set"]
    if "calibrated_rank_v5" in df.columns:
        df["precursor_rank"] = pd.to_numeric(df["calibrated_rank_v5"], errors="coerce")
    else:
        df["precursor_rank"] = pd.to_numeric(df["rank"], errors="coerce")
    if "calibrated_score_v5" in df.columns:
        df["precursor_score"] = pd.to_numeric(df["calibrated_score_v5"], errors="coerce")
    elif "total_score_v5" in df.columns:
        df["precursor_score"] = pd.to_numeric(df["total_score_v5"], errors="coerce")
    else:
        df["precursor_score"] = -df["precursor_rank"]
    for col, default in [
        ("exact", False),
        ("f1", 0.0),
        ("jaccard", 0.0),
        ("element_coverage", np.nan),
        ("missing_element_count", np.nan),
        ("extra_element_count", np.nan),
        ("candidate_size", np.nan),
        ("retrieval_score", 0.0),
        ("mlp_score", 0.0),
        ("family_score", 0.0),
        ("open_vocab_score", 0.0),
    ]:
        if col not in df.columns:
            df[col] = default
    df["precursor_exact"] = df["exact"].astype(str).str.lower().isin(["true", "1"])
    return df.sort_values(["sample_id", "precursor_rank"], kind="mergesort")


def load_conditions(path: Path, top_k: int) -> pd.DataFrame:
    cols = [
        "sample_id",
        "reaction_method",
        "temperature_c",
        "time_h",
        "atmosphere",
        "solvent",
        "condition_source",
        "model_score",
        "retrieval_score",
        "method_template_score",
        "condition_prior_score",
        "temperature_plausibility_score",
        "time_plausibility_score",
        "atmosphere_probability",
        "condition_total_score",
        "condition_calibrated_score",
        "condition_rank_calibrated",
        "true_temperature_c",
        "true_time_h",
        "true_atmosphere",
        "true_solvent",
        "strict_hit_if_eval",
        "relaxed_hit_if_eval",
    ]
    df = read_csv_topk(path, ["condition_rank_calibrated", "condition_rank_raw"], top_k, cols)
    df["condition_rank"] = pd.to_numeric(df.get("condition_rank_calibrated", df.get("condition_rank_raw")), errors="coerce")
    df["condition_score"] = pd.to_numeric(df.get("condition_calibrated_score", df.get("condition_total_score")), errors="coerce")
    return df.sort_values(["sample_id", "condition_rank"], kind="mergesort")


def minmax(s: pd.Series) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    lo, hi = vals.quantile(0.01), vals.quantile(0.99)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return vals.fillna(0.0)
    return ((vals - lo) / (hi - lo)).clip(0, 1).fillna(0.0)


def build_split(split: str, precursor_path: Path, condition_path: Path, output_dir: Path, top_precursors: int, top_conditions: int) -> pd.DataFrame:
    prec = load_precursors(precursor_path, top_precursors)
    cond = load_conditions(condition_path, top_conditions)
    prec["precursor_score_norm"] = minmax(prec["precursor_score"])
    cond["condition_score_norm"] = minmax(cond["condition_score"])
    rows: List[pd.DataFrame] = []
    cond_cols = [
        "sample_id",
        "temperature_c",
        "time_h",
        "atmosphere",
        "solvent",
        "condition_source",
        "condition_rank",
        "condition_score",
        "condition_score_norm",
        "model_score",
        "retrieval_score",
        "method_template_score",
        "condition_prior_score",
        "temperature_plausibility_score",
        "time_plausibility_score",
        "atmosphere_probability",
        "true_temperature_c",
        "true_time_h",
        "true_atmosphere",
        "true_solvent",
        "strict_hit_if_eval",
        "relaxed_hit_if_eval",
    ]
    for sid, pg in prec.groupby("sample_id", sort=False):
        cg = cond[cond["sample_id"] == sid]
        if cg.empty:
            continue
        merged = pg.merge(cg[cond_cols], on="sample_id", how="inner", suffixes=("_precursor", "_condition"))
        rows.append(merged)
    if not rows:
        raise RuntimeError(f"No route candidates generated for {split}")
    df = pd.concat(rows, ignore_index=True)
    df["route_candidate_id"] = df["sample_id"].astype(str) + "::p" + df["precursor_rank"].astype(int).astype(str) + "::c" + df["condition_rank"].astype(int).astype(str)
    df["route_total_score"] = (
        1.25 * df["precursor_score_norm"]
        + 1.00 * df["condition_score_norm"]
        + 0.25 * pd.to_numeric(df["element_coverage"], errors="coerce").fillna(0.0)
        - 0.25 * np.log1p(pd.to_numeric(df["precursor_rank"], errors="coerce").fillna(top_precursors))
        - 0.15 * np.log1p(pd.to_numeric(df["condition_rank"], errors="coerce").fillna(top_conditions))
    )
    df["route_exact_strict"] = (df["precursor_exact"] & (pd.to_numeric(df["strict_hit_if_eval"], errors="coerce").fillna(0) > 0.5)).astype(int)
    df["route_exact_relaxed"] = (df["precursor_exact"] & (pd.to_numeric(df["relaxed_hit_if_eval"], errors="coerce").fillna(0) > 0.5)).astype(int)
    df["route_usable_relaxed"] = ((pd.to_numeric(df["jaccard"], errors="coerce").fillna(0.0) >= 0.5) & (pd.to_numeric(df["relaxed_hit_if_eval"], errors="coerce").fillna(0) > 0.5)).astype(int)
    df = df.sort_values(["sample_id", "route_total_score"], ascending=[True, False], kind="mergesort")
    df["route_rank_raw"] = df.groupby("sample_id", sort=False).cumcount() + 1
    out_path = output_dir / f"{split}_route_candidates.csv"
    df.to_csv(out_path, index=False)
    metrics = route_metrics(df, "route_rank_raw")
    write_json(output_dir / f"{split}_route_candidate_metrics.json", metrics)
    return df


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Stage35 full route candidates v2 from Stage2 chemistry-checked precursors and calibrated Stage3 conditions.")
    ap.add_argument("--val_precursors", default="outputs/evaluation/stage2_candidate_pool_v5_20260610/val_candidate_sets_repaired.csv")
    ap.add_argument("--test_precursors", default="outputs/evaluation/stage2_score_calibration_v5_20260610/test_candidate_sets_calibrated_v5.csv")
    ap.add_argument("--condition_dir", default="outputs/evaluation/stage3_condition_calibration_v2_20260610")
    ap.add_argument("--output_dir", default="outputs/evaluation/stage35_route_candidates_v2_20260610")
    ap.add_argument("--top_precursors", type=int, default=20)
    ap.add_argument("--top_conditions", type=int, default=10)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args)
    write_json(out / "config.json", config)
    summaries: Dict[str, Any] = {}
    for split, precursor_path in [("val", Path(args.val_precursors)), ("test", Path(args.test_precursors))]:
        condition_path = Path(args.condition_dir) / f"{split}_condition_candidates_calibrated_v2.csv"
        df = build_split(split, precursor_path, condition_path, out, args.top_precursors, args.top_conditions)
        summaries[split] = route_metrics(df, "route_rank_raw")
    write_json(out / "stage35_route_candidate_v2_summary.json", summaries)
    print(json.dumps(to_builtin(summaries), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
