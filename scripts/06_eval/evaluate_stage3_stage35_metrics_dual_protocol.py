#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


CORE_METHODS = {"solid_state", "solution", "melt_arc"}


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


def condition_flags(df: pd.DataFrame, protocol: str) -> pd.DataFrame:
    out = df.copy()
    if protocol == "missing_aware":
        out["_strict"] = pd.to_numeric(out["strict_hit_if_eval"], errors="coerce").fillna(0).astype(int)
        out["_relaxed"] = pd.to_numeric(out["relaxed_hit_if_eval"], errors="coerce").fillna(0).astype(int)
    else:
        known = pd.to_numeric(out.get("atmosphere_known_mask", 0), errors="coerce").fillna(0).astype(int) == 1
        atm_ok = known & (out["atmosphere"].astype(str) == out["true_atmosphere"].astype(str))
        temp_err = pd.to_numeric(out["temp_error"], errors="coerce").fillna(np.inf)
        time_err = pd.to_numeric(out["time_error"], errors="coerce").fillna(np.inf)
        out["_strict"] = ((temp_err <= 100) & (time_err <= 24) & atm_ok).astype(int)
        out["_relaxed"] = ((temp_err <= 200) & (time_err <= 48) & atm_ok).astype(int)
    return out


def condition_metrics(df: pd.DataFrame, protocol: str, rank_col: str = "condition_rank_calibrated_v3") -> Dict[str, Any]:
    d = condition_flags(df, protocol)
    out: Dict[str, Any] = {"n_samples": int(d["sample_id"].nunique()), "n_candidates": int(len(d))}
    for k in [1, 5, 10, 20]:
        sub = d[d[rank_col] <= k]
        g = sub.groupby("sample_id", sort=False)
        out[f"top{k}_strict_condition"] = float(g["_strict"].max().mean()) if len(g) else 0.0
        out[f"top{k}_relaxed_condition"] = float(g["_relaxed"].max().mean()) if len(g) else 0.0
    g = d.groupby("sample_id", sort=False)
    out["oracle_strict_condition"] = float(g["_strict"].max().mean()) if len(g) else 0.0
    out["oracle_relaxed_condition"] = float(g["_relaxed"].max().mean()) if len(g) else 0.0
    out["core"] = condition_metrics_simple(d[d["reaction_method"].isin(CORE_METHODS)], rank_col)
    out["by_method"] = {str(m): condition_metrics_simple(g, rank_col) for m, g in d.groupby("reaction_method", sort=False)}
    return out


def condition_metrics_simple(d: pd.DataFrame, rank_col: str) -> Dict[str, float]:
    if d.empty:
        return {}
    out = {}
    for k in [1, 10, 20]:
        g = d[d[rank_col] <= k].groupby("sample_id", sort=False)
        out[f"top{k}_relaxed_condition"] = float(g["_relaxed"].max().mean()) if len(g) else 0.0
        out[f"top{k}_strict_condition"] = float(g["_strict"].max().mean()) if len(g) else 0.0
    return out


def route_flags(df: pd.DataFrame, protocol: str) -> pd.DataFrame:
    out = df.copy()
    if protocol == "missing_aware":
        out["_cond_strict"] = pd.to_numeric(out["strict_condition_hit_if_eval"], errors="coerce").fillna(0).astype(int)
        out["_cond_relaxed"] = pd.to_numeric(out["relaxed_condition_hit_if_eval"], errors="coerce").fillna(0).astype(int)
    else:
        known = pd.to_numeric(out.get("atmosphere_known_mask", 0), errors="coerce").fillna(0).astype(int) == 1
        atm_ok = known & (out["atmosphere"].astype(str) == out["true_atmosphere"].astype(str))
        temp_err = pd.to_numeric(out["temp_error"], errors="coerce").fillna(np.inf)
        time_err = pd.to_numeric(out["time_error"], errors="coerce").fillna(np.inf)
        out["_cond_strict"] = ((temp_err <= 100) & (time_err <= 24) & atm_ok).astype(int)
        out["_cond_relaxed"] = ((temp_err <= 200) & (time_err <= 48) & atm_ok).astype(int)
    exact = out["precursor_exact_if_eval"].astype(bool)
    jac = pd.to_numeric(out["precursor_jaccard_if_eval"], errors="coerce").fillna(0.0)
    out["_strict_route"] = (exact & (out["_cond_strict"] > 0)).astype(int)
    out["_relaxed_route"] = (exact & (out["_cond_relaxed"] > 0)).astype(int)
    out["_usable_relaxed_route"] = ((jac >= 0.5) & (out["_cond_relaxed"] > 0)).astype(int)
    return out


def route_metrics(df: pd.DataFrame, protocol: str, rank_col: str = "route_rank_raw") -> Dict[str, Any]:
    d = route_flags(df, protocol)
    out: Dict[str, Any] = {"n_samples": int(d["sample_id"].nunique()), "n_candidates": int(len(d))}
    for k in [1, 10, 200, 400]:
        g = d[d[rank_col] <= k].groupby("sample_id", sort=False)
        out[f"top{k}_strict_route"] = float(g["_strict_route"].max().mean()) if len(g) else 0.0
        out[f"top{k}_relaxed_route"] = float(g["_relaxed_route"].max().mean()) if len(g) else 0.0
        out[f"top{k}_usable_relaxed_route"] = float(g["_usable_relaxed_route"].max().mean()) if len(g) else 0.0
    out["core"] = route_metrics_simple(d[d["reaction_method"].isin(CORE_METHODS)], rank_col)
    out["by_method"] = {str(m): route_metrics_simple(g, rank_col) for m, g in d.groupby("reaction_method", sort=False)}
    return out


def route_metrics_simple(d: pd.DataFrame, rank_col: str) -> Dict[str, float]:
    if d.empty:
        return {}
    out = {}
    for k in [1, 10, 200, 400]:
        g = d[d[rank_col] <= k].groupby("sample_id", sort=False)
        out[f"top{k}_relaxed_route"] = float(g["_relaxed_route"].max().mean()) if len(g) else 0.0
        out[f"top{k}_strict_route"] = float(g["_strict_route"].max().mean()) if len(g) else 0.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate Stage3/Stage35 v3 using missing-aware and strict-comparable protocols.")
    ap.add_argument("--condition_dir", default="outputs/evaluation/stage3_condition_calibration_v3_final_20260612")
    ap.add_argument("--route_dir", default="outputs/evaluation/stage35_route_candidates_v3_final_20260612")
    ap.add_argument("--output_dir", default="outputs/evaluation/stage3_stage35_v3_dual_protocol_20260612")
    ap.add_argument("--splits", default="val,test")
    args = ap.parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    cond_ma: Dict[str, Any] = {}
    cond_strict: Dict[str, Any] = {}
    route_ma: Dict[str, Any] = {}
    route_strict: Dict[str, Any] = {}
    for split in splits:
        cond = pd.read_csv(Path(args.condition_dir) / f"{split}_condition_candidates_calibrated.csv")
        route = pd.read_csv(Path(args.route_dir) / f"{split}_route_candidates.csv")
        cond_ma[split] = condition_metrics(cond, "missing_aware")
        cond_strict[split] = condition_metrics(cond, "strict")
        route_ma[split] = route_metrics(route, "missing_aware")
        route_strict[split] = route_metrics(route, "strict")
    write_json(outdir / "stage3_condition_metrics_missing_aware.json", cond_ma)
    write_json(outdir / "stage3_condition_metrics_strict_comparable.json", cond_strict)
    write_json(outdir / "stage35_route_metrics_missing_aware.json", route_ma)
    write_json(outdir / "stage35_route_metrics_strict_comparable.json", route_strict)
    report = [
        "# Stage3/Stage35 v3 Dual Protocol Metrics",
        "",
        "Protocol A missing-aware: missing atmosphere does not penalize atmosphere match.",
        "Protocol B strict-comparable: missing/unknown atmosphere is treated as not evaluable and therefore fails atmosphere match.",
        "",
        "## Test Summary",
        "",
        json.dumps(to_builtin({
            "condition_missing_aware": cond_ma.get("test", {}),
            "condition_strict_comparable": cond_strict.get("test", {}),
            "route_missing_aware": route_ma.get("test", {}),
            "route_strict_comparable": route_strict.get("test", {}),
        }), ensure_ascii=False, indent=2),
    ]
    (outdir / "dual_protocol_metrics_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(to_builtin({"condition_missing_aware": cond_ma.get("test", {}), "route_missing_aware": route_ma.get("test", {})}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
