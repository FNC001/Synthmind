#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import pandas as pd


CORE_METHODS = {"solid_state", "solution", "melt_arc"}
SCORE_COLS = [
    "precursor_score",
    "condition_score",
    "precursor_score_norm",
    "condition_score_norm",
    "precursor_confidence",
    "condition_confidence",
    "method_template_score",
    "retrieval_score",
    "multimodal_template_score",
    "precursor_condition_compatibility_score",
    "reaction_method_prior_score",
    "atmosphere_probability",
    "solvent_probability",
    "temperature_point_score",
    "temperature_bin_score",
    "time_point_score",
    "time_bin_score",
    "open_generated_penalty",
    "repair_penalty",
    "route_total_score_raw",
]


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


def metric(df: pd.DataFrame, rank_col: str) -> Dict[str, float]:
    out = {"n_samples": int(df["sample_id"].nunique()), "n_candidates": int(len(df))}
    for k in [1, 3, 5, 10, 50, 100, 200, 400]:
        sub = df[df[rank_col] <= k]
        g = sub.groupby("sample_id", sort=False)
        if len(g):
            out[f"top{k}_strict_route"] = float(g["strict_route_hit_if_eval"].max().mean())
            out[f"top{k}_relaxed_route"] = float(g["relaxed_route_hit_if_eval"].max().mean())
            out[f"top{k}_usable_relaxed_route"] = float(g["usable_relaxed_route_hit_if_eval"].max().mean())
    core = df[df["reaction_method"].isin(CORE_METHODS)]
    cg = core[core[rank_col] <= 10].groupby("sample_id", sort=False)
    out["core_top10_relaxed_route"] = float(cg["relaxed_route_hit_if_eval"].max().mean()) if len(cg) else 0.0
    return out


def objective(m: Mapping[str, float]) -> float:
    return (
        0.30 * m["top1_relaxed_route"]
        + 0.20 * m["top1_strict_route"]
        + 0.20 * m["top10_relaxed_route"]
        + 0.10 * m["top10_strict_route"]
        + 0.10 * m["core_top10_relaxed_route"]
        + 0.10 * m["top10_usable_relaxed_route"]
    )


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in SCORE_COLS:
        vals = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0) if col in df.columns else pd.Series(0.0, index=df.index)
        lo, hi = vals.quantile(0.01), vals.quantile(0.99)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            out[col] = ((vals - lo) / (hi - lo)).clip(0, 1)
        else:
            out[col] = vals
    out["precursor_rank_score"] = 1.0 / pd.to_numeric(df["precursor_rank"], errors="coerce").clip(lower=1).fillna(999)
    out["condition_rank_score"] = 1.0 / pd.to_numeric(df["condition_rank"], errors="coerce").clip(lower=1).fillna(999)
    return out


def weights() -> Iterable[Dict[str, float]]:
    yield {"route_total_score_raw": 1.0}
    yield {"precursor_score_norm": 1.2, "condition_score_norm": 1.2, "precursor_condition_compatibility_score": 0.4, "precursor_rank_score": 0.4, "condition_rank_score": 0.3}
    yield {"precursor_score_norm": 1.0, "condition_score_norm": 1.5, "condition_confidence": 0.8, "atmosphere_probability": 0.4, "open_generated_penalty": -0.8, "repair_penalty": -0.5}
    yield {"precursor_score_norm": 1.5, "condition_score_norm": 1.0, "precursor_confidence": 0.6, "temperature_point_score": 0.4, "time_point_score": 0.4}


def apply(df: pd.DataFrame, norm: pd.DataFrame, w: Mapping[str, float]) -> pd.DataFrame:
    out = df.copy()
    score = np.zeros(len(out), dtype=float)
    for k, v in w.items():
        score += float(v) * norm.get(k, 0.0)
    out["route_calibrated_score_v3"] = score
    out = out.sort_values(["sample_id", "route_calibrated_score_v3", "route_total_score_raw"], ascending=[True, False, False], kind="mergesort")
    out["route_rank_calibrated_v3"] = out.groupby("sample_id", sort=False).cumcount() + 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate Stage35 route scores v3 using validation candidates.")
    ap.add_argument("--candidate_dir", default="outputs/evaluation/stage35_route_candidates_v3_20260610")
    ap.add_argument("--output_dir", default="outputs/evaluation/stage35_route_score_calibration_v3_20260610")
    args = ap.parse_args()
    cand = Path(args.candidate_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    val = pd.read_csv(cand / "val_route_candidates.csv")
    test = pd.read_csv(cand / "test_route_candidates.csv")
    vn = normalize(val)
    rows = []
    best = None
    for i, w in enumerate(weights()):
        rr = apply(val, vn, w)
        m = metric(rr, "route_rank_calibrated_v3")
        obj = objective(m)
        rows.append({"search_id": i, "objective": obj, **m, "weights": json.dumps(w, sort_keys=True)})
        if best is None or obj > best["objective"]:
            best = {"search_id": i, "objective": obj, "weights": w, "metrics": m}
    assert best is not None
    pd.DataFrame(rows).sort_values("objective", ascending=False).to_csv(outdir / "val_weight_search.csv", index=False)
    write_json(outdir / "best_weights.json", best)
    outputs = {}
    for split, df in [("val", val), ("test", test)]:
        rr = apply(df, normalize(df), best["weights"])
        rr.to_csv(outdir / f"{split}_route_candidates_calibrated.csv", index=False)
        outputs[split] = metric(rr, "route_rank_calibrated_v3")
    write_json(outdir / "test_route_metrics_calibrated.json", outputs["test"])
    write_json(outdir / "route_calibration_metrics_v3.json", outputs)
    report = ["# Stage35 Route Score Calibration v3", "", json.dumps(to_builtin({"best": best, "metrics": outputs}), ensure_ascii=False, indent=2)]
    (outdir / "route_score_calibration_v3_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(to_builtin({"best": best, "metrics": outputs}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
