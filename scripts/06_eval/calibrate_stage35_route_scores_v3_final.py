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
    "contains_open_generated_precursor",
    "contains_repair_precursor",
]


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def add_protocol_flags(df: pd.DataFrame, protocol: str) -> pd.DataFrame:
    out = df.copy()
    exact = out["precursor_exact_if_eval"].astype(bool)
    jac = pd.to_numeric(out["precursor_jaccard_if_eval"], errors="coerce").fillna(0.0)
    if protocol == "strict_comparable":
        known = pd.to_numeric(out.get("atmosphere_known_mask", 0), errors="coerce").fillna(0).astype(int) == 1
        atm_ok = known & (out["atmosphere"].astype(str) == out["true_atmosphere"].astype(str))
        temp_err = pd.to_numeric(out["temp_error"], errors="coerce").fillna(np.inf)
        time_err = pd.to_numeric(out["time_error"], errors="coerce").fillna(np.inf)
        cond_strict = (temp_err <= 100) & (time_err <= 24) & atm_ok
        cond_relaxed = (temp_err <= 200) & (time_err <= 48) & atm_ok
    else:
        cond_strict = pd.to_numeric(out["strict_condition_hit_if_eval"], errors="coerce").fillna(0) > 0.5
        cond_relaxed = pd.to_numeric(out["relaxed_condition_hit_if_eval"], errors="coerce").fillna(0) > 0.5
    out["_strict_route"] = (exact & cond_strict).astype(int)
    out["_relaxed_route"] = (exact & cond_relaxed).astype(int)
    out["_usable_relaxed_route"] = ((jac >= 0.5) & cond_relaxed).astype(int)
    return out


def metric(df: pd.DataFrame, rank_col: str, protocol: str) -> Dict[str, float]:
    d = add_protocol_flags(df, protocol)
    out: Dict[str, float] = {"n_samples": int(d["sample_id"].nunique()), "n_candidates": int(len(d))}
    for k in [1, 3, 5, 10, 50, 100, 200, 400]:
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


def objective(m: Mapping[str, float]) -> float:
    return (
        0.30 * m.get("top1_relaxed_route", 0.0)
        + 0.20 * m.get("top1_strict_route", 0.0)
        + 0.20 * m.get("top10_relaxed_route", 0.0)
        + 0.10 * m.get("top10_strict_route", 0.0)
        + 0.10 * m.get("core_top10_relaxed_route", 0.0)
        + 0.10 * m.get("top10_usable_relaxed_route", 0.0)
    )


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in SCORE_COLS:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        else:
            vals = pd.Series(0.0, index=df.index)
        lo, hi = vals.quantile(0.01), vals.quantile(0.99)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            out[col] = ((vals - lo) / (hi - lo)).clip(0, 1)
        else:
            out[col] = vals
    out["precursor_rank_score"] = 1.0 / pd.to_numeric(df["precursor_rank"], errors="coerce").clip(lower=1).fillna(999)
    out["condition_rank_score"] = 1.0 / pd.to_numeric(df["condition_rank"], errors="coerce").clip(lower=1).fillna(999)
    out["rank_product_score"] = out["precursor_rank_score"] * out["condition_rank_score"]
    return out


def weights() -> Iterable[Dict[str, float]]:
    yield {"route_total_score_raw": 1.0}
    yield {
        "precursor_score_norm": 1.2,
        "condition_score_norm": 1.2,
        "precursor_condition_compatibility_score": 0.4,
        "precursor_rank_score": 0.4,
        "condition_rank_score": 0.3,
    }
    yield {
        "precursor_score_norm": 1.0,
        "condition_score_norm": 1.5,
        "condition_confidence": 0.8,
        "atmosphere_probability": 0.4,
        "open_generated_penalty": -0.8,
        "repair_penalty": -0.5,
    }
    yield {
        "precursor_score_norm": 1.5,
        "condition_score_norm": 1.0,
        "precursor_confidence": 0.6,
        "temperature_point_score": 0.4,
        "time_point_score": 0.4,
    }
    yield {
        "route_total_score_raw": 0.8,
        "precursor_rank_score": 0.5,
        "condition_rank_score": 0.5,
        "rank_product_score": 0.7,
        "contains_open_generated_precursor": -0.2,
        "contains_repair_precursor": -0.1,
    }


def apply(df: pd.DataFrame, norm: pd.DataFrame, w: Mapping[str, float]) -> pd.DataFrame:
    out = df.copy()
    score = np.zeros(len(out), dtype=float)
    for key, val in w.items():
        if key in norm.columns:
            score += float(val) * norm[key].to_numpy(float)
    out["route_calibrated_score_v3_final"] = score
    out = out.sort_values(
        ["sample_id", "route_calibrated_score_v3_final", "route_total_score_raw"],
        ascending=[True, False, False],
        kind="mergesort",
    )
    out["route_rank_calibrated_v3_final"] = out.groupby("sample_id", sort=False).cumcount() + 1
    return out


def calibrate_protocol(val: pd.DataFrame, test: pd.DataFrame, outdir: Path, protocol: str) -> Dict[str, Any]:
    rows = []
    best = None
    vn = normalize(val)
    for idx, w in enumerate(weights()):
        ranked = apply(val, vn, w)
        m = metric(ranked, "route_rank_calibrated_v3_final", protocol)
        obj = objective(m)
        row = {"search_id": idx, "objective": obj, **m, "weights": json.dumps(w, sort_keys=True)}
        rows.append(row)
        if best is None or obj > best["objective"]:
            best = {"search_id": idx, "objective": obj, "weights": w, "val_metrics": m}
    assert best is not None
    pd.DataFrame(rows).sort_values("objective", ascending=False).to_csv(outdir / f"val_weight_search_{protocol}.csv", index=False)
    write_json(outdir / f"best_weights_{protocol}.json", best)
    val_ranked = apply(val, normalize(val), best["weights"])
    test_ranked = apply(test, normalize(test), best["weights"])
    result = {
        "protocol": protocol,
        "best": best,
        "val": metric(val_ranked, "route_rank_calibrated_v3_final", protocol),
        "test": metric(test_ranked, "route_rank_calibrated_v3_final", protocol),
    }
    write_json(outdir / f"test_route_metrics_{protocol}.json", result["test"])
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate final Stage35 route scores v3 under missing-aware and strict-comparable protocols.")
    ap.add_argument("--candidate_dir", default="outputs/evaluation/stage35_route_candidates_v3_final_20260612")
    ap.add_argument("--output_dir", default="outputs/evaluation/stage35_route_score_calibration_v3_final_20260612")
    args = ap.parse_args()

    cand = Path(args.candidate_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    val = pd.read_csv(cand / "val_route_candidates.csv")
    test = pd.read_csv(cand / "test_route_candidates.csv")

    outputs = {
        "missing_aware": calibrate_protocol(val, test, outdir, "missing_aware"),
        "strict_comparable": calibrate_protocol(val, test, outdir, "strict_comparable"),
    }
    write_json(outdir / "route_calibration_metrics_v3_final.json", outputs)
    report = [
        "# Stage35 Route Score Calibration v3 Final",
        "",
        "This final calibration evaluates the same candidate pool under missing-aware and strict-comparable protocols.",
        "Full calibrated candidate CSVs are intentionally not written because the route candidate tables are large.",
        "",
        "```json",
        json.dumps(to_builtin(outputs), ensure_ascii=False, indent=2),
        "```",
    ]
    (outdir / "route_score_calibration_v3_final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(to_builtin(outputs), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
