#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

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


def flags(df: pd.DataFrame, protocol: str) -> pd.DataFrame:
    out = df.copy()
    temp_err = pd.to_numeric(out["temp_error"], errors="coerce").fillna(np.inf)
    time_err = pd.to_numeric(out["time_error"], errors="coerce").fillna(np.inf)
    if protocol == "strict_comparable":
        known = pd.to_numeric(out.get("atmosphere_known_mask", 0), errors="coerce").fillna(0).astype(int) == 1
        atm_ok = known & (out["atmosphere"].astype(str) == out["true_atmosphere"].astype(str))
        out["_strict"] = ((temp_err <= 100) & (time_err <= 24) & atm_ok).astype(int)
        out["_relaxed"] = ((temp_err <= 200) & (time_err <= 48) & atm_ok).astype(int)
    else:
        out["_strict"] = pd.to_numeric(out["strict_hit_if_eval"], errors="coerce").fillna(0).astype(int)
        out["_relaxed"] = pd.to_numeric(out["relaxed_hit_if_eval"], errors="coerce").fillna(0).astype(int)
    return out


def metrics(df: pd.DataFrame, protocol: str, rank_col: str) -> Dict[str, float]:
    d = flags(df, protocol)
    out: Dict[str, float] = {"n_samples": int(d["sample_id"].nunique()), "n_candidates": int(len(d))}
    for k in [1, 5, 10, 20]:
        sub = d[d[rank_col] <= k]
        g = sub.groupby("sample_id", sort=False)
        out[f"top{k}_strict_condition"] = float(g["_strict"].max().mean()) if len(g) else 0.0
        out[f"top{k}_relaxed_condition"] = float(g["_relaxed"].max().mean()) if len(g) else 0.0
    g = d.groupby("sample_id", sort=False)
    out["oracle_strict_condition"] = float(g["_strict"].max().mean()) if len(g) else 0.0
    out["oracle_relaxed_condition"] = float(g["_relaxed"].max().mean()) if len(g) else 0.0
    core = d[d["reaction_method"].isin(CORE_METHODS)]
    cg = core[core[rank_col] <= 10].groupby("sample_id", sort=False)
    out["core_top10_relaxed_condition"] = float(cg["_relaxed"].max().mean()) if len(cg) else 0.0
    return out


def objective(m: Mapping[str, float], protocol: str) -> float:
    if protocol == "strict_comparable":
        return 0.30 * m["top1_relaxed_condition"] + 0.20 * m["top1_strict_condition"] + 0.25 * m["top10_relaxed_condition"] + 0.15 * m["top10_strict_condition"] + 0.10 * m["core_top10_relaxed_condition"]
    return 0.25 * m["top1_relaxed_condition"] + 0.20 * m["top1_strict_condition"] + 0.25 * m["top10_relaxed_condition"] + 0.15 * m["top10_strict_condition"] + 0.10 * m["core_top10_relaxed_condition"]


def weights() -> Iterable[Dict[str, float]]:
    yield {"total_score_raw": 1.0}
    yield {"missing_aware_score": 1.0}
    yield {"strict_comparable_score": 1.0}
    yield {"total_score_raw": 0.6, "missing_aware_score": 0.8, "atmosphere_class_probability": 0.25, "solvent_class_probability": 0.1}
    yield {"strict_comparable_score": 0.9, "atmosphere_known_probability": 0.3, "atmosphere_class_probability": 0.35}
    yield {"temperature_point_score": 1.0, "time_point_score": 1.0, "retrieval_score": 0.5, "multimodal_score": 0.5, "method_expert_score": 0.2}


def score(df: pd.DataFrame, w: Mapping[str, float]) -> pd.Series:
    s = pd.Series(0.0, index=df.index)
    for col, val in w.items():
        s += float(val) * pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0.0)
    return s


def apply(df: pd.DataFrame, w: Mapping[str, float]) -> pd.DataFrame:
    out = df.copy()
    out["condition_calibrated_score_v4"] = score(out, w)
    out = out.sort_values(["sample_id", "condition_calibrated_score_v4", "total_score_raw"], ascending=[True, False, False], kind="mergesort")
    out["condition_rank_calibrated_v4"] = out.groupby("sample_id", sort=False).cumcount() + 1
    return out


def calibrate(val: pd.DataFrame, test: pd.DataFrame, protocol: str, outdir: Path) -> Dict[str, Any]:
    rows = []
    best = None
    for i, w in enumerate(weights()):
        ranked = apply(val, w)
        m = metrics(ranked, protocol, "condition_rank_calibrated_v4")
        obj = objective(m, protocol)
        rows.append({"search_id": i, "objective": obj, **m, "weights": json.dumps(w, sort_keys=True)})
        if best is None or obj > best["objective"]:
            best = {"search_id": i, "objective": obj, "weights": w, "val_metrics": m}
    assert best is not None
    pd.DataFrame(rows).sort_values("objective", ascending=False).to_csv(outdir / f"val_weight_search_{protocol}.csv", index=False)
    write_json(outdir / f"best_weights_{protocol}.json", best)
    test_m = metrics(apply(test, best["weights"]), protocol, "condition_rank_calibrated_v4")
    write_json(outdir / f"test_metrics_{protocol}.json", test_m)
    return {"best": best, "test": test_m}


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate Stage3 condition candidate scores v4 on val and evaluate test.")
    ap.add_argument("--candidate_dir", default="outputs/evaluation/stage3_condition_candidates_v4_20260612")
    ap.add_argument("--output_dir", default="outputs/evaluation/stage3_condition_calibration_v4_20260612")
    args = ap.parse_args()
    cand = Path(args.candidate_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    data = {s: pd.read_csv(cand / f"{s}_condition_candidates.csv") for s in ["train", "val", "test"]}
    missing = calibrate(data["val"], data["test"], "missing_aware", outdir)
    strict = calibrate(data["val"], data["test"], "strict_comparable", outdir)
    chosen = missing["best"]["weights"]
    outputs = {}
    for split, df in data.items():
        ranked = apply(df, chosen)
        ranked.to_csv(outdir / f"{split}_condition_candidates_calibrated.csv", index=False)
        outputs[split] = {
            "missing_aware": metrics(ranked, "missing_aware", "condition_rank_calibrated_v4"),
            "strict_comparable": metrics(ranked, "strict_comparable", "condition_rank_calibrated_v4"),
        }
    summary = {"missing_aware": missing, "strict_comparable": strict, "chosen_for_route": "missing_aware", "outputs": outputs, "config": vars(args)}
    write_json(outdir / "condition_calibration_v4_summary.json", summary)
    report = ["# Stage3 Condition Score Calibration v4", "", "```json", json.dumps(to_builtin(summary), ensure_ascii=False, indent=2), "```"]
    (outdir / "condition_calibration_v4_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
