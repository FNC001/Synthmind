#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from metrics_registry import (  # noqa: E402
    BASELINE_THRESHOLDS,
    DEFAULT_OUTPUT_DIR,
    build_registry,
    number,
    pct,
    read_json,
    write_json,
)


WEAK_METHODS = {"hydro_solvothermal", "precipitation", "flux_molten_salt", "other"}
CORE_METHODS = {"solid_state", "solution", "melt_arc"}


def abs_path(project_root: Path, path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = project_root / p
    return p


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def metric_table_from_records(records: List[Dict[str, Any]], key_col: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in records:
        key = str(row.get(key_col, "unknown"))
        out[key] = row
    return out


def collect_candidate_distribution(path: Path, rank_col: str = "calibrated_rank_v5") -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    usecols = None
    header = pd.read_csv(path, nrows=0)
    cols = set(header.columns)
    wanted = [
        "sample_index",
        "sample_id",
        "id",
        "reaction_method",
        "candidate_source",
        "candidate_source_mix",
        "exact",
        "f1",
        "jaccard",
        rank_col,
        "rank",
        "contains_open_generated_precursor",
        "contains_repair_precursor",
        "chemistry_check_status",
        "precursor_exact_if_eval",
        "precursor_f1_if_eval",
        "precursor_jaccard_if_eval",
    ]
    usecols = [c for c in wanted if c in cols]
    df = pd.read_csv(path, usecols=usecols)
    if df.empty:
        return {"exists": True, "path": str(path), "rows": 0}
    sample_col = "sample_id" if "sample_id" in df.columns else "id" if "id" in df.columns else "sample_index"
    rank = rank_col if rank_col in df.columns else "rank" if "rank" in df.columns else None
    source_cols = [c for c in ("candidate_source", "candidate_source_mix") if c in df.columns]
    source_text = pd.Series([""] * len(df))
    for col in source_cols:
        source_text = source_text.str.cat(df[col].fillna("").astype(str), sep="|")
    is_open = (
        df["contains_open_generated_precursor"].astype(bool)
        if "contains_open_generated_precursor" in df.columns
        else source_text.str.contains("open", case=False, regex=False)
    )
    is_repair = (
        df["contains_repair_precursor"].astype(bool)
        if "contains_repair_precursor" in df.columns
        else source_text.str.contains("repair", case=False, regex=False)
    )
    if "exact" in df.columns:
        exact = df["exact"].astype(bool)
    elif "precursor_exact_if_eval" in df.columns:
        exact = df["precursor_exact_if_eval"].fillna(False).astype(bool)
    else:
        exact = pd.Series([False] * len(df))
    if "f1" in df.columns:
        f1 = pd.to_numeric(df["f1"], errors="coerce")
    elif "precursor_f1_if_eval" in df.columns:
        f1 = pd.to_numeric(df["precursor_f1_if_eval"], errors="coerce")
    else:
        f1 = pd.Series([0.0] * len(df))
    if "jaccard" in df.columns:
        jaccard = pd.to_numeric(df["jaccard"], errors="coerce")
    elif "precursor_jaccard_if_eval" in df.columns:
        jaccard = pd.to_numeric(df["precursor_jaccard_if_eval"], errors="coerce")
    else:
        jaccard = pd.Series([0.0] * len(df))
    source_mix = {}
    if "candidate_source" in df.columns:
        source_mix = df["candidate_source"].fillna("unknown").value_counts().to_dict()
    by_sample = df.assign(_open=is_open, _repair=is_repair, _exact=exact, _f1=f1, _jaccard=jaccard).groupby(sample_col)
    sample_summary = by_sample.agg(
        any_open=("_open", "max"),
        any_repair=("_repair", "max"),
        best_f1=("_f1", "max"),
        best_jaccard=("_jaccard", "max"),
        any_exact=("_exact", "max"),
    )
    top1_exact = None
    top10_exact = None
    top20_exact = None
    if rank:
        rank_values = pd.to_numeric(df[rank], errors="coerce")
        tmp = df.assign(_rank=rank_values, _exact=exact)
        top1_exact = tmp[tmp["_rank"] <= 1].groupby(sample_col)["_exact"].max().mean()
        top10_exact = tmp[tmp["_rank"] <= 10].groupby(sample_col)["_exact"].max().mean()
        top20_exact = tmp[tmp["_rank"] <= 20].groupby(sample_col)["_exact"].max().mean()
    return {
        "exists": True,
        "path": str(path),
        "rows": int(len(df)),
        "samples": int(sample_summary.shape[0]),
        "candidate_open_generated_rate": float(is_open.mean()),
        "candidate_repair_rate": float(is_repair.mean()),
        "sample_any_open_generated_rate": float(sample_summary["any_open"].mean()),
        "sample_any_repair_rate": float(sample_summary["any_repair"].mean()),
        "mean_best_f1": float(sample_summary["best_f1"].mean()),
        "mean_best_jaccard": float(sample_summary["best_jaccard"].mean()),
        "top1_exact": None if pd.isna(top1_exact) else float(top1_exact),
        "top10_exact": None if pd.isna(top10_exact) else float(top10_exact),
        "top20_exact": None if pd.isna(top20_exact) else float(top20_exact),
        "candidate_source_mix": source_mix,
    }


def assess_weak_methods(by_method: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    weak = []
    for row in by_method:
        method = str(row.get("reaction_method", ""))
        top10 = number(row.get("top10_exact"))
        top500 = number(row.get("top500_exact"))
        top1 = number(row.get("top1_exact"))
        flagged = method in WEAK_METHODS or (top10 is not None and top10 < 0.55) or (top500 is not None and top500 < 0.75)
        if flagged:
            weak.append(
                {
                    "reaction_method": method,
                    "n_samples": row.get("n_samples"),
                    "top1_exact": top1,
                    "top10_exact": top10,
                    "top200_exact": number(row.get("top200_exact")),
                    "top500_exact": top500,
                    "reason": "requested_weak_method" if method in WEAK_METHODS else "low_top10_or_top500",
                }
            )
    weak.sort(key=lambda r: (r.get("top10_exact") is None, r.get("top10_exact") or 9.0))
    return weak


def build_diagnosis(project_root: Path, output_dir: Path) -> Dict[str, Any]:
    registry = build_registry(project_root, output_dir, include_experiments=False)
    records = registry.records
    baselines = records["baselines"]

    all_metrics = baselines.get("stage2_v5_all_test", {}).get("metrics", {})
    core_metrics = baselines.get("stage2_core_calibrated_test", {}).get("metrics", {})
    oof_metrics = baselines.get("stage2_train_oof_v4_approx", {}).get("metrics", {})
    by_method = baselines.get("stage2_v5_by_reaction_method", {}).get("records", [])
    by_failure = baselines.get("stage2_v5_by_failure_type", {}).get("records", [])
    by_source = baselines.get("stage2_v5_by_candidate_source", {}).get("records", [])

    val_dist = collect_candidate_distribution(abs_path(project_root, "outputs/evaluation/stage2_candidate_pool_v5_20260610/val_candidate_sets_repaired.csv"), rank_col="rank")
    test_dist = collect_candidate_distribution(
        abs_path(project_root, "outputs/evaluation/stage2_score_calibration_v5_20260610/test_candidate_sets_calibrated_v5.csv")
    )
    train_oof_dist = collect_candidate_distribution(
        abs_path(project_root, "outputs/evaluation/stage2_train_oof_top20_candidates_v4_20260612/train_oof_top20_precursor_candidates.csv"),
        rank_col="precursor_rank",
    )

    v4_failure_summary_path = abs_path(
        project_root, "outputs/evaluation/stage2_v5_failure_decomposition_20260610/failure_decomposition_summary.json"
    )
    v4_failure_summary = read_json(v4_failure_summary_path) if v4_failure_summary_path.exists() else {}

    stage2_thresholds = BASELINE_THRESHOLDS["stage2"]
    all_pass = {
        "top1_gt_v5": (number(all_metrics.get("top1_exact")) or 0.0) > stage2_thresholds["all_top1_exact"],
        "top10_ge_v5": (number(all_metrics.get("top10_exact")) or 0.0) >= stage2_thresholds["all_top10_exact"],
        "top500_ge_v5": (number(all_metrics.get("top500_exact")) or 0.0) >= stage2_thresholds["all_top500_exact"],
    }
    core_pass = {
        "top1_ge_core_final": (number(core_metrics.get("top1_exact")) or 0.0) >= stage2_thresholds["core_top1_exact"],
        "top10_ge_core_target": (number(core_metrics.get("top10_exact")) or 0.0) >= stage2_thresholds["core_top10_exact"],
        "top500_ge_core_target": (number(core_metrics.get("top500_exact")) or 0.0) >= stage2_thresholds["core_top500_exact"],
    }

    oof_top1 = number(oof_metrics.get("top1_exact"))
    test_top1 = number(all_metrics.get("top1_exact"))
    oof_gap = None if oof_top1 is None or test_top1 is None else test_top1 - oof_top1
    oof_recommend = bool(
        oof_gap is not None
        and oof_gap > 0.08
        and (number(oof_metrics.get("top10_exact")) or 0.0) < (number(all_metrics.get("top10_exact")) or 0.0) - 0.06
    )

    weak_methods = assess_weak_methods(by_method)
    core_plateau = bool(
        abs((number(core_metrics.get("top500_exact")) or 0.0) - stage2_thresholds["core_top500_exact"]) < 0.005
        and (number(core_metrics.get("top1_exact")) or 0.0) < stage2_thresholds["core_top1_exact"]
    )

    diagnosis = {
        "registry": records,
        "stage2_metrics": {
            "all_method_test": all_metrics,
            "core_method_test": core_metrics,
            "train_oof_v4_approx": oof_metrics,
            "all_method_threshold_checks": all_pass,
            "core_method_threshold_checks": core_pass,
        },
        "distribution": {
            "train_oof": train_oof_dist,
            "val": val_dist,
            "test": test_dist,
            "train_oof_vs_test": {
                "top1_exact_gap_test_minus_oof": oof_gap,
                "top10_exact_gap_test_minus_oof": (
                    (number(all_metrics.get("top10_exact")) or 0.0) - (number(oof_metrics.get("top10_exact")) or 0.0)
                    if oof_metrics
                    else None
                ),
                "mean_f1_gap_test_minus_oof": (
                    (number(test_dist.get("mean_best_f1")) or 0.0) - (number(oof_metrics.get("mean_best_f1_top20")) or 0.0)
                    if oof_metrics
                    else None
                ),
            },
        },
        "by_reaction_method": by_method,
        "weak_methods": weak_methods,
        "core_method_plateau": core_plateau,
        "failure_types": {
            "v5_calibrated_by_original_failure_type": metric_table_from_records(by_failure, "failure_type"),
            "v5_by_candidate_source": metric_table_from_records(by_source, "candidate_source"),
            "v4_origin_failure_decomposition": v4_failure_summary,
        },
        "recommendations": {
            "run_expensive_stage2_kfold_neural_oof": oof_recommend,
            "why": [
                "Current train OOF is explicitly an approximation, not complete neural K-fold OOF.",
                f"Train OOF top1 is {pct(oof_top1)} versus v5 calibrated test top1 {pct(test_top1)}.",
                "Open-generated and repair rates are high in train OOF, so real K-fold neural scores may improve calibration.",
            ],
            "priority_actions": [],
        },
    }
    actions = diagnosis["recommendations"]["priority_actions"]
    if weak_methods:
        actions.append("Improve weak-method candidate generation/ranking for hydro_solvothermal, precipitation, flux_molten_salt, and other.")
    if not all_pass["top1_gt_v5"]:
        actions.append("Run top1/top10 Stage2 score calibration or hard-negative reranking before changing default Stage2.")
    if core_plateau:
        actions.append("Treat core methods as near coverage plateau; focus on top1/top10 ranking rather than top500 coverage.")
    if oof_recommend:
        actions.append("Run K=3 smoke true neural OOF first, then K=5 if it improves OOF top1/top10 without hurting top20.")
    else:
        actions.append("Keep v4 fold-safe OOF approximation until a cheaper Stage2 calibration/core-selector experiment passes validation.")
    return diagnosis


def render_markdown(d: Dict[str, Any]) -> str:
    all_m = d["stage2_metrics"]["all_method_test"]
    core_m = d["stage2_metrics"]["core_method_test"]
    oof_m = d["stage2_metrics"]["train_oof_v4_approx"]
    lines = [
        "# Stage2 Bottleneck Diagnosis",
        "",
        "## Current TopK",
        "",
        "| slice | top1 | top10 | top20 | top200 | top500 | best F1@500 | best Jaccard@500 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| all-method v5 calibrated test | {pct(all_m.get('top1_exact'))} | {pct(all_m.get('top10_exact'))} | {pct(all_m.get('top20_exact'))} | {pct(all_m.get('top200_exact'))} | {pct(all_m.get('top500_exact'))} | {pct(all_m.get('top500_best_f1'))} | {pct(all_m.get('top500_best_jaccard'))} |",
        f"| core calibrated test | {pct(core_m.get('top1_exact'))} | {pct(core_m.get('top10_exact'))} | {pct(core_m.get('top20_exact'))} | {pct(core_m.get('top200_exact'))} | {pct(core_m.get('top500_exact'))} | {pct(core_m.get('top500_best_f1'))} | {pct(core_m.get('top500_best_jaccard'))} |",
        f"| train OOF v4 approximation | {pct(oof_m.get('top1_exact'))} | {pct(oof_m.get('top10_exact'))} | {pct(oof_m.get('top20_exact'))} | n/a | n/a | {pct(oof_m.get('mean_best_f1_top20'))} | {pct(oof_m.get('mean_best_jaccard_top20'))} |",
        "",
        "## OOF / Val / Test Distribution",
        "",
        "| split | samples | top1 exact | top10 exact | top20 exact | mean best F1 | mean best Jaccard | open-generated | repair |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("train_oof", "val", "test"):
        item = d["distribution"][name]
        top1 = item.get("top1_exact")
        top10 = item.get("top10_exact")
        top20 = item.get("top20_exact")
        open_rate = item.get("candidate_open_generated_rate")
        repair_rate = item.get("candidate_repair_rate")
        if name == "train_oof":
            top1 = oof_m.get("top1_exact", top1)
            top10 = oof_m.get("top10_exact", top10)
            top20 = oof_m.get("top20_exact", top20)
            open_rate = oof_m.get("open_generated_rate", open_rate)
            repair_rate = oof_m.get("repair_rate", repair_rate)
        lines.append(
            f"| {name} | {item.get('samples', 'n/a')} | {pct(top1)} | {pct(top10)} | {pct(top20)} | {pct(item.get('mean_best_f1') or oof_m.get('mean_best_f1_top20') if name == 'train_oof' else item.get('mean_best_f1'))} | {pct(item.get('mean_best_jaccard') or oof_m.get('mean_best_jaccard_top20') if name == 'train_oof' else item.get('mean_best_jaccard'))} | {pct(open_rate)} | {pct(repair_rate)} |"
        )
    lines.extend(["", "## Weak Reaction Methods", "", "| method | n | top1 | top10 | top200 | top500 | reason |", "|---|---:|---:|---:|---:|---:|---|"])
    for row in d["weak_methods"]:
        lines.append(
            f"| {row.get('reaction_method')} | {row.get('n_samples')} | {pct(row.get('top1_exact'))} | {pct(row.get('top10_exact'))} | {pct(row.get('top200_exact'))} | {pct(row.get('top500_exact'))} | {row.get('reason')} |"
        )
    lines.extend(["", "## Decision", ""])
    rec = d["recommendations"]
    lines.append(f"- Run expensive Stage2 K-fold neural OOF now: **{rec['run_expensive_stage2_kfold_neural_oof']}**")
    for action in rec["priority_actions"]:
        lines.append(f"- {action}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose Stage2 precursor-candidate bottlenecks for SynPred auto-improvement.")
    ap.add_argument("--project_root", default=".", help="Repository root.")
    ap.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR / "stage2_diagnosis"), help="Diagnosis output directory.")
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    output_dir = abs_path(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnosis = build_diagnosis(project_root, output_dir.parent)
    json_path = output_dir / "stage2_bottleneck_diagnosis.json"
    md_path = output_dir / "stage2_bottleneck_diagnosis.md"
    write_json(json_path, diagnosis)
    md_path.write_text(render_markdown(diagnosis), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "report": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
