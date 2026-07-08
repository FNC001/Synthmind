#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

DEFAULT_OUTPUT_DIR = Path("outputs/auto_improve/synpred_auto_v1_20260612/stage2_stage3_coupling")
CORE_METHODS = {"solid_state", "solution", "melt_arc"}


def abs_path(project_root: Path, path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = project_root / p
    return p


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def pct(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{100.0 * float(value):.2f}%"
    except (TypeError, ValueError):
        return str(value)


def read_route_candidates(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    header = pd.read_csv(path, nrows=0)
    cols = set(header.columns)
    wanted = [
        "sample_index",
        "sample_id",
        "id",
        "reaction_method",
        "precursor_rank",
        "condition_rank_calibrated_v3",
        "condition_rank",
        "route_rank_raw",
        "exact",
        "jaccard",
        "f1",
        "precursor_exact_if_eval",
        "precursor_jaccard_if_eval",
        "relaxed_condition_hit_if_eval",
        "strict_condition_hit_if_eval",
        "relaxed_route_hit_if_eval",
        "strict_route_hit_if_eval",
        "usable_relaxed_route_hit_if_eval",
        "contains_open_generated_precursor",
        "contains_repair_precursor",
    ]
    usecols = [c for c in wanted if c in cols]
    df = pd.read_csv(path, usecols=usecols)
    if "sample_id" not in df.columns:
        df["sample_id"] = df["id"] if "id" in df.columns else df["sample_index"].astype(str)
    if "condition_rank_calibrated_v3" not in df.columns and "condition_rank" in df.columns:
        df["condition_rank_calibrated_v3"] = df["condition_rank"]
    return df


def bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    return df[col].fillna(False).astype(bool)


def numeric_series(df: pd.DataFrame, col: str, default: float = 10**9) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def summarize_samples(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_precursor_rank"] = numeric_series(df, "precursor_rank")
    df["_condition_rank"] = numeric_series(df, "condition_rank_calibrated_v3")
    df["_route_rank"] = numeric_series(df, "route_rank_raw")
    precursor_exact = bool_series(df, "precursor_exact_if_eval")
    if "precursor_exact_if_eval" not in df.columns and "exact" in df.columns:
        precursor_exact = bool_series(df, "exact")
    df["_precursor_exact"] = precursor_exact
    df["_precursor_jaccard"] = numeric_series(df, "precursor_jaccard_if_eval", default=0.0)
    if "precursor_jaccard_if_eval" not in df.columns and "jaccard" in df.columns:
        df["_precursor_jaccard"] = numeric_series(df, "jaccard", default=0.0)
    df["_relaxed_condition_hit"] = bool_series(df, "relaxed_condition_hit_if_eval")
    df["_strict_condition_hit"] = bool_series(df, "strict_condition_hit_if_eval")
    df["_relaxed_route_hit"] = bool_series(df, "relaxed_route_hit_if_eval")
    df["_strict_route_hit"] = bool_series(df, "strict_route_hit_if_eval")
    df["_usable_relaxed_route_hit"] = bool_series(df, "usable_relaxed_route_hit_if_eval")
    df["_open"] = bool_series(df, "contains_open_generated_precursor")
    df["_repair"] = bool_series(df, "contains_repair_precursor")

    rows: List[Dict[str, Any]] = []
    for sample_id, g in df.groupby("sample_id", sort=False):
        reaction_method = str(g["reaction_method"].iloc[0]) if "reaction_method" in g.columns else "unknown"
        stage2_top1 = bool(((g["_precursor_rank"] <= 1) & g["_precursor_exact"]).any())
        stage2_top10 = bool(((g["_precursor_rank"] <= 10) & g["_precursor_exact"]).any())
        stage2_top20 = bool(((g["_precursor_rank"] <= 20) & g["_precursor_exact"]).any())
        stage2_any = bool(g["_precursor_exact"].any())
        stage2_jaccard_ge_05 = bool((g["_precursor_jaccard"] >= 0.5).any())
        stage3_top1 = bool(((g["_condition_rank"] <= 1) & g["_relaxed_condition_hit"]).any())
        stage3_top10 = bool(((g["_condition_rank"] <= 10) & g["_relaxed_condition_hit"]).any())
        stage3_any = bool(g["_relaxed_condition_hit"].any())
        route_top1 = bool(((g["_route_rank"] <= 1) & g["_relaxed_route_hit"]).any())
        route_top10 = bool(((g["_route_rank"] <= 10) & g["_relaxed_route_hit"]).any())
        route_any = bool(g["_relaxed_route_hit"].any())
        if stage2_top1 and stage3_top1:
            bucket = "stage2_top1_exact__stage3_top1_relaxed_hit"
        elif stage2_top1 and not stage3_top1:
            bucket = "stage2_top1_exact__stage3_top1_relaxed_miss"
        elif stage2_top10 and stage3_top10:
            bucket = "stage2_top10_exact__stage3_top10_relaxed_hit"
        elif not stage2_top10 and stage3_any:
            bucket = "stage2_top10_no_exact__stage3_condition_hit_exists"
        elif stage2_jaccard_ge_05 and stage3_any:
            bucket = "stage2_jaccard_ge_0_5__stage3_condition_hit"
        elif not stage2_any:
            bucket = "stage2_fails_completely"
        elif not stage3_any:
            bucket = "stage3_condition_fails_completely"
        else:
            bucket = "combined_or_stage35_ranking_mismatch"
        rows.append(
            {
                "sample_id": sample_id,
                "reaction_method": reaction_method,
                "is_core_method": reaction_method in CORE_METHODS,
                "stage2_top1_exact": stage2_top1,
                "stage2_top10_exact": stage2_top10,
                "stage2_top20_exact": stage2_top20,
                "stage2_any_exact": stage2_any,
                "stage2_jaccard_ge_0_5": stage2_jaccard_ge_05,
                "stage3_top1_relaxed_hit": stage3_top1,
                "stage3_top10_relaxed_hit": stage3_top10,
                "stage3_any_relaxed_hit": stage3_any,
                "route_top1_relaxed_hit": route_top1,
                "route_top10_relaxed_hit": route_top10,
                "route_any_relaxed_hit": route_any,
                "contains_open_generated_precursor": bool(g["_open"].any()),
                "contains_repair_precursor": bool(g["_repair"].any()),
                "bucket": bucket,
                "recommended_next_action": recommend_action(bucket, route_top1, route_top10),
            }
        )
    return pd.DataFrame(rows)


def recommend_action(bucket: str, route_top1: bool, route_top10: bool) -> str:
    if route_top1:
        return "already_top1_relaxed_route_hit"
    if bucket == "stage2_top10_no_exact__stage3_condition_hit_exists":
        return "prioritize_stage2_candidate_generation_or_ranking"
    if bucket == "stage2_fails_completely":
        return "prioritize_stage2_oov_cleanup_and_candidate_generation"
    if bucket in {"stage2_top1_exact__stage3_top1_relaxed_miss", "stage3_condition_fails_completely"}:
        return "prioritize_stage3_condition_candidate_improvement"
    if bucket in {"stage2_top1_exact__stage3_top1_relaxed_hit", "stage2_top10_exact__stage3_top10_relaxed_hit"}:
        return "prioritize_stage35_ranking_or_meta_calibration"
    if route_top10:
        return "prioritize_stage35_top1_pairwise_ranking"
    return "inspect_distribution_mismatch"


def count_table(df: pd.DataFrame, group_cols: List[str]) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    grouped = df.groupby(group_cols, dropna=False).agg(
        n_samples=("sample_id", "count"),
        route_top1_relaxed_rate=("route_top1_relaxed_hit", "mean"),
        route_top10_relaxed_rate=("route_top10_relaxed_hit", "mean"),
        stage2_top10_exact_rate=("stage2_top10_exact", "mean"),
        stage3_top10_relaxed_rate=("stage3_top10_relaxed_hit", "mean"),
    )
    return grouped.reset_index().to_dict(orient="records")


def build_diagnosis(project_root: Path, route_candidates: Path) -> Dict[str, Any]:
    route_df = read_route_candidates(route_candidates)
    sample_df = summarize_samples(route_df)
    bucket_counts = sample_df["bucket"].value_counts().rename_axis("bucket").reset_index(name="n_samples")
    bucket_counts["fraction"] = bucket_counts["n_samples"] / max(len(sample_df), 1)
    action_counts = sample_df["recommended_next_action"].value_counts().rename_axis("action").reset_index(name="n_samples")
    action_counts["fraction"] = action_counts["n_samples"] / max(len(sample_df), 1)

    decomposition = {
        "route_candidates": str(route_candidates),
        "n_route_candidate_rows": int(len(route_df)),
        "n_samples": int(len(sample_df)),
        "overall_rates": {
            "stage2_top1_exact": float(sample_df["stage2_top1_exact"].mean()),
            "stage2_top10_exact": float(sample_df["stage2_top10_exact"].mean()),
            "stage2_top20_exact": float(sample_df["stage2_top20_exact"].mean()),
            "stage2_any_exact": float(sample_df["stage2_any_exact"].mean()),
            "stage3_top1_relaxed_hit": float(sample_df["stage3_top1_relaxed_hit"].mean()),
            "stage3_top10_relaxed_hit": float(sample_df["stage3_top10_relaxed_hit"].mean()),
            "stage3_any_relaxed_hit": float(sample_df["stage3_any_relaxed_hit"].mean()),
            "route_top1_relaxed_hit": float(sample_df["route_top1_relaxed_hit"].mean()),
            "route_top10_relaxed_hit": float(sample_df["route_top10_relaxed_hit"].mean()),
            "route_any_relaxed_hit": float(sample_df["route_any_relaxed_hit"].mean()),
            "contains_open_generated_precursor": float(sample_df["contains_open_generated_precursor"].mean()),
            "contains_repair_precursor": float(sample_df["contains_repair_precursor"].mean()),
        },
        "bucket_counts": bucket_counts.to_dict(orient="records"),
        "recommended_action_counts": action_counts.to_dict(orient="records"),
        "by_reaction_method": count_table(sample_df, ["reaction_method", "bucket"]),
        "by_core_method": count_table(sample_df, ["is_core_method", "bucket"]),
        "by_open_generated": count_table(sample_df, ["contains_open_generated_precursor", "bucket"]),
        "by_repair": count_table(sample_df, ["contains_repair_precursor", "bucket"]),
        "oov_note": "No explicit OOV flag was present in route candidates; open-generated and repair flags are reported as proxy distribution slices.",
    }
    top_action = action_counts.iloc[0]["action"] if not action_counts.empty else "none"
    decomposition["primary_bottleneck"] = infer_primary_bottleneck(top_action, decomposition["overall_rates"])
    return {"decomposition": decomposition, "sample_buckets": sample_df.to_dict(orient="records")}


def infer_primary_bottleneck(top_action: str, rates: Dict[str, float]) -> str:
    if rates.get("stage2_top10_exact", 0.0) < 0.55:
        return "stage2_precursor_candidates"
    if rates.get("stage3_top10_relaxed_hit", 0.0) < 0.60:
        return "stage3_condition_candidates"
    if rates.get("route_top10_relaxed_hit", 0.0) > rates.get("route_top1_relaxed_hit", 0.0) + 0.08:
        return "stage35_route_ranking"
    if "stage2" in top_action:
        return "stage2_precursor_candidates"
    if "stage3" in top_action:
        return "stage3_condition_candidates"
    if "stage35" in top_action:
        return "stage35_route_ranking"
    return "distribution_mismatch_or_label_quality"


def render_markdown(d: Dict[str, Any]) -> str:
    dec = d["decomposition"]
    rates = dec["overall_rates"]
    lines = [
        "# Stage2-Stage3 Coupling Diagnosis",
        "",
        f"Route candidates: `{dec['route_candidates']}`",
        "",
        "## Overall",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in (
        "stage2_top1_exact",
        "stage2_top10_exact",
        "stage2_top20_exact",
        "stage3_top1_relaxed_hit",
        "stage3_top10_relaxed_hit",
        "route_top1_relaxed_hit",
        "route_top10_relaxed_hit",
        "route_any_relaxed_hit",
    ):
        lines.append(f"| {key} | {pct(rates.get(key))} |")
    lines.extend(["", f"Primary bottleneck: **{dec['primary_bottleneck']}**", "", "## Failure Buckets", "", "| bucket | n | fraction |", "|---|---:|---:|"])
    for row in dec["bucket_counts"]:
        lines.append(f"| {row['bucket']} | {row['n_samples']} | {pct(row['fraction'])} |")
    lines.extend(["", "## Recommended Actions", "", "| action | n | fraction |", "|---|---:|---:|"])
    for row in dec["recommended_action_counts"]:
        lines.append(f"| {row['action']} | {row['n_samples']} | {pct(row['fraction'])} |")
    lines.extend(["", f"Note: {dec['oov_note']}", ""])
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Decompose full-route failures into Stage2, Stage3, Stage35, and mismatch buckets.")
    ap.add_argument("--project_root", default=".", help="Repository root.")
    ap.add_argument(
        "--route_candidates",
        default="outputs/evaluation/stage35_route_candidates_v3_final_20260612/test_route_candidates.csv",
        help="Stage35 route candidate CSV to diagnose.",
    )
    ap.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    route_candidates = abs_path(project_root, args.route_candidates)
    output_dir = abs_path(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnosis = build_diagnosis(project_root, route_candidates)
    json_path = output_dir / "stage2_stage3_coupling_diagnosis.json"
    md_path = output_dir / "stage2_stage3_coupling_report.md"
    csv_path = output_dir / "stage2_stage3_coupling_sample_buckets.csv"
    write_json(json_path, diagnosis["decomposition"])
    pd.DataFrame(diagnosis["sample_buckets"]).to_csv(csv_path, index=False)
    md_path.write_text(render_markdown(diagnosis), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "report": str(md_path), "sample_buckets": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
