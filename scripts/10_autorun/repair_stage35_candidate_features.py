#!/usr/bin/env python3
"""Repair inference-safe Stage35 candidate chemistry features.

This fixes split-schema drift where one split has safe precursor chemistry
features and another split lacks them, causing zero-filled element coverage,
candidate size, and compatibility scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
EVAL_DIR = REPO_ROOT / "scripts" / "06_eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from stage35_feature_repair_utils import repair_precursor_chem_features, repair_summary  # noqa: E402


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(payload), indent=2, sort_keys=True), encoding="utf-8")


def numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def recompute_v3_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    missing = numeric(out, "missing_element_count")
    extra = numeric(out, "extra_element_count")
    out["chemistry_check_status"] = np.where((missing <= 0) & (extra <= 0), "ok", "failed")
    out["precursor_condition_compatibility_score"] = numeric(out, "element_coverage")
    if "precursor_confidence" in out.columns:
        out["precursor_confidence"] = numeric(out, "precursor_score_norm") * (
            1.0 - 0.1 * missing
        ).clip(lower=0.0)
    if "condition_confidence" in out.columns:
        out["condition_confidence"] = numeric(out, "condition_score_norm")
    out["reaction_method_prior_score"] = out.get("reaction_method", pd.Series("", index=out.index)).astype(str).isin(CORE_METHODS).astype(float) * 0.2
    if "route_total_score_raw" in out.columns:
        top_precursors = max(float(numeric(out, "precursor_rank", 20.0).max()), 20.0)
        top_conditions = max(float(numeric(out, "condition_rank", 20.0).max()), 20.0)
        out["route_total_score_raw"] = (
            1.2 * numeric(out, "precursor_score_norm")
            + 1.1 * numeric(out, "condition_score_norm")
            + 0.4 * numeric(out, "precursor_condition_compatibility_score")
            + 0.2 * numeric(out, "reaction_method_prior_score")
            - 0.18 * np.log1p(numeric(out, "precursor_rank", top_precursors))
            - 0.12 * np.log1p(numeric(out, "condition_rank", top_conditions))
            - 0.15 * numeric(out, "contains_open_generated_precursor")
            - 0.10 * numeric(out, "contains_repair_precursor")
            - numeric(out, "open_generated_penalty")
            - numeric(out, "repair_penalty")
        )
        out = out.sort_values(["sample_id", "route_total_score_raw"], ascending=[True, False], kind="mergesort")
        out["route_rank_raw"] = out.groupby("sample_id", sort=False).cumcount() + 1
    return out


def route_metrics(df: pd.DataFrame, rank_col: str = "route_rank_raw") -> dict[str, float]:
    out: dict[str, float] = {"num_samples": float(df["sample_id"].astype(str).nunique()), "num_candidates": float(len(df))}
    for k in (1, 10, 200):
        top = df[pd.to_numeric(df[rank_col], errors="coerce").fillna(10**9) <= k]
        grouped = top.groupby(top["sample_id"].astype(str), sort=False)
        if len(grouped) == 0:
            continue
        for col, name in [
            ("relaxed_route_hit_if_eval", "missing"),
            ("strict_route_hit_if_eval", "strict"),
            ("usable_relaxed_route_hit_if_eval", "usable"),
        ]:
            if col in top.columns:
                out[f"{name}_top{k}_relaxed_route"] = float(pd.to_numeric(grouped[col].max(), errors="coerce").fillna(0).mean())
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--report_md", default="")
    parser.add_argument("--repair_zeroish_columns", type=int, default=1)
    parser.add_argument("--recompute_scores", type=int, default=1)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    summary_json = Path(args.summary_json)
    report_md = Path(args.report_md) if args.report_md else summary_json.with_suffix(".md")

    before = pd.read_csv(input_csv)
    repaired = repair_precursor_chem_features(before, repair_zeroish_columns=bool(args.repair_zeroish_columns))
    if args.recompute_scores:
        repaired = recompute_v3_scores(repaired)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    repaired.to_csv(output_csv, index=False)

    payload = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "repair_zeroish_columns": bool(args.repair_zeroish_columns),
        "recompute_scores": bool(args.recompute_scores),
        "feature_repair_summary": repair_summary(before, repaired),
        "route_metrics_after": route_metrics(repaired) if "sample_id" in repaired.columns and "route_rank_raw" in repaired.columns else {},
    }
    write_json(summary_json, payload)
    lines = [
        "# Stage35 Candidate Feature Repair Report",
        "",
        f"- Input: `{input_csv}`",
        f"- Output: `{output_csv}`",
        f"- Rows: {len(repaired):,}",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(to_builtin(payload["feature_repair_summary"]), indent=2, sort_keys=True),
        "```",
        "",
        "## Route Metrics After Repair",
        "",
        "```json",
        json.dumps(to_builtin(payload["route_metrics_after"]), indent=2, sort_keys=True),
        "```",
    ]
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(to_builtin(payload["route_metrics_after"]), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
