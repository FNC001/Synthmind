#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def pick_existing(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def write_md(df: pd.DataFrame, path: Path, title: str, note: str = "", cols: list[str] | None = None) -> None:
    lines = [f"# {title}", ""]
    if note:
        lines.append(note)
        lines.append("")
    lines.append(f"- Rows: `{len(df)}`")
    lines.append("")
    if len(df) == 0:
        lines.append("No records.")
    else:
        if cols:
            use_cols = pick_existing(df, cols)
            lines.append(df[use_cols].to_markdown(index=False))
        else:
            lines.append(df.to_markdown(index=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default="/Users/wyc/SynPred")
    ap.add_argument("--batch_name", default="batch_500")
    ap.add_argument("--top_n_per_case", type=int, default=5)
    args = ap.parse_args()

    project_root = Path(args.project_root)
    batch_dir = project_root / "outputs" / "batch_adaptive" / args.batch_name

    master_csv = batch_dir / "master_status_standard.csv"
    routes_csv = batch_dir / "batch_recommended_routes_topn_review_refined.csv"
    recovery_plan_csv = batch_dir / "recovery_plan" / "recovery_plan.csv"

    if not master_csv.exists():
        master_csv = batch_dir / "master_status.csv"
    if not routes_csv.exists():
        routes_csv = batch_dir / "batch_recommended_routes_topn.csv"

    master = pd.read_csv(master_csv) if master_csv.exists() else pd.DataFrame()
    routes = pd.read_csv(routes_csv) if routes_csv.exists() else pd.DataFrame()
    recovery_plan = read_csv_if_exists(recovery_plan_csv)

    out_dir = batch_dir / "manual_review"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Case-level manual review cases.
    manual_case_ids = set()

    if len(recovery_plan) > 0 and "recovery_module" in recovery_plan.columns:
        tmp = recovery_plan[
            recovery_plan["recovery_module"].astype(str).isin([
                "manual_or_rule_recovery",
                "inspect_pipeline_log",
            ])
        ].copy()
        if "case_id" in tmp.columns:
            manual_case_ids.update(tmp["case_id"].dropna().astype(str).tolist())

    if len(master) > 0:
        if "refined_case_status" in master.columns:
            tmp = master[
                master["refined_case_status"].astype(str).isin([
                    "needs_manual_check",
                    "needs_manual_check_refined",
                ])
            ].copy()
            if "case_id" in tmp.columns:
                manual_case_ids.update(tmp["case_id"].dropna().astype(str).tolist())

        if "final_case_status" in master.columns:
            tmp = master[
                master["final_case_status"].astype(str).isin([
                    "pipeline_failed",
                    "needs_manual_or_rule_recovery",
                ])
            ].copy()
            if "case_id" in tmp.columns:
                manual_case_ids.update(tmp["case_id"].dropna().astype(str).tolist())

    manual_case_ids = sorted(manual_case_ids)

    if len(master) > 0 and "case_id" in master.columns:
        manual_cases = master[master["case_id"].astype(str).isin(manual_case_ids)].copy()
    else:
        manual_cases = pd.DataFrame({"case_id": manual_case_ids})

    manual_cases_csv = out_dir / "manual_review_cases.csv"
    manual_cases_md = out_dir / "manual_review_cases.md"

    case_cols = [
        "case_id",
        "final_case_status",
        "refined_case_status",
        "refined_case_reason",
        "condition_support_score",
        "top1_condition_support_score",
        "top1_final_score",
        "top1_score",
        "top1_precursor_set",
        "problem_type",
        "recommended_action",
        "input_poscar",
        "pipeline_log",
    ]

    manual_cases.to_csv(manual_cases_csv, index=False)
    write_md(
        manual_cases,
        manual_cases_md,
        "Manual Review Cases",
        "Cases selected by recovery plan, refined status, or pipeline failure.",
        case_cols,
    )

    # Route-level manual review routes.
    manual_routes = pd.DataFrame()

    if len(routes) > 0 and "case_id" in routes.columns:
        route_mask = routes["case_id"].astype(str).isin(manual_case_ids)

        if "route_refined_status" in routes.columns:
            route_mask |= routes["route_refined_status"].astype(str).isin([
                "route_needs_manual_check",
            ])

        if "review_severity" in routes.columns:
            route_mask |= routes["review_severity"].astype(str).isin([
                "needs_manual_check",
            ])

        manual_routes = routes[route_mask].copy()

    # Sort manual routes if score exists.
    score_col = None
    for c in [
        "final_recommendation_score",
        "stage35_v43_safe_strict_score",
        "real_stage3_condition_reference_support_score",
    ]:
        if c in manual_routes.columns:
            score_col = c
            break

    if score_col and len(manual_routes) > 0:
        manual_routes[score_col] = pd.to_numeric(manual_routes[score_col], errors="coerce")
        manual_routes = manual_routes.sort_values(["case_id", score_col], ascending=[True, False])

    manual_routes_csv = out_dir / "manual_review_routes.csv"
    manual_routes_md = out_dir / "manual_review_routes.md"

    route_cols = [
        "case_id",
        "case_status",
        "case_refined_status",
        "refined_case_status",
        "route_refined_status",
        "route_refined_reason",
        "review_severity",
        "precursor_set",
        "final_recommendation_score",
        "stage35_v43_safe_strict_score",
        "real_stage3_condition_reference_support_score",
        "condition_distribution_support_score",
        "final_recommendation_status",
        "input_poscar",
    ]

    manual_routes.to_csv(manual_routes_csv, index=False)
    write_md(
        manual_routes,
        manual_routes_md,
        "Manual Review Routes",
        "Routes selected from manual-review cases or route-level manual-check flags.",
        route_cols,
    )

    # Summary.
    summary_rows = [
        {"item": "n_manual_cases", "value": len(manual_cases)},
        {"item": "n_manual_routes", "value": len(manual_routes)},
        {"item": "n_unique_manual_case_ids", "value": len(manual_case_ids)},
    ]
    summary = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "manual_review_summary.csv"
    summary_md = out_dir / "manual_review_summary.md"
    summary.to_csv(summary_csv, index=False)
    write_md(summary, summary_md, "Manual Review Summary")

    print("[SAVE]", manual_cases_csv)
    print("[SAVE]", manual_cases_md)
    print("[SAVE]", manual_routes_csv)
    print("[SAVE]", manual_routes_md)
    print("[SAVE]", summary_csv)
    print("[SAVE]", summary_md)
    print(f"[INFO] n_manual_cases = {len(manual_cases)}")
    print(f"[INFO] n_manual_routes = {len(manual_routes)}")


if __name__ == "__main__":
    main()
