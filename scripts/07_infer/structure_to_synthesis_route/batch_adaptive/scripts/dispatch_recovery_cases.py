#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def save_subset(df: pd.DataFrame, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    md_path = path.with_suffix(".md")
    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- Number of cases: `{len(df)}`")
    lines.append("")

    if len(df) == 0:
        lines.append("No cases.")
    else:
        cols = [
            "case_id",
            "final_case_status",
            "refined_case_status",
            "refined_case_reason",
            "refined_condition_support_score",
            "refined_final_score",
            "problem_type",
            "recommended_action",
            "n_stage2_candidates",
            "n_stage3_conditions",
            "n_final_routes",
            "top1_condition_support_score",
            "input_poscar",
        ]
        cols = [c for c in cols if c in df.columns]
        lines.append(df[cols].to_markdown(index=False))

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[SAVE]", path)
    print("[SAVE]", md_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_root", default="/Users/wyc/SynPred/outputs/batch_adaptive")
    ap.add_argument("--batch_name", default="batch_001")
    args = ap.parse_args()

    batch_dir = Path(args.batch_root) / args.batch_name
    master_csv = batch_dir / "master_status.csv"

    if not master_csv.exists():
        raise FileNotFoundError(f"Missing master_status.csv: {master_csv}")

    df = pd.read_csv(master_csv)

    dispatch_dir = batch_dir / "recovery_dispatch"
    dispatch_dir.mkdir(parents=True, exist_ok=True)

    status = df["final_case_status"].fillna("")
    refined = df["refined_case_status"].fillna("") if "refined_case_status" in df.columns else status
    problem = df["problem_type"].fillna("")

    stage2_recovery = df[
        (status == "needs_stage2_recovery") |
        (problem == "stage2_no_candidate")
    ].copy()

    stage3_recovery = df[
        (status == "needs_stage3_recovery") |
        (problem == "stage3_no_condition")
    ].copy()

    condition_reexport = df[
        (status == "needs_condition_reexport") |
        (problem == "condition_support_too_low") |
        (problem == "condition_support_moderate")
    ].copy()

    route_finalizer_recovery = df[
        (status == "needs_route_finalization_recovery") |
        (problem == "no_final_route")
    ].copy()

    manual_or_rule_recovery = df[
        (status == "needs_manual_or_rule_recovery") |
        (problem == "major_audit_warning")
    ].copy()

    pipeline_failed = df[
        (status == "pipeline_failed") |
        (problem == "pipeline_failed")
    ].copy()

    review_only = df[
        status == "review_required_only"
    ].copy()

    high_confidence_review = df[
        refined == "high_confidence_review"
    ].copy()

    medium_confidence_review = df[
        refined == "medium_confidence_review"
    ].copy()

    low_confidence_review = df[
        refined == "low_confidence_review"
    ].copy()

    needs_manual_check_refined = df[
        refined == "needs_manual_check"
    ].copy()

    passed = df[
        (status == "pass") | refined.isin(["pass_high_confidence", "pass_medium_confidence", "pass_low_confidence"])
    ].copy()

    save_subset(stage2_recovery, dispatch_dir / "needs_stage2_recovery.csv", "Needs Stage2 Recovery")
    save_subset(stage3_recovery, dispatch_dir / "needs_stage3_recovery.csv", "Needs Stage3 Recovery")
    save_subset(condition_reexport, dispatch_dir / "needs_condition_reexport.csv", "Needs Condition Re-export")
    save_subset(route_finalizer_recovery, dispatch_dir / "needs_route_finalizer_recovery.csv", "Needs Route Finalizer Recovery")
    save_subset(manual_or_rule_recovery, dispatch_dir / "needs_manual_or_rule_recovery.csv", "Needs Manual or Rule Recovery")
    save_subset(pipeline_failed, dispatch_dir / "pipeline_failed.csv", "Pipeline Failed")
    save_subset(review_only, dispatch_dir / "review_required_only.csv", "Review Required Only")
    save_subset(high_confidence_review, dispatch_dir / "high_confidence_review.csv", "High Confidence Review")
    save_subset(medium_confidence_review, dispatch_dir / "medium_confidence_review.csv", "Medium Confidence Review")
    save_subset(low_confidence_review, dispatch_dir / "low_confidence_review.csv", "Low Confidence Review")
    save_subset(needs_manual_check_refined, dispatch_dir / "needs_manual_check_refined.csv", "Needs Manual Check Refined")
    save_subset(passed, dispatch_dir / "pass.csv", "Pass")

    summary = pd.DataFrame([
        {"bucket": "needs_stage2_recovery", "n_cases": len(stage2_recovery)},
        {"bucket": "needs_stage3_recovery", "n_cases": len(stage3_recovery)},
        {"bucket": "needs_condition_reexport", "n_cases": len(condition_reexport)},
        {"bucket": "needs_route_finalizer_recovery", "n_cases": len(route_finalizer_recovery)},
        {"bucket": "needs_manual_or_rule_recovery", "n_cases": len(manual_or_rule_recovery)},
        {"bucket": "pipeline_failed", "n_cases": len(pipeline_failed)},
        {"bucket": "review_required_only", "n_cases": len(review_only)},
        {"bucket": "high_confidence_review", "n_cases": len(high_confidence_review)},
        {"bucket": "medium_confidence_review", "n_cases": len(medium_confidence_review)},
        {"bucket": "low_confidence_review", "n_cases": len(low_confidence_review)},
        {"bucket": "needs_manual_check_refined", "n_cases": len(needs_manual_check_refined)},
        {"bucket": "pass", "n_cases": len(passed)},
    ])

    summary_csv = dispatch_dir / "recovery_dispatch_summary.csv"
    summary_md = dispatch_dir / "recovery_dispatch_summary.md"

    summary.to_csv(summary_csv, index=False)
    summary_md.write_text(
        "# Recovery Dispatch Summary\n\n" + summary.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )

    print("[SAVE]", summary_csv)
    print("[SAVE]", summary_md)
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
