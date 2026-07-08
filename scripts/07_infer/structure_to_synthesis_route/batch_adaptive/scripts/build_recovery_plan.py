#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


RECOVERY_STATUSES = {
    "pipeline_failed",
    "needs_stage2_recovery",
    "needs_stage3_recovery",
    "needs_condition_reexport",
    "needs_route_finalization_recovery",
    "needs_manual_or_rule_recovery",
}


def infer_recovery_module(row) -> tuple[str, str, str]:
    """
    Return:
      recovery_module
      recovery_reason
      suggested_command_type
    """
    status = str(row.get("final_case_status", ""))
    refined = str(row.get("refined_case_status", ""))
    problem = str(row.get("problem_type", ""))

    has_stage2 = bool(row.get("has_stage2_candidates", False))
    has_stage3 = bool(row.get("has_stage3_conditions", False))
    has_final = bool(row.get("has_final_recommendation", False))

    if status == "pipeline_failed":
        return (
            "inspect_pipeline_log",
            "Layer-1 pipeline failed before producing reliable outputs.",
            "manual_debug",
        )

    if status == "needs_stage2_recovery" or problem == "stage2_no_candidate":
        return (
            "stage2_recovery",
            "No precursor candidates were produced by Stage2.",
            "rerun_stage2_with_stronger_sampling",
        )

    if status == "needs_stage3_recovery" or problem == "stage3_no_condition":
        if has_stage2 and not has_stage3:
            return (
                "stage3_gap_recovery",
                "Stage2 candidates exist but Stage3 condition candidates are missing.",
                "run_stage3_feature_audit_then_regenerate",
            )
        return (
            "stage3_gap_recovery",
            "Stage3 recovery requested.",
            "run_stage3_feature_audit_then_regenerate",
        )

    if status == "needs_condition_reexport" or "condition_support" in problem:
        return (
            "condition_reexport",
            "Stage3 conditions exist but condition support is weak.",
            "rerun_stage3_condition_export",
        )

    if status == "needs_route_finalization_recovery" or problem == "no_final_route":
        if has_stage2 and has_stage3 and not has_final:
            return (
                "route_finalizer_recovery",
                "Stage2 and Stage3 outputs exist but no final recommended route was produced.",
                "rerun_route_finalizer",
            )
        return (
            "route_finalizer_recovery",
            "Final route table is missing or empty.",
            "rerun_route_finalizer",
        )

    if status == "needs_manual_or_rule_recovery" or refined == "needs_manual_check":
        return (
            "manual_or_rule_recovery",
            "Major audit warning or suspicious rule-level issue.",
            "manual_or_rule_review",
        )

    return (
        "none",
        "No heavy recovery required.",
        "none",
    )


def build_command(project_root: str, batch_name: str, row, command_type: str) -> str:
    case_id = row["case_id"]

    if command_type == "manual_debug":
        log = row.get("pipeline_log", "")
        return f'echo "[MANUAL] inspect pipeline log for {case_id}: {log}"'

    if command_type == "rerun_stage2_with_stronger_sampling":
        return (
            f'echo "[TODO Stage2 recovery] {case_id}: '
            f'increase n_samples/top_k, enable composition fallback, retrieval, ExtraTrees baseline"'
        )

    if command_type == "run_stage3_feature_audit_then_regenerate":
        return (
            f'echo "[TODO Stage3 recovery] {case_id}: '
            f'run feature-source audit -> regenerate Stage3 NPZ -> re-export conditions"'
        )

    if command_type == "rerun_stage3_condition_export":
        return (
            f'echo "[TODO condition re-export] {case_id}: '
            f'rerun Stage3 condition export with more samples / clipping diagnostic"'
        )

    if command_type == "rerun_route_finalizer":
        return (
            f'echo "[TODO route finalizer recovery] {case_id}: '
            f'rerun summarize_routes / finalizer from existing Stage2+Stage3 outputs"'
        )

    if command_type == "manual_or_rule_review":
        return (
            f'echo "[MANUAL/RULE] {case_id}: inspect precursor/condition/audit warning"'
        )

    return f'echo "[NOOP] {case_id}"'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default="/Users/wyc/SynPred")
    ap.add_argument("--batch_root", default="/Users/wyc/SynPred/outputs/batch_adaptive")
    ap.add_argument("--batch_name", default="batch_001")
    args = ap.parse_args()

    batch_dir = Path(args.batch_root) / args.batch_name
    standard_csv = batch_dir / "master_status_standard.csv"
    full_csv = batch_dir / "master_status.csv"

    if standard_csv.exists():
        df = pd.read_csv(standard_csv)
        source_csv = standard_csv
    elif full_csv.exists():
        df = pd.read_csv(full_csv)
        source_csv = full_csv
    else:
        raise FileNotFoundError(
            f"Missing both {standard_csv} and {full_csv}"
        )

    out_dir = batch_dir / "recovery_plan"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    commands = []

    for _, row in df.iterrows():
        status = str(row.get("final_case_status", ""))
        refined = str(row.get("refined_case_status", ""))

        needs_recovery = (
            status in RECOVERY_STATUSES
            or refined in ["needs_manual_check"]
        )

        if not needs_recovery:
            continue

        module, reason, cmd_type = infer_recovery_module(row)
        cmd = build_command(args.project_root, args.batch_name, row, cmd_type)

        rec = row.to_dict()
        rec["recovery_module"] = module
        rec["recovery_reason"] = reason
        rec["suggested_command_type"] = cmd_type
        rec["suggested_command"] = cmd

        rows.append(rec)
        commands.append(cmd)

    plan = pd.DataFrame(rows)

    out_csv = out_dir / "recovery_plan.csv"
    out_md = out_dir / "recovery_plan.md"
    out_sh = out_dir / "recovery_commands.sh"
    summary_csv = out_dir / "recovery_plan_summary.csv"
    summary_md = out_dir / "recovery_plan_summary.md"

    plan.to_csv(out_csv, index=False)

    if len(plan) > 0:
        summary = (
            plan.groupby(["recovery_module", "suggested_command_type"])
            .size()
            .reset_index(name="n_cases")
        )
    else:
        summary = pd.DataFrame(
            columns=["recovery_module", "suggested_command_type", "n_cases"]
        )

    summary.to_csv(summary_csv, index=False)

    md = []
    md.append("# Recovery Plan")
    md.append("")
    md.append(f"- Batch name: `{args.batch_name}`")
    md.append(f"- Source status table: `{source_csv}`")
    md.append(f"- Number of recovery cases: `{len(plan)}`")
    md.append("")

    if len(plan) == 0:
        md.append("No cases require heavy recovery.")
    else:
        cols = [
            "case_id",
            "final_case_status",
            "refined_case_status",
            "problem_type",
            "recovery_module",
            "recovery_reason",
            "suggested_command_type",
            "condition_support_score",
            "top1_final_score",
            "top1_precursor_set",
        ]
        cols = [c for c in cols if c in plan.columns]
        md.append(plan[cols].to_markdown(index=False))

    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    smd = []
    smd.append("# Recovery Plan Summary")
    smd.append("")
    if len(summary) == 0:
        smd.append("No recovery cases.")
    else:
        smd.append(summary.to_markdown(index=False))
    summary_md.write_text("\n".join(smd) + "\n", encoding="utf-8")

    sh = []
    sh.append("#!/usr/bin/env bash")
    sh.append("set -euo pipefail")
    sh.append("")
    sh.append(f'echo "Recovery commands for batch: {args.batch_name}"')
    sh.append("")

    if len(commands) == 0:
        sh.append('echo "No recovery commands generated."')
    else:
        for i, cmd in enumerate(commands, start=1):
            sh.append("")
            sh.append(f'echo "===== RECOVERY TASK {i}/{len(commands)} ====="')
            sh.append(cmd)

    out_sh.write_text("\n".join(sh) + "\n", encoding="utf-8")
    out_sh.chmod(0o755)

    print("[SAVE]", out_csv)
    print("[SAVE]", out_md)
    print("[SAVE]", summary_csv)
    print("[SAVE]", summary_md)
    print("[SAVE]", out_sh)
    print()
    if len(summary) > 0:
        print(summary.to_string(index=False))
    else:
        print("No recovery cases.")


if __name__ == "__main__":
    main()
