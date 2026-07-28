#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
13_auto_recover_and_reaudit.py

Automated recovery + re-audit loop for batch_reliability.

For each case that needs recovery:
1. Determine the recovery action based on status.
2. Re-run the appropriate pipeline steps.
3. Re-audit the case outputs.
4. Update the case_status.json and master_status.

Supports:
  --dry_run   : plan only, no execution
  --execute   : actually run recovery commands
  --limit N   : process at most N cases
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    load_config,
    get_paths,
    mkdir,
    now,
    read_status_table,
    run_command,
    save_json,
    write_table_and_md,
)

# Also import audit function from batch_adaptive.
BATCH_ADAPTIVE_DIR = Path(__file__).resolve().parent.parent.parent / "batch_adaptive"
sys.path.insert(0, str(BATCH_ADAPTIVE_DIR))
from run_adaptive_batch_pipeline import (
    audit_case_outputs,
    build_standard_case_status,
    write_case_status,
)


RECOVERY_ACTIONS = {
    "needs_condition_reexport": "condition_reexport_start_from",
    "needs_stage3_recovery": "stage3_recovery_start_from",
    "needs_stage2_recovery": "stage2_recovery_start_from",
    "needs_route_finalization_recovery": "route_finalizer_start_from",
    "pipeline_failed": "make_infer_split",
}


def main():
    ap = argparse.ArgumentParser(description="Auto-recover failed/degraded cases and re-audit.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--target_statuses", nargs="*", default=None,
                    help="Override which statuses to recover. Default: from config.")
    args = ap.parse_args()

    dry_run = not args.execute or args.dry_run
    cfg = load_config(args.config)
    paths = get_paths(cfg)
    df = read_status_table(cfg)

    execution_cfg = cfg.get("execution", {})
    target_statuses = args.target_statuses or cfg.get("target_statuses", list(RECOVERY_ACTIONS.keys()))

    targets = df[df["final_case_status"].astype(str).isin(target_statuses)].copy()
    if args.limit and args.limit > 0:
        targets = targets.head(args.limit)

    out_dir = paths["reliability_root"] / "auto_recover_and_reaudit"
    log_dir = out_dir / "logs"
    mkdir(out_dir)
    mkdir(log_dir)

    print("=" * 60)
    print("Auto Recovery + Re-audit")
    print(f"  dry_run         = {dry_run}")
    print(f"  target_statuses = {target_statuses}")
    print(f"  n_targets       = {len(targets)}")
    print(f"  out_dir         = {out_dir}")
    print("=" * 60)

    results = []

    for _, row in targets.iterrows():
        case_id = str(row["case_id"])
        status = str(row["final_case_status"])

        # Determine start_from step.
        action_key = RECOVERY_ACTIONS.get(status, "")
        if action_key and action_key in execution_cfg:
            start_from = execution_cfg[action_key]
        elif action_key:
            start_from = action_key
        else:
            start_from = "make_infer_split"

        log_file = log_dir / f"{case_id}_recovery.log"

        cmd = [
            "python",
            str(paths["pipeline_dir"] / "run_pipeline.py"),
            "--config", str(paths["pipeline_config"]),
            "--infer_name", case_id,
            "--start_from", start_from,
        ]

        print(f"\n[RECOVER] {case_id} (status={status}, start_from={start_from})")

        rc = run_command(cmd, log_file, dry_run=dry_run)

        # Re-audit after recovery.
        new_status = {}
        if not dry_run and rc == 0:
            new_status = audit_case_outputs(
                project_root=paths["project_root"],
                case_id=case_id,
                audit_cfg=cfg.get("thresholds", {}),
            )
            new_status["input_poscar"] = str(row.get("input_poscar", ""))
            new_status["pipeline_log"] = str(log_file)
            new_status["recovery_source_status"] = status
            new_status["recovery_start_from"] = start_from

            # Write updated case_status.json.
            adaptive_root = paths["adaptive_root"]
            case_status_path = adaptive_root / "cases" / case_id / "case_status.json"
            if case_status_path.parent.exists():
                write_case_status(case_status_path, new_status)
                standard = build_standard_case_status(new_status)
                write_case_status(case_status_path.with_name("case_status_standard.json"), standard)

        results.append({
            "case_id": case_id,
            "original_status": status,
            "start_from": start_from,
            "return_code": rc,
            "recovery_status": "planned" if dry_run else ("success" if rc == 0 else "failed"),
            "new_final_status": new_status.get("final_case_status", ""),
            "log_file": str(log_file),
            "command": " ".join(cmd),
        })

    # Save results.
    result_df = pd.DataFrame(results)
    write_table_and_md(
        result_df,
        out_dir / "auto_recovery_results.csv",
        out_dir / "auto_recovery_results.md",
        "Auto Recovery + Re-audit Results",
        cols=["case_id", "original_status", "start_from", "return_code", "recovery_status", "new_final_status"],
    )

    # If not dry_run, rebuild master_status with updated statuses.
    if not dry_run and len(results) > 0:
        _rebuild_master_status(paths, cfg)

    summary = {
        "timestamp": now(),
        "dry_run": dry_run,
        "n_targets": len(targets),
        "n_success": sum(1 for r in results if r["recovery_status"] == "success"),
        "n_failed": sum(1 for r in results if r["recovery_status"] == "failed"),
        "n_planned": sum(1 for r in results if r["recovery_status"] == "planned"),
    }
    save_json(out_dir / "auto_recovery_summary.json", summary)

    print("\n" + "=" * 60)
    print(f"[DONE] {summary['n_success']} recovered, {summary['n_failed']} failed, {summary['n_planned']} planned")
    print(f"[SAVE] {out_dir / 'auto_recovery_results.csv'}")
    print("=" * 60)


def _rebuild_master_status(paths: dict, cfg: dict):
    """Rebuild master_status.csv from individual case_status.json files."""
    adaptive_root = paths["adaptive_root"]
    cases_dir = adaptive_root / "cases"

    if not cases_dir.exists():
        return

    all_status = []
    for case_dir in sorted(cases_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        status_path = case_dir / "case_status.json"
        if status_path.exists():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
                all_status.append(status)
            except Exception:
                pass

    if not all_status:
        return

    master_df = pd.DataFrame(all_status)
    master_csv = adaptive_root / "master_status.csv"
    master_json = adaptive_root / "master_status.json"

    standard_list = [build_standard_case_status(x) for x in all_status]
    standard_df = pd.DataFrame(standard_list)
    standard_csv = adaptive_root / "master_status_standard.csv"
    standard_json = adaptive_root / "master_status_standard.json"

    master_df.to_csv(master_csv, index=False)
    master_json.write_text(json.dumps(all_status, ensure_ascii=False, indent=2), encoding="utf-8")
    standard_df.to_csv(standard_csv, index=False)
    standard_json.write_text(json.dumps(standard_list, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[REBUILD] master_status updated: {len(all_status)} cases")


if __name__ == "__main__":
    main()
