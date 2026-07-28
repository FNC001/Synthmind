#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from _common import load_config, get_paths, read_status_table, run_command, write_table_and_md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    dry_run = not args.execute or args.dry_run
    cfg = load_config(args.config)
    paths = get_paths(cfg)
    df = read_status_table(cfg)

    targets = df[df["final_case_status"].astype(str).isin([
        "needs_stage3_recovery",
        "needs_condition_reexport",
    ])].copy()

    out_dir = paths["reliability_root"] / "export_stage3_conditions"
    log_dir = out_dir / "logs"
    rows = []

    for _, r in targets.iterrows():
        case_id = r["case_id"]
        cmd = [
            "python",
            str(paths["pipeline_dir"] / "run_pipeline.py"),
            "--config",
            str(paths["pipeline_config"]),
            "--infer_name",
            case_id,
            "--start_from",
            "run_stage3_flow",
        ]
        log_file = log_dir / f"{case_id}.log"
        rc = run_command(cmd, log_file, dry_run=dry_run)
        rows.append({
            "case_id": case_id,
            "return_code": rc,
            "status": "planned" if dry_run else ("finished" if rc == 0 else "failed"),
            "log_file": str(log_file),
            "command": " ".join(cmd),
        })

    out = pd.DataFrame(rows)
    write_table_and_md(
        out,
        out_dir / "export_stage3_conditions_results.csv",
        out_dir / "export_stage3_conditions_results.md",
        "Export Stage3 Conditions Results",
    )

    print("[SAVE]", out_dir / "export_stage3_conditions_results.csv")


if __name__ == "__main__":
    main()
