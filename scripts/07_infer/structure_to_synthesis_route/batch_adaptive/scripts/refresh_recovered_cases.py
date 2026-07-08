#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import pandas as pd


def run(cmd: list[str], log_file: Path, dry_run: bool) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("[CMD] " + " ".join(map(str, cmd)) + "\n")
        f.write(f"[DRY_RUN] {dry_run}\n\n")
        f.flush()

        if dry_run:
            return 0

        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        f.write(f"\n[RETURN_CODE] {proc.returncode}\n")
        return proc.returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default="/Users/wyc/SynPred")
    ap.add_argument("--batch_root", default="/Users/wyc/SynPred/outputs/batch_adaptive")
    ap.add_argument("--batch_name", default="batch_001")
    ap.add_argument("--config", required=True)
    ap.add_argument("--plan_csv", default="")
    ap.add_argument("--only_modules", default="")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    dry_run = True
    if args.execute:
        dry_run = False
    if args.dry_run:
        dry_run = True

    project_root = Path(args.project_root)
    batch_dir = Path(args.batch_root) / args.batch_name

    if args.plan_csv:
        plan_csv = Path(args.plan_csv)
    else:
        plan_csv = batch_dir / "recovery_plan" / "recovery_plan.csv"

    if not plan_csv.exists():
        raise FileNotFoundError(f"Missing recovery plan: {plan_csv}")

    plan = pd.read_csv(plan_csv)

    if args.only_modules.strip():
        mods = {x.strip() for x in args.only_modules.split(",") if x.strip()}
        plan = plan[plan["recovery_module"].isin(mods)].copy()

    case_ids = plan["case_id"].dropna().astype(str).unique().tolist()

    out_dir = batch_dir / "refresh_recovered"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for i, case_id in enumerate(case_ids, start=1):
        cmd = [
            "python",
            str(project_root / "scripts/07_infer/structure_to_synthesis_route/batch_adaptive/run_adaptive_batch_pipeline.py"),
            "--config",
            str(Path(args.config)),
            "--mode",
            "audit_only",
            "--case_id_filter",
            case_id,
            "--force",
        ]

        log_file = out_dir / f"{case_id}.refresh.log"
        rc = run(cmd, log_file, dry_run=dry_run)

        rows.append({
            "case_id": case_id,
            "refresh_status": "planned" if dry_run else ("finished" if rc == 0 else "failed"),
            "return_code": rc,
            "log_file": str(log_file),
            "command": " ".join(cmd),
        })

    df = pd.DataFrame(rows)
    out_csv = out_dir / "refresh_recovered_cases_results.csv"
    out_md = out_dir / "refresh_recovered_cases_results.md"

    df.to_csv(out_csv, index=False)

    lines = []
    lines.append("# Refresh Recovered Cases Results")
    lines.append("")
    lines.append(f"- Batch: `{args.batch_name}`")
    lines.append(f"- Recovery plan: `{plan_csv}`")
    lines.append(f"- Dry run: `{dry_run}`")
    lines.append(f"- Number of cases: `{len(df)}`")
    lines.append("")
    if len(df) > 0:
        cols = ["case_id", "refresh_status", "return_code", "log_file"]
        lines.append(df[cols].to_markdown(index=False))
    else:
        lines.append("No recovered cases to refresh.")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[SAVE]", out_csv)
    print("[SAVE]", out_md)
    if len(df) > 0:
        print(df.to_string(index=False))
    else:
        print("No recovered cases to refresh.")


if __name__ == "__main__":
    main()
