#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def tail_text(path: Path, n: int = 120) -> str:
    if not path.exists():
        return f"[MISSING LOG] {path}"
    try:
        lines = path.read_text(errors="ignore").splitlines()
        return "\n".join(lines[-n:])
    except Exception as e:
        return f"[READ ERROR] {path}: {e}"


def pick_existing(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default="/Users/wyc/SynPred")
    ap.add_argument("--batch_name", default="batch_500")
    ap.add_argument("--tail_lines", type=int, default=120)
    args = ap.parse_args()

    project_root = Path(args.project_root)
    batch_dir = project_root / "outputs" / "batch_adaptive" / args.batch_name

    master_csv = batch_dir / "master_status_standard.csv"
    if not master_csv.exists():
        master_csv = batch_dir / "master_status.csv"

    if not master_csv.exists():
        raise FileNotFoundError(f"Missing master status csv: {master_csv}")

    df = pd.read_csv(master_csv)

    if "final_case_status" not in df.columns:
        raise ValueError(f"final_case_status column not found in {master_csv}")

    failed = df[df["final_case_status"].astype(str).isin([
        "pipeline_failed",
        "failed",
        "error",
    ])].copy()

    out_dir = batch_dir / "failed_case_inspection"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for _, r in failed.iterrows():
        case_id = str(r.get("case_id", ""))
        pipeline_log = str(r.get("pipeline_log", ""))

        if pipeline_log and pipeline_log != "nan":
            log_path = Path(pipeline_log)
        else:
            log_path = batch_dir / "logs" / f"{case_id}.log"

        log_tail = tail_text(log_path, args.tail_lines)

        # Save individual tail file.
        tail_file = out_dir / f"{case_id}.log_tail.txt"
        tail_file.write_text(log_tail + "\n", encoding="utf-8")

        rows.append({
            "case_id": case_id,
            "final_case_status": r.get("final_case_status", ""),
            "problem_type": r.get("problem_type", ""),
            "recommended_action": r.get("recommended_action", ""),
            "input_poscar": r.get("input_poscar", ""),
            "pipeline_log": str(log_path),
            "log_tail_file": str(tail_file),
        })

    out = pd.DataFrame(rows)
    out_csv = out_dir / "failed_cases_inspection.csv"
    out_md = out_dir / "failed_cases_inspection.md"
    out.to_csv(out_csv, index=False)

    lines = []
    lines.append("# Failed Cases Inspection")
    lines.append("")
    lines.append(f"- Batch name: `{args.batch_name}`")
    lines.append(f"- Failed cases: `{len(out)}`")
    lines.append(f"- Tail lines per case: `{args.tail_lines}`")
    lines.append("")

    if len(out) == 0:
        lines.append("No failed cases.")
    else:
        display_cols = pick_existing(out, [
            "case_id",
            "final_case_status",
            "problem_type",
            "recommended_action",
            "input_poscar",
            "pipeline_log",
            "log_tail_file",
        ])
        lines.append(out[display_cols].to_markdown(index=False))
        lines.append("")

        for _, rr in out.iterrows():
            lines.append(f"## {rr['case_id']}")
            lines.append("")
            lines.append(f"- Log: `{rr['pipeline_log']}`")
            lines.append(f"- Tail file: `{rr['log_tail_file']}`")
            lines.append("")
            lines.append("```text")
            lines.append(Path(rr["log_tail_file"]).read_text(errors="ignore"))
            lines.append("```")
            lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[SAVE]", out_csv)
    print("[SAVE]", out_md)
    print(f"[INFO] n_failed = {len(out)}")


if __name__ == "__main__":
    main()
