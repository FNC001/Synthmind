#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from _common import load_config, get_paths, run_command, write_table_and_md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    dry_run = not args.execute or args.dry_run
    cfg = load_config(args.config)
    paths = get_paths(cfg)

    manifest_csv = paths["reliability_root"] / "stage3_input_extension" / "stage3_input_extension_manifest.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(manifest_csv)

    manifest = pd.read_csv(manifest_csv)
    out_dir = paths["reliability_root"] / "regenerate_stage3_npz"
    log_dir = out_dir / "logs"
    rows = []

    for _, r in manifest.iterrows():
        case_id = r["case_id"]
        start_from = r.get("planned_start_from", "build_stage3_features")

        cmd = [
            "python",
            str(paths["pipeline_dir"] / "run_pipeline.py"),
            "--config",
            str(paths["pipeline_config"]),
            "--infer_name",
            case_id,
            "--start_from",
            str(start_from),
        ]

        log_file = log_dir / f"{case_id}.log"
        rc = run_command(cmd, log_file, dry_run=dry_run)

        rows.append({
            "case_id": case_id,
            "start_from": start_from,
            "return_code": rc,
            "status": "planned" if dry_run else ("finished" if rc == 0 else "failed"),
            "log_file": str(log_file),
            "command": " ".join(cmd),
        })

    out = pd.DataFrame(rows)
    write_table_and_md(
        out,
        out_dir / "regenerate_stage3_npz_results.csv",
        out_dir / "regenerate_stage3_npz_results.md",
        "Regenerate Stage3 NPZ / Features Results",
    )

    print("[SAVE]", out_dir / "regenerate_stage3_npz_results.csv")


if __name__ == "__main__":
    main()
