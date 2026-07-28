#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from _common import load_config, get_paths, read_status_table, write_table_and_md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = get_paths(cfg)
    df = read_status_table(cfg)

    targets = df[df["final_case_status"].astype(str).isin([
        "pipeline_failed", "needs_stage3_recovery", "needs_condition_reexport"
    ])].copy()

    rows = []
    for _, r in targets.iterrows():
        poscar = Path(str(r.get("input_poscar", "")))
        rows.append({
            "case_id": r["case_id"],
            "input_poscar": str(poscar),
            "poscar_exists": poscar.exists(),
            "recovery_status": "available" if poscar.exists() else "missing_input_poscar",
            "suggested_action": "use_existing_poscar" if poscar.exists() else "recover_from_mp_or_original_archive",
        })

    out_cols = [
        "case_id",
        "input_poscar",
        "poscar_exists",
        "recovery_status",
        "suggested_action",
    ]
    out = pd.DataFrame(rows, columns=out_cols)
    out_dir = paths["reliability_root"] / "recover_missing_structures"
    write_table_and_md(
        out,
        out_dir / "structure_source_recovery_manifest.csv",
        out_dir / "structure_source_recovery_manifest.md",
        "Structure Source Recovery Manifest",
    )

    print("[SAVE]", out_dir / "structure_source_recovery_manifest.csv")


if __name__ == "__main__":
    main()
