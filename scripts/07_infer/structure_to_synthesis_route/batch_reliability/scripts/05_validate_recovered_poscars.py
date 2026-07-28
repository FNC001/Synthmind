#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from _common import load_config, get_paths, write_table_and_md


def validate_poscar(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing_file"
    try:
        lines = path.read_text(errors="ignore").splitlines()
        if len(lines) < 8:
            return False, "too_few_lines"
        scale = float(lines[1].strip().split()[0])
        # lattice lines
        for i in [2, 3, 4]:
            vals = lines[i].split()
            if len(vals) < 3:
                return False, f"bad_lattice_line_{i+1}"
            [float(x) for x in vals[:3]]
        return True, "ok"
    except Exception as e:
        return False, f"parse_error:{e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = get_paths(cfg)

    src_csv = paths["reliability_root"] / "recover_missing_structures" / "structure_source_recovery_manifest.csv"
    if not src_csv.exists():
        raise FileNotFoundError(src_csv)

    src = pd.read_csv(src_csv)

    required_cols = ["case_id", "input_poscar"]
    missing_cols = [c for c in required_cols if c not in src.columns]
    if missing_cols and len(src) > 0:
        raise ValueError(f"Source recovery manifest missing columns: {missing_cols}; file={src_csv}")

    rows = []
    if len(src) > 0:
        for _, r in src.iterrows():
            poscar = Path(str(r["input_poscar"]))
            ok, reason = validate_poscar(poscar)
            rows.append({
                "case_id": r["case_id"],
                "input_poscar": str(poscar),
                "poscar_valid": ok,
                "validation_reason": reason,
            })

    out_cols = [
        "case_id",
        "input_poscar",
        "poscar_valid",
        "validation_reason",
    ]
    out = pd.DataFrame(rows, columns=out_cols)
    out_dir = paths["reliability_root"] / "validate_recovered_poscars"
    write_table_and_md(
        out,
        out_dir / "poscar_validation_audit.csv",
        out_dir / "poscar_validation_audit.md",
        "POSCAR Validation Audit",
    )

    print("[SAVE]", out_dir / "poscar_validation_audit.csv")


if __name__ == "__main__":
    main()
