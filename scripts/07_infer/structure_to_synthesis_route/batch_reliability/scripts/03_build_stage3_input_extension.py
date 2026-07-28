#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from _common import load_config, get_paths, write_table_and_md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = get_paths(cfg)

    audit_csv = paths["reliability_root"] / "stage3_feature_source_audit" / "stage3_feature_source_audit.csv"
    if not audit_csv.exists():
        raise FileNotFoundError(f"Missing audit csv: {audit_csv}")

    audit = pd.read_csv(audit_csv)

    required_cols = ["has_stage3_hybrid_csv", "has_stage3_conditioned_x"]
    missing_cols = [c for c in required_cols if c not in audit.columns]

    rows = []

    if len(audit) == 0:
        need_extension = audit.copy()
    elif missing_cols:
        raise ValueError(f"Audit table missing required columns: {missing_cols}; file={audit_csv}")
    else:
        need_extension = audit[
            (~audit["has_stage3_hybrid_csv"].astype(bool)) |
            (~audit["has_stage3_conditioned_x"].astype(bool))
        ].copy()
    for _, r in need_extension.iterrows():
        case_id = r["case_id"]
        rows.append({
            "case_id": case_id,
            "need_stage3_hybrid_regeneration": not bool(r["has_stage3_hybrid_csv"]),
            "need_conditioned_x_regeneration": not bool(r["has_stage3_conditioned_x"]),
            "source_infer_root": r["infer_root"],
            "planned_start_from": "build_stage3_features" if not bool(r["has_stage3_hybrid_csv"]) else "build_stage3_conditioned_table",
            "status": "planned",
        })

    manifest_cols = [
        "case_id",
        "need_stage3_hybrid_regeneration",
        "need_conditioned_x_regeneration",
        "source_infer_root",
        "planned_start_from",
        "status",
    ]
    manifest = pd.DataFrame(rows, columns=manifest_cols)

    out_dir = paths["reliability_root"] / "stage3_input_extension"
    write_table_and_md(
        manifest,
        out_dir / "stage3_input_extension_manifest.csv",
        out_dir / "stage3_input_extension_manifest.md",
        "Stage3 Input Extension Manifest",
    )

    print("[SAVE]", out_dir / "stage3_input_extension_manifest.csv")


if __name__ == "__main__":
    main()
