#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from _common import load_config, get_paths


def read_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = get_paths(cfg)
    root = paths["reliability_root"]

    audit = read_if_exists(root / "audit_batch" / "abnormal_cases.csv")
    compact_watchlist = read_if_exists(root / "audit_batch" / "compact_watchlist_imported.csv")
    stage3_audit = read_if_exists(root / "stage3_feature_source_audit" / "stage3_feature_source_audit.csv")
    ext = read_if_exists(root / "stage3_input_extension" / "stage3_input_extension_manifest.csv")
    poscar_val = read_if_exists(root / "validate_recovered_poscars" / "poscar_validation_audit.csv")
    regen = read_if_exists(root / "regenerate_stage3_npz" / "regenerate_stage3_npz_results.csv")
    export = read_if_exists(root / "export_stage3_conditions" / "export_stage3_conditions_results.csv")
    normalized = read_if_exists(root / "normalize_recovered_candidates" / "normalized_recovered_candidates.csv")
    merge = read_if_exists(root / "merge_recovered_results" / "merged_recovered_results.csv")
    closure = read_if_exists(root / "merge_recovered_results" / "recovery_closure_status.csv")
    clip = read_if_exists(root / "clipping_diagnostic" / "clipping_diagnostic_cases.csv")

    report_dir = root / "gap_closure_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = [
        {"section": "abnormal_cases", "n_rows": len(audit)},
        {"section": "compact_watchlist_imported", "n_rows": len(compact_watchlist)},
        {"section": "stage3_feature_source_audit", "n_rows": len(stage3_audit)},
        {"section": "stage3_input_extension", "n_rows": len(ext)},
        {"section": "poscar_validation", "n_rows": len(poscar_val)},
        {"section": "regenerate_stage3_npz", "n_rows": len(regen)},
        {"section": "export_stage3_conditions", "n_rows": len(export)},
        {"section": "normalize_recovered_candidates", "n_rows": len(normalized)},
        {"section": "merge_recovered_results", "n_rows": len(merge)},
        {"section": "recovery_closure_status", "n_rows": len(closure)},
        {"section": "clipping_diagnostic", "n_rows": len(clip)},
    ]
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(report_dir / "gap_closure_summary.csv", index=False)

    md = []
    md.append("# Batch Gap Closure Report")
    md.append("")
    md.append(f"- Batch: `{paths['batch_name']}`")
    md.append("")
    md.append("## Summary")
    md.append("")
    md.append(summary.to_markdown(index=False))
    md.append("")

    def add_section(title, df):
        md.append(f"## {title}")
        md.append("")
        if len(df) == 0:
            md.append("No records.")
        else:
            md.append(df.head(50).to_markdown(index=False))
        md.append("")

    add_section("Abnormal Cases", audit)
    add_section("Imported Compact Watchlist", compact_watchlist)
    add_section("Stage3 Feature Source Audit", stage3_audit)
    add_section("Stage3 Input Extension", ext)
    add_section("POSCAR Validation", poscar_val)
    add_section("Regenerate Stage3 NPZ Results", regen)
    add_section("Export Stage3 Conditions Results", export)
    add_section("Normalized Recovered Candidates", normalized)
    add_section("Merged Recovered Results", merge)
    add_section("Recovery Closure Status", closure)
    add_section("Clipping Diagnostic Cases", clip)

    (report_dir / "gap_closure_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("[SAVE]", report_dir / "gap_closure_summary.csv")
    print("[SAVE]", report_dir / "gap_closure_report.md")


if __name__ == "__main__":
    main()
