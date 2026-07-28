#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    root = Path(args.project_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    v32_root = root / "outputs/benchmark_100_v32_mp_metadata_aware_stage3_alignment"

    stage3_csv = v32_root / "input_from_v30/stage3_condition_candidates_v30.csv"
    meta_csv = v32_root / "mp_metadata_table_v32/v32_mp_metadata_table.csv"

    if not stage3_csv.exists():
        raise SystemExit(f"[ERROR] missing Stage3 candidates: {stage3_csv}")

    if not meta_csv.exists():
        raise SystemExit(f"[ERROR] missing MP metadata table: {meta_csv}")

    stage3 = pd.read_csv(stage3_csv)
    meta = pd.read_csv(meta_csv)

    if "case_id" not in stage3.columns:
        raise SystemExit("[ERROR] Stage3 candidates missing case_id column")

    if "mp_id" not in meta.columns:
        raise SystemExit("[ERROR] metadata table missing mp_id column")

    # V30 Stage3 candidate case_id is mp-* id.
    stage3["mp_id"] = stage3["case_id"].astype(str)

    keep_meta_cols = [
        "mp_id",
        "formula",
        "elements",
        "chemsys",
        "n_elements",
        "mp_family",
        "crystal_system",
        "spacegroup_symbol",
        "spacegroup_number",
        "metadata_source_type",
    ]
    keep_meta_cols = [c for c in keep_meta_cols if c in meta.columns]

    merged = stage3.merge(
        meta[keep_meta_cols],
        on="mp_id",
        how="left",
        suffixes=("", "_mpmeta"),
    )

    merged["has_mp_metadata"] = merged["formula"].fillna("").astype(str).str.len() > 0
    merged["has_mp_elements"] = merged["elements"].fillna("").astype(str).str.len() > 0
    merged["metadata_alignment_status"] = merged["has_mp_metadata"].map(
        lambda x: "metadata_attached" if x else "missing_metadata"
    )

    # Write full table
    out_csv = out / "v32_stage3_candidates_with_mp_metadata.csv"
    out_preview = out / "v32_stage3_candidates_with_mp_metadata_preview.md"
    out_summary = out / "v32_stage3_candidates_with_mp_metadata_summary.json"

    merged.to_csv(out_csv, index=False)

    preview_cols = [
        "case_id",
        "mp_id",
        "formula",
        "elements",
        "mp_family",
        "candidate_temperature_c",
        "candidate_time_h",
        "candidate_condition_score",
        "candidate_condition_rank",
        "candidate_source",
        "stage3_model",
        "stage2_model",
        "source_split",
        "has_mp_metadata",
    ]
    preview_cols = [c for c in preview_cols if c in merged.columns]

    merged[preview_cols].head(80).to_markdown(out_preview, index=False)

    summary = {
        "status": "pass" if int(merged["has_mp_metadata"].sum()) > 0 else "blocked",
        "input_stage3_candidates": str(stage3_csv),
        "input_mp_metadata": str(meta_csv),
        "output_csv": str(out_csv),
        "n_stage3_candidate_rows": int(len(merged)),
        "n_stage3_unique_mp_ids": int(merged["mp_id"].nunique()),
        "n_rows_with_mp_metadata": int(merged["has_mp_metadata"].sum()),
        "n_rows_missing_mp_metadata": int((~merged["has_mp_metadata"]).sum()),
        "metadata_coverage_ratio": float(merged["has_mp_metadata"].mean()) if len(merged) else 0.0,
        "n_unique_mp_ids_with_metadata": int(merged.loc[merged["has_mp_metadata"], "mp_id"].nunique()),
        "candidate_source_counts": merged["candidate_source"].value_counts(dropna=False).to_dict()
        if "candidate_source" in merged.columns else {},
        "mp_family_counts": merged["mp_family"].value_counts(dropna=False).to_dict()
        if "mp_family" in merged.columns else {},
        "interpretation": "V32 attaches formula/elements/family metadata to real Stage3 MDN/Flow condition candidates, enabling composition-aware external alignment.",
    }

    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[SAVE]", out_csv)
    print("[SAVE]", out_preview)
    print("[SAVE]", out_summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
