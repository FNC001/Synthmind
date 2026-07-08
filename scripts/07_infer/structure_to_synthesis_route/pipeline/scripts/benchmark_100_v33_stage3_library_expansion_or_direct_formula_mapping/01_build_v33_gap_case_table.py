#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import json
import pandas as pd


def save_md(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_markdown(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    root = Path(args.project_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    input_dir = root / "outputs/benchmark_100_v33_stage3_library_expansion_or_direct_formula_mapping/input_from_v32"

    unaligned_p = input_dir / "v32_metadata_aware_alignment_unaligned_cases.csv"
    review_p = input_dir / "v32_metadata_aware_alignment_bad_or_review_cases.csv"

    dfs = []

    if unaligned_p.exists():
        u = pd.read_csv(unaligned_p)
        u["gap_type"] = "unaligned"
        if "review_reasons" not in u.columns:
            u["review_reasons"] = "no_metadata_aware_stage3_alignment_under_strict_gate"
        dfs.append(u)

    if review_p.exists():
        r = pd.read_csv(review_p)
        r["gap_type"] = "review"
        dfs.append(r)

    if not dfs:
        raise FileNotFoundError("No V32 gap/review inputs found in input_from_v32.")

    gap = pd.concat(dfs, ignore_index=True, sort=False)

    keep_cols = [
        "gap_type",
        "external_case_id",
        "external_formula",
        "external_elements",
        "external_family",
        "mp_id",
        "mp_formula",
        "mp_elements",
        "mp_family",
        "element_jaccard",
        "family_compatibility",
        "formula_exact_match",
        "condition_distribution_support",
        "alignment_score",
        "review_reasons",
    ]
    for c in keep_cols:
        if c not in gap.columns:
            gap[c] = ""

    gap = gap[keep_cols].drop_duplicates()

    # Conservative priority
    def priority(row):
        reasons = str(row.get("review_reasons", ""))
        if row["gap_type"] == "unaligned":
            return 1
        if "family_mismatch" in reasons:
            return 2
        if "weak_element_overlap" in reasons:
            return 3
        if "low_condition_support" in reasons:
            return 4
        return 5

    gap["gap_priority"] = gap.apply(priority, axis=1)
    gap = gap.sort_values(["gap_priority", "external_case_id"]).reset_index(drop=True)

    out_csv = out / "v33_gap_case_table.csv"
    out_md = out / "v33_gap_case_table.md"
    out_json = out / "v33_gap_case_table_summary.json"

    gap.to_csv(out_csv, index=False)
    save_md(gap, out_md)

    summary = {
        "status": "pass",
        "n_gap_rows": int(len(gap)),
        "n_external_gap_cases": int(gap["external_case_id"].nunique()),
        "gap_type_counts": gap["gap_type"].value_counts().to_dict(),
        "priority_counts": gap["gap_priority"].value_counts().sort_index().to_dict(),
        "output_csv": str(out_csv),
        "interpretation": "V33 gap table collects V32 unaligned and review cases for direct formula-to-MP mapping and Stage3 library expansion planning."
    }

    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[SAVE] {out_csv}")
    print(f"[SAVE] {out_md}")
    print(f"[SAVE] {out_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
