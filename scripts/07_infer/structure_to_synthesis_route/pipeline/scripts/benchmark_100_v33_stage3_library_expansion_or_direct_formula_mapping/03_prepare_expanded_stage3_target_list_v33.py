#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import json
import pandas as pd


def save_md(df, path):
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

    v33 = root / "outputs/benchmark_100_v33_stage3_library_expansion_or_direct_formula_mapping"
    mapping_p = v33 / "direct_formula_mp_mapping_v33/v33_direct_formula_mp_mapping.csv"
    stage3_p = root / "outputs/benchmark_100_v32_mp_metadata_aware_stage3_alignment/stage3_candidates_with_metadata_v32/v32_stage3_candidates_with_mp_metadata.csv"

    if not mapping_p.exists():
        raise FileNotFoundError(f"Missing mapping file: {mapping_p}")

    mapping = pd.read_csv(mapping_p)
    mapped = mapping[mapping["mapping_status"] == "mapped"].copy()

    if stage3_p.exists():
        stage3 = pd.read_csv(stage3_p, usecols=lambda c: c in ["mp_id", "case_id", "formula", "elements", "mp_family"])
        existing_mp_ids = set(stage3["mp_id"].dropna().astype(str).unique())
    else:
        existing_mp_ids = set()

    rows = []
    for _, r in mapped.iterrows():
        mpid = str(r.get("direct_mp_id", ""))
        already_has_stage3 = mpid in existing_mp_ids

        reason = []
        if r.get("gap_type") == "unaligned":
            reason.append("v32_unaligned")
        if r.get("gap_type") == "review":
            reason.append("v32_review")
        if r.get("direct_mapping_type") == "formula_exact":
            reason.append("formula_exact_direct_mp_match")
        else:
            reason.append(str(r.get("direct_mapping_type", "")))

        rows.append({
            "external_case_id": r.get("external_case_id", ""),
            "external_formula": r.get("external_formula", ""),
            "external_elements": r.get("external_elements", ""),
            "external_family": r.get("external_family", ""),
            "direct_mp_id": mpid,
            "mp_formula": r.get("mp_formula", ""),
            "mp_elements": r.get("mp_elements", ""),
            "mp_family": r.get("mp_family", ""),
            "direct_mapping_type": r.get("direct_mapping_type", ""),
            "already_has_real_stage3_candidates": bool(already_has_stage3),
            "stage3_expansion_needed": not bool(already_has_stage3),
            "reason_for_expansion": ";".join(reason),
            "recommended_action": "export_or_generate_real_stage3_mdn_flow_candidates" if not already_has_stage3 else "reuse_existing_real_stage3_candidates",
        })

    target = pd.DataFrame(rows)

    # Keep top-priority one mapping per external case for concise target list
    if len(target):
        target["priority_sort"] = target["direct_mapping_type"].map({"formula_exact": 1}).fillna(2)
        target = target.sort_values(["external_case_id", "priority_sort", "direct_mp_id"])
        target_top1 = target.groupby("external_case_id", as_index=False).head(1).drop(columns=["priority_sort"])
    else:
        target_top1 = target

    out_csv = out / "v33_expanded_stage3_target_list.csv"
    out_md = out / "v33_expanded_stage3_target_list.md"
    out_json = out / "v33_expanded_stage3_target_list_summary.json"

    target_top1.to_csv(out_csv, index=False)
    save_md(target_top1, out_md)

    summary = {
        "status": "pass",
        "n_expansion_target_rows": int(len(target_top1)),
        "n_external_cases_with_direct_mapping": int(target_top1["external_case_id"].nunique()) if len(target_top1) else 0,
        "n_already_has_stage3": int(target_top1["already_has_real_stage3_candidates"].sum()) if len(target_top1) else 0,
        "n_stage3_expansion_needed": int(target_top1["stage3_expansion_needed"].sum()) if len(target_top1) else 0,
        "expansion_needed_cases": target_top1.loc[target_top1["stage3_expansion_needed"], "external_case_id"].tolist() if len(target_top1) else [],
        "output_csv": str(out_csv),
        "interpretation": "V33 expansion target list identifies direct MP targets that need real Stage3 MDN/Flow candidate export or generation."
    }

    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[SAVE] {out_csv}")
    print(f"[SAVE] {out_md}")
    print(f"[SAVE] {out_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
