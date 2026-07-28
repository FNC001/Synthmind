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

    ext_csv = v32_root / "input_from_v28/v28_recipe_confidence_scores.csv"
    align_csv = v32_root / "metadata_aware_alignment_v32/v32_metadata_aware_external_stage3_alignment_summary.csv"

    if not ext_csv.exists():
        raise SystemExit(f"[ERROR] missing external table: {ext_csv}")
    if not align_csv.exists():
        raise SystemExit(f"[ERROR] missing alignment summary: {align_csv}")

    ext = pd.read_csv(ext_csv)
    align = pd.read_csv(align_csv)

    all_cases = set(ext["case_id"].astype(str))
    aligned_cases = set(align["external_case_id"].astype(str))
    missing_cases = sorted(all_cases - aligned_cases)

    bad_or_review = []

    for _, r in align.iterrows():
        reasons = []

        elem = float(r.get("element_jaccard", 0))
        fam = float(r.get("family_compatibility", 0))
        exact = float(r.get("formula_exact_match", 0))
        score = float(r.get("alignment_score", 0))
        cond = float(r.get("condition_distribution_support", 0))

        if exact < 1 and elem < 0.5:
            reasons.append("weak_element_overlap")
        if fam <= 0:
            reasons.append("family_mismatch")
        if score < 0.45:
            reasons.append("low_alignment_score")
        if cond < 0.05:
            reasons.append("low_condition_support")

        if reasons:
            item = r.to_dict()
            item["review_reasons"] = ";".join(reasons)
            bad_or_review.append(item)

    missing_rows = []
    if missing_cases:
        ext_small = ext[ext["case_id"].astype(str).isin(missing_cases)].copy()
        for _, r in ext_small.iterrows():
            missing_rows.append({
                "external_case_id": r.get("case_id"),
                "external_formula": r.get("target_formula"),
                "external_elements": r.get("target_elements"),
                "external_family": r.get("v28_target_family"),
                "review_reasons": "no_metadata_aware_stage3_alignment_under_strict_gate",
            })

    bad_df = pd.DataFrame(bad_or_review)
    missing_df = pd.DataFrame(missing_rows)

    out_bad = out / "v32_metadata_aware_alignment_bad_or_review_cases.csv"
    out_bad_md = out / "v32_metadata_aware_alignment_bad_or_review_cases.md"
    out_missing = out / "v32_metadata_aware_alignment_unaligned_cases.csv"
    out_missing_md = out / "v32_metadata_aware_alignment_unaligned_cases.md"
    out_summary = out / "v32_metadata_aware_alignment_audit_summary.json"
    out_summary_md = out / "v32_metadata_aware_alignment_audit_summary.md"

    if bad_df.empty:
        pd.DataFrame(columns=[
            "external_case_id", "external_formula", "mp_id", "mp_formula",
            "element_jaccard", "family_compatibility", "condition_distribution_support",
            "alignment_score", "review_reasons"
        ]).to_csv(out_bad, index=False)
        Path(out_bad_md).write_text("| status |\n|:--|\n| no bad_or_review aligned cases |\n", encoding="utf-8")
    else:
        bad_df.to_csv(out_bad, index=False)
        bad_df.head(100).to_markdown(out_bad_md, index=False)

    if missing_df.empty:
        pd.DataFrame(columns=[
            "external_case_id", "external_formula", "external_elements",
            "external_family", "review_reasons"
        ]).to_csv(out_missing, index=False)
        Path(out_missing_md).write_text("| status |\n|:--|\n| no unaligned external cases |\n", encoding="utf-8")
    else:
        missing_df.to_csv(out_missing, index=False)
        missing_df.to_markdown(out_missing_md, index=False)

    status = "pass_with_review" if missing_cases or len(bad_or_review) > 0 else "pass"

    summary = {
        "status": status,
        "n_external_cases": int(len(all_cases)),
        "n_aligned_cases": int(len(aligned_cases)),
        "n_unaligned_cases": int(len(missing_cases)),
        "unaligned_cases": missing_cases,
        "n_bad_or_review_aligned_cases": int(len(bad_or_review)),
        "mean_alignment_score": float(align["alignment_score"].mean()) if len(align) else None,
        "mean_element_jaccard": float(align["element_jaccard"].mean()) if len(align) else None,
        "mean_family_compatibility": float(align["family_compatibility"].mean()) if len(align) else None,
        "mean_condition_distribution_support": float(align["condition_distribution_support"].mean()) if len(align) else None,
        "n_formula_exact_matches": int((align["formula_exact_match"] > 0).sum()) if "formula_exact_match" in align.columns else 0,
        "audit_interpretation": (
            "V32 metadata-aware alignment is chemically stricter than V31. "
            "Unaligned cases indicate that the real Stage3 library lacks a sufficiently matched mp-* candidate "
            "under the strict formula/element/family gate."
        ),
        "output_bad_or_review_csv": str(out_bad),
        "output_unaligned_cases_csv": str(out_missing),
    }

    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [
        "# V32 Metadata-Aware Alignment Audit Summary",
        "",
        f"- Audit status: `{summary['status']}`",
        f"- External cases: `{summary['n_external_cases']}`",
        f"- Aligned cases: `{summary['n_aligned_cases']}`",
        f"- Unaligned cases: `{summary['n_unaligned_cases']}`",
        f"- Bad/review aligned cases: `{summary['n_bad_or_review_aligned_cases']}`",
        f"- Formula exact matches: `{summary['n_formula_exact_matches']}`",
        f"- Mean alignment score: `{summary['mean_alignment_score']}`",
        f"- Mean element Jaccard: `{summary['mean_element_jaccard']}`",
        f"- Mean family compatibility: `{summary['mean_family_compatibility']}`",
        f"- Mean condition support: `{summary['mean_condition_distribution_support']}`",
        "",
        "## Interpretation",
        "",
        summary["audit_interpretation"],
        "",
    ]

    if missing_cases:
        md += [
            "## Unaligned cases",
            "",
            ", ".join(f"`{c}`" for c in missing_cases),
            "",
            "These cases should not be force-matched unless the Stage3 candidate library is expanded or a direct MP/formula mapping is added.",
            "",
        ]

    out_summary_md.write_text("\n".join(md), encoding="utf-8")

    print("[SAVE]", out_bad)
    print("[SAVE]", out_bad_md)
    print("[SAVE]", out_missing)
    print("[SAVE]", out_missing_md)
    print("[SAVE]", out_summary)
    print("[SAVE]", out_summary_md)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
