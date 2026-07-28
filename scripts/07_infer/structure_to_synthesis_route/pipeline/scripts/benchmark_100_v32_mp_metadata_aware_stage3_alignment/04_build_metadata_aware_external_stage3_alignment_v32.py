#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_elements(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return set()
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return set()
    if ";" in s:
        return set([v.strip() for v in s.split(";") if v.strip()])
    if "," in s:
        return set([v.strip() for v in s.split(",") if v.strip()])
    if "-" in s:
        return set([v.strip() for v in s.split("-") if v.strip()])
    # fallback: parse formula-like string
    return set(re.findall(r"[A-Z][a-z]?", s))


def normalize_family(x):
    s = str(x or "").lower().strip()
    if not s or s == "nan":
        return "unknown"
    if "phosphate" in s:
        return "phosphate"
    if "oxide" in s:
        return "oxide"
    if "sulfate" in s:
        return "sulfate"
    if "sulfide" in s:
        return "sulfide"
    if "selen" in s:
        return "selenide"
    if "nitride" in s:
        return "nitride"
    if "halide" in s:
        return "halide"
    if "carbonate" in s:
        return "carbonate"
    return s


def family_compatible(ext_family, mp_family):
    ef = normalize_family(ext_family)
    mf = normalize_family(mp_family)

    if ef == "unknown" or mf == "unknown":
        return 0.0

    if ef == mf:
        return 1.0

    # compatible aliases
    if ef == "phosphate" and "phosphate" in mf:
        return 1.0
    if ef == "oxide" and "oxide" in mf:
        return 1.0
    if ef == "carbonate" and "carbonate" in mf:
        return 1.0
    if ef == "sulfate" and ("sulfate" in mf or "sulfide" in mf):
        return 0.7
    if ef == "sulfide" and ("sulfate" in mf or "sulfide" in mf):
        return 0.7
    if ef == "selenide" and ("selen" in mf):
        return 1.0
    if ef == "nitride" and "nitride" in mf:
        return 1.0
    if ef == "halide" and "halide" in mf:
        return 1.0

    return 0.0


def jaccard(a, b):
    a = set(a)
    b = set(b)
    if not a and not b:
        return 0.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def condition_support_for_group(temp, time_h, g):
    temps = pd.to_numeric(g["candidate_temperature_c"], errors="coerce").dropna()
    times = pd.to_numeric(g["candidate_time_h"], errors="coerce").dropna()

    if len(temps) < 3 or len(times) < 3:
        return 0.0, np.nan, np.nan, np.nan, np.nan

    temp_center = float(temps.median())
    time_center = float(times.median())

    temp_width = float(max(temps.quantile(0.75) - temps.quantile(0.25), 50.0))
    time_width = float(max(times.quantile(0.75) - times.quantile(0.25), 2.0))

    temp_z = abs(float(temp) - temp_center) / temp_width
    time_z = abs(float(time_h) - time_center) / time_width

    # smooth decay; 1 means near the retrieved Stage3 distribution center
    support = math.exp(-0.5 * (temp_z ** 2 + time_z ** 2))
    support = max(0.0, min(1.0, support))

    return support, temp_center, time_center, temp_width, time_width


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--top_k", type=int, default=10)
    args = ap.parse_args()

    root = Path(args.project_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    v32_root = root / "outputs/benchmark_100_v32_mp_metadata_aware_stage3_alignment"

    ext_csv = v32_root / "input_from_v28/v28_recipe_confidence_scores.csv"
    stage3_csv = v32_root / "stage3_candidates_with_metadata_v32/v32_stage3_candidates_with_mp_metadata.csv"

    if not ext_csv.exists():
        raise SystemExit(f"[ERROR] missing external recipe table: {ext_csv}")
    if not stage3_csv.exists():
        raise SystemExit(f"[ERROR] missing metadata-attached Stage3 candidates: {stage3_csv}")

    ext = pd.read_csv(ext_csv)
    stg = pd.read_csv(stage3_csv)

    required_ext = ["case_id", "target_formula", "target_elements", "v28_target_family", "v28_temperature_c", "v28_time_h"]
    missing_ext = [c for c in required_ext if c not in ext.columns]
    if missing_ext:
        raise SystemExit(f"[ERROR] external table missing columns: {missing_ext}")

    required_stg = ["mp_id", "formula", "elements", "mp_family", "candidate_temperature_c", "candidate_time_h"]
    missing_stg = [c for c in required_stg if c not in stg.columns]
    if missing_stg:
        raise SystemExit(f"[ERROR] Stage3 table missing columns: {missing_stg}")

    # Collapse real Stage3 candidates to one distribution per mp_id.
    rows = []

    grouped = stg.groupby("mp_id", dropna=False)

    stage3_summary_rows = []
    for mp_id, g in grouped:
        first = g.iloc[0]
        stage3_summary_rows.append({
            "mp_id": mp_id,
            "mp_formula": first.get("formula", ""),
            "mp_elements": first.get("elements", ""),
            "mp_family": first.get("mp_family", ""),
            "n_stage3_candidates": int(len(g)),
            "candidate_source_counts": dict(g["candidate_source"].value_counts()) if "candidate_source" in g.columns else {},
            "mean_candidate_condition_score": float(pd.to_numeric(g.get("candidate_condition_score", pd.Series(dtype=float)), errors="coerce").mean()),
            "median_temperature_c": float(pd.to_numeric(g["candidate_temperature_c"], errors="coerce").median()),
            "median_time_h": float(pd.to_numeric(g["candidate_time_h"], errors="coerce").median()),
        })

    stage3_summary = pd.DataFrame(stage3_summary_rows)

    for _, er in ext.iterrows():
        ext_case_id = er["case_id"]
        ext_formula = str(er["target_formula"])
        ext_elements = parse_elements(er["target_elements"])
        ext_family = er["v28_target_family"]
        ext_temp = float(er["v28_temperature_c"])
        ext_time = float(er["v28_time_h"])

        candidate_rows = []

        for _, sr in stage3_summary.iterrows():
            mp_id = sr["mp_id"]
            mp_elements = parse_elements(sr["mp_elements"])
            mp_family = sr["mp_family"]

            elem_score = jaccard(ext_elements, mp_elements)
            fam_score = family_compatible(ext_family, mp_family)

            # Strict metadata-aware gate:
            # 1) exact formula is always allowed;
            # 2) otherwise require meaningful element overlap;
            # 3) if overlap is weak, family/condition alone must not dominate.
            formula_score = 1.0 if str(sr["mp_formula"]).replace(" ", "") == ext_formula.replace(" ", "") else 0.0

            if formula_score < 1.0:
                # For binary targets, require at least one shared target element AND nonzero element Jaccard.
                # For ternary/quaternary targets, require Jaccard >= 0.40 unless family is very strong.
                if len(ext_elements) <= 2:
                    if elem_score < 0.25:
                        continue
                else:
                    if elem_score < 0.40:
                        continue

                # Prevent family-only false positives such as NaCl -> SnPbF4.
                if elem_score < 0.50 and fam_score < 1.0:
                    continue

            g = grouped.get_group(mp_id)
            cond_support, t_center, h_center, t_width, h_width = condition_support_for_group(ext_temp, ext_time, g)

            # More chemistry-dominant score:
            # formula exact match and element overlap dominate;
            # family is secondary;
            # condition distribution support only refines among chemically plausible candidates.
            alignment_score = (
                0.55 * elem_score
                + 0.20 * fam_score
                + 0.15 * formula_score
                + 0.10 * cond_support
            )

            candidate_rows.append({
                "external_case_id": ext_case_id,
                "external_formula": ext_formula,
                "external_elements": ";".join(sorted(ext_elements)),
                "external_family": ext_family,
                "external_temperature_c": ext_temp,
                "external_time_h": ext_time,
                "mp_id": mp_id,
                "mp_formula": sr["mp_formula"],
                "mp_elements": sr["mp_elements"],
                "mp_family": mp_family,
                "n_stage3_candidates": sr["n_stage3_candidates"],
                "element_jaccard": elem_score,
                "family_compatibility": fam_score,
                "formula_exact_match": formula_score,
                "condition_distribution_support": cond_support,
                "stage3_temperature_center_c": t_center,
                "stage3_time_center_h": h_center,
                "stage3_temperature_width_c": t_width,
                "stage3_time_width_h": h_width,
                "alignment_score": alignment_score,
                "alignment_mode": "metadata_aware_formula_elements_family_condition",
            })

        if candidate_rows:
            tmp = pd.DataFrame(candidate_rows)
            tmp = tmp.sort_values(
                ["alignment_score", "element_jaccard", "family_compatibility", "condition_distribution_support"],
                ascending=False,
            ).head(args.top_k)
            rows.extend(tmp.to_dict("records"))

    align = pd.DataFrame(rows)

    out_topk = out / "v32_metadata_aware_external_to_stage3_alignment_topk.csv"
    out_preview = out / "v32_metadata_aware_external_to_stage3_alignment_topk_preview.md"
    out_case_summary = out / "v32_metadata_aware_external_stage3_alignment_summary.csv"
    out_case_summary_md = out / "v32_metadata_aware_external_stage3_alignment_summary.md"
    out_summary = out / "v32_metadata_aware_external_stage3_alignment_summary.json"

    if align.empty:
        summary = {
            "status": "blocked",
            "reason": "no metadata-aware alignment rows generated",
            "output_dir": str(out),
        }
        out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        raise SystemExit("[BLOCKED] no alignment rows generated")

    align.to_csv(out_topk, index=False)

    preview_cols = [
        "external_case_id", "external_formula", "external_elements", "external_family",
        "mp_id", "mp_formula", "mp_elements", "mp_family",
        "element_jaccard", "family_compatibility", "formula_exact_match",
        "condition_distribution_support", "alignment_score",
        "external_temperature_c", "external_time_h",
        "stage3_temperature_center_c", "stage3_time_center_h",
    ]
    preview_cols = [c for c in preview_cols if c in align.columns]
    align[preview_cols].head(100).to_markdown(out_preview, index=False)

    case_summary = (
        align.sort_values(["external_case_id", "alignment_score"], ascending=[True, False])
        .groupby("external_case_id")
        .head(1)
        .copy()
    )

    case_summary = case_summary[[
        "external_case_id", "external_formula", "external_elements", "external_family",
        "mp_id", "mp_formula", "mp_elements", "mp_family",
        "element_jaccard", "family_compatibility", "formula_exact_match",
        "condition_distribution_support", "alignment_score",
        "alignment_mode",
    ]]

    case_summary.to_csv(out_case_summary, index=False)
    case_summary.to_markdown(out_case_summary_md, index=False)

    summary = {
        "status": "pass",
        "n_external_cases": int(ext["case_id"].nunique()),
        "n_stage3_library_mp_ids": int(stage3_summary["mp_id"].nunique()),
        "n_alignment_rows": int(len(align)),
        "top_k": int(args.top_k),
        "n_cases_with_alignment": int(case_summary["external_case_id"].nunique()),
        "mean_top_alignment_score": float(case_summary["alignment_score"].mean()),
        "mean_top_element_jaccard": float(case_summary["element_jaccard"].mean()),
        "mean_top_family_compatibility": float(case_summary["family_compatibility"].mean()),
        "mean_top_condition_support": float(case_summary["condition_distribution_support"].mean()),
        "n_formula_exact_matches": int((case_summary["formula_exact_match"] > 0).sum()),
        "alignment_mode": "metadata_aware_formula_elements_family_condition",
        "output_topk_csv": str(out_topk),
        "output_case_summary_csv": str(out_case_summary),
        "interpretation": "V32 aligns external benchmark cases to real Stage3 mp-* condition distributions using MP metadata: formula, element overlap, family compatibility, and condition-distribution support.",
    }

    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[SAVE]", out_topk)
    print("[SAVE]", out_preview)
    print("[SAVE]", out_case_summary)
    print("[SAVE]", out_case_summary_md)
    print("[SAVE]", out_summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
