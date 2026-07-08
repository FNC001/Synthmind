#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import json
import re
import pandas as pd


def normalize_formula(s):
    if pd.isna(s):
        return ""
    s = str(s).strip()
    s = s.replace("_", "")
    s = s.replace(" ", "")
    s = s.replace("(", "")
    s = s.replace(")", "")
    return s


def parse_elements_from_formula(formula):
    if pd.isna(formula):
        return ""
    toks = re.findall(r"[A-Z][a-z]?", str(formula))
    return ";".join(sorted(set(toks)))


def elem_set(x):
    if pd.isna(x) or str(x).strip() == "":
        return set()
    return set([t.strip() for t in str(x).replace(",", ";").split(";") if t.strip()])


def jaccard(a, b):
    a, b = elem_set(a), elem_set(b)
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


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
    gap_p = v33 / "stage3_library_gap_analysis_v33/v33_gap_case_table.csv"
    mp_p = v33 / "input_from_v32/v32_mp_metadata_table.csv"

    if not gap_p.exists():
        raise FileNotFoundError(f"Missing gap table: {gap_p}")
    if not mp_p.exists():
        raise FileNotFoundError(f"Missing MP metadata table: {mp_p}")

    gap = pd.read_csv(gap_p)
    mp = pd.read_csv(mp_p)

    gap["external_formula_norm"] = gap["external_formula"].map(normalize_formula)
    gap["external_elements_norm"] = gap["external_elements"].fillna("")
    mp["mp_formula_norm"] = mp["formula"].map(normalize_formula)

    rows = []

    for _, g in gap.iterrows():
        ext_case = g.get("external_case_id", "")
        ext_formula = g.get("external_formula", "")
        ext_formula_norm = g.get("external_formula_norm", "")
        ext_elements = g.get("external_elements_norm", "")
        ext_family = g.get("external_family", "")
        gap_type = g.get("gap_type", "")
        review_reasons = g.get("review_reasons", "")

        exact = mp[mp["mp_formula_norm"] == ext_formula_norm].copy()

        if len(exact) == 0:
            # fallback: same element set, not formula exact
            ext_e = elem_set(ext_elements)
            if not ext_e:
                ext_e = elem_set(parse_elements_from_formula(ext_formula))
            cand = mp.copy()
            cand["element_jaccard_direct"] = cand["elements"].map(lambda x: len(elem_set(x) & ext_e) / max(1, len(elem_set(x) | ext_e)))
            cand = cand[cand["element_jaccard_direct"] >= 0.999].copy()
            cand["direct_mapping_type"] = "same_element_set_no_formula_exact"
        else:
            cand = exact.copy()
            cand["element_jaccard_direct"] = cand["elements"].map(lambda x: jaccard(ext_elements, x))
            cand["direct_mapping_type"] = "formula_exact"

        if len(cand) == 0:
            rows.append({
                "external_case_id": ext_case,
                "external_formula": ext_formula,
                "external_elements": ext_elements,
                "external_family": ext_family,
                "gap_type": gap_type,
                "review_reasons": review_reasons,
                "direct_mp_id": "",
                "mp_formula": "",
                "mp_elements": "",
                "mp_family": "",
                "direct_mapping_type": "no_direct_mp_match",
                "element_jaccard_direct": 0.0,
                "mapping_priority": 9,
                "mapping_status": "missing",
            })
            continue

        cand = cand.sort_values(["direct_mapping_type", "source_priority", "mp_id"], ascending=[True, True, True]).head(10)

        for _, m in cand.iterrows():
            mapping_type = m.get("direct_mapping_type", "")
            priority = 1 if mapping_type == "formula_exact" else 2

            rows.append({
                "external_case_id": ext_case,
                "external_formula": ext_formula,
                "external_elements": ext_elements,
                "external_family": ext_family,
                "gap_type": gap_type,
                "review_reasons": review_reasons,
                "direct_mp_id": m.get("mp_id", ""),
                "mp_formula": m.get("formula", ""),
                "mp_elements": m.get("elements", ""),
                "mp_family": m.get("mp_family", ""),
                "direct_mapping_type": mapping_type,
                "element_jaccard_direct": float(m.get("element_jaccard_direct", 0.0)),
                "mapping_priority": priority,
                "mapping_status": "mapped",
            })

    res = pd.DataFrame(rows)
    res = res.sort_values(["external_case_id", "mapping_priority", "direct_mp_id"]).reset_index(drop=True)

    out_csv = out / "v33_direct_formula_mp_mapping.csv"
    out_md = out / "v33_direct_formula_mp_mapping.md"
    out_json = out / "v33_direct_formula_mp_mapping_summary.json"

    res.to_csv(out_csv, index=False)
    save_md(res.head(200), out_md)

    summary = {
        "status": "pass",
        "n_mapping_rows": int(len(res)),
        "n_external_gap_cases": int(res["external_case_id"].nunique()),
        "n_mapped_cases": int(res[res["mapping_status"] == "mapped"]["external_case_id"].nunique()),
        "n_missing_cases": int(res[res["mapping_status"] == "missing"]["external_case_id"].nunique()),
        "mapping_type_counts": res["direct_mapping_type"].value_counts().to_dict(),
        "missing_cases": sorted(res.loc[res["mapping_status"] == "missing", "external_case_id"].unique().tolist()),
        "output_csv": str(out_csv),
        "interpretation": "V33 direct formula-to-MP mapping searches exact formula matches first, then same-element-set matches, using the V32 MP metadata table."
    }

    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[SAVE] {out_csv}")
    print(f"[SAVE] {out_md}")
    print(f"[SAVE] {out_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
