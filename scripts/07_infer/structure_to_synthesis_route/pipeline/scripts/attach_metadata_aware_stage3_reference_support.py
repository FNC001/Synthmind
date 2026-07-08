#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


ELEMENT_RE = re.compile(r"([A-Z][a-z]?)")
WEAK_SHARED_ONLY_ELEMENTS = {"O", "F", "Cl", "Br", "I", "N", "S", "Se"}


def to_float(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default



def safe_json_float(x):
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None


def safe_json_str(x):
    try:
        if x is None:
            return ""
        if isinstance(x, float) and not math.isfinite(x):
            return ""
        sx = str(x)
        if sx.lower() in {"nan", "none"}:
            return ""
        return sx
    except Exception:
        return ""

def to_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def clip01(x: float, default=0.0) -> float:
    if not np.isfinite(x):
        return default
    return max(0.0, min(1.0, float(x)))


def parse_elements(x) -> set[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return set()

    s = str(x).strip()
    if not s:
        return set()

    if ";" in s:
        return {e.strip() for e in s.split(";") if e.strip()}

    return set(ELEMENT_RE.findall(s))


def normalize_case_id(x: str) -> str:
    s = to_str(x)
    if not s:
        return ""

    # pipeline infer ids sometimes look like:
    # infer_00001__P16Se8O24
    # external V28/V32 ids look like:
    # external_case_001
    # benchmark ids look like:
    # benchmark_001
    return s


def infer_external_case_id_from_route_row(row: pd.Series) -> str:
    """
    Try to recover external_case_id for joining V32 alignment summary.

    Priority:
      1. explicit external_case_id
      2. case_id if already external_case_*
      3. sample_id/material_id/infer_name pattern if available
      4. empty string
    """
    for c in ["external_case_id", "case_id"]:
        if c in row.index:
            v = to_str(row.get(c))
            if v.startswith("external_case_"):
                return v

    # Common benchmark_001 -> external_case_001 mapping.
    for c in ["infer_name", "benchmark_id"]:
        if c in row.index:
            v = to_str(row.get(c))
            m = re.search(r"benchmark[_-](\d+)", v)
            if m:
                return f"external_case_{int(m.group(1)):03d}"

    # Some rows may only carry sample_id/material_id like infer_00001__...
    # This is not safely mappable to external_case_001 unless caller adds external_case_id.
    return ""


def chemistry_gate_factor(row: pd.Series) -> tuple[float, str]:
    formula_exact = to_float(row.get("formula_exact_match", 0.0), 0.0)
    elem_jaccard = to_float(row.get("element_jaccard", 0.0), 0.0)
    family = to_float(row.get("family_compatibility", 0.0), 0.0)

    external_elements = parse_elements(row.get("external_elements", ""))
    mp_elements = parse_elements(row.get("mp_elements", ""))

    shared = external_elements & mp_elements
    shared_core = shared - WEAK_SHARED_ONLY_ELEMENTS

    if formula_exact >= 1.0:
        return 1.0, "formula_exact_match"

    if len(shared_core) == 0:
        return 0.15, "anion_only_or_weak_shared_element_overlap"

    if elem_jaccard >= 0.75 and family >= 1.0:
        return 0.85, "strong_element_family_match"

    if elem_jaccard >= 0.50 and family >= 1.0:
        return 0.65, "medium_element_family_match"

    if elem_jaccard >= 0.40:
        return 0.45, "partial_element_match"

    if family <= 0:
        return 0.25, "family_mismatch"

    return 0.25, "weak_metadata_match"


def level_from_support(score: float) -> str:
    if not np.isfinite(score):
        return "missing"
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def warning_from_metadata_support(
    support: float,
    alignment_score: float,
    gate_factor: float,
    gate_reason: str,
    element_jaccard: float,
    family_compatibility: float,
    formula_exact: float,
) -> tuple[str, str, str]:
    """
    Return:
      warning_level, recommendation_status, warning_reason
    """
    reasons = []

    if not np.isfinite(support):
        return "major_warning", "review_required", "missing_metadata_aware_support"

    if formula_exact >= 1.0:
        if support >= 0.35:
            return "no_warning", "recommended", "formula_exact_match"
        return "minor_warning", "recommended_with_validation", "formula_exact_match_but_low_condition_support"

    if gate_factor <= 0.15:
        reasons.append(gate_reason)

    if element_jaccard < 0.40:
        reasons.append("low_element_jaccard")

    if family_compatibility <= 0:
        reasons.append("family_mismatch")

    if alignment_score < 0.35:
        reasons.append("low_alignment_score")

    if support < 0.25:
        reasons.append("very_low_metadata_aware_support")
    elif support < 0.45:
        reasons.append("low_metadata_aware_support")

    if any(r in reasons for r in ["very_low_metadata_aware_support", "family_mismatch"]):
        return "major_warning", "review_required", ";".join(reasons)

    if reasons:
        return "minor_warning", "recommended_with_validation", ";".join(reasons)

    return "no_warning", "recommended", ""


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Attach V32 metadata-aware Stage3 reference support to pipeline_v3 route table."
    )
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--metadata_alignment_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--output_md", required=True)
    ap.add_argument("--summary_json", required=True)
    ap.add_argument("--top_n", type=int, default=30)
    args = ap.parse_args()

    input_csv = Path(args.input_csv)
    metadata_alignment_csv = Path(args.metadata_alignment_csv)
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    summary_json = Path(args.summary_json)

    if not input_csv.exists():
        raise FileNotFoundError(input_csv)

    if not metadata_alignment_csv.exists():
        raise FileNotFoundError(metadata_alignment_csv)

    routes = pd.read_csv(input_csv)
    align = pd.read_csv(metadata_alignment_csv)

    required_align = [
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
    ]
    missing = [c for c in required_align if c not in align.columns]
    if missing:
        raise RuntimeError(f"metadata_alignment_csv missing required columns: {missing}")

    # Keep top alignment per external case.
    align = align.copy()
    align["alignment_score"] = pd.to_numeric(align["alignment_score"], errors="coerce").fillna(0.0)
    align = (
        align.sort_values(["external_case_id", "alignment_score"], ascending=[True, False])
        .drop_duplicates("external_case_id", keep="first")
        .reset_index(drop=True)
    )

    gate_records = []
    for _, r in align.iterrows():
        gate, reason = chemistry_gate_factor(r)
        alignment_score = clip01(to_float(r.get("alignment_score"), 0.0), 0.0)
        condition_support = clip01(to_float(r.get("condition_distribution_support"), 0.0), 0.0)

        # Alignment already contains chemistry + condition, but we additionally gate it
        # so weak chemistry cannot be rescued only by condition distribution.
        support = clip01(alignment_score * gate, 0.0)

        elem_jaccard = clip01(to_float(r.get("element_jaccard"), 0.0), 0.0)
        fam = clip01(to_float(r.get("family_compatibility"), 0.0), 0.0)
        formula_exact = clip01(to_float(r.get("formula_exact_match"), 0.0), 0.0)

        warning, status, warning_reason = warning_from_metadata_support(
            support=support,
            alignment_score=alignment_score,
            gate_factor=gate,
            gate_reason=reason,
            element_jaccard=elem_jaccard,
            family_compatibility=fam,
            formula_exact=formula_exact,
        )

        row = r.to_dict()
        row.update(
            {
                "metadata_aware_stage3_chemistry_gate_factor": round(float(gate), 6),
                "metadata_aware_stage3_chemistry_gate_reason": reason,
                "metadata_aware_stage3_reference_support_score": round(float(support), 6),
                "metadata_aware_stage3_reference_level": level_from_support(support),
                "metadata_aware_stage3_reference_warning_level": warning,
                "metadata_aware_stage3_reference_warning_reason": warning_reason,
                "metadata_aware_stage3_reference_recommendation_status": status,
                "metadata_aware_stage3_raw_condition_distribution_support": round(float(condition_support), 6),
            }
        )
        gate_records.append(row)

    align2 = pd.DataFrame(gate_records)

    # Route table join key.
    out = routes.copy()

    if "external_case_id" not in out.columns:
        out["external_case_id"] = out.apply(infer_external_case_id_from_route_row, axis=1)

    # IMPORTANT:
    # Do NOT infer external_case_id from a path such as benchmark_001.
    # In pipeline_v3, benchmark_001 may be an arbitrary inference case and does
    # not necessarily correspond to V28/V32 external_case_001.
    #
    # Instead, when external_case_id is absent, use a conservative chemistry
    # match between the route target formula/elements and the V32 external
    # case formula/elements. If no chemically exact/near-exact match exists,
    # leave external_case_id empty and fall back downstream.
    if out["external_case_id"].fillna("").astype(str).str.len().eq(0).all():
        ext_lookup = align2.copy()
        ext_lookup["_external_element_set"] = ext_lookup["external_elements"].apply(parse_elements)
        ext_lookup["_external_formula_norm"] = ext_lookup["external_formula"].astype(str).str.replace(" ", "", regex=False)

        inferred_ids = []
        inferred_reasons = []

        for _, rr in out.iterrows():
            route_formula = ""
            for fc in ["target_formula", "formula", "formula_x", "formula_y", "material_id", "sample_id"]:
                if fc in rr.index and to_str(rr.get(fc)):
                    route_formula = to_str(rr.get(fc))
                    break

            route_elements = parse_elements(route_formula)
            if not route_elements:
                inferred_ids.append("")
                inferred_reasons.append("no_route_formula_or_elements")
                continue

            route_formula_norm = route_formula.replace(" ", "")

            # 1) exact formula match if available
            exact = ext_lookup[ext_lookup["_external_formula_norm"].eq(route_formula_norm)]
            if len(exact):
                inferred_ids.append(str(exact.iloc[0]["external_case_id"]))
                inferred_reasons.append("exact_formula_match")
                continue

            # 2) exact element-set match
            candidates = []
            for _, er in ext_lookup.iterrows():
                ext_elements = er["_external_element_set"]
                if not ext_elements:
                    continue

                inter = route_elements & ext_elements
                union = route_elements | ext_elements
                jac = len(inter) / max(len(union), 1)

                # require exact element set or very high overlap
                if route_elements == ext_elements:
                    candidates.append((1.0, er["external_case_id"], "exact_element_set_match"))
                elif jac >= 0.80:
                    candidates.append((jac, er["external_case_id"], "high_element_jaccard_match"))

            if candidates:
                candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
                inferred_ids.append(str(candidates[0][1]))
                inferred_reasons.append(candidates[0][2])
            else:
                inferred_ids.append("")
                inferred_reasons.append("no_safe_external_case_match")

        out["external_case_id"] = inferred_ids
        out["metadata_aware_stage3_external_case_inference_reason"] = inferred_reasons

    rename_map = {
        "mp_id": "metadata_aware_stage3_mp_id",
        "mp_formula": "metadata_aware_stage3_mp_formula",
        "mp_elements": "metadata_aware_stage3_mp_elements",
        "mp_family": "metadata_aware_stage3_mp_family",
        "external_formula": "metadata_aware_stage3_external_formula",
        "external_elements": "metadata_aware_stage3_external_elements",
        "external_family": "metadata_aware_stage3_external_family",
        "element_jaccard": "metadata_aware_stage3_element_jaccard",
        "family_compatibility": "metadata_aware_stage3_family_compatibility",
        "formula_exact_match": "metadata_aware_stage3_formula_exact_match",
        "condition_distribution_support": "metadata_aware_stage3_condition_distribution_support",
        "alignment_score": "metadata_aware_stage3_alignment_score",
    }

    keep_cols = ["external_case_id"] + list(rename_map.keys()) + [
        "metadata_aware_stage3_chemistry_gate_factor",
        "metadata_aware_stage3_chemistry_gate_reason",
        "metadata_aware_stage3_reference_support_score",
        "metadata_aware_stage3_reference_level",
        "metadata_aware_stage3_reference_warning_level",
        "metadata_aware_stage3_reference_warning_reason",
        "metadata_aware_stage3_reference_recommendation_status",
        "metadata_aware_stage3_raw_condition_distribution_support",
    ]

    attach = align2[[c for c in keep_cols if c in align2.columns]].rename(columns=rename_map)

    out = out.merge(attach, on="external_case_id", how="left")

    out["metadata_aware_stage3_reference_alignment_status"] = np.where(
        out["metadata_aware_stage3_reference_support_score"].notna(),
        "metadata_aware_alignment_attached",
        "missing_metadata_aware_alignment",
    )

    # Missing alignment should not block pipeline; it becomes fallback.
    missing_mask = out["metadata_aware_stage3_reference_support_score"].isna()
    out.loc[missing_mask, "metadata_aware_stage3_reference_warning_level"] = "missing"
    out.loc[missing_mask, "metadata_aware_stage3_reference_warning_reason"] = "missing_metadata_aware_alignment"
    out.loc[missing_mask, "metadata_aware_stage3_reference_recommendation_status"] = "fallback_to_real_stage3_condition_reference"
    out.loc[missing_mask, "metadata_aware_stage3_reference_level"] = "missing"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    preview_cols = [
        "external_case_id",
        "precursor_set",
        "temperature_c",
        "time_h",
        "metadata_aware_stage3_reference_alignment_status",
        "metadata_aware_stage3_reference_support_score",
        "metadata_aware_stage3_reference_level",
        "metadata_aware_stage3_reference_warning_level",
        "metadata_aware_stage3_reference_warning_reason",
        "metadata_aware_stage3_mp_id",
        "metadata_aware_stage3_mp_formula",
        "metadata_aware_stage3_element_jaccard",
        "metadata_aware_stage3_family_compatibility",
        "metadata_aware_stage3_formula_exact_match",
        "metadata_aware_stage3_alignment_score",
        "metadata_aware_stage3_chemistry_gate_factor",
        "metadata_aware_stage3_chemistry_gate_reason",
    ]
    preview_cols = [c for c in preview_cols if c in out.columns]

    lines = []
    lines.append("# Metadata-aware Stage3 Reference Support\n")
    lines.append(f"- input_csv: `{input_csv}`")
    lines.append(f"- metadata_alignment_csv: `{metadata_alignment_csv}`")
    lines.append(f"- n_routes: {len(out)}")
    lines.append(f"- top_n: {args.top_n}")
    lines.append("")
    if preview_cols:
        lines.append(out[preview_cols].head(args.top_n).to_markdown(index=False))
    else:
        lines.append("No preview columns available.")
    lines.append("")

    output_md.write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "input_csv": str(input_csv),
        "metadata_alignment_csv": str(metadata_alignment_csv),
        "output_csv": str(output_csv),
        "output_md": str(output_md),
        "n_routes": int(len(out)),
        "n_metadata_alignment_cases": int(align2["external_case_id"].nunique()) if len(align2) else 0,
        "n_routes_with_metadata_aware_alignment": int(
            out["metadata_aware_stage3_reference_alignment_status"]
            .astype(str)
            .eq("metadata_aware_alignment_attached")
            .sum()
        ),
        "n_routes_missing_metadata_aware_alignment": int(
            out["metadata_aware_stage3_reference_alignment_status"]
            .astype(str)
            .eq("missing_metadata_aware_alignment")
            .sum()
        ),
        "metadata_aware_reference_level_counts": out["metadata_aware_stage3_reference_level"]
        .astype(str)
        .value_counts(dropna=False)
        .to_dict(),
        "metadata_aware_reference_warning_level_counts": out[
            "metadata_aware_stage3_reference_warning_level"
        ]
        .astype(str)
        .value_counts(dropna=False)
        .to_dict(),
        "metadata_aware_reference_recommendation_status_counts": out[
            "metadata_aware_stage3_reference_recommendation_status"
        ]
        .astype(str)
        .value_counts(dropna=False)
        .to_dict(),
        "mean_metadata_aware_stage3_reference_support_score": float(
            pd.to_numeric(out["metadata_aware_stage3_reference_support_score"], errors="coerce").mean()
        )
        if pd.to_numeric(out["metadata_aware_stage3_reference_support_score"], errors="coerce").notna().any()
        else None,
        "top1_metadata_aware_stage3_reference_support_score": float(
            to_float(out.iloc[0].get("metadata_aware_stage3_reference_support_score"), np.nan)
        )
        if len(out)
        else None,
        "top1_metadata_aware_stage3_mp_id": str(out.iloc[0].get("metadata_aware_stage3_mp_id", ""))
        if len(out)
        else "",
        "top1_metadata_aware_stage3_mp_formula": str(out.iloc[0].get("metadata_aware_stage3_mp_formula", ""))
        if len(out)
        else "",
        "claim_boundary": "metadata_aware_stage3_reference_support_is_internal_alignment_not_experimental_validation",
        "interpretation": (
            "This step attaches V32 metadata-aware Stage3 reference support. "
            "The support score combines metadata-aware alignment with an additional chemistry gate, "
            "so formula/elements/family consistency is prioritized over condition-only similarity."
        ),
    }

    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[SAVE] {output_csv}")
    print(f"[SAVE] {output_md}")
    print(f"[SAVE] {summary_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
