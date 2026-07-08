#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd


DEFAULT_NORMALIZATION = {
    "Sr(CO3)": "SrCO3",
    "Sr2CO3": "SrCO3",
    "Sb2O3via": "Sb2O3",
}

# 这些不一定绝对错误，但在最终推荐里应该降低置信度或提示人工复核。
DEFAULT_UNUSUAL_PRECURSORS = {
    "SrO2": "unusual_strontium_peroxide_precursor",
    "SrF2": "fluoride_precursor_may_introduce_f_or_specific_flux_route",
    "LaSb": "contains_non_target_la_and_intermetallic_precursor",
    "SrFeO2.875": "contains_non_target_fe_complex_oxide",
    "SrCO3-6Fe2O3": "contains_non_target_fe_mixed_precursor",
}

# 单质前驱体不一定不能用，但对固相氧化物路线来说一般应提示人工复核。
ELEMENTAL_PRECURSOR_PATTERN = re.compile(r"^[A-Z][a-z]?$")

# 识别化学式中的元素符号。
ELEMENT_RE = re.compile(r"([A-Z][a-z]?)")


def split_precursor_set(x: object) -> list[str]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return []
    s = str(x).strip()
    if not s:
        return []
    # 兼容 ; 或 || 分隔
    parts = re.split(r"\s*;\s*|\s*\|\|\s*", s)
    return [p.strip() for p in parts if p.strip()]


def join_precursor_set(parts: list[str]) -> str:
    return "; ".join([p for p in parts if p])


def load_json_dict(path: str | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def normalize_precursors(parts: list[str], normalization: dict[str, str]) -> tuple[list[str], list[str]]:
    out = []
    notes = []
    for p in parts:
        np = normalization.get(p, p)
        if np != p:
            notes.append(f"normalized:{p}->{np}")
        out.append(np)
    return out, notes


def extract_elements_from_formula_like(s: str) -> set[str]:
    if not s:
        return set()
    return set(ELEMENT_RE.findall(str(s)))


def infer_target_elements(row: pd.Series, ignore_elements: set[str]) -> set[str]:
    # 优先从已有 element_hit / element_hit_recomputed 旁推，不可靠时再从 formula 提取。
    for col in ["target_elements", "target_core_elements"]:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            raw = str(row[col]).replace(",", ";")
            elems = {x.strip() for x in raw.split(";") if x.strip()}
            elems = {e for e in elems if re.match(r"^[A-Z][a-z]?$", e)}
            if elems:
                return elems - ignore_elements

    for col in ["formula_x", "formula", "material_id", "sample_id"]:
        if col in row and pd.notna(row[col]):
            elems = extract_elements_from_formula_like(str(row[col]))
            elems = elems - ignore_elements
            # material_id / sample_id 里可能有 infer 等字符串，过滤一下明显非元素内容由正则已处理。
            if elems:
                return elems

    # 如果没有 formula，就从 element_hit 里取，虽然这只代表已命中的元素。
    for col in ["element_hit_recomputed", "element_hit"]:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            elems = {x.strip() for x in str(row[col]).split(";") if x.strip()}
            elems = {e for e in elems if re.match(r"^[A-Z][a-z]?$", e)}
            if elems:
                return elems - ignore_elements

    return set()


def precursor_elements(parts: list[str], ignore_elements: set[str]) -> set[str]:
    elems: set[str] = set()
    for p in parts:
        elems |= extract_elements_from_formula_like(p)
    return elems - ignore_elements


def is_elemental_precursor(p: str, target_elements: set[str]) -> bool:
    if not ELEMENTAL_PRECURSOR_PATTERN.match(p):
        return False
    # 单质如果是目标元素，也提示，但不作为严重错误。
    return p in target_elements


def score_route_qc(
    normalized_parts: list[str],
    target_elements: set[str],
    ignore_elements: set[str],
    unusual_precursors: dict[str, str],
    penalty_elemental: float,
    penalty_unusual: float,
    penalty_extra_element: float,
    penalty_missing_element: float,
) -> dict:
    warnings: list[str] = []

    p_elems = precursor_elements(normalized_parts, ignore_elements)

    missing = sorted(target_elements - p_elems)
    extra = sorted(p_elems - target_elements)

    if missing:
        warnings.append("missing_target_elements:" + ",".join(missing))
    if extra:
        warnings.append("extra_non_target_elements:" + ",".join(extra))

    elemental = []
    unusual = []

    for p in normalized_parts:
        if is_elemental_precursor(p, target_elements):
            elemental.append(p)
        if p in unusual_precursors:
            unusual.append(p)

    if elemental:
        warnings.append("elemental_precursors:" + ",".join(elemental))
    if unusual:
        warnings.append(
            "unusual_precursors:"
            + ",".join([f"{p}({unusual_precursors[p]})" for p in unusual])
        )

    score = 1.0
    score -= penalty_missing_element * len(missing)
    score -= penalty_extra_element * len(extra)
    score -= penalty_elemental * len(elemental)
    score -= penalty_unusual * len(unusual)
    score = max(0.0, min(1.0, score))

    if missing:
        level = "fail"
        status = "review_required"
    elif score >= 0.85 and not warnings:
        level = "pass"
        status = "recommended"
    elif score >= 0.65:
        level = "minor_warning"
        status = "recommended_with_validation"
    else:
        level = "major_warning"
        status = "review_required"

    return {
        "precursor_qc_score": round(score, 4),
        "precursor_qc_level": level,
        "precursor_qc_status": status,
        "precursor_qc_warnings": "; ".join(warnings),
        "target_elements_qc": ";".join(sorted(target_elements)),
        "precursor_elements_qc": ";".join(sorted(p_elems)),
        "missing_target_elements_qc": ";".join(missing),
        "extra_elements_qc": ";".join(extra),
        "has_missing_target_elements_qc": int(bool(missing)),
        "has_extra_elements_qc": int(bool(extra)),
        "has_elemental_precursor_qc": int(bool(elemental)),
        "has_unusual_precursor_qc": int(bool(unusual)),
    }


def to_markdown_table(df: pd.DataFrame, top_n: int) -> str:
    if df.empty:
        return "_No records._"

    cols = [
        "final_route_rank",
        "stage35_v21_rank",
        "precursor_rank",
        "normalized_precursor_set",
        "temperature_c",
        "time_h",
        "stage3_score",
        "stage35_v21_score",
        "element_coverage",
        "precursor_qc_score",
        "precursor_qc_level",
        "precursor_qc_status",
        "precursor_qc_warnings",
    ]
    cols = [c for c in cols if c in df.columns]
    show = df[cols].head(top_n).copy()

    for c in show.columns:
        if pd.api.types.is_numeric_dtype(show[c]):
            show[c] = show[c].round(4)

    show = show.fillna("")
    return show.to_markdown(index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--output_md", required=True)
    ap.add_argument("--summary_json", required=True)

    ap.add_argument("--precursor_col", default="precursor_set")
    ap.add_argument("--top_n", type=int, default=30)

    ap.add_argument("--normalization_json", default=None)
    ap.add_argument("--unusual_precursors_json", default=None)

    ap.add_argument("--ignore_elements", default="H,O")

    ap.add_argument("--penalty_elemental", type=float, default=0.08)
    ap.add_argument("--penalty_unusual", type=float, default=0.15)
    ap.add_argument("--penalty_extra_element", type=float, default=0.18)
    ap.add_argument("--penalty_missing_element", type=float, default=0.45)

    ap.add_argument(
        "--sort_by_qc",
        action="store_true",
        help="If enabled, sort routes by precursor_qc_score before existing rank columns.",
    )

    args = ap.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    summary_json = Path(args.summary_json)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"input_csv not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if args.precursor_col not in df.columns:
        raise ValueError(
            f"precursor column `{args.precursor_col}` not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    ignore_elements = {x.strip() for x in args.ignore_elements.split(",") if x.strip()}

    normalization = dict(DEFAULT_NORMALIZATION)
    normalization.update(load_json_dict(args.normalization_json))

    unusual_precursors = dict(DEFAULT_UNUSUAL_PRECURSORS)
    unusual_precursors.update(load_json_dict(args.unusual_precursors_json))

    normalized_sets = []
    normalization_notes_all = []
    qc_rows = []

    for _, row in df.iterrows():
        raw_parts = split_precursor_set(row.get(args.precursor_col, ""))
        norm_parts, norm_notes = normalize_precursors(raw_parts, normalization)

        target_elems = infer_target_elements(row, ignore_elements)

        qc = score_route_qc(
            normalized_parts=norm_parts,
            target_elements=target_elems,
            ignore_elements=ignore_elements,
            unusual_precursors=unusual_precursors,
            penalty_elemental=args.penalty_elemental,
            penalty_unusual=args.penalty_unusual,
            penalty_extra_element=args.penalty_extra_element,
            penalty_missing_element=args.penalty_missing_element,
        )

        normalized_sets.append(join_precursor_set(norm_parts))
        normalization_notes_all.append("; ".join(norm_notes))
        qc_rows.append(qc)

    qc_df = pd.DataFrame(qc_rows)
    out = df.copy()
    out["normalized_precursor_set"] = normalized_sets
    out["precursor_normalization_notes"] = normalization_notes_all

    for c in qc_df.columns:
        out[c] = qc_df[c]

    if args.sort_by_qc:
        sort_cols = []
        ascending = []

        if "precursor_qc_score" in out.columns:
            sort_cols.append("precursor_qc_score")
            ascending.append(False)
        if "stage35_v21_score" in out.columns:
            sort_cols.append("stage35_v21_score")
            ascending.append(False)
        elif "final_route_rank" in out.columns:
            sort_cols.append("final_route_rank")
            ascending.append(True)

        if sort_cols:
            out = out.sort_values(sort_cols, ascending=ascending, kind="mergesort").reset_index(drop=True)
            out["qc_adjusted_route_rank"] = range(1, len(out) + 1)

    out.to_csv(output_csv, index=False)

    level_counts = out["precursor_qc_level"].fillna("").value_counts().to_dict()
    status_counts = out["precursor_qc_status"].fillna("").value_counts().to_dict()

    summary = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "output_md": str(output_md),
        "n_routes": int(len(out)),
        "top_n": int(args.top_n),
        "precursor_col": args.precursor_col,
        "normalization_rules_count": len(normalization),
        "unusual_precursors_count": len(unusual_precursors),
        "ignore_elements": sorted(ignore_elements),
        "penalties": {
            "penalty_elemental": args.penalty_elemental,
            "penalty_unusual": args.penalty_unusual,
            "penalty_extra_element": args.penalty_extra_element,
            "penalty_missing_element": args.penalty_missing_element,
        },
        "precursor_qc_level_counts": level_counts,
        "precursor_qc_status_counts": status_counts,
        "n_with_normalization": int((out["precursor_normalization_notes"].fillna("") != "").sum()),
        "n_with_extra_elements": int(pd.to_numeric(out["has_extra_elements_qc"], errors="coerce").fillna(0).sum()),
        "n_with_elemental_precursor": int(pd.to_numeric(out["has_elemental_precursor_qc"], errors="coerce").fillna(0).sum()),
        "n_with_unusual_precursor": int(pd.to_numeric(out["has_unusual_precursor_qc"], errors="coerce").fillna(0).sum()),
        "n_with_missing_target_elements": int(pd.to_numeric(out["has_missing_target_elements_qc"], errors="coerce").fillna(0).sum()),
        "claim_boundary": "qc_is_rule_based_not_experimental_validation",
    }

    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append("# Route precursor QC report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(summary, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## Top routes with precursor QC")
    lines.append("")
    lines.append(to_markdown_table(out, args.top_n))
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `normalized_precursor_set` applies simple precursor-name normalization rules.")
    lines.append("- `precursor_qc_score` is a rule-based route quality indicator.")
    lines.append("- `precursor_qc_level` flags whether a route passes precursor-level checks.")
    lines.append("- This QC is not experimental proof; it only highlights routes that need additional chemical review.")
    lines.append("")

    output_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"[SAVE] {output_csv}")
    print(f"[SAVE] {output_md}")
    print(f"[SAVE] {summary_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
