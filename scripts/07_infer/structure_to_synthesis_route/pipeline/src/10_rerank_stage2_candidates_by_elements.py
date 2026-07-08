#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd


ELEMENTS = {
    "H","He","Li","Be","B","C","N","O","F","Ne",
    "Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca",
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr",
    "Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn",
    "Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd",
    "Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb",
    "Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th",
    "Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm",
    "Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds",
    "Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og",
}

COMMON_SOURCE_ELEMENTS = {"H", "O", "C", "N", "P", "S", "Cl", "F", "I", "Br"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_list_cell(v: Any) -> List[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if v is None or pd.isna(v):
        return []
    s = str(v).strip()
    if not s:
        return []
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass
    if ";" in s:
        return [x.strip() for x in s.split(";") if x.strip()]
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    return [s]


def extract_elements_from_formula(s: str) -> Set[str]:
    toks = re.findall(r"[A-Z][a-z]?", str(s))
    return {t for t in toks if t in ELEMENTS}


def precursor_elements(v: Any) -> Set[str]:
    elems: Set[str] = set()
    for item in parse_list_cell(v):
        elems |= extract_elements_from_formula(item)
    return elems


def target_elements_from_row(row: pd.Series) -> Set[str]:
    for c in ["target_formula", "formula", "pretty_formula", "composition", "material_formula", "formula_x", "formula_y"]:
        if c in row and str(row[c]).strip() and str(row[c]).lower() != "nan":
            elems = extract_elements_from_formula(str(row[c]))
            if elems:
                return elems

    for c in ["sample_id", "material_id"]:
        if c in row:
            s = str(row[c])
            if "__" in s:
                s = s.split("__", 1)[1]
            elems = extract_elements_from_formula(s)
            if elems:
                return elems

    return set()


def main() -> None:
    ap = argparse.ArgumentParser(description="Composition-aware rerank for Stage2 precursor candidates.")
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--summary_json", default="")
    ap.add_argument("--precursor_col", default="precursor_set")
    ap.add_argument("--rank_col", default="rank")
    ap.add_argument("--top_n", type=int, default=30)
    ap.add_argument("--ignore_target_elements", default="O,H")
    ap.add_argument("--coverage_weight", type=float, default=10.0)
    ap.add_argument("--extra_penalty_weight", type=float, default=2.0)
    ap.add_argument("--rank_weight", type=float, default=0.03)
    args = ap.parse_args()

    input_csv = Path(args.input_csv).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    summary_json = Path(args.summary_json).expanduser().resolve() if args.summary_json else output_csv.with_suffix(".summary.json")

    df = pd.read_csv(input_csv)
    if args.precursor_col not in df.columns:
        raise KeyError(f"Missing precursor column: {args.precursor_col}. columns={list(df.columns)}")

    ignore_target = {x.strip() for x in args.ignore_target_elements.split(",") if x.strip()}
    group_cols = [c for c in ["sample_id", "material_id"] if c in df.columns]
    grouped = [(None, df)] if not group_cols else df.groupby(group_cols, dropna=False, sort=False)

    out_parts = []
    group_summaries = []

    for group_key, g in grouped:
        g = g.copy()
        if args.rank_col in g.columns:
            g = g.sort_values(args.rank_col)

        target_elems = target_elements_from_row(g.iloc[0])
        target_core = target_elems - ignore_target

        coverages = []
        extra_penalties = []
        hit_elems_list = []
        missing_elems_list = []
        precursor_elems_list = []
        scores = []

        max_rank = max(float(g[args.rank_col].max()) if args.rank_col in g.columns else len(g), 1.0)

        for _, row in g.iterrows():
            pe = precursor_elements(row[args.precursor_col])
            precursor_elems_list.append(";".join(sorted(pe)))

            if target_core:
                hit = pe & target_core
                missing = target_core - pe
                coverage = len(hit) / len(target_core)
            else:
                hit = set()
                missing = set()
                coverage = 0.0

            # Penalize non-target, non-common-source elements.
            extra = pe - target_core - COMMON_SOURCE_ELEMENTS
            extra_penalty = len(extra) / max(1, len(pe))

            old_rank = float(row[args.rank_col]) if args.rank_col in g.columns else float(len(scores))
            rank_penalty = old_rank / max_rank

            score = (
                args.coverage_weight * coverage
                - args.extra_penalty_weight * extra_penalty
                - args.rank_weight * rank_penalty
            )

            coverages.append(coverage)
            extra_penalties.append(extra_penalty)
            hit_elems_list.append(";".join(sorted(hit)))
            missing_elems_list.append(";".join(sorted(missing)))
            scores.append(score)

        g["target_elements"] = ";".join(sorted(target_elems))
        g["target_core_elements"] = ";".join(sorted(target_core))
        g["precursor_elements"] = precursor_elems_list
        g["element_hit"] = hit_elems_list
        g["element_missing"] = missing_elems_list
        g["element_coverage"] = coverages
        g["extra_element_penalty"] = extra_penalties
        g["element_rerank_score"] = scores

        g = g.sort_values(
            ["element_rerank_score", "element_coverage", args.rank_col if args.rank_col in g.columns else "element_rerank_score"],
            ascending=[False, False, True],
        ).head(int(args.top_n)).copy()

        g["original_rank"] = g[args.rank_col] if args.rank_col in g.columns else range(len(g))
        g[args.rank_col] = range(1, len(g) + 1)

        out_parts.append(g)

        group_summaries.append({
            "group": str(group_key),
            "target_elements": sorted(target_elems),
            "target_core_elements": sorted(target_core),
            "input_rows": int(len(g)),
            "top_n": int(args.top_n),
            "max_element_coverage": float(max(coverages) if coverages else 0.0),
            "n_candidates_with_coverage_gt0": int(sum(c > 0 for c in coverages)),
            "n_candidates_full_coverage": int(sum(c >= 1.0 for c in coverages)),
        })

    out = pd.concat(out_parts, ignore_index=True) if out_parts else df.iloc[0:0].copy()

    ensure_dir(output_csv.parent)
    out.to_csv(output_csv, index=False)

    summary = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "input_rows": int(len(df)),
        "output_rows": int(len(out)),
        "top_n": int(args.top_n),
        "coverage_weight": float(args.coverage_weight),
        "extra_penalty_weight": float(args.extra_penalty_weight),
        "rank_weight": float(args.rank_weight),
        "ignore_target_elements": sorted(ignore_target),
        "groups": group_summaries,
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[SAVE]", output_csv)
    print("[SAVE]", summary_json)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
