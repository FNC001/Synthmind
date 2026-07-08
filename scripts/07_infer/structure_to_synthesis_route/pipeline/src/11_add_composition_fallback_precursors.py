#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Set

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

COMMON_PRECURSORS: Dict[str, List[str]] = {
    "Li": ["Li2CO3", "LiNO3", "LiOH"],
    "Na": ["Na2CO3", "NaNO3", "NaOH"],
    "K": ["K2CO3", "KNO3"],
    "Mg": ["MgO", "MgCO3", "Mg(NO3)2"],
    "Ca": ["CaCO3", "CaO", "Ca(NO3)2"],
    "Sr": ["SrCO3", "SrO", "Sr(NO3)2"],
    "Ba": ["BaCO3", "BaO", "Ba(NO3)2"],
    "Al": ["Al2O3", "Al(NO3)3", "Al(OH)3"],
    "Ga": ["Ga2O3", "Ga(NO3)3"],
    "In": ["In2O3", "In(NO3)3"],
    "Ti": ["TiO2", "TiO"],
    "Zr": ["ZrO2", "Zr(NO3)4"],
    "Hf": ["HfO2"],
    "V": ["V2O5", "VO2"],
    "Nb": ["Nb2O5"],
    "Ta": ["Ta2O5"],
    "Cr": ["Cr2O3", "Cr(NO3)3"],
    "Mn": ["MnO2", "Mn2O3", "MnO", "MnCO3"],
    "Fe": ["Fe2O3", "Fe3O4", "Fe(NO3)3·9H2O"],
    "Co": ["Co3O4", "CoO", "Co(NO3)2·6H2O"],
    "Ni": ["NiO", "Ni(NO3)2·6H2O"],
    "Cu": ["CuO", "Cu(NO3)2"],
    "Zn": ["ZnO", "Zn(NO3)2"],
    "La": ["La2O3", "La(NO3)3"],
    "Ce": ["CeO2", "Ce2(CO3)3"],
    "Pr": ["Pr6O11", "Pr(NO3)3"],
    "Nd": ["Nd2O3", "Nd(NO3)3"],
    "Sm": ["Sm2O3", "Sm(NO3)3"],
    "Eu": ["Eu2O3", "Eu(NO3)3"],
    "Gd": ["Gd2O3", "Gd(NO3)3"],
    "Y": ["Y2O3", "Y(NO3)3"],
    "Si": ["SiO2", "Si"],
    "Ge": ["GeO2", "Ge"],
    "Sn": ["SnO2", "SnO", "Sn"],
    "Pb": ["PbO", "PbO2", "Pb(NO3)2"],
    "P": ["NH4H2PO4", "(NH4)2HPO4", "H3PO4"],
    "S": ["S", "NH4HSO4"],
    "Se": ["Se", "SeO2"],
    "Te": ["Te", "TeO2"],
    "As": ["As2O3", "As2O5"],
    "Sb": ["Sb2O3", "Sb2O5", "Sb"],
    "Bi": ["Bi2O3", "Bi(NO3)3"],
    "Mo": ["MoO3", "(NH4)6Mo7O24·4H2O"],
    "W": ["WO3", "(NH4)10W12O41"],
}


def extract_elements_from_formula(s: str) -> Set[str]:
    toks = re.findall(r"[A-Z][a-z]?", str(s))
    return {t for t in toks if t in ELEMENTS}


def target_formula_from_row(row: pd.Series) -> str:
    for c in ["formula", "formula_x", "formula_y", "target_formula", "pretty_formula"]:
        if c in row and str(row[c]).strip() and str(row[c]).lower() != "nan":
            return str(row[c]).strip()
    for c in ["sample_id", "material_id"]:
        if c in row:
            s = str(row[c])
            if "__" in s:
                return s.split("__", 1)[1]
    return ""


def make_fallback_sets(target_elems: List[str], max_sets: int) -> List[List[str]]:
    pools = []
    for e in target_elems:
        pools.append(COMMON_PRECURSORS.get(e, [e]))

    if not pools:
        return []

    # Deterministic combinations: all-first, then vary one element at a time.
    sets = []
    base = [p[0] for p in pools]
    sets.append(base)

    max_len = max(len(p) for p in pools)
    for j in range(max_len):
        cand = []
        for p in pools:
            cand.append(p[min(j, len(p) - 1)])
        if cand not in sets:
            sets.append(cand)

    for i, p in enumerate(pools):
        for alt in p:
            cand = list(base)
            cand[i] = alt
            if cand not in sets:
                sets.append(cand)
            if len(sets) >= max_sets:
                return sets

    return sets[:max_sets]


def main() -> None:
    ap = argparse.ArgumentParser(description="Add composition-complete fallback precursor sets.")
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--summary_json", default="")
    ap.add_argument("--top_n_fallback", type=int, default=20)
    ap.add_argument("--ignore_elements", default="O,H")
    ap.add_argument("--rank_col", default="rank")
    ap.add_argument("--precursor_col", default="precursor_set")
    args = ap.parse_args()

    input_csv = Path(args.input_csv).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    summary_json = Path(args.summary_json).expanduser().resolve() if args.summary_json else output_csv.with_suffix(".summary.json")

    df = pd.read_csv(input_csv)
    ignore = {x.strip() for x in args.ignore_elements.split(",") if x.strip()}

    rows = [df]
    added = []

    group_cols = [c for c in ["sample_id", "material_id"] if c in df.columns]
    grouped = [(None, df)] if not group_cols else df.groupby(group_cols, dropna=False, sort=False)

    for group_key, g in grouped:
        first = g.iloc[0]
        formula = target_formula_from_row(first)
        elems = sorted(extract_elements_from_formula(formula) - ignore)

        fallback_sets = make_fallback_sets(elems, int(args.top_n_fallback))
        if not fallback_sets:
            continue

        base = first.to_dict()
        start_rank = int(df[args.rank_col].max()) + 1 if args.rank_col in df.columns else len(df) + 1

        new_rows = []
        for i, precs in enumerate(fallback_sets):
            r = dict(base)
            r[args.rank_col] = start_rank + i
            r["original_rank"] = r.get(args.rank_col, start_rank + i)
            r["set_key"] = " || ".join(precs)
            r[args.precursor_col] = "; ".join(precs)
            r["n_precursors"] = len(precs)
            r["count"] = 0
            r["frequency"] = 0.0
            r["decode_method"] = "composition_fallback"
            r["decode_methods_seen"] = "['composition_fallback']"
            r["sample_rank_min"] = start_rank + i
            r["sample_rank_mean"] = start_rank + i
            r["sample_rank_max"] = start_rank + i
            r["is_composition_fallback"] = True
            new_rows.append(r)

        added.extend(new_rows)

    if added:
        add_df = pd.DataFrame(added)
        if "is_composition_fallback" not in df.columns:
            df["is_composition_fallback"] = False
        out = pd.concat([df, add_df], ignore_index=True, sort=False)
    else:
        out = df.copy()
        out["is_composition_fallback"] = False

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    summary = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "input_rows": int(len(df)),
        "added_fallback_rows": int(len(added)),
        "output_rows": int(len(out)),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[SAVE]", output_csv)
    print("[SAVE]", summary_json)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
