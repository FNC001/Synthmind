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


CANDIDATE_SET_COLS = [
    "precursor_set",
    "decoded_precursor_set",
    "candidate_precursor_set",
    "pred_precursor_set",
    "labels",
    "pred_labels",
    "sampled_labels",
]


def extract_elements_from_formula(s: str) -> Set[str]:
    toks = re.findall(r"[A-Z][a-z]?", str(s))
    return {t for t in toks if t in ELEMENTS}


def formula_from_row(row: pd.Series) -> str:
    for c in [
        "formula", "formula_x", "formula_y", "target_formula",
        "pretty_formula", "composition", "material_formula"
    ]:
        if c in row.index:
            v = str(row[c]).strip()
            if v and v.lower() != "nan":
                return v

    for c in ["sample_id", "material_id", "split_group", "id"]:
        if c in row.index:
            s = str(row[c]).strip()
            if "__" in s:
                return s.split("__", 1)[1]

    return ""


def parse_list_cell(v: Any) -> List[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if v is None:
        return []
    s = str(v).strip()
    if not s or s.lower() == "nan":
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


def get_precursor_labels(row: pd.Series) -> List[str]:
    for c in CANDIDATE_SET_COLS:
        if c in row.index:
            labels = parse_list_cell(row[c])
            if labels:
                return labels
    return []


def stable_set_key(labels: List[str]) -> str:
    return " || ".join(sorted([str(x).strip() for x in labels if str(x).strip()]))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Composition-constrained rerank/filter for Stage2 sampled precursor candidates."
    )
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--summary_json", default="")
    ap.add_argument("--ignore_target_elements", default="H,O")
    ap.add_argument("--min_coverage", type=float, default=0.0)
    ap.add_argument("--drop_zero_overlap", action="store_true")
    ap.add_argument("--coverage_weight", type=float, default=20.0)
    ap.add_argument("--extra_penalty_weight", type=float, default=5.0)
    ap.add_argument("--rank_weight", type=float, default=0.01)
    ap.add_argument("--top_n_per_sample", type=int, default=0)
    ap.add_argument("--dedup", action="store_true")
    args = ap.parse_args()

    input_csv = Path(args.input_csv).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    summary_json = (
        Path(args.summary_json).expanduser().resolve()
        if args.summary_json
        else output_csv.with_suffix(".summary.json")
    )

    ignore = {x.strip() for x in str(args.ignore_target_elements).split(",") if x.strip()}
    df = pd.read_csv(input_csv)

    if df.empty:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        summary_json.write_text(json.dumps({
            "input_csv": str(input_csv),
            "output_csv": str(output_csv),
            "input_rows": 0,
            "output_rows": 0,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[SAVE]", output_csv)
        print("[SAVE]", summary_json)
        return

    rows = []
    group_cols = ["sample_id"] if "sample_id" in df.columns else []

    if not group_cols:
        df["sample_id"] = "infer_sample"
        group_cols = ["sample_id"]

    for sid, g in df.groupby(group_cols[0], sort=False):
        g = g.copy()
        target_formula = formula_from_row(g.iloc[0])
        target_core = extract_elements_from_formula(target_formula) - ignore

        local_rows = []
        seen = set()

        for local_i, (_, row) in enumerate(g.iterrows()):
            labels = get_precursor_labels(row)
            key = stable_set_key(labels)

            if args.dedup and key in seen:
                continue
            seen.add(key)

            precursor_elements = set()
            for lab in labels:
                precursor_elements |= extract_elements_from_formula(lab)

            hit = target_core & precursor_elements
            missing = target_core - precursor_elements
            extra = precursor_elements - target_core - ignore

            coverage = len(hit) / max(1, len(target_core))
            extra_penalty = len(extra) / max(1, len(precursor_elements - ignore))
            zero_overlap = len(hit) == 0

            if coverage < float(args.min_coverage):
                continue
            if args.drop_zero_overlap and zero_overlap:
                continue

            old_rank = None
            for rc in ["sample_rank", "rank", "candidate_rank"]:
                if rc in row.index:
                    try:
                        old_rank = float(row[rc])
                        break
                    except Exception:
                        pass
            if old_rank is None:
                old_rank = float(local_i)

            score = (
                float(args.coverage_weight) * coverage
                - float(args.extra_penalty_weight) * extra_penalty
                - float(args.rank_weight) * old_rank
            )

            rec = row.to_dict()
            rec["composition_constraint_target_formula"] = target_formula
            rec["composition_constraint_target_elements"] = ";".join(sorted(target_core))
            rec["composition_constraint_precursor_elements"] = ";".join(sorted(precursor_elements))
            rec["composition_constraint_element_hit"] = ";".join(sorted(hit))
            rec["composition_constraint_element_missing"] = ";".join(sorted(missing))
            rec["composition_constraint_extra_elements"] = ";".join(sorted(extra))
            rec["composition_constraint_coverage"] = float(coverage)
            rec["composition_constraint_extra_penalty"] = float(extra_penalty)
            rec["composition_constraint_zero_overlap"] = bool(zero_overlap)
            rec["composition_constraint_score"] = float(score)
            rec["composition_constraint_old_rank"] = float(old_rank)
            rec["composition_constraint_set_key"] = key
            local_rows.append(rec)

        local_rows = sorted(
            local_rows,
            key=lambda r: (
                -float(r["composition_constraint_score"]),
                -float(r["composition_constraint_coverage"]),
                float(r["composition_constraint_extra_penalty"]),
                float(r["composition_constraint_old_rank"]),
            ),
        )

        if int(args.top_n_per_sample) > 0:
            local_rows = local_rows[: int(args.top_n_per_sample)]

        for new_rank, rec in enumerate(local_rows):
            rec["composition_constraint_rank"] = int(new_rank)
            rows.append(rec)

    out = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    summary = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "input_rows": int(len(df)),
        "output_rows": int(len(out)),
        "ignore_target_elements": sorted(ignore),
        "min_coverage": float(args.min_coverage),
        "drop_zero_overlap": bool(args.drop_zero_overlap),
        "coverage_weight": float(args.coverage_weight),
        "extra_penalty_weight": float(args.extra_penalty_weight),
        "rank_weight": float(args.rank_weight),
        "top_n_per_sample": int(args.top_n_per_sample),
        "dedup": bool(args.dedup),
    }

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[SAVE]", output_csv)
    print("[SAVE]", summary_json)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
