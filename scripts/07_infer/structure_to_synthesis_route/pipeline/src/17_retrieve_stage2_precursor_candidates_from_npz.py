#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, List, Set, Tuple

import numpy as np
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


def extract_elements_from_formula(s: str) -> Set[str]:
    toks = re.findall(r"[A-Z][a-z]?", str(s))
    return {t for t in toks if t in ELEMENTS}


def formula_from_row(row: pd.Series) -> str:
    for c in [
        "formula", "formula_x", "formula_y", "target_formula",
        "pretty_formula", "composition", "material_formula"
    ]:
        if c in row and str(row[c]).strip() and str(row[c]).lower() != "nan":
            return str(row[c]).strip()

    for c in ["sample_id", "material_id", "split_group", "id"]:
        if c in row:
            s = str(row[c])
            if "__" in s:
                return s.split("__", 1)[1]

    return ""


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def load_precursor_names(path: Path) -> List[str]:
    obj = json.loads(path.read_text())

    if isinstance(obj, list):
        return [str(x).replace("label_prec__", "") for x in obj]

    if isinstance(obj, dict):
        # id -> name
        if all(str(k).isdigit() for k in obj.keys()):
            return [str(obj[str(i)]).replace("label_prec__", "") for i in range(len(obj))]

        # name -> id
        inv = sorted([(int(v), k) for k, v in obj.items()], key=lambda x: x[0])
        return [str(k).replace("label_prec__", "") for _, k in inv]

    raise TypeError(f"Unsupported precursor_names type: {type(obj)}")


def choose_label_matrix(npz: Any, vocab_size: int) -> Tuple[str, np.ndarray]:
    preferred = ["y_multi_hot", "y", "y_set", "labels", "target", "targets"]

    for k in preferred:
        if k in npz.files:
            arr = npz[k]
            if arr.ndim == 2 and arr.shape[1] == vocab_size:
                return k, arr

    for k in npz.files:
        arr = npz[k]
        if arr.ndim == 2 and arr.shape[1] == vocab_size:
            return k, arr

    raise RuntimeError(
        f"Cannot find 2D label matrix with vocab_size={vocab_size}. "
        f"npz keys={npz.files}"
    )


def decode_precursors(y_row: np.ndarray, precursor_names: List[str], threshold: float) -> List[str]:
    idx = np.where(y_row > threshold)[0].tolist()
    return [precursor_names[i] for i in idx if i < len(precursor_names)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrieve historical precursor candidates from meta.csv + npz labels.")
    ap.add_argument("--target_csv", required=True)
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--splits", default="train,val,test,gold_train_holdout")
    ap.add_argument("--precursor_names_json", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--summary_json", default="")
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--min_similarity", type=float, default=0.0)
    ap.add_argument("--label_threshold", type=float, default=0.5)
    ap.add_argument("--ignore_target_elements", default="H,O")
    args = ap.parse_args()

    target_csv = Path(args.target_csv).expanduser().resolve()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    precursor_names_json = Path(args.precursor_names_json).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    summary_json = (
        Path(args.summary_json).expanduser().resolve()
        if args.summary_json
        else output_csv.with_suffix(".summary.json")
    )

    ignore = {x.strip() for x in args.ignore_target_elements.split(",") if x.strip()}
    precursor_names = load_precursor_names(precursor_names_json)
    vocab_size = len(precursor_names)

    target_df = pd.read_csv(target_csv)
    if target_df.empty:
        raise RuntimeError(f"Empty target_csv: {target_csv}")

    target_row = target_df.iloc[0]
    target_formula = formula_from_row(target_row)
    target_core = extract_elements_from_formula(target_formula) - ignore

    if not target_core:
        raise RuntimeError(f"Cannot infer target elements from target_csv={target_csv}")

    rows = []
    seen_sets = set()
    split_summaries = []

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        meta_p = dataset_dir / f"{split}_meta.csv"
        npz_p = dataset_dir / f"{split}.npz"

        if not meta_p.exists() or not npz_p.exists():
            split_summaries.append({
                "split": split,
                "meta": str(meta_p),
                "npz": str(npz_p),
                "status": "missing",
            })
            continue

        meta = pd.read_csv(meta_p)
        npz = np.load(npz_p, allow_pickle=True)
        label_key, y = choose_label_matrix(npz, vocab_size)

        n = min(len(meta), y.shape[0])
        local_used = 0

        for i in range(n):
            row = meta.iloc[i]
            source_formula = formula_from_row(row)
            source_core = extract_elements_from_formula(source_formula) - ignore
            sim = jaccard(target_core, source_core)

            if sim < float(args.min_similarity):
                continue

            precs = decode_precursors(y[i], precursor_names, float(args.label_threshold))
            if not precs:
                continue

            set_key = " || ".join(sorted(precs))
            if set_key in seen_sets:
                continue
            seen_sets.add(set_key)

            precursor_elems = set()
            for p in precs:
                precursor_elems |= extract_elements_from_formula(p)

            coverage = len(target_core & precursor_elems) / max(1, len(target_core))

            rows.append({
                "rank": len(rows) + 1,
                "set_key": set_key,
                "precursor_set": "; ".join(precs),
                "n_precursors": len(precs),
                "count": 0,
                "frequency": 0.0,
                "sample_id": str(target_row.get("sample_id", target_row.get("material_id", ""))),
                "formula": target_formula,
                "formula_x": target_formula,
                "formula_y": target_formula,
                "doi": "",
                "split_group": str(target_row.get("split_group", target_row.get("sample_id", ""))),
                "decode_method": "retrieval_npz",
                "decode_methods_seen": "['retrieval_npz']",
                "sample_rank_min": len(rows) + 1,
                "sample_rank_mean": len(rows) + 1,
                "sample_rank_max": len(rows) + 1,
                "is_retrieval_candidate": True,
                "retrieval_source_formula": source_formula,
                "retrieval_source_elements": ";".join(sorted(source_core)),
                "retrieval_similarity": sim,
                "retrieval_element_coverage": coverage,
                "retrieval_source_split": split,
                "retrieval_label_key": label_key,
                "retrieval_source_index": int(i),
                "retrieval_source_file": str(npz_p),
            })
            local_used += 1

        split_summaries.append({
            "split": split,
            "meta_rows": int(len(meta)),
            "npz_rows": int(y.shape[0]),
            "label_key": label_key,
            "used_rows_before_dedup_or_topk": int(local_used),
        })

    columns = [
        "rank", "set_key", "precursor_set", "n_precursors",
        "count", "frequency", "sample_id", "formula", "formula_x", "formula_y",
        "doi", "split_group", "decode_method", "decode_methods_seen",
        "sample_rank_min", "sample_rank_mean", "sample_rank_max",
        "is_retrieval_candidate", "retrieval_source_formula",
        "retrieval_source_elements", "retrieval_similarity",
        "retrieval_element_coverage",
        "retrieval_source_split", "retrieval_label_key", "retrieval_source_index",
        "retrieval_source_file",
    ]

    out = pd.DataFrame(rows, columns=columns)

    if not out.empty:
        out = out.sort_values(
            ["retrieval_element_coverage", "retrieval_similarity", "n_precursors"],
            ascending=[False, False, True],
        ).head(int(args.top_k)).copy()
        out["rank"] = range(1, len(out) + 1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    summary = {
        "target_csv": str(target_csv),
        "target_formula": target_formula,
        "target_core_elements": sorted(target_core),
        "dataset_dir": str(dataset_dir),
        "precursor_names_json": str(precursor_names_json),
        "vocab_size": int(vocab_size),
        "retrieved_rows": int(len(out)),
        "output_csv": str(output_csv),
        "top_k": int(args.top_k),
        "min_similarity": float(args.min_similarity),
        "label_threshold": float(args.label_threshold),
        "splits": split_summaries,
    }

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[SAVE]", output_csv)
    print("[SAVE]", summary_json)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
