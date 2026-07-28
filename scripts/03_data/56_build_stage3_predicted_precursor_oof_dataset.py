#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


FORMULA_RE = re.compile(r"([A-Z][a-z]?)")
LIGHT_OR_COUNTER = {"H", "O", "C", "N", "F", "Cl", "Br", "I", "S", "P"}
CORE_METHODS = {"solid_state", "solution", "melt_arc"}
FAMILY_PATTERNS = {
    "acetate": re.compile(r"CH3COO|C2H3O2|acetate", re.I),
    "carbonate": re.compile(r"CO3|carbonate", re.I),
    "elemental": re.compile(r"^[A-Z][a-z]?$"),
    "halide": re.compile(r"Cl|Br|I|F|chloride|bromide|iodide|fluoride", re.I),
    "hydroxide": re.compile(r"OH|hydroxide", re.I),
    "nitrate": re.compile(r"NO3|nitrate", re.I),
    "oxide": re.compile(r"O[0-9]*|oxide", re.I),
    "phosphate": re.compile(r"PO4|phosphate", re.I),
    "sulfate": re.compile(r"SO4|sulfate", re.I),
}
COMMON_OXIDES = {
    "Li": "Li2O", "Na": "Na2O", "K": "K2O", "Mg": "MgO", "Ca": "CaO", "Sr": "SrO", "Ba": "BaO",
    "Al": "Al2O3", "Ti": "TiO2", "Zr": "ZrO2", "Hf": "HfO2", "V": "V2O5", "Nb": "Nb2O5",
    "Ta": "Ta2O5", "Cr": "Cr2O3", "Mo": "MoO3", "W": "WO3", "Mn": "MnO2", "Fe": "Fe2O3",
    "Co": "Co3O4", "Ni": "NiO", "Cu": "CuO", "Zn": "ZnO", "Y": "Y2O3", "La": "La2O3",
    "Ce": "CeO2", "Pr": "Pr6O11", "Nd": "Nd2O3", "Sm": "Sm2O3", "Gd": "Gd2O3", "Bi": "Bi2O3",
    "Si": "SiO2", "Sn": "SnO2", "Pb": "PbO", "B": "B2O3", "P": "P2O5",
}


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_builtin(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def parse_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [str(x) for x in obj if str(x).strip()]
    except Exception:
        pass
    return [x.strip() for x in re.split(r"\s*\+\s*|;", text) if x.strip()]


def dump_list(items: Sequence[str]) -> str:
    return json.dumps([str(x) for x in items if str(x).strip()], ensure_ascii=False)


def elements(text: str) -> Set[str]:
    return set(FORMULA_RE.findall(str(text)))


def target_source_elements(formula: str) -> Set[str]:
    elems = elements(formula) - {"O"}
    return elems or elements(formula)


def extra_forbidden(labels: Iterable[str], target: Set[str]) -> Set[str]:
    out: Set[str] = set()
    for lab in labels:
        out |= elements(str(lab)) - target - LIGHT_OR_COUNTER
    return out


def generated_formula(el: str, method: str, target: Set[str]) -> str:
    if method == "melt_arc":
        return el
    if el in {"Li", "Na", "K"}:
        return f"{el}2CO3" if "O" in target else el
    if el in {"Mg", "Ca", "Sr", "Ba"}:
        return f"{el}CO3" if "O" in target else el
    return COMMON_OXIDES.get(el, el)


def repair_set(labels: Sequence[str], formula: str, method: str) -> Tuple[List[str], List[str]]:
    target = target_source_elements(formula)
    kept: List[str] = []
    sources: List[str] = []
    covered: Set[str] = set()
    for lab in labels:
        src = elements(lab) & target
        if not src or extra_forbidden([lab], target):
            continue
        if src <= covered:
            continue
        kept.append(str(lab))
        sources.append("pseudo_neighbor_kept")
        covered |= src
    for el in sorted(target - covered):
        kept.append(generated_formula(el, method, target))
        sources.append("pseudo_repair_generated")
    return kept, sources


def set_metrics(true_labels: Sequence[str], pred_labels: Sequence[str]) -> Dict[str, float]:
    t = set(str(x) for x in true_labels if str(x).strip())
    p = set(str(x) for x in pred_labels if str(x).strip())
    inter = len(t & p)
    precision = inter / len(p) if p else 0.0
    recall = inter / len(t) if t else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = len(t | p)
    return {
        "precursor_exact": float(t == p),
        "precursor_precision": precision,
        "precursor_recall": recall,
        "precursor_f1_to_true": f1,
        "precursor_jaccard_to_true": inter / union if union else 1.0,
    }


def family_counts(labels: Sequence[str]) -> Dict[str, float]:
    counts = {name: 0.0 for name in FAMILY_PATTERNS}
    counts["other_salt"] = 0.0
    counts["unknown"] = 0.0
    for lab in labels:
        matched = False
        for name, pat in FAMILY_PATTERNS.items():
            if pat.search(str(lab)):
                counts[name] += 1.0
                matched = True
                break
        if not matched:
            counts["unknown"] += 1.0
    n = max(float(len(labels)), 1.0)
    out = {}
    for name, val in counts.items():
        out[f"precursor_family_count__{name}"] = float(val)
        out[f"precursor_family_frac__{name}"] = float(val / n)
    return out


def write_table(df: pd.DataFrame, path_base: Path) -> None:
    csv_path = path_base.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(path_base.with_suffix(".parquet"), index=False)
    except Exception as exc:
        path_base.with_suffix(".parquet.SKIPPED.txt").write_text(str(exc), encoding="utf-8")


def pseudo_predict_train(train: pd.DataFrame, feature_cols: Sequence[str], n_neighbors: int) -> pd.DataFrame:
    out = train.copy()
    x = out[list(feature_cols)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    x = StandardScaler().fit_transform(x)
    pred_sets: List[List[str]] = [[] for _ in range(len(out))]
    pred_sources: List[List[str]] = [[] for _ in range(len(out))]
    for method, idx_values in out.groupby("reaction_method", sort=False).groups.items():
        idx = np.asarray(list(idx_values), dtype=int)
        if len(idx) < 2:
            for i in idx:
                labels, src = repair_set([], str(out.at[i, "formula"]), str(method))
                pred_sets[i], pred_sources[i] = labels, src
            continue
        nn = NearestNeighbors(n_neighbors=min(max(2, n_neighbors), len(idx)), metric="euclidean")
        nn.fit(x[idx])
        _, nbr = nn.kneighbors(x[idx])
        for local_i, row_i in enumerate(idx):
            chosen: List[str] = []
            for local_j in nbr[local_i]:
                cand_i = int(idx[int(local_j)])
                if cand_i == row_i:
                    continue
                chosen = parse_list(out.at[cand_i, "predicted_precursor_set_chem_checked"]) or parse_list(out.at[cand_i, "true_precursor_set"])
                if chosen:
                    break
            repaired, sources = repair_set(chosen, str(out.at[row_i, "formula"]), str(method))
            pred_sets[row_i], pred_sources[row_i] = repaired, sources
    for i in range(len(out)):
        true_set = parse_list(out.at[i, "true_precursor_set"])
        pred = pred_sets[i]
        out.at[i, "raw_predicted_precursor_set"] = dump_list(pred)
        out.at[i, "predicted_precursor_set_chem_checked"] = dump_list(pred)
        out.at[i, "precursors_text"] = " + ".join(pred)
        out.at[i, "precursor_input_source"] = "pseudo_predicted"
        out.at[i, "precursor_input_mode"] = "pseudo_predicted"
        out.at[i, "precursor_source_mix"] = dump_list(pred_sources[i])
        out.at[i, "contains_open_generated_precursor"] = int(any("generated" in s for s in pred_sources[i]))
        out.at[i, "contains_repair_precursor"] = int(any("repair" in s for s in pred_sources[i]))
        out.at[i, "contains_raw_model_precursor"] = 0
        out.at[i, "precursor_set_size"] = len(pred)
        target = target_source_elements(str(out.at[i, "formula"]))
        covered = set().union(*(elements(p) & target for p in pred)) if pred else set()
        out.at[i, "target_source_elements"] = dump_list(sorted(target))
        out.at[i, "covered_source_elements"] = dump_list(sorted(covered))
        out.at[i, "missing_source_elements"] = dump_list(sorted(target - covered))
        out.at[i, "extra_forbidden_elements"] = dump_list(sorted(extra_forbidden(pred, target)))
        out.at[i, "precursor_check_status"] = "ok" if target <= covered and not extra_forbidden(pred, target) else "repaired_with_residual_issue"
        out.at[i, "precursor_confidence_score"] = float(np.clip(1.0 - 0.15 * len(target - covered) - 0.1 * len(extra_forbidden(pred, target)), 0, 1))
        for k, v in set_metrics(true_set, pred).items():
            out.at[i, k] = v
        fam = family_counts(pred)
        for k, v in fam.items():
            if k in out.columns:
                out.at[i, k] = v
            else:
                out[k] = 0.0
                out.at[i, k] = v
    return out


def normalize_split(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = df.copy()
    out["precursor_input_mode"] = mode
    out["contains_repair_precursor"] = out.get("contains_repair_precursor", 0)
    out = out.rename(columns={"precursor_f1": "precursor_f1_to_true", "precursor_jaccard": "precursor_jaccard_to_true"})
    if "precursor_f1_to_true" not in out.columns:
        out["precursor_f1_to_true"] = 0.0
    if "precursor_jaccard_to_true" not in out.columns:
        out["precursor_jaccard_to_true"] = 0.0
    return out


def summarize(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "rows": int(len(df)),
        "precursor_input_mode": {str(k): int(v) for k, v in df["precursor_input_mode"].value_counts().items()},
        "check_status": {str(k): int(v) for k, v in df["precursor_check_status"].value_counts().items()},
        "mean_precursor_f1": float(pd.to_numeric(df["precursor_f1_to_true"], errors="coerce").mean()),
        "mean_precursor_jaccard": float(pd.to_numeric(df["precursor_jaccard_to_true"], errors="coerce").mean()),
        "open_generated_rows": int(pd.to_numeric(df.get("contains_open_generated_precursor", 0), errors="coerce").fillna(0).sum()),
        "repair_rows": int(pd.to_numeric(df.get("contains_repair_precursor", 0), errors="coerce").fillna(0).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Stage3 v3 pseudo-OOF predicted-precursor dataset to reduce train/test precursor input mismatch.")
    ap.add_argument("--input_dir", default="data/interim/generative/stage3_condition_dataset_chem_checked/method_stratified_v5_20260610")
    ap.add_argument("--output_dir", default="data/interim/generative/stage3_condition_dataset_predprec_oof_v3_20260610")
    ap.add_argument("--n_neighbors", type=int, default=8)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    schema = json.loads((input_dir / "schema.json").read_text(encoding="utf-8"))
    feature_cols = [c for c in schema.get("feature_cols", []) if c.startswith("feat_")]
    if not feature_cols:
        probe = pd.read_csv(input_dir / "train.csv", nrows=1)
        feature_cols = [c for c in probe.columns if c.startswith("feat_")]

    train = pd.read_csv(input_dir / "train.csv")
    val = pd.read_csv(input_dir / "val.csv")
    test = pd.read_csv(input_dir / "test.csv")
    train_v3 = pseudo_predict_train(train, feature_cols, args.n_neighbors)
    val_v3 = normalize_split(val, "val_predicted")
    test_v3 = normalize_split(test, "test_predicted")
    train_v3["split"] = "train"
    val_v3["split"] = "val"
    test_v3["split"] = "test"

    for split, df in [("train", train_v3), ("val", val_v3), ("test", test_v3)]:
        write_table(df, output_dir / split)
        meta_cols = [c for c in [
            "sample_index", "sample_id", "material_id", "formula", "reaction_method", "is_core_method",
            "source_dataset", "split", "precursor_input_mode", "precursor_input_source",
            "precursor_check_status", "precursor_set_size", "precursor_f1_to_true",
            "precursor_jaccard_to_true", "contains_open_generated_precursor", "contains_repair_precursor",
        ] if c in df.columns]
        df[meta_cols].to_csv(output_dir / f"{split}_meta.csv", index=False)

    schema["stage3_v3_predicted_precursor"] = {
        "source": str(input_dir),
        "train_mode": "pseudo_predicted_by_leave-one-neighbor_retrieval",
        "feature_cols": feature_cols,
    }
    write_json(output_dir / "schema.json", schema)
    summary = {"config": vars(args), "splits": {s: summarize(d) for s, d in [("train", train_v3), ("val", val_v3), ("test", test_v3)]}}
    write_json(output_dir / "summary.json", summary)
    report = ["# Stage3 Predicted-Precursor OOF/Pseudo Dataset v3", "", json.dumps(to_builtin(summary), ensure_ascii=False, indent=2)]
    (output_dir / "stage3_predprec_oof_dataset_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
