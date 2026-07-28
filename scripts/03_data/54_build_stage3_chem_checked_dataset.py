#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set

import numpy as np
import pandas as pd


CORE_METHODS = {"solid_state", "solution", "melt_arc"}
FORMULA_RE = re.compile(r"([A-Z][a-z]?)")
LIGHT_OR_COUNTER = {"H", "O", "C", "N", "F", "Cl", "Br", "I", "S", "P"}
DIATOMIC = {"H": "H2", "N": "N2", "O": "O2", "F": "F2", "Cl": "Cl2", "Br": "Br2", "I": "I2"}
ALKALI = {"Li", "Na", "K", "Rb", "Cs"}
ALKALINE = {"Mg", "Ca", "Sr", "Ba"}
LANTHANOIDS = {"La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"}
COMMON_OXIDES = {
    "Li": "Li2O", "Na": "Na2O", "K": "K2O", "Rb": "Rb2O", "Cs": "Cs2O",
    "Mg": "MgO", "Ca": "CaO", "Sr": "SrO", "Ba": "BaO", "Al": "Al2O3",
    "Ga": "Ga2O3", "In": "In2O3", "Si": "SiO2", "Ge": "GeO2", "Sn": "SnO2",
    "Pb": "PbO", "B": "B2O3", "P": "P2O5", "Ti": "TiO2", "Zr": "ZrO2",
    "Hf": "HfO2", "V": "V2O5", "Nb": "Nb2O5", "Ta": "Ta2O5", "Cr": "Cr2O3",
    "Mo": "MoO3", "W": "WO3", "Mn": "MnO2", "Fe": "Fe2O3", "Co": "Co3O4",
    "Ni": "NiO", "Cu": "CuO", "Zn": "ZnO", "Ag": "Ag2O", "Cd": "CdO",
    "Hg": "HgO", "As": "As2O5", "Sb": "Sb2O3", "Bi": "Bi2O3", "Ru": "RuO2",
    "Rh": "Rh2O3", "Pd": "PdO", "Os": "OsO2", "Ir": "IrO2", "Pt": "PtO2",
    "Au": "Au2O3", "Th": "ThO2", "U": "UO2", "Np": "NpO2", "Pu": "PuO2",
}
for _el in LANTHANOIDS:
    COMMON_OXIDES.setdefault(_el, f"{_el}2O3")
COMMON_VALENCE = {
    "Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1, "Ag": 1,
    "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2, "Zn": 2, "Cd": 2, "Hg": 2,
    "Al": 3, "Ga": 3, "In": 3, "Sc": 3, "Y": 3, "La": 3, "Ce": 3, "Pr": 3,
    "Nd": 3, "Sm": 3, "Eu": 3, "Gd": 3, "Tb": 3, "Dy": 3, "Ho": 3, "Er": 3,
    "Tm": 3, "Yb": 3, "Lu": 3, "Ti": 4, "Zr": 4, "Hf": 4, "Sn": 4, "Pb": 2,
    "Bi": 3, "V": 5, "Nb": 5, "Ta": 5, "Cr": 3, "Mo": 6, "W": 6, "Mn": 2,
    "Fe": 3, "Co": 2, "Ni": 2, "Cu": 2, "Ru": 3, "Rh": 3, "Pd": 2, "Os": 4,
    "Ir": 4, "Pt": 4, "Au": 3, "Th": 4, "U": 4, "Np": 4, "Pu": 4,
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(to_builtin(obj), ensure_ascii=False, indent=2), encoding="utf-8")


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
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    return {k: arr[k] for k in arr.files}


def parse_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    try:
        obj = json.loads(str(value))
        if isinstance(obj, list):
            return [str(x) for x in obj if str(x).strip()]
    except Exception:
        pass
    return []


def elements(text: str) -> Set[str]:
    return set(FORMULA_RE.findall(str(text)))


def target_source_elements(formula: str) -> Set[str]:
    elems = elements(formula) - {"O"}
    return elems or elements(formula)


def precursor_source_elements(labels: Iterable[str], target: Set[str]) -> Set[str]:
    out: Set[str] = set()
    for lab in labels:
        out |= (elements(str(lab)) & target)
    return out


def extra_forbidden_elements(labels: Iterable[str], target: Set[str]) -> Set[str]:
    out: Set[str] = set()
    for lab in labels:
        out |= elements(str(lab)) - target - LIGHT_OR_COUNTER
    return out


def nitrate_formula(el: str) -> str:
    v = int(COMMON_VALENCE.get(el, 3))
    return f"{el}NO3" if v <= 1 else f"{el}(NO3){v}"


def carbonate_formula(el: str) -> str:
    if el in ALKALI:
        return f"{el}2CO3"
    if el in ALKALINE:
        return f"{el}CO3"
    return COMMON_OXIDES.get(el, el)


def generated_formula(el: str, method: str, target: Set[str]) -> str:
    if method == "melt_arc":
        return DIATOMIC.get(el, el)
    if el in {"F", "Cl", "Br", "I", "H", "N"}:
        return DIATOMIC.get(el, el)
    if el in {"S", "Se", "Te", "C"}:
        return el
    if el == "P":
        return "P2O5" if "O" in target else "P"
    if el == "B":
        return "H3BO3" if "O" in target else "B"
    if method == "solution":
        return nitrate_formula(el) if "O" in target else f"{el}Cl{COMMON_VALENCE.get(el, 3)}"
    if method == "solid_state" and "O" in target:
        return carbonate_formula(el) if el in ALKALI or el in ALKALINE else COMMON_OXIDES.get(el, f"{el}2O3")
    return COMMON_OXIDES.get(el, DIATOMIC.get(el, el)) if "O" in target else DIATOMIC.get(el, el)


def build_label_index(labels: Sequence[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for lab in labels:
        for el in elements(str(lab)):
            out.setdefault(el, []).append(str(lab))
    for el in out:
        out[el] = sorted(set(out[el]), key=lambda x: (len(x), x))
    return out


def repair_precursor_set(labels: Sequence[str], formula: str, method: str, label_index: Mapping[str, Sequence[str]]) -> tuple[List[str], List[str]]:
    target = target_source_elements(formula)
    chosen: List[str] = []
    sources: List[str] = []
    covered: Set[str] = set()
    for lab in labels:
        src = elements(str(lab)) & target
        if not src:
            continue
        if extra_forbidden_elements([str(lab)], target):
            continue
        if src <= covered:
            continue
        chosen.append(str(lab))
        sources.append("model_or_true_kept")
        covered |= src
    for el in sorted(target - covered):
        candidate = None
        for lab in label_index.get(el, []):
            if not extra_forbidden_elements([lab], target) and (elements(lab) & target):
                candidate = lab
                break
        if candidate is None:
            candidate = generated_formula(el, method, target)
            source = "open_generated"
        else:
            source = "known_vocab_repair"
        chosen.append(candidate)
        sources.append(source)
        covered |= (elements(candidate) & target) or {el}
    return chosen, sources


def set_metrics(true_labels: Sequence[str], pred_labels: Sequence[str]) -> Dict[str, float]:
    true_set = set(str(x) for x in true_labels if str(x))
    pred_set = set(str(x) for x in pred_labels if str(x))
    inter = len(true_set & pred_set)
    precision = inter / len(pred_set) if pred_set else 0.0
    recall = inter / len(true_set) if true_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = len(true_set | pred_set)
    return {
        "precursor_exact": float(pred_set == true_set),
        "precursor_precision": precision,
        "precursor_recall": recall,
        "precursor_f1": f1,
        "precursor_jaccard": inter / union if union else 1.0,
    }


def decode_y_sets(y: np.ndarray, names: Sequence[str]) -> List[List[str]]:
    out: List[List[str]] = []
    for i in range(y.shape[0]):
        idx = np.where(y[i] > 0)[0].tolist()
        out.append([str(names[j]) for j in idx if 0 <= j < len(names)])
    return out


def inverse_continuous(y_norm: np.ndarray, schema: Mapping[str, Any], names: Sequence[str]) -> np.ndarray:
    out = np.asarray(y_norm, dtype=np.float32).copy()
    stats = schema.get("continuous_schema", {}) or {}
    for j, name in enumerate(names):
        st = stats.get(name, {}) or {}
        out[:, j] = out[:, j] * float(st.get("std", 1.0)) + float(st.get("mean", 0.0))
    return out


def decode_discrete(y_disc: np.ndarray, schema: Mapping[str, Any], names: Sequence[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    ds = schema.get("discrete_schema", {}) or {}
    for j, name in enumerate(names):
        vocab = (ds.get(name, {}) or {}).get("vocab", [])
        vals = []
        for idx in y_disc[:, j].astype(int).tolist():
            vals.append(str(vocab[idx]) if 0 <= idx < len(vocab) else str(idx))
        out[name] = vals
    return out


def multihot_from_labels(labels_list: Sequence[Sequence[str]], vocab: Sequence[str]) -> np.ndarray:
    index = {str(x): i for i, x in enumerate(vocab)}
    y = np.zeros((len(labels_list), len(vocab)), dtype=np.float32)
    for i, labels in enumerate(labels_list):
        for lab in labels:
            j = index.get(str(lab))
            if j is not None:
                y[i, j] = 1.0
    return y


def load_top1_candidates(path: Path, rank_col: str = "rank") -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    usecols = None
    df = pd.read_csv(path, usecols=usecols)
    if rank_col not in df.columns:
        if "calibrated_rank_v5" in df.columns:
            rank_col = "calibrated_rank_v5"
        elif "rank" in df.columns:
            rank_col = "rank"
        else:
            raise ValueError(f"No rank column found in {path}")
    df = df[df[rank_col].astype(int) == 1].copy()
    if "id" not in df.columns:
        raise ValueError(f"Candidate file {path} lacks id column")
    keep = [c for c in [
        "id", "pred_precursors", "candidate_set", "candidate_source", "candidate_source_mix",
        "total_score_v5", "calibrated_score_v5", "f1", "jaccard", "exact",
        "element_coverage", "missing_element_count", "extra_element_count",
    ] if c in df.columns]
    return df[keep].drop_duplicates("id", keep="first")


def load_candidate_map(split: str, args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    candidates: List[pd.DataFrame] = []
    if split == "test" and args.test_candidate_csv:
        p = Path(args.test_candidate_csv)
        candidates.append(load_top1_candidates(p, rank_col="calibrated_rank_v5" if "calibrated" in p.name else "rank"))
    elif split == "val" and args.val_candidate_csv:
        candidates.append(load_top1_candidates(Path(args.val_candidate_csv), rank_col="rank"))
    if not candidates:
        return {}
    df = pd.concat(candidates, ignore_index=True)
    return {str(r["id"]): r.to_dict() for _, r in df.iterrows()}


def family_features(labels: Sequence[str], family_lookup: Mapping[str, str], families: Sequence[str]) -> Dict[str, float]:
    counts = {f"precursor_family_count__{fam}": 0.0 for fam in families}
    for lab in labels:
        fam = family_lookup.get(str(lab), "unknown")
        key = f"precursor_family_count__{fam}"
        if key in counts:
            counts[key] += 1.0
    n = max(len(labels), 1)
    for fam in families:
        counts[f"precursor_family_frac__{fam}"] = counts[f"precursor_family_count__{fam}"] / n
    return counts


def write_table(df: pd.DataFrame, out_dir: Path, split: str) -> None:
    csv_path = out_dir / f"{split}.csv"
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(out_dir / f"{split}.parquet", index=False)
    except Exception as exc:
        (out_dir / f"{split}.parquet.SKIPPED.txt").write_text(
            f"Parquet export skipped: {exc!r}\n", encoding="utf-8"
        )


def build_split(split: str, args: argparse.Namespace, schema: Mapping[str, Any], family_lookup: Mapping[str, str], families: Sequence[str]) -> Dict[str, Any]:
    stage2_dir = Path(args.stage2_dataset_dir)
    stage3_dir = Path(args.stage3_dataset_dir)
    out_dir = Path(args.output_dir)

    s2 = load_npz(stage2_dir / f"{split}.npz")
    s3 = load_npz(stage3_dir / f"{split}.npz")
    meta = pd.read_csv(stage2_dir / f"{split}_meta.csv")
    names = [str(x) for x in load_json(stage2_dir / "precursor_names.json")]
    label_index = build_label_index(names)
    stage3_vocab = [str(x) for x in schema.get("precursor_vocab", [])]
    cont_names = list(schema.get("continuous_cols", []))
    disc_names = list(schema.get("discrete_cols", []))
    feature_cols = list(schema.get("feature_cols", []))

    true_sets = decode_y_sets(np.asarray(s2["y_multi_hot"]), names)
    cand_map = load_candidate_map(split, args)
    y_cont_raw = inverse_continuous(np.asarray(s3["y_cond_continuous"]), schema, cont_names)
    y_disc_decoded = decode_discrete(np.asarray(s3["y_cond_discrete"]), schema, disc_names)
    x = np.asarray(s3["x"], dtype=np.float32)

    rows: List[Dict[str, Any]] = []
    pred_sets: List[List[str]] = []
    missing_counts = []
    extra_counts = []
    for i, row in meta.iterrows():
        sid = str(row.get("id", s3["sample_id"][i]))
        formula = str(row.get("formula", ""))
        method = str(row.get("reaction_method", ""))
        true_labels = true_sets[i]
        cand = cand_map.get(sid)
        if cand is not None:
            raw_pred_labels = parse_list(cand.get("candidate_set", cand.get("pred_precursors", "[]")))
            source_mix = str(cand.get("candidate_source_mix", cand.get("candidate_source", "stage2_v5_repaired")))
            precursor_score = float(cand.get("calibrated_score_v5", cand.get("total_score_v5", 0.0)) or 0.0)
            input_source = "stage2_v5_repaired_top1"
        else:
            raw_pred_labels = true_labels
            source_mix = "true_train_or_no_candidate_fallback"
            precursor_score = 1.0
            input_source = "true_precursor_fallback"
        pred_labels, repair_sources = repair_precursor_set(raw_pred_labels, formula, method, label_index)
        if repair_sources:
            source_mix = json.dumps(repair_sources, ensure_ascii=False)

        target = target_source_elements(formula)
        covered = precursor_source_elements(pred_labels, target)
        missing = sorted(target - covered)
        extra = sorted(extra_forbidden_elements(pred_labels, target))
        missing_counts.append(len(missing))
        extra_counts.append(len(extra))
        pred_sets.append(pred_labels)

        sm = set_metrics(true_labels, pred_labels)
        rec: Dict[str, Any] = {
            "sample_index": i,
            "sample_id": sid,
            "id": sid,
            "material_id": row.get("material_id", sid),
            "formula": formula,
            "reaction_method": method,
            "is_core_method": method in CORE_METHODS,
            "source_dataset": row.get("source_dataset", ""),
            "split": split,
            "true_precursor_set": json.dumps(true_labels, ensure_ascii=False),
            "raw_predicted_precursor_set": json.dumps(raw_pred_labels, ensure_ascii=False),
            "predicted_precursor_set_chem_checked": json.dumps(pred_labels, ensure_ascii=False),
            "precursors_text": " + ".join(pred_labels),
            "precursor_input_source": input_source,
            "precursor_source_mix": source_mix,
            "contains_open_generated_precursor": int(any(s == "open_generated" for s in repair_sources)),
            "contains_known_vocab_precursor": int(any(s == "known_vocab_repair" for s in repair_sources)),
            "contains_raw_model_precursor": int(any(s == "model_or_true_kept" for s in repair_sources)),
            "precursor_set_size": len(pred_labels),
            "target_source_elements": json.dumps(sorted(target), ensure_ascii=False),
            "covered_source_elements": json.dumps(sorted(covered), ensure_ascii=False),
            "missing_source_elements": json.dumps(missing, ensure_ascii=False),
            "extra_forbidden_elements": json.dumps(extra, ensure_ascii=False),
            "precursor_check_status": "ok" if not missing and not extra else "needs_review",
            "precursor_confidence_score": precursor_score,
            **sm,
        }
        for j, name in enumerate(cont_names):
            rec[name.replace("target_", "")] = float(y_cont_raw[i, j])
            rec[f"has_{name.replace('target_', '')}"] = int(s3["y_cond_continuous_mask"][i, j] > 0.5)
        for name in disc_names:
            rec[name.replace("target_", "")] = y_disc_decoded[name][i]
        rec.update(family_features(pred_labels, family_lookup, families))
        for j, col in enumerate(feature_cols):
            rec[col] = float(x[i, j])
        rows.append(rec)

    df = pd.DataFrame(rows)
    write_table(df, out_dir, split)
    y_pred = multihot_from_labels(pred_sets, stage3_vocab)
    np.savez_compressed(
        out_dir / f"{split}.npz",
        x=x.astype(np.float32),
        y_set=y_pred.astype(np.float32),
        y_cond_continuous=np.asarray(s3["y_cond_continuous"], dtype=np.float32),
        y_cond_continuous_mask=np.asarray(s3["y_cond_continuous_mask"], dtype=np.float32),
        y_cond_discrete=np.asarray(s3["y_cond_discrete"], dtype=np.int64),
        y_cond_discrete_mask=np.asarray(s3["y_cond_discrete_mask"], dtype=np.float32),
        sample_id=np.asarray(df["sample_id"].tolist(), dtype=object),
    )
    df[[
        "sample_index", "sample_id", "material_id", "formula", "reaction_method",
        "is_core_method", "source_dataset", "split", "precursor_input_source",
        "precursor_check_status", "precursor_set_size", "precursor_f1",
        "precursor_jaccard", "contains_open_generated_precursor",
    ]].to_csv(out_dir / f"{split}_meta.csv", index=False)

    return {
        "split": split,
        "n_rows": int(len(df)),
        "method_counts": df["reaction_method"].value_counts().to_dict(),
        "core_rows": int(df["is_core_method"].sum()),
        "precursor_check_ok": int((df["precursor_check_status"] == "ok").sum()),
        "missing_source_element_rows": int((df["missing_source_elements"] != "[]").sum()),
        "extra_forbidden_element_rows": int((df["extra_forbidden_elements"] != "[]").sum()),
        "mean_precursor_f1": float(df["precursor_f1"].mean()),
        "mean_precursor_jaccard": float(df["precursor_jaccard"].mean()),
        "open_generated_rows": int(df["contains_open_generated_precursor"].sum()),
        "input_source_counts": df["precursor_input_source"].value_counts().to_dict(),
        "x_shape": list(x.shape),
        "y_set_shape": list(y_pred.shape),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Stage3 chemistry-checked condition dataset from fixed splits.")
    ap.add_argument("--stage2_dataset_dir", default="data/interim/generative/stage2_setpred_dataset/descriptor/route_method_stratified_20260610_relaxed_only")
    ap.add_argument("--stage3_dataset_dir", default="data/interim/generative/stage3_condition_dataset_mixed/route_method_stratified_units_normalized_20260610_poscar_geom1024")
    ap.add_argument("--ontology_csv", default="data/interim/ontology/precursor_ontology_v4_20260610/precursor_ontology.csv")
    ap.add_argument("--val_candidate_csv", default="outputs/evaluation/stage2_candidate_pool_v5_20260610/val_candidate_sets_repaired.csv")
    ap.add_argument("--test_candidate_csv", default="outputs/evaluation/stage2_score_calibration_v5_20260610/test_candidate_sets_calibrated_v5.csv")
    ap.add_argument("--output_dir", default="data/interim/generative/stage3_condition_dataset_chem_checked/method_stratified_v5_20260610")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)
    schema = load_json(Path(args.stage3_dataset_dir) / "schema.json")
    ont = pd.read_csv(args.ontology_csv)
    family_lookup = dict(zip(ont["canonical_precursor"].astype(str), ont["precursor_family"].astype(str)))
    families = sorted(set(str(x) for x in ont["precursor_family"].dropna().astype(str).tolist()) | {"unknown"})

    summaries = [build_split(split, args, schema, family_lookup, families) for split in ["train", "val", "test"]]
    out_schema = dict(schema)
    out_schema["chem_checked"] = {
        "precursor_input": "train uses true precursor fallback; val/test use Stage2 v5 repaired/calibrated top1 when available",
        "families": families,
        "core_methods": sorted(CORE_METHODS),
    }
    write_json(out_dir / "schema.json", out_schema)
    summary = {
        "config": vars(args),
        "splits": {s["split"]: s for s in summaries},
    }
    write_json(out_dir / "summary.json", summary)
    lines = [
        "# Stage3 Chemistry-Checked Dataset Report",
        "",
        f"- Output: `{out_dir}`",
        f"- Stage2 source: `{args.stage2_dataset_dir}`",
        f"- Stage3 source: `{args.stage3_dataset_dir}`",
        "",
        "| split | rows | core rows | check ok | missing rows | extra rows | mean F1 | mean Jaccard | open-generated rows |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['split']} | {s['n_rows']} | {s['core_rows']} | {s['precursor_check_ok']} | "
            f"{s['missing_source_element_rows']} | {s['extra_forbidden_element_rows']} | "
            f"{s['mean_precursor_f1']:.4f} | {s['mean_precursor_jaccard']:.4f} | {s['open_generated_rows']} |"
        )
    lines += [
        "",
        "Notes:",
        "- Train split uses the true precursor set as chemistry-checked input because no no-leakage Stage2 v5 train predictions are available.",
        "- Validation/test splits use repaired/calibrated Stage2 v5 top1 candidate sets when available.",
        "- `model_precursors_text` from GNoME inference is not used for this supervised Stage3 dataset.",
    ]
    (out_dir / "stage3_chem_checked_dataset_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
