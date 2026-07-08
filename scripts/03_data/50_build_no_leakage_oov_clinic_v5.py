#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from pymatgen.core import Composition
except Exception:  # pragma: no cover
    Composition = None  # type: ignore


COMMON_NAME_ALIASES = {
    "lithium carbonate": "Li2CO3",
    "sodium carbonate": "Na2CO3",
    "potassium carbonate": "K2CO3",
    "aluminum nitrate nonahydrate": "Al(NO3)3·9H2O",
    "aluminium nitrate nonahydrate": "Al(NO3)3·9H2O",
    "cobalt acetate tetrahydrate": "Co(CH3COO)2·4H2O",
    "cobalt nitrate hexahydrate": "Co(NO3)2·6H2O",
    "nickel nitrate hexahydrate": "Ni(NO3)2·6H2O",
    "iron(iii) nitrate nonahydrate": "Fe(NO3)3·9H2O",
    "ferric nitrate nonahydrate": "Fe(NO3)3·9H2O",
    "ammonium dihydrogen phosphate": "NH4H2PO4",
    "diammonium hydrogen phosphate": "(NH4)2HPO4",
    "ammonium metavanadate": "NH4VO3",
    "boric acid": "H3BO3",
    "oxalic acid": "C2H2O4",
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
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    return {k: arr[k] for k in arr.files}


def get_y(pack: Dict[str, np.ndarray]) -> np.ndarray:
    for key in ["y_multi_hot", "y", "labels", "targets"]:
        if key in pack:
            return (np.asarray(pack[key]) > 0).astype(np.int8)
    raise KeyError(f"No label matrix found in npz keys={list(pack)}")


def label_counts(dataset_dir: Path, split: str, names: List[str]) -> Counter:
    y = get_y(load_npz(dataset_dir / f"{split}.npz"))
    return Counter({str(names[j]): int(y[:, j].sum()) for j in range(y.shape[1]) if int(y[:, j].sum())})


def formula_key(label: str) -> str:
    if Composition is None:
        return ""
    try:
        return Composition(label).alphabetical_formula.replace(" ", "")
    except Exception:
        return ""


def clean_candidate(label: str) -> Tuple[str, str, str, float]:
    raw = str(label).strip()
    s = raw
    low = s.lower().strip()
    if low in COMMON_NAME_ALIASES:
        return COMMON_NAME_ALIASES[low], "common_name_alias", "general_chemistry_rule", 0.90
    s2 = re.sub(r"^[αβγδ]-", "", s)
    if s2 != s:
        return s2, "phase_prefix_remove", "train_pattern", 0.85
    s2 = re.sub(r"^[Tt]-", "", s)
    if s2 != s:
        return s2, "phase_prefix_remove", "train_pattern", 0.80
    s2 = re.sub(r"\([0-9.]+\s*[mM]\)$", "", s).strip()
    if s2 != s:
        return s2, "concentration_suffix_remove", "train_pattern", 0.85
    s2 = re.sub(r"(?i)\s*(aq|solid|powder|anhydrous|solution)$", "", s).strip()
    if s2 != s:
        return s2, "state_suffix_remove", "manual_general_rule", 0.80
    # Broken hydrates such as Ca(NO3)2.4H2 are generalizable, but only patch
    # when the corrected hydrate or base appears in train/val.
    s2 = re.sub(r"\.([0-9]+)H2$", r"·\1H2O", s)
    if s2 != s:
        return s2, "broken_hydrate_h2_to_h2o", "general_chemistry_rule", 0.95
    hydrate_match = re.search(r"([·.])([0-9.]+)?H2O$", s, re.I)
    if hydrate_match:
        base = s[: hydrate_match.start()]
        return base, "hydrate_to_base", "general_chemistry_rule", 0.90
    # Solvent/adduct cleanup. Keep conservative and require target label to exist.
    for sep in ["-", "·"]:
        if sep in s:
            left, right = s.split(sep, 1)
            if re.search(r"(?i)HCl|HNO3|H2O|C[0-9]*H[0-9]+OH|CH3OH|EtOH|DMF|DMSO|NH3", right):
                return left, "solvent_adduct_suffix_remove", "manual_general_rule", 0.82
    roman = {
        "(i)": "", "(ii)": "", "(iii)": "", "(iv)": "", "(v)": "", "(vi)": "",
        " i ": " ", " ii ": " ", " iii ": " ", " iv ": " ",
    }
    low2 = f" {low} "
    for k, v in roman.items():
        low2 = low2.replace(k, v)
    if low2.strip() != low:
        return low2.strip(), "roman_oxidation_text_cleanup", "manual_general_rule", 0.75
    return raw, "none", "none", 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a no-leakage Stage2 v5 OOV clinic and alias patch from train/val only.")
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--ontology_csv", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--patch_output_dir", required=True)
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.output_dir).resolve()
    patch_dir = Path(args.patch_output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    patch_dir.mkdir(parents=True, exist_ok=True)

    names = [str(x) for x in load_json(dataset_dir / "precursor_names.json")]
    train_counts = label_counts(dataset_dir, "train", names)
    val_counts = label_counts(dataset_dir, "val", names)
    train_val_labels = set(train_counts) | set(val_counts)
    ont = pd.read_csv(args.ontology_csv)
    ont_map = ont.set_index("canonical_precursor").to_dict(orient="index")

    rows = []
    patches = []
    for lab in sorted(train_val_labels):
        train_count = int(train_counts.get(lab, 0))
        val_count = int(val_counts.get(lab, 0))
        seen = "+".join([s for s, c in [("train", train_count), ("val", val_count)] if c])
        info = ont_map.get(lab, {})
        cleaned, patch_type, rule_source, conf = clean_candidate(lab)
        alias_candidate = ""
        suggested_patch = ""
        allowed = False
        if patch_type != "none":
            candidates = [cleaned]
            base_key = formula_key(cleaned)
            if base_key:
                candidates += [x for x in train_val_labels if formula_key(x) == base_key][:3]
            for cand in candidates:
                if cand in train_val_labels and cand != lab:
                    alias_candidate = cand
                    suggested_patch = cand
                    allowed = rule_source != "test_pattern"
                    break
        row = {
            "label": lab,
            "split_seen": seen,
            "train_count": train_count,
            "val_count": val_count,
            "family": info.get("precursor_family", "unknown"),
            "parse_status": info.get("parse_status", "failed"),
            "parse_error": info.get("parse_error", ""),
            "formula_guess": info.get("canonical_formula", lab),
            "cleaned_formula": cleaned,
            "alias_candidate": alias_candidate,
            "suggested_patch": suggested_patch,
            "patch_type": patch_type,
            "confidence": conf,
            "generalizable_rule": rule_source not in {"none", "test_pattern"},
            "allowed_for_v5_patch": bool(allowed),
            "rule_source": rule_source,
        }
        rows.append(row)
        if allowed:
            patches.append({
                "raw_label": lab,
                "patched_label": suggested_patch,
                "patch_type": patch_type,
                "confidence": conf,
                "rule_source": rule_source,
                "reason": f"train/val no-leakage rule: {patch_type}",
            })

    clinic = pd.DataFrame(rows)
    patch_df = pd.DataFrame(patches).drop_duplicates(["raw_label", "patched_label"]) if patches else pd.DataFrame(columns=["raw_label", "patched_label", "patch_type", "confidence", "rule_source", "reason"])
    clinic.to_csv(out_dir / "oov_clinic_train_val.csv", index=False)
    patch_df.to_csv(patch_dir / "precursor_alias_patch_v5.csv", index=False)
    summary = {
        "n_train_val_labels": int(len(train_val_labels)),
        "n_clinic_rows": int(len(clinic)),
        "n_allowed_patches": int(len(patch_df)),
        "forbidden_test_derived_patch_count": int((patch_df.get("rule_source", pd.Series(dtype=str)) == "test_pattern").sum()) if len(patch_df) else 0,
        "patch_type_counts": patch_df["patch_type"].value_counts().to_dict() if len(patch_df) else {},
        "rule_source_counts": patch_df["rule_source"].value_counts().to_dict() if len(patch_df) else {},
        "parse_failed_train_val": int((clinic["parse_status"] != "ok").sum()),
        "artifacts": {
            "clinic_csv": str(out_dir / "oov_clinic_train_val.csv"),
            "patch_csv": str(patch_dir / "precursor_alias_patch_v5.csv"),
            "summary": str(out_dir / "oov_clinic_summary.json"),
        },
    }
    write_json(out_dir / "oov_clinic_summary.json", summary)
    write_json(patch_dir / "summary.json", summary)
    report = ["# Stage2 v5 No-Leakage OOV Clinic", ""]
    for k, v in summary.items():
        if k != "artifacts":
            report.append(f"- {k}: {v}")
    (out_dir / "oov_clinic_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
