#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

import pandas as pd
from pymatgen.core import Composition


LIGHT_BYPRODUCT_ELEMENTS = {"O", "H", "C", "N"}
COUNTERION_ELEMENTS = {"O", "H", "C", "N", "F", "Cl", "Br", "I", "S", "P"}
SOURCE_SKIP = {"O"}
DIATOMIC = {"H": "H2", "N": "N2", "O": "O2", "F": "F2", "Cl": "Cl2", "Br": "Br2", "I": "I2"}

ALKALI = {"Li", "Na", "K", "Rb", "Cs"}
ALKALINE = {"Mg", "Ca", "Sr", "Ba"}
LANTHANOIDS = {"La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"}
COMMON_OXIDES = {
    "Li": "Li2O", "Na": "Na2O", "K": "K2O", "Rb": "Rb2O", "Cs": "Cs2O",
    "Be": "BeO", "Mg": "MgO", "Ca": "CaO", "Sr": "SrO", "Ba": "BaO",
    "B": "B2O3", "Al": "Al2O3", "Ga": "Ga2O3", "In": "In2O3", "Tl": "Tl2O3",
    "Si": "SiO2", "Ge": "GeO2", "Sn": "SnO2", "Pb": "PbO", "P": "P2O5", "S": "SO3",
    "Sc": "Sc2O3", "Y": "Y2O3", "Ti": "TiO2", "Zr": "ZrO2", "Hf": "HfO2",
    "V": "V2O5", "Nb": "Nb2O5", "Ta": "Ta2O5", "Cr": "Cr2O3", "Mo": "MoO3", "W": "WO3",
    "Mn": "MnO2", "Fe": "Fe2O3", "Co": "Co3O4", "Ni": "NiO", "Cu": "CuO", "Zn": "ZnO",
    "Ag": "Ag2O", "Cd": "CdO", "Hg": "HgO",
    "As": "As2O5", "Sb": "Sb2O3", "Bi": "Bi2O3",
    "Ru": "RuO2", "Rh": "Rh2O3", "Pd": "PdO", "Os": "OsO2", "Ir": "IrO2", "Pt": "PtO2", "Au": "Au2O3",
    "Th": "ThO2", "U": "UO2", "Np": "NpO2", "Pu": "PuO2",
}
for _el in LANTHANOIDS:
    COMMON_OXIDES.setdefault(_el, f"{_el}2O3")

COMMON_VALENCE = {
    "Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1, "Ag": 1,
    "Be": 2, "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2, "Zn": 2, "Cd": 2, "Hg": 2,
    "Al": 3, "Ga": 3, "In": 3, "Sc": 3, "Y": 3, "La": 3, "Ce": 3, "Pr": 3, "Nd": 3,
    "Sm": 3, "Eu": 3, "Gd": 3, "Tb": 3, "Dy": 3, "Ho": 3, "Er": 3, "Tm": 3, "Yb": 3, "Lu": 3,
    "Ti": 4, "Zr": 4, "Hf": 4, "Sn": 4, "Pb": 2, "Bi": 3,
    "V": 5, "Nb": 5, "Ta": 5, "Cr": 3, "Mo": 6, "W": 6,
    "Mn": 2, "Fe": 3, "Co": 2, "Ni": 2, "Cu": 2,
    "Ru": 3, "Rh": 3, "Pd": 2, "Os": 4, "Ir": 4, "Pt": 4, "Au": 3,
    "Th": 4, "U": 4, "Np": 4, "Pu": 4,
}


def parse_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    try:
        obj = json.loads(str(value))
        if isinstance(obj, list):
            return [str(x) for x in obj]
    except Exception:
        pass
    return []


@lru_cache(maxsize=200000)
def formula_elements(value: str) -> Set[str]:
    text = str(value).replace("·", "+")
    try:
        return {str(el) for el in Composition(text).elements}
    except Exception:
        return set(re.findall(r"[A-Z][a-z]?", text))


def target_source_elements(formula: str) -> Set[str]:
    return {el for el in formula_elements(formula) if el not in SOURCE_SKIP}


def precursor_source_elements(precursor: str, target: Set[str]) -> Set[str]:
    elems = formula_elements(precursor)
    if not elems:
        return set()
    if elems <= target | LIGHT_BYPRODUCT_ELEMENTS:
        return elems & target
    # Salts often contain non-product counterions; avoid counting those as source
    # unless they are explicitly present in the target.
    return {el for el in elems if el in target}


def has_forbidden_extra(precursor: str, target: Set[str]) -> bool:
    elems = formula_elements(precursor)
    extras = elems - target - COUNTERION_ELEMENTS
    return bool(extras)


def hydrate_clean(label: str) -> str:
    return str(label).replace("·", ".")


def nitrate_formula(el: str) -> str:
    v = int(COMMON_VALENCE.get(el, 3))
    if v <= 1:
        return f"{el}NO3"
    return f"{el}(NO3){v}"


def chloride_formula(el: str) -> str:
    v = int(COMMON_VALENCE.get(el, 3))
    if v <= 1:
        return f"{el}Cl"
    return f"{el}Cl{v}"


def carbonate_formula(el: str) -> str:
    if el in ALKALI:
        return f"{el}2CO3"
    if el in ALKALINE:
        return f"{el}CO3"
    v = int(COMMON_VALENCE.get(el, 3))
    if v == 2:
        return f"{el}CO3"
    if v == 3:
        return f"{el}2(CO3)3"
    return COMMON_OXIDES.get(el, el)


def elemental_formula(el: str) -> str:
    return DIATOMIC.get(el, el)


def generated_formula_for_element(el: str, method: str, target: Set[str]) -> str:
    if method == "melt_arc":
        return elemental_formula(el)
    if el in {"F", "Cl", "Br", "I", "H", "N"}:
        return elemental_formula(el)
    if el in {"S", "Se", "Te", "C"}:
        return el
    if el == "P":
        return "P2O5" if "O" in target else "P"
    if el == "B":
        return "H3BO3" if "O" in target else "B"
    if method == "solution":
        if "O" in target:
            return nitrate_formula(el)
        return chloride_formula(el)
    if method == "solid_state":
        if "O" in target:
            if el in ALKALI or el in ALKALINE:
                return carbonate_formula(el)
            return COMMON_OXIDES.get(el, f"{el}2O3")
        return elemental_formula(el)
    return COMMON_OXIDES.get(el, elemental_formula(el)) if "O" in target else elemental_formula(el)


def family_bonus(label: str, method: str, target: Set[str]) -> float:
    s = hydrate_clean(label)
    if method == "melt_arc":
        return 5.0 if re.fullmatch(r"[A-Z][a-z]?[0-9]?", s) or s in DIATOMIC.values() else 0.0
    if method == "solution":
        if "NO3" in s:
            return 4.0
        if any(x in s for x in ["Cl", "Br", "I", "F"]):
            return 2.2
        if "C2H3O2" in s or "CH3COO" in s:
            return 2.0
    if method == "solid_state":
        if "O" in target and ("CO3" in s or re.search(r"O[0-9]?$", s)):
            return 4.0
    return 0.0


def build_label_index(all_labels: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    index: Dict[str, List[Dict[str, Any]]] = {}
    for label in all_labels:
        elems = formula_elements(label)
        rec = {
            "label": str(label),
            "elements": elems,
            "length": len(str(label)),
        }
        for el in elems:
            index.setdefault(el, []).append(rec)
    for el in index:
        index[el].sort(key=lambda r: (r["length"], r["label"]))
    return index


def choose_known_candidate(
    element: str,
    target: Set[str],
    method: str,
    top_labels: Sequence[Mapping[str, Any]],
    label_index: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Tuple[str, float, str] | None:
    candidates: List[Tuple[float, str, str]] = []
    top_map = {str(x.get("precursor")): float(x.get("probability", 0.0) or 0.0) for x in top_labels}
    indexed = [str(x.get("label")) for x in label_index.get(element, [])]
    search_labels = list(top_map.keys()) + indexed
    seen: Set[str] = set()
    for label in search_labels:
        if not label or label in seen:
            continue
        seen.add(label)
        elems = formula_elements(label)
        if element not in elems:
            continue
        if has_forbidden_extra(label, target):
            continue
        if not precursor_source_elements(label, target):
            continue
        extra_soft = len((elems - target) - LIGHT_BYPRODUCT_ELEMENTS)
        score = 10.0 * top_map.get(label, 0.0) + family_bonus(label, method, target) - 0.15 * extra_soft - 0.002 * len(label)
        candidates.append((score, label, "model_top20" if label in top_map else "known_vocab"))
    if not candidates:
        return None
    candidates.sort(reverse=True, key=lambda x: x[0])
    score, label, source = candidates[0]
    return label, score, source


def route_text(row: Mapping[str, Any]) -> str:
    method_map = {"solid_state": "固相法", "solution": "溶液法", "melt_arc": "熔融/电弧熔炼法"}
    solvent = str(row.get("pred_solvent", ""))
    parts = [
        f"方法: {method_map.get(str(row.get('reaction_method')), row.get('reaction_method'))}",
        f"前驱体: {row.get('precursors_text')}",
        f"温度: {float(row.get('pred_temperature_c', 0.0)):.1f} °C",
        f"时间: {float(row.get('pred_time_h', 0.0)):.1f} h",
        f"气氛: {row.get('pred_atmosphere')}",
    ]
    if solvent and solvent != "<UNK_OR_MISSING>" and solvent.lower() != "nan":
        parts.append(f"溶剂: {solvent}")
    return "; ".join(parts)


def chem_checked_precursors_indexed(row: Mapping[str, Any], label_index: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    target = target_source_elements(str(row["formula"]))
    method = str(row.get("reaction_method", ""))
    top_labels = parse_list(row.get("top20_precursor_labels", "[]"))
    model_labels = parse_list(row.get("predicted_precursors", "[]"))
    if top_labels and isinstance(top_labels[0], str):
        top_labels = [{"precursor": x, "probability": 0.0} for x in top_labels]

    chosen: List[str] = []
    sources: List[str] = []
    covered: Set[str] = set()

    for label in model_labels:
        src = precursor_source_elements(label, target)
        if not src or has_forbidden_extra(label, target):
            continue
        if src <= covered:
            continue
        chosen.append(label)
        sources.append("model_kept")
        covered |= src

    for el in sorted(target - covered):
        known = choose_known_candidate(el, target, method, top_labels, label_index)
        if known is not None:
            label, _, source = known
        else:
            label, source = generated_formula_for_element(el, method, target), "open_generated"
        if label not in chosen:
            chosen.append(label)
            sources.append(source)
            covered |= precursor_source_elements(label, target) or ({el} if el in formula_elements(label) else set())

    missing = sorted(target - covered)
    extra = sorted(set().union(*(formula_elements(x) for x in chosen)) - target - COUNTERION_ELEMENTS) if chosen else []
    status = "ok" if not missing and not extra else "needs_review"
    return {
        "model_precursors_text": " + ".join(model_labels),
        "predicted_precursors_chem_checked": json.dumps(chosen, ensure_ascii=False),
        "precursors_text": " + ".join(chosen),
        "precursor_check_status": status,
        "target_source_elements": json.dumps(sorted(target), ensure_ascii=False),
        "covered_source_elements": json.dumps(sorted(covered), ensure_ascii=False),
        "missing_source_elements": json.dumps(missing, ensure_ascii=False),
        "extra_forbidden_elements": json.dumps(extra, ensure_ascii=False),
        "precursor_source_mix": json.dumps(sources, ensure_ascii=False),
    }


def process_file(input_csv: Path, output_csv: Path, label_index: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    df = pd.read_csv(input_csv)
    checks = [chem_checked_precursors_indexed(row, label_index) for row in df.to_dict("records")]
    cdf = pd.DataFrame(checks)
    out = pd.concat([df, cdf], axis=1)
    method_map = {"solid_state": "固相法", "solution": "溶液法", "melt_arc": "熔融/电弧熔炼法"}
    out["synthesis_method_cn"] = out["reaction_method"].map(method_map).fillna(out["reaction_method"])
    out["synthesis_route_text_cn"] = out.apply(route_text, axis=1)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return {
        "input": str(input_csv),
        "output": str(output_csv),
        "n_rows": int(len(out)),
        "ok_ratio": float((out["precursor_check_status"] == "ok").mean()),
        "needs_review": int((out["precursor_check_status"] != "ok").sum()),
        "source_mix_counts": out["precursor_source_mix"].value_counts().head(20).to_dict(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Chemistry-constrained postprocess for Genome selected precursor text.")
    ap.add_argument("--input_dir", default="outputs/inference/genome_selected_best_current_20260611")
    ap.add_argument("--stage2_precursor_names", default="data/interim/generative/stage2_setpred_dataset/descriptor/core_methods_ss_solution_meltarc_20260610_relaxed_only/precursor_names.json")
    args = ap.parse_args()

    base = Path(args.input_dir).expanduser().resolve()
    all_labels = [str(x) for x in json.loads(Path(args.stage2_precursor_names).read_text(encoding="utf-8"))]
    label_index = build_label_index(all_labels)
    summaries = []
    summaries.append(process_file(
        base / "genome_selected_recommended_top1.csv",
        base / "genome_selected_recommended_top1_chem_checked.csv",
        label_index,
    ))
    summaries.append(process_file(
        base / "genome_selected_predictions.csv",
        base / "genome_selected_predictions_chem_checked.csv",
        label_index,
    ))
    (base / "precursor_chem_check_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
