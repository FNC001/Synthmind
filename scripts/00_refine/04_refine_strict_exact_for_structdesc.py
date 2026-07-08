#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

AUX_SPECIES = {
    "O2", "H2O", "CO2", "NH3", "N2", "H2", "Ar", "He",
    "[OH-]", "[NO3-]", "[Cl-]", "[SO4-2]", "[SO4--]", "[PO4-3]",
    "H+", "[Na+]", "[K+]",
}

CONTAMINATION_PATTERNS = [
    r"polymer gel electrolyte",
    r"polymer-coated",
    r"aluminum laminates",
    r"BET surface area",
    r"porosity analyzer",
    r"electrolyte",
    r"electrochemical",
    r"battery",
    r"coin cell",
    r"cathode",
    r"anode",
    r"separator",
]

VARIABLE_PATTERNS = [
    r"\bx\b",
    r"\by\b",
    r"\bz\b",
    r"δ",
    r"\bLn\s*=",
    r"\bRE\s*=",
    r"\d+(?:\.\d+)?\s*[-–]\s*[xyz]",
    r"[xyz]\s*=",
]

TOO_GENERIC_PATTERNS = [
    r"according to the previous method",
    r"according to the literature",
    r"prepared under similar conditions",
    r"similar experimental conditions",
]

FORMULA_TOKEN_PAT = re.compile(r"\b(?:[A-Z][a-z]?\d*){2,}\b")
ELEMENT_PAT = re.compile(r"([A-Z][a-z]?)(\d*(?:\.\d+)?)")

STRUCTURAL_METALS = frozenset({
    "Li", "Na", "K", "Rb", "Cs",
    "Be", "Mg", "Ca", "Sr", "Ba",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
    "Al", "Ga", "In", "Tl", "Ge", "Sn", "Pb", "Sb", "Bi", "Te", "Se",
    "Si", "P", "B",
})

COMMON_NONSTRUCTURAL = frozenset({"H", "O", "C", "N", "S", "F", "Cl", "Br", "I"})


def extract_elements_from_precursor(name: Optional[str]) -> Set[str]:
    if not name:
        return set()
    return set(ELEMENT_PAT.findall(name)[i][0] for i in range(len(ELEMENT_PAT.findall(name))))


def check_precursor_element_consistency(
    main_precursors: List[str], target_formula: Optional[str]
) -> Tuple[List[str], List[str]]:
    """
    Check if main precursors contain metal elements not present in the target formula.
    Returns (severe_reasons, mild_reasons).
    """
    severe: List[str] = []
    mild: List[str] = []

    if not target_formula or not main_precursors:
        return severe, mild

    target_elements: Set[str] = set()
    for el, _ in ELEMENT_PAT.findall(target_formula):
        target_elements.add(el)

    prec_metals: Set[str] = set()
    for p in main_precursors:
        if p in AUX_SPECIES:
            continue
        for el, _ in ELEMENT_PAT.findall(p):
            if el in STRUCTURAL_METALS:
                prec_metals.add(el)

    target_metals = target_elements & STRUCTURAL_METALS
    extra_metals = prec_metals - target_metals

    if len(extra_metals) >= 2:
        severe.append("precursor_element_mismatch_severe")
    elif len(extra_metals) == 1:
        mild.append("precursor_element_mismatch_mild")

    return severe, mild


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_nested(obj: Any, path: str, default: Any = None) -> Any:
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def normalize_text(s: Any) -> str:
    if s is None:
        return ""
    return " ".join(str(s).replace("\n", " ").split())


def normalize_formula_string(s: Any) -> Optional[str]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    s = s.replace("⋅", "·").replace("∙", "·")
    s = s.replace(" ", "")
    return s


def has_pattern(text: str, patterns: Sequence[str]) -> bool:
    return any(re.search(p, text, flags=re.I) for p in patterns)


def is_variable_formula_like(s: Optional[str]) -> bool:
    if not s:
        return False
    return bool(re.search(r"[xyzδ]", s, flags=re.I))


def canonical_simple_formula(formula: Optional[str]) -> Optional[str]:
    formula = normalize_formula_string(formula)
    if not formula:
        return None
    if any(ch in formula for ch in "()[]·+-"):
        return None
    if re.search(r"[xyzδ]", formula, flags=re.I):
        return None
    parts = ELEMENT_PAT.findall(formula)
    if not parts:
        return None
    joined = "".join(el + num for el, num in parts)
    if joined != formula:
        return None
    counts: Dict[str, float] = {}
    for el, num in parts:
        counts[el] = counts.get(el, 0.0) + (float(num) if num else 1.0)
    ordered = sorted(counts.items())
    out = []
    for el, amt in ordered:
        if abs(amt - round(amt)) < 1e-8:
            iamt = int(round(amt))
            out.append(el if iamt == 1 else f"{el}{iamt}")
        else:
            out.append(f"{el}{amt:g}")
    return "".join(out)


def canonical_formula_key(formula: Optional[str]) -> Optional[str]:
    norm = normalize_formula_string(formula)
    if not norm:
        return None
    return canonical_simple_formula(norm) or norm


def extract_formula_tokens(text: str) -> Set[str]:
    tokens = set(FORMULA_TOKEN_PAT.findall(text or ""))
    return {t for t in tokens if len(t) >= 3}


def target_formula_candidates(row: Dict[str, Any]) -> List[str]:
    vals = [
        row.get("mp_formula"),
        row.get("synth_formula"),
        get_nested(row, "raw_synthesis_record.target.material_formula"),
        row.get("parent_formula"),
    ]
    out: List[str] = []
    seen: Set[str] = set()
    for v in vals:
        nv = normalize_formula_string(v)
        if nv and nv not in seen:
            out.append(nv)
            seen.add(nv)
    return out


def precursor_formula_list(row: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    precs = get_nested(row, "raw_synthesis_record.precursors", None) or row.get("precursors") or []
    for p in precs:
        cand = None
        if isinstance(p, dict):
            cand = p.get("material_formula") or p.get("name") or p.get("material") or p.get("material_string")
        else:
            cand = p
        cand = normalize_formula_string(cand)
        if cand and cand not in seen:
            out.append(cand)
            seen.add(cand)
    return out


def split_precursors(precursors: Sequence[str]) -> Tuple[List[str], List[str]]:
    main, aux = [], []
    for p in precursors:
        if p in AUX_SPECIES:
            aux.append(p)
        else:
            main.append(p)
    return main, aux


def _multi_target_precursor_safe(row: Dict[str, Any]) -> bool:
    """Check if precursor metals are a subset of target metals (safe for series-shared recovery)."""
    target_formula = normalize_formula_string(
        row.get("mp_formula") or row.get("synth_formula")
    )
    if not target_formula:
        return False
    target_metals = set()
    for m in ELEMENT_PAT.findall(target_formula):
        if m[0] in STRUCTURAL_METALS:
            target_metals.add(m[0])
    if not target_metals:
        return False

    raw_precs = get_nested(row, "raw_synthesis_record.precursors", []) or []
    prec_metals: Set[str] = set()
    for p in raw_precs:
        if not isinstance(p, dict):
            continue
        formula = p.get("material_formula", "") or ""
        for m in ELEMENT_PAT.findall(formula):
            if m[0] in STRUCTURAL_METALS:
                prec_metals.add(m[0])
    return prec_metals <= target_metals


def split_variable_reasons(row: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    severe: List[str] = []
    mild: List[str] = []
    rxn = normalize_text(get_nested(row, "raw_synthesis_record.reaction_string", ""))
    text = normalize_text(row.get("synthesis_text", ""))
    targets = get_nested(row, "raw_synthesis_record.targets_string", []) or []
    target_amount_vars = get_nested(row, "raw_synthesis_record.target.amounts_vars", {}) or {}

    if len(targets) > 1:
        if _multi_target_precursor_safe(row):
            mild.append("multiple_targets_series_shared")
        else:
            severe.append("multiple_targets_precursor_contaminated")
    if target_amount_vars:
        severe.append("target_has_amount_variables")
    if has_pattern(rxn, VARIABLE_PATTERNS):
        severe.append("variable_pattern_in_reaction")
    if has_pattern(text, VARIABLE_PATTERNS):
        mild.append("variable_pattern_in_text")
    for p in precursor_formula_list(row):
        if is_variable_formula_like(p):
            severe.append("variable_precursor_formula")
            break
    return sorted(set(severe)), sorted(set(mild))


def paragraph_contamination_reasons(text: str) -> Tuple[List[str], List[str]]:
    severe: List[str] = []
    mild: List[str] = []
    if has_pattern(text, CONTAMINATION_PATTERNS):
        mild.append("paragraph_contamination")
    if has_pattern(text, TOO_GENERIC_PATTERNS):
        mild.append("too_generic_text")
    if len(text.strip()) < 30:
        severe.append("text_too_short")
    return sorted(set(severe)), sorted(set(mild))


def target_text_conflict(row: Dict[str, Any], text: str) -> bool:
    targets = target_formula_candidates(row)
    target_keys = {canonical_formula_key(x) for x in targets if canonical_formula_key(x)}
    text_hits = extract_formula_tokens(text)
    if not text_hits:
        return False

    precursor_keys = {canonical_formula_key(x) for x in precursor_formula_list(row) if canonical_formula_key(x)}
    aux_keys = {canonical_formula_key(x) for x in AUX_SPECIES if canonical_formula_key(x)}
    filtered_keys = {canonical_formula_key(x) for x in text_hits if canonical_formula_key(x)}
    filtered_keys = {k for k in filtered_keys if k and k not in precursor_keys and k not in aux_keys}
    if not filtered_keys:
        return False
    if filtered_keys & target_keys:
        return False
    return True


def precursor_text_conflict(row: Dict[str, Any], text: str) -> bool:
    precursor_keys = {canonical_formula_key(x) for x in precursor_formula_list(row) if canonical_formula_key(x)}
    if not precursor_keys:
        return False
    trigger = re.search(r"starting materials|precursors|used as starting materials", text, flags=re.I)
    if not trigger:
        return False
    text_hits = {canonical_formula_key(x) for x in extract_formula_tokens(text) if canonical_formula_key(x)}
    target_keys = {canonical_formula_key(x) for x in target_formula_candidates(row) if canonical_formula_key(x)}
    text_hits = {x for x in text_hits if x and x not in target_keys}
    if not text_hits:
        return False
    if precursor_keys & text_hits:
        return False
    return True


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def celsius_from_cond_item(item: Dict[str, Any]) -> List[float]:
    vals: List[float] = []
    values = item.get("values", []) or []
    min_v = safe_float(item.get("min_value"))
    max_v = safe_float(item.get("max_value"))
    unit = str(item.get("units") or "").strip().lower().replace(" ", "")

    raw_values: List[float] = []
    for v in values:
        vv = safe_float(v)
        if vv is not None:
            raw_values.append(vv)
    if not raw_values and min_v is not None and max_v is not None:
        raw_values.append((min_v + max_v) / 2.0)
    elif not raw_values and max_v is not None:
        raw_values.append(max_v)
    elif not raw_values and min_v is not None:
        raw_values.append(min_v)

    for v in raw_values:
        if unit in {"k", "kelvin"}:
            v = v - 273.15
        elif unit in {"f", "°f", "℉", "fahrenheit"}:
            v = (v - 32.0) * 5.0 / 9.0
        if 0 <= v <= 2000:
            vals.append(round(v, 4))
    return vals


def hours_from_cond_item(item: Dict[str, Any]) -> List[float]:
    vals: List[float] = []
    values = item.get("values", []) or []
    min_v = safe_float(item.get("min_value"))
    max_v = safe_float(item.get("max_value"))
    unit = str(item.get("units") or "").strip().lower().replace(" ", "")

    raw_values: List[float] = []
    for v in values:
        vv = safe_float(v)
        if vv is not None:
            raw_values.append(vv)
    if not raw_values and min_v is not None and max_v is not None:
        raw_values.append((min_v + max_v) / 2.0)
    elif not raw_values and max_v is not None:
        raw_values.append(max_v)
    elif not raw_values and min_v is not None:
        raw_values.append(min_v)

    for v in raw_values:
        if unit in {"h", "hr", "hrs", "hour", "hours"}:
            v = v
        elif unit in {"min", "mins", "minute", "minutes"}:
            v = v / 60.0
        elif unit in {"s", "sec", "secs", "second", "seconds"}:
            v = v / 3600.0
        elif unit in {"day", "days"}:
            v = v * 24.0
        if 0 <= v <= 500:
            vals.append(round(v, 4))
    return vals


ATM_MAP = {
    "air": "air",
    "ar": "ar",
    "argon": "ar",
    "n2": "n2",
    "nitrogen": "n2",
    "o2": "o2",
    "oxygen": "o2",
    "co2": "co2",
    "carbon dioxide": "co2",
    "vacuum": "vacuum",
    "h2": "h2",
    "hydrogen": "h2",
    "he": "he",
    "helium": "he",
    "nh3": "nh3",
    "ammonia": "nh3",
    "inert": "inert",
}

SOLVENT_MAP = {
    "water": "water",
    "h2o": "water",
    "ethanol": "ethanol",
    "methanol": "methanol",
    "acetone": "acetone",
    "isopropanol": "isopropanol",
    "ipa": "isopropanol",
    "ethylene glycol": "ethylene_glycol",
    "glycerol": "glycerol",
    "dmf": "dmf",
    "dimethylformamide": "dmf",
    "dmso": "dmso",
    "dimethyl sulfoxide": "dmso",
    "acetonitrile": "acetonitrile",
    "toluene": "toluene",
    "hexane": "hexane",
    "nh3 solution": "ammonia_solution",
    "aqueous nh3 solution": "ammonia_solution",
}


def normalize_atm(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = " ".join(s.split())
    return ATM_MAP.get(s)


def normalize_solvent(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = " ".join(s.split())
    return SOLVENT_MAP.get(s)


def normalize_synth_type(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower().replace("_", "-")
    s = " ".join(s.split())
    if s == "coprecipitation":
        s = "co-precipitation"
    if s == "molten salt":
        s = "molten-salt"
    return s or None


def extract_conditions_from_operations(row: Dict[str, Any]) -> Dict[str, Any]:
    ops = get_nested(row, "raw_synthesis_record.operations", []) or []
    temps: List[float] = []
    times: List[float] = []
    atms: List[str] = []
    solvents: List[str] = []
    heatlike_count = 0

    for op in ops:
        if not isinstance(op, dict):
            continue
        typ = str(op.get("type") or "")
        cond = op.get("conditions", {}) or {}
        if typ in {"HeatingOperation", "DryingOperation", "AnnealingOperation", "CalciningOperation"}:
            heatlike_count += 1

        for item in cond.get("heating_temperature", []) or []:
            if isinstance(item, dict):
                temps.extend(celsius_from_cond_item(item))
        for item in cond.get("heating_time", []) or []:
            if isinstance(item, dict):
                times.extend(hours_from_cond_item(item))
        for a in cond.get("heating_atmosphere", []) or []:
            aa = normalize_atm(a)
            if aa:
                atms.append(aa)

        # solution_synthesis schema: conditions use "temperature"/"time"/"atmosphere" directly
        temp_direct = cond.get("temperature")
        if isinstance(temp_direct, dict) and temp_direct.get("values"):
            temps.extend(celsius_from_cond_item(temp_direct))
        time_direct = cond.get("time")
        if isinstance(time_direct, dict) and time_direct.get("values"):
            times.extend(hours_from_cond_item(time_direct))
        atm_direct = cond.get("atmosphere")
        if isinstance(atm_direct, list):
            for a in atm_direct:
                aa = normalize_atm(a)
                if aa:
                    atms.append(aa)
        elif isinstance(atm_direct, str):
            aa = normalize_atm(atm_direct)
            if aa:
                atms.append(aa)

        ss = normalize_solvent(cond.get("mixing_media"))
        if ss:
            solvents.append(ss)

    temps = sorted(set(temps))
    times = sorted(set(times))
    atms = list(Counter(atms).keys())
    solvents = list(Counter(solvents).keys())

    return {
        "temperature_c": max(temps) if temps else None,
        "time_h": sum(times) if times and sum(times) <= 500 else (max(times) if times else None),
        "atmosphere": atms[0] if atms else None,
        "solvent": solvents[0] if solvents else None,
        "all_temps_clean": temps,
        "all_times_clean": times,
        "all_atmos_clean": atms,
        "all_solvents_clean": solvents,
        "n_heatlike_ops": heatlike_count,
    }


def extract_conditions_row_fallback(row: Dict[str, Any]) -> Dict[str, Any]:
    temp = safe_float(row.get("max_temperature_c"))
    if temp is None or not (0 <= temp <= 2000):
        temp = None

    time_h = safe_float(row.get("total_time_h"))
    if time_h is None or not (0 <= time_h <= 500):
        time_h = None

    atm: Optional[str] = None
    for cand in [row.get("atmosphere")] + list(row.get("all_atmospheres") or []):
        atm = normalize_atm(cand)
        if atm:
            break

    solvent: Optional[str] = None
    for cand in [row.get("solvent")] + list(row.get("all_solvents") or []):
        solvent = normalize_solvent(cand)
        if solvent:
            break

    return {
        "temperature_c": temp,
        "time_h": time_h,
        "atmosphere": atm,
        "solvent": solvent,
    }


def merge_conditions_for_relaxed(op_cond: Dict[str, Any], fallback_cond: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "temperature_c": op_cond["temperature_c"] if op_cond["temperature_c"] is not None else fallback_cond["temperature_c"],
        "time_h": op_cond["time_h"] if op_cond["time_h"] is not None else fallback_cond["time_h"],
        "atmosphere": op_cond["atmosphere"] if op_cond["atmosphere"] is not None else fallback_cond["atmosphere"],
        "solvent": op_cond["solvent"] if op_cond["solvent"] is not None else fallback_cond["solvent"],
    }


def build_base_out_row(
    row: Dict[str, Any],
    main_precursors: List[str],
    aux_precursors: List[str],
    op_cond: Dict[str, Any],
    fallback_cond: Dict[str, Any],
) -> Dict[str, Any]:
    doi_list = row.get("dois") or []
    doi = doi_list[0] if doi_list else None
    target_formula = normalize_formula_string(
        row.get("mp_formula") or row.get("synth_formula") or get_nested(row, "raw_synthesis_record.target.material_formula")
    )
    return {
        "id": row.get("id"),
        "synth_uid": row.get("synth_uid"),
        "source_dataset": row.get("source_dataset"),
        "record_index": row.get("record_index"),
        "material_id": row.get("material_id"),
        "formula": target_formula,
        "mp_formula": normalize_formula_string(row.get("mp_formula")),
        "synth_formula": normalize_formula_string(row.get("synth_formula")),
        "parent_formula": normalize_formula_string(row.get("parent_formula")),
        "doi": doi,
        "dois": doi_list,
        "split_group": doi or row.get("synth_uid") or row.get("material_id") or target_formula,
        "poscar_path": row.get("poscar_path"),
        "summary_json_path": row.get("summary_json_path"),
        "provenance_json_path": row.get("provenance_json_path"),
        "reaction_string": get_nested(row, "raw_synthesis_record.reaction_string"),
        "synthesis_type": normalize_synth_type(row.get("synthesis_type") or get_nested(row, "raw_synthesis_record.synthesis_type")),
        "synthesis_text": normalize_text(row.get("synthesis_text")),
        "main_precursors": main_precursors,
        "aux_precursors": aux_precursors,
        "n_main_precursors": len(main_precursors),
        "n_aux_precursors": len(aux_precursors),
        "temperature_c_op": op_cond["temperature_c"],
        "time_h_op": op_cond["time_h"],
        "atmosphere_op": op_cond["atmosphere"],
        "solvent_op": op_cond["solvent"],
        "all_temps_clean": op_cond["all_temps_clean"],
        "all_times_clean": op_cond["all_times_clean"],
        "all_atmos_clean": op_cond["all_atmos_clean"],
        "all_solvents_clean": op_cond["all_solvents_clean"],
        "n_heatlike_ops": op_cond["n_heatlike_ops"],
        "temperature_c_fallback": fallback_cond["temperature_c"],
        "time_h_fallback": fallback_cond["time_h"],
        "atmosphere_fallback": fallback_cond["atmosphere"],
        "solvent_fallback": fallback_cond["solvent"],
    }


def assess_row(row: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, List[str]], Dict[str, Any], Dict[str, Any]]:
    text = normalize_text(row.get("synthesis_text"))
    precursor_formulas = precursor_formula_list(row)
    main_precursors, aux_precursors = split_precursors(precursor_formulas)
    var_severe, var_mild = split_variable_reasons(row)
    text_severe, text_mild = paragraph_contamination_reasons(text)
    op_cond = extract_conditions_from_operations(row)
    fallback_cond = extract_conditions_row_fallback(row)

    severe_reasons: List[str] = []
    mild_reasons: List[str] = []

    if not target_formula_candidates(row):
        severe_reasons.append("missing_target_formula")
    if not main_precursors:
        severe_reasons.append("no_main_precursors")

    severe_reasons.extend(var_severe)
    mild_reasons.extend(var_mild)
    severe_reasons.extend(text_severe)
    mild_reasons.extend(text_mild)

    target_formula = normalize_formula_string(
        row.get("mp_formula") or row.get("synth_formula")
    )
    prec_severe, prec_mild = check_precursor_element_consistency(main_precursors, target_formula)
    severe_reasons.extend(prec_severe)
    mild_reasons.extend(prec_mild)

    if target_text_conflict(row, text):
        mild_reasons.append("target_text_conflict")
    if precursor_text_conflict(row, text):
        mild_reasons.append("precursor_text_conflict")

    rxn = normalize_text(get_nested(row, "raw_synthesis_record.reaction_string", ""))
    if not rxn:
        mild_reasons.append("missing_reaction_string")
    elif "==" not in rxn and "->" not in rxn:
        mild_reasons.append("missing_reaction_arrow")

    op_has_any_condition = any(
        op_cond[k] is not None for k in ["temperature_c", "time_h", "atmosphere", "solvent"]
    )
    fallback_has_any_condition = any(
        fallback_cond[k] is not None for k in ["temperature_c", "time_h", "atmosphere", "solvent"]
    )
    if not op_has_any_condition:
        mild_reasons.append("no_operation_level_conditions")
    if op_cond["n_heatlike_ops"] == 0:
        mild_reasons.append("no_heatlike_operation")
    if not fallback_has_any_condition:
        mild_reasons.append("no_row_level_condition_fallback")

    out = build_base_out_row(row, main_precursors, aux_precursors, op_cond, fallback_cond)
    meta = {
        "severe": sorted(set(severe_reasons)),
        "mild": sorted(set(mild_reasons)),
    }
    return out, meta, op_cond, fallback_cond


def make_stage3_variant(base: Dict[str, Any], cond: Dict[str, Any], source: str) -> Dict[str, Any]:
    out = dict(base)
    out["temperature_c"] = cond["temperature_c"]
    out["time_h"] = cond["time_h"]
    out["atmosphere"] = cond["atmosphere"]
    out["solvent"] = cond["solvent"]
    out["condition_source"] = source
    return out


def update_reason_counter(counter: Counter[str], rows: Iterable[Dict[str, Any]], key: str) -> None:
    for row in rows:
        for reason in row.get(key, []) or []:
            counter[reason] += 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refine strict_exact_only.jsonl into stage2/stage3 gold + relaxed sets for structdesc prediction."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="/Users/wyc/SynPred/data/raw/strict_exact_only.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/Users/wyc/SynPred/data/interim/refined/structdesc_refined",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.input_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hard_fail_for_stage2_relaxed = {
        "missing_target_formula",
        "no_main_precursors",
        "multiple_targets_precursor_contaminated",
        "target_has_amount_variables",
        "variable_pattern_in_reaction",
        "variable_precursor_formula",
        "text_too_short",
        "precursor_element_mismatch_severe",
    }

    gold_blocking_mild = {
        "too_generic_text",
        "paragraph_contamination",
        "target_text_conflict",
        "precursor_text_conflict",
        "variable_pattern_in_text",
        "precursor_element_mismatch_mild",
        "multiple_targets_series_shared",
    }

    stage2_gold: List[Dict[str, Any]] = []
    stage2_relaxed: List[Dict[str, Any]] = []
    stage3_gold: List[Dict[str, Any]] = []
    stage3_relaxed: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []

    for row in rows:
        base_out, meta, op_cond, fallback_cond = assess_row(row)
        severe = meta["severe"]
        mild = meta["mild"]

        base_out["_refine_severe_reasons"] = severe
        base_out["_refine_mild_reasons"] = mild

        stage2_relaxed_ok = not any(r in hard_fail_for_stage2_relaxed for r in severe)
        stage2_gold_ok = stage2_relaxed_ok and not any(r in gold_blocking_mild for r in mild)

        op_has_any_condition = any(op_cond[k] is not None for k in ["temperature_c", "time_h", "atmosphere", "solvent"])
        relaxed_cond = merge_conditions_for_relaxed(op_cond, fallback_cond)
        relaxed_has_any_condition = any(relaxed_cond[k] is not None for k in ["temperature_c", "time_h", "atmosphere", "solvent"])

        stage3_gold_ok = stage2_gold_ok and op_has_any_condition and base_out.get("n_heatlike_ops", 0) > 0
        stage3_relaxed_ok = stage2_relaxed_ok and relaxed_has_any_condition

        if stage2_gold_ok:
            stage2_gold.append(dict(base_out))
        if stage2_relaxed_ok:
            stage2_relaxed.append(dict(base_out))
        if stage3_gold_ok:
            stage3_gold.append(make_stage3_variant(base_out, op_cond, source="operation"))
        if stage3_relaxed_ok:
            source = "operation" if op_has_any_condition else "row_fallback"
            stage3_relaxed.append(make_stage3_variant(base_out, relaxed_cond, source=source))
        if not (stage2_relaxed_ok or stage3_relaxed_ok):
            dropped.append(dict(base_out))

    write_jsonl(out_dir / "stage2_gold.jsonl", stage2_gold)
    write_jsonl(out_dir / "stage2_train_relaxed.jsonl", stage2_relaxed)
    write_jsonl(out_dir / "stage3_gold.jsonl", stage3_gold)
    write_jsonl(out_dir / "stage3_train_relaxed.jsonl", stage3_relaxed)
    write_jsonl(out_dir / "dropped_records.jsonl", dropped)

    severe_counter: Counter[str] = Counter()
    mild_counter: Counter[str] = Counter()
    update_reason_counter(severe_counter, stage2_relaxed, "_refine_severe_reasons")
    update_reason_counter(mild_counter, stage2_relaxed, "_refine_mild_reasons")

    dropped_severe_counter: Counter[str] = Counter()
    dropped_mild_counter: Counter[str] = Counter()
    update_reason_counter(dropped_severe_counter, dropped, "_refine_severe_reasons")
    update_reason_counter(dropped_mild_counter, dropped, "_refine_mild_reasons")

    summary = {
        "input_total": len(rows),
        "stage2_gold": len(stage2_gold),
        "stage2_train_relaxed": len(stage2_relaxed),
        "stage3_gold": len(stage3_gold),
        "stage3_train_relaxed": len(stage3_relaxed),
        "dropped_records": len(dropped),
    }

    reason_summary = {
        "stage2_relaxed_severe_reason_counts": dict(severe_counter.most_common()),
        "stage2_relaxed_mild_reason_counts": dict(mild_counter.most_common()),
        "dropped_severe_reason_counts": dict(dropped_severe_counter.most_common()),
        "dropped_mild_reason_counts": dict(dropped_mild_counter.most_common()),
        "stage3_relaxed_condition_source_counts": dict(Counter(r.get("condition_source") for r in stage3_relaxed).most_common()),
    }

    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "reason_summary.json", reason_summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
