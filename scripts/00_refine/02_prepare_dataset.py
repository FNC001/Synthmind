import os
import re
import json
import glob
from pathlib import Path
from difflib import SequenceMatcher
from typing import Any

import pandas as pd
from pymatgen.core import Composition, Structure


# =========================================================
# Config
# =========================================================
CONFIG = {
    # ---------- MP archive ----------
    "MP_METADATA_CSV": "/Users/wyc/SynPred/data/raw/mp_full_archive_export/mp_full_archive_metadata.csv",
    "MP_POSCAR_DIR": "/Users/wyc/SynPred/data/raw/mp_full_archive_export/poscar",

    # ---------- synthesis datasets ----------
    "SYN_FILES": {
        "solid_state": "/Users/wyc/SynPred/data/raw/solid-state_dataset_20200713.json",
        "solution_synthesis": "/Users/wyc/SynPred/data/raw/solutionsynthesis_dataset_202185.json",
    },

    # ---------- outputs ----------
    "OUTPUT_DIR": "/Users/wyc/SynPred/data/raw/mp_synth_direct_aligned",

    # ---------- matching ----------
    "TOPK_PER_RECORD": 3,
    "MIN_KEEP_SCORE": 35,
    "BEST_ONLY_MIN_SCORE": 55,
    "COMP_TOL": 0.015,
    "TITLE_SIM_THRESHOLD": 0.75,

    # ---------- runtime ----------
    "DEBUG_EVERY": 500,
    "RESUME": False,
    "PRINT_JSON_SHAPE": True,
}


# =========================================================
# Regex
# =========================================================
DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
TEMP_PATTERN = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*(°?\s*[CFK]|celsius|fahrenheit|kelvin|℃|℉|k)\b",
    re.I
)
TIME_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours|min|mins|minute|minutes|s|sec|secs|second|seconds|d|day|days)\b",
    re.I
)
PH_PATTERN = re.compile(r"\bpH\s*[:=]?\s*(\d+(?:\.\d+)?)", re.I)


# =========================================================
# Basic IO
# =========================================================
def safe_mkdir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_json_or_jsonl(path: str | Path) -> Any:
    """
    自动兼容普通 JSON 和 JSONL
    """
    path = Path(path)

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        pass

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_json(obj: Any, path: str | Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(records: list[dict], path: str | Path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl_record(path: str | Path, record: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# =========================================================
# General utils
# =========================================================
def normalize_text(s: Any) -> str | None:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    s = re.sub(r"\s+", " ", s)
    return s


def safe_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def unique_keep_order(seq):
    out = []
    seen = set()
    for x in seq:
        key = json.dumps(x, ensure_ascii=False, sort_keys=True) if isinstance(x, (dict, list)) else str(x)
        if key not in seen:
            out.append(x)
            seen.add(key)
    return out


def title_similarity(a: str | None, b: str | None) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def parse_formula(formula: Any) -> str | None:
    if formula is None:
        return None
    formula = str(formula).strip().replace(" ", "")
    if not formula:
        return None
    try:
        return Composition(formula).reduced_formula
    except Exception:
        return None

def rounded_parent_formula(formula: str | None, tol: float = 0.12) -> str | None:
    """
    把接近整数的化学计量数四舍五入到母相，用于：
    Y0.95VO4 -> YVO4
    Ca0.1Y0.9FeO3 -> YFeO3
    Ba1.99Eu0.01MgB2O6 -> Ba2MgB2O6

    不会处理 0.5/0.33 这类真正固溶体成分，它们会保持原样。
    """
    if not formula:
        return None

    try:
        comp = Composition(formula)
        ed = comp.get_el_amt_dict()
        new_ed = {}

        for el, amt in ed.items():
            nearest = int(amt + 0.5)
            if abs(amt - nearest) <= tol:
                amt = float(nearest)
            if amt > 1e-8:
                new_ed[el] = amt

        if not new_ed:
            return None

        return Composition(new_ed).reduced_formula
    except Exception:
        return None
from pymatgen.core import Composition


def get_el_amt_dict_safe(formula: str | None) -> dict[str, float]:
    if not formula:
        return {}
    try:
        return {k: float(v) for k, v in Composition(formula).get_el_amt_dict().items()}
    except Exception:
        return {}


def parent_match_profile(s_formula: str | None, parent_formula: str | None) -> dict | None:
    """
    用“原始化学计量”而不是原子分数判断掺杂是否足够轻。
    例如：
      Ca0.1Y0.9FeO3 -> YFeO3
      Li1Ti0.05Ni0.95O2 -> LiNiO2
      Ba1.95Eu0.05Mg1B2O6 -> Ba2MgB2O6
    """
    if not s_formula or not parent_formula:
        return None

    rp = rounded_parent_formula(s_formula)
    if rp != parent_formula:
        return None

    s = get_el_amt_dict_safe(s_formula)
    p = get_el_amt_dict_safe(parent_formula)
    if not s or not p:
        return None

    extra_elems = [e for e in s if e not in p and s[e] > 1e-8]
    extra_amount = sum(s[e] for e in extra_elems)

    missing_parent_amount = sum(max(p[e] - s.get(e, 0.0), 0.0) for e in p)

    shared_l1 = sum(abs(s.get(e, 0.0) - p.get(e, 0.0)) for e in p if e in s)
    max_shared_dev = max([abs(s.get(e, 0.0) - p.get(e, 0.0)) for e in p if e in s] + [0.0])

    profile = {
        "extra_elems": extra_elems,
        "n_extra_elems": len(extra_elems),
        "extra_amount": extra_amount,
        "missing_parent_amount": missing_parent_amount,
        "shared_l1": shared_l1,
        "max_shared_dev": max_shared_dev,
    }
    return profile


def is_light_parent_match(s_formula: str | None, parent_formula: str | None) -> bool:
    """
    严格母相回退规则：
    - rounded_parent_formula 必须命中
    - 最多 1 个新掺杂元素
    - 掺杂量 <= 0.15
    - 母相元素总偏移 <= 0.20
    - 单元素最大偏移 <= 0.15
    """
    prof = parent_match_profile(s_formula, parent_formula)
    if prof is None:
        return False

    return (
        prof["n_extra_elems"] <= 1
        and prof["extra_amount"] <= 0.15
        and prof["missing_parent_amount"] <= 0.15
        and prof["shared_l1"] <= 0.20
        and prof["max_shared_dev"] <= 0.15
    )


def classify_match_level(s_row: dict, mp_row: dict, reason: str) -> str:
    rs = set(reason.split("|")) if reason else set()

    if "exact_formula" in rs:
        return "exact"

    if "parent_formula_strict" in rs:
        return "parent_strict"

    if "doi_overlap" in rs or any(x.startswith("title_sim=") for x in rs):
        return "literature_supported"

    return "weak"

def composition_dict_to_formula(comp: dict[str, Any]) -> str | None:
    try:
        c = Composition({k: float(v) for k, v in comp.items() if float(v) != 0})
        return c.reduced_formula
    except Exception:
        return None


def composition_to_fraction_dict(formula: str | None) -> dict[str, float]:
    if not formula:
        return {}
    try:
        comp = Composition(formula)
        ed = comp.get_el_amt_dict()
        total = sum(ed.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in ed.items()}
    except Exception:
        return {}


def comp_distance(frac_a: dict[str, float], frac_b: dict[str, float]) -> float:
    keys = sorted(set(frac_a) | set(frac_b))
    return sum(abs(frac_a.get(k, 0.0) - frac_b.get(k, 0.0)) for k in keys)


def same_element_set(frac_a: dict[str, float], frac_b: dict[str, float]) -> bool:
    return set(frac_a.keys()) == set(frac_b.keys())


def get_chemsys(formula: str | None) -> str | None:
    if not formula:
        return None
    try:
        els = sorted(Composition(formula).get_el_amt_dict().keys())
        return "-".join(els)
    except Exception:
        return None


def get_anonymous_formula(formula: str | None) -> str | None:
    if not formula:
        return None
    try:
        return Composition(formula).anonymized_formula
    except Exception:
        return None


def extract_dois_in_obj(obj: Any) -> list[str]:
    found = set()

    def walk(x):
        if x is None:
            return
        if isinstance(x, str):
            for m in DOI_PATTERN.findall(x):
                found.add(m.strip().rstrip(".,;]})"))
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple, set)):
            for item in x:
                walk(item)

    walk(obj)
    return sorted(found)


def flatten_strings_from_obj(obj: Any, key_contains: tuple[str, ...]) -> list[str]:
    out = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                k_low = str(k).lower()
                if any(token in k_low for token in key_contains):
                    if isinstance(v, str):
                        vv = normalize_text(v)
                        if vv:
                            out.append(vv)
                    elif isinstance(v, list):
                        for item in v:
                            if isinstance(item, str):
                                ii = normalize_text(item)
                                if ii:
                                    out.append(ii)
                walk(v)
        elif isinstance(x, list):
            for item in x:
                walk(item)

    walk(obj)
    return unique_keep_order(out)


# =========================================================
# JSON structure helpers
# =========================================================
def looks_like_record(d: dict) -> bool:
    if not isinstance(d, dict) or not d:
        return False

    keys = {str(k).lower() for k in d.keys()}
    hint_tokens = [
        "formula", "composition", "target", "product",
        "precursor", "reactant", "reagent",
        "synthesis", "procedure", "step", "method",
        "temperature", "temp", "time", "duration",
        "doi", "title", "solvent", "atmosphere", "ph",
        "reaction", "targets_string", "paragraph_string"
    ]
    hit = sum(any(tok in k for tok in hint_tokens) for k in keys)
    primitive_count = sum(not isinstance(v, (dict, list)) for v in d.values())
    primitive_ratio = primitive_count / max(len(d), 1)
    return hit >= 1 or primitive_ratio >= 0.5


def collect_record_candidates(obj: Any) -> list[dict]:
    out = []

    def walk(x):
        if isinstance(x, dict):
            if looks_like_record(x):
                out.append(x)
                return
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            if x and all(isinstance(i, dict) for i in x):
                record_like = sum(looks_like_record(i) for i in x)
                if record_like >= max(1, len(x) // 3):
                    out.extend(x)
                    return
            for item in x:
                walk(item)

    walk(obj)
    return out


def iter_records(obj: Any) -> list[dict]:
    """
    兼容：
    - list[dict]
    - {"reactions": [...]}
    - {"data": [...]}
    - {"id1": {...}, "id2": {...}}
    """
    if isinstance(obj, list):
        dict_items = [x for x in obj if isinstance(x, dict)]
        if dict_items:
            return dict_items

    if isinstance(obj, dict):
        for key in ["data", "records", "results", "items", "entries", "reactions", "dataset"]:
            v = obj.get(key)
            if isinstance(v, list) and v and any(isinstance(i, dict) for i in v):
                return [i for i in v if isinstance(i, dict)]
            if isinstance(v, dict):
                vals = list(v.values())
                if vals and all(isinstance(i, dict) for i in vals):
                    return vals

        vals = list(obj.values())
        if vals and all(isinstance(i, dict) for i in vals):
            return vals

    candidates = collect_record_candidates(obj)
    uniq = []
    seen = set()
    for r in candidates:
        key = json.dumps(r, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            uniq.append(r)
            seen.add(key)

    if uniq:
        return uniq

    raise ValueError("无法自动识别 JSON 记录结构。")


def inspect_json_shape(obj: Any, max_depth: int = 2, prefix: str = "root"):
    if max_depth < 0:
        return

    if isinstance(obj, dict):
        keys = list(obj.keys())[:20]
        print(f"{prefix}: dict, keys={keys}", flush=True)
        for k in keys[:10]:
            inspect_json_shape(obj[k], max_depth - 1, f"{prefix}.{k}")
    elif isinstance(obj, list):
        print(f"{prefix}: list, len={len(obj)}", flush=True)
        for i, item in enumerate(obj[:3]):
            inspect_json_shape(item, max_depth - 1, f"{prefix}[{i}]")
    else:
        print(f"{prefix}: {type(obj).__name__}", flush=True)


# =========================================================
# Value search in synthesis records
# =========================================================
FORMULA_KEYS = [
    "formula", "reduced_formula", "pretty_formula", "target_formula",
    "composition_formula", "chemical_formula", "material_formula", "product_formula"
]

TITLE_KEYS = [
    "title", "paper_title", "article_title", "publication_title"
]

DOI_KEYS = [
    "doi", "DOI", "paper_doi", "article_doi", "publication_doi"
]

PRECURSOR_TOKENS = (
    "precursor", "precursors", "reactant", "reactants", "reagent", "reagents",
    "starting_material", "starting materials", "source_material", "feedstock"
)

STEP_TOKENS = (
    "step", "steps", "procedure", "process", "operations", "operation",
    "synthesis", "experimental", "method", "recipe"
)

TEMP_TOKENS = (
    "temperature", "temp", "calcination_temperature", "annealing_temperature",
    "heating_temperature", "reaction_temperature"
)

TIME_TOKENS = (
    "time", "duration", "heating_time", "annealing_time", "calcination_time",
    "reaction_time"
)

ATM_TOKENS = (
    "atmosphere", "gas", "ambient", "environment"
)

SOLVENT_TOKENS = (
    "solvent", "solvents", "medium"
)


def find_values_by_key_tokens(obj: Any, tokens: tuple[str, ...]) -> list[Any]:
    out = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                k_low = str(k).lower()
                if any(tok in k_low for tok in tokens):
                    out.append(v)
                walk(v)
        elif isinstance(x, list):
            for item in x:
                walk(item)

    walk(obj)
    return out


def extract_formula_from_record(rec: dict) -> str | None:
    # 1) 常见直接字段
    for key in FORMULA_KEYS:
        if key in rec:
            f = parse_formula(rec[key])
            if f:
                return f

    # 2) solid-state 数据常见 targets_string
    if "targets_string" in rec:
        ts = rec["targets_string"]
        if isinstance(ts, list):
            for x in ts:
                f = parse_formula(x)
                if f:
                    return f
        elif isinstance(ts, str):
            f = parse_formula(ts)
            if f:
                return f

    # 3) 宽松递归公式字段
    for k, v in rec.items():
        if isinstance(v, str) and "formula" in str(k).lower():
            f = parse_formula(v)
            if f:
                return f
        if isinstance(v, dict) and ("comp" in str(k).lower() or "composition" in str(k).lower()):
            f = composition_dict_to_formula(v)
            if f:
                return f

    # 4) reaction.right_side 回退
    reaction = rec.get("reaction")
    if isinstance(reaction, dict):
        right_side = reaction.get("right_side")
        if isinstance(right_side, list):
            for x in right_side:
                f = parse_formula(x)
                if f:
                    return f
        elif isinstance(right_side, str):
            f = parse_formula(right_side)
            if f:
                return f

    return None


def extract_title_from_record(rec: dict) -> str | None:
    for k in TITLE_KEYS:
        if k in rec:
            t = normalize_text(rec[k])
            if t:
                return t

    for k, v in rec.items():
        if isinstance(v, str) and "title" in str(k).lower():
            t = normalize_text(v)
            if t:
                return t
    return None


def extract_synthesis_text(rec: dict) -> str | None:
    for key in ["paragraph_string", "paragraph", "text", "procedure", "experimental", "method"]:
        if key in rec and isinstance(rec[key], str) and rec[key].strip():
            return rec[key].strip()
    return None


def extract_dois_from_record(rec: dict) -> list[str]:
    out = []
    for k in DOI_KEYS:
        if k in rec:
            v = rec[k]
            if isinstance(v, str):
                out.extend(DOI_PATTERN.findall(v))
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, str):
                        out.extend(DOI_PATTERN.findall(item))

    out.extend(extract_dois_in_obj(rec))
    out = [x.strip().rstrip(".,;]})") for x in out]
    return unique_keep_order(out)


def normalize_precursor_item(x: Any) -> dict | None:
    if isinstance(x, str):
        s = normalize_text(x)
        if not s:
            return None
        return {"name": s}

    if isinstance(x, dict):
        name = None
        amount = None
        unit = None
        formula = None

        for k, v in x.items():
            k_low = str(k).lower()
            if name is None and k_low in {"name", "precursor", "reactant", "reagent", "compound", "chemical", "material"}:
                name = normalize_text(v)
            if formula is None and "formula" in k_low:
                formula = normalize_text(v)
            if amount is None and any(t in k_low for t in ["amount", "quantity", "mass", "mole", "mol", "weight"]):
                amount = normalize_text(v)
            if unit is None and "unit" in k_low:
                unit = normalize_text(v)

        if not name and formula:
            name = formula

        if not name:
            blob = normalize_text(json.dumps(x, ensure_ascii=False))
            if not blob:
                return None
            return {"name": blob}

        out = {"name": name}
        if formula:
            out["formula"] = formula
        if amount:
            out["amount"] = amount
        if unit:
            out["unit"] = unit
        return out

    return None


def extract_precursors(rec: dict) -> list[dict]:
    precursors = []

    # 1) reaction.left_side 优先
    reaction = rec.get("reaction")
    if isinstance(reaction, dict):
        left_side = reaction.get("left_side")
        if isinstance(left_side, list):
            for x in left_side:
                nz = normalize_precursor_item(x)
                if nz:
                    precursors.append(nz)

    # 2) 其他 precursor/reactant/reagent 键
    if not precursors:
        raw = find_values_by_key_tokens(rec, PRECURSOR_TOKENS)
        for item in raw:
            if isinstance(item, list):
                for z in item:
                    nz = normalize_precursor_item(z)
                    if nz:
                        precursors.append(nz)
            else:
                nz = normalize_precursor_item(item)
                if nz:
                    precursors.append(nz)

    return unique_keep_order(precursors)


def parse_temperature_value(x: Any) -> float | None:
    if isinstance(x, (int, float)):
        return float(x)

    if not isinstance(x, str):
        return None

    s = x.strip()
    m = TEMP_PATTERN.search(s)
    if not m:
        return None

    value = float(m.group(1))
    unit = m.group(2).lower().replace(" ", "")

    if unit in ["c", "°c", "℃", "celsius"]:
        return value
    if unit in ["k", "kelvin"]:
        return value - 273.15
    if unit in ["f", "°f", "℉", "fahrenheit"]:
        return (value - 32.0) * 5.0 / 9.0

    return None


def parse_time_hours(x: Any) -> float | None:
    if isinstance(x, (int, float)):
        return float(x)

    if not isinstance(x, str):
        return None

    s = x.strip().lower()
    m = TIME_PATTERN.search(s)
    if not m:
        return None

    value = float(m.group(1))
    unit = m.group(2)

    if unit in ["h", "hr", "hrs", "hour", "hours"]:
        return value
    if unit in ["min", "mins", "minute", "minutes"]:
        return value / 60.0
    if unit in ["s", "sec", "secs", "second", "seconds"]:
        return value / 3600.0
    if unit in ["d", "day", "days"]:
        return value * 24.0

    return None


def infer_atmosphere_from_text(text: str | None) -> str | None:
    if not text:
        return None
    t = text.lower()
    for atm in ["air", "oxygen", "o2", "nitrogen", "n2", "argon", "ar", "vacuum", "reducing", "inert"]:
        if atm in t:
            return atm
    return None


def infer_solvent_from_text(text: str | None) -> str | None:
    if not text:
        return None
    t = text.lower()
    for s in ["water", "ethanol", "methanol", "isopropanol", "dmf", "dmso", "acetone", "toluene"]:
        if s in t:
            return s
    return None


def normalize_step_name(name: str | None) -> str | None:
    if not name:
        return None
    name_low = name.lower()
    for token in [
        "mix", "grind", "milling", "ball mill", "stir", "dry", "drying",
        "calcine", "calcination", "anneal", "annealing", "heat", "heating",
        "sinter", "hydrothermal", "solvothermal", "wash", "filter", "press"
    ]:
        if token in name_low:
            return token
    return name


def normalize_step_object(step: Any) -> dict | None:
    if isinstance(step, str):
        s = normalize_text(step)
        if not s:
            return None
        return {
            "operation": normalize_step_name(s),
            "text": s,
            "temperature_c": parse_temperature_value(s),
            "time_h": parse_time_hours(s),
            "atmosphere": infer_atmosphere_from_text(s),
            "solvent": infer_solvent_from_text(s),
        }

    if isinstance(step, dict):
        out = {
            "operation": None,
            "text": None,
            "temperature_c": None,
            "time_h": None,
            "atmosphere": None,
            "solvent": None,
        }

        for k, v in step.items():
            k_low = str(k).lower()

            if out["operation"] is None and any(tok in k_low for tok in ["name", "step", "operation", "process", "type"]):
                out["operation"] = normalize_step_name(normalize_text(v))

            if out["text"] is None and any(tok in k_low for tok in ["description", "detail", "text", "note"]):
                out["text"] = normalize_text(v)

            if out["temperature_c"] is None and "temp" in k_low:
                out["temperature_c"] = parse_temperature_value(v)

            if out["time_h"] is None and any(tok in k_low for tok in ["time", "duration"]):
                out["time_h"] = parse_time_hours(v)

            if out["atmosphere"] is None and any(tok in k_low for tok in ["atmosphere", "gas", "ambient", "environment"]):
                out["atmosphere"] = normalize_text(v)

            if out["solvent"] is None and "solvent" in k_low:
                out["solvent"] = normalize_text(v)

        blob = normalize_text(json.dumps(step, ensure_ascii=False))
        if out["text"] is None:
            out["text"] = blob
        if out["temperature_c"] is None and blob:
            out["temperature_c"] = parse_temperature_value(blob)
        if out["time_h"] is None and blob:
            out["time_h"] = parse_time_hours(blob)
        if out["atmosphere"] is None and blob:
            out["atmosphere"] = infer_atmosphere_from_text(blob)
        if out["solvent"] is None and blob:
            out["solvent"] = infer_solvent_from_text(blob)
        if out["operation"] is None and blob:
            out["operation"] = normalize_step_name(blob)

        if all(v is None for v in out.values()):
            return None
        return out

    return None


def extract_steps(rec: dict) -> list[dict]:
    steps = []

    # 先从显式步骤字段里找
    raw_steps = find_values_by_key_tokens(rec, STEP_TOKENS)
    for item in raw_steps:
        if isinstance(item, list):
            for step in item:
                parsed = normalize_step_object(step)
                if parsed:
                    steps.append(parsed)
        else:
            parsed = normalize_step_object(item)
            if parsed:
                steps.append(parsed)

    # 再从 synthesis_text 构造
    synth_text = extract_synthesis_text(rec)
    if synth_text:
        step_from_text = normalize_step_object(synth_text)
        if step_from_text:
            steps.append(step_from_text)

    return unique_keep_order(steps)


def extract_temperatures(rec: dict) -> list[float]:
    out = []
    vals = find_values_by_key_tokens(rec, TEMP_TOKENS)
    for v in vals:
        if isinstance(v, list):
            for z in v:
                t = parse_temperature_value(z)
                if t is not None:
                    out.append(t)
        else:
            t = parse_temperature_value(v)
            if t is not None:
                out.append(t)

    synth_text = extract_synthesis_text(rec)
    if synth_text:
        t = parse_temperature_value(synth_text)
        if t is not None:
            out.append(t)

    return out


def extract_times(rec: dict) -> list[float]:
    out = []
    vals = find_values_by_key_tokens(rec, TIME_TOKENS)
    for v in vals:
        if isinstance(v, list):
            for z in v:
                t = parse_time_hours(z)
                if t is not None:
                    out.append(t)
        else:
            t = parse_time_hours(v)
            if t is not None:
                out.append(t)

    synth_text = extract_synthesis_text(rec)
    if synth_text:
        t = parse_time_hours(synth_text)
        if t is not None:
            out.append(t)

    return out


def extract_atmospheres(rec: dict) -> list[str]:
    out = []

    vals = find_values_by_key_tokens(rec, ATM_TOKENS)
    for v in vals:
        if isinstance(v, str):
            s = normalize_text(v)
            if s:
                out.append(s)
        elif isinstance(v, list):
            for z in v:
                if isinstance(z, str):
                    s = normalize_text(z)
                    if s:
                        out.append(s)

    synth_text = extract_synthesis_text(rec)
    if synth_text:
        s = infer_atmosphere_from_text(synth_text)
        if s:
            out.append(s)

    return unique_keep_order(out)


def extract_solvents(rec: dict) -> list[str]:
    out = []

    vals = find_values_by_key_tokens(rec, SOLVENT_TOKENS)
    for v in vals:
        if isinstance(v, str):
            s = normalize_text(v)
            if s:
                out.append(s)
        elif isinstance(v, list):
            for z in v:
                if isinstance(z, str):
                    s = normalize_text(z)
                    if s:
                        out.append(s)

    synth_text = extract_synthesis_text(rec)
    if synth_text:
        s = infer_solvent_from_text(synth_text)
        if s:
            out.append(s)

    return unique_keep_order(out)


def extract_ph(rec: dict) -> float | None:
    blob = json.dumps(rec, ensure_ascii=False)
    m = PH_PATTERN.search(blob)
    if m:
        return float(m.group(1))
    return None


def build_synthesis_features(rec: dict) -> dict:
    formula = extract_formula_from_record(rec)
    parent_formula = rounded_parent_formula(formula)
    title = extract_title_from_record(rec)
    synthesis_text = extract_synthesis_text(rec)
    dois = extract_dois_from_record(rec)
    precursors = extract_precursors(rec)
    steps = extract_steps(rec)
    temps = extract_temperatures(rec)
    times = extract_times(rec)
    atmospheres = extract_atmospheres(rec)
    solvents = extract_solvents(rec)
    ph = extract_ph(rec)

    max_temp = max(temps) if temps else None
    total_time = sum(times) if times else None

    if steps:
        step_temps = [x.get("temperature_c") for x in steps if x.get("temperature_c") is not None]
        step_times = [x.get("time_h") for x in steps if x.get("time_h") is not None]
        if max_temp is None and step_temps:
            max_temp = max(step_temps)
        if total_time is None and step_times:
            total_time = sum(step_times)

    return {
        "formula": formula,
        "parent_formula": parent_formula,
        "title": title,
        "synthesis_text": synthesis_text,
        "dois": dois,
        "precursors": precursors,
        "steps": steps,
        "temperatures_c": temps,
        "times_h": times,
        "max_temperature_c": max_temp,
        "total_time_h": total_time,
        "atmosphere": atmospheres[0] if atmospheres else None,
        "all_atmospheres": atmospheres,
        "solvent": solvents[0] if solvents else None,
        "all_solvents": solvents,
        "ph": ph,
        "chemsys": get_chemsys(formula),
        "anonymous_formula": get_anonymous_formula(formula),
        "comp_frac": composition_to_fraction_dict(formula),
    }

# =========================================================
# MP loading
# =========================================================
def find_poscar_for_mpid(mpid: str, poscar_dir: str | Path) -> str | None:
    pat = str(Path(poscar_dir) / f"{mpid}_*.vasp")
    hits = glob.glob(pat)
    return hits[0] if hits else None


def build_mp_entries() -> pd.DataFrame:
    df = pd.read_csv(CONFIG["MP_METADATA_CSV"])
    entries = []

    for i, row in df.iterrows():
        mpid = row.get("material_id")
        formula = row.get("formula_pretty")
        formula = None if pd.isna(formula) else parse_formula(formula)

        summary_path = row.get("summary_json_path")
        doi_path = row.get("doi_json_path")
        prov_path = row.get("provenance_json_path")

        summary = read_json(summary_path) if isinstance(summary_path, str) and os.path.exists(summary_path) else {}
        doi_json = read_json(doi_path) if isinstance(doi_path, str) and os.path.exists(doi_path) else {}
        prov_json = read_json(prov_path) if isinstance(prov_path, str) and os.path.exists(prov_path) else {}

        if not formula and isinstance(summary, dict) and "structure" in summary:
            try:
                structure = Structure.from_dict(summary["structure"])
                formula = structure.composition.reduced_formula
            except Exception:
                pass

        all_dois = set()
        all_dois.update(extract_dois_in_obj(doi_json))
        all_dois.update(extract_dois_in_obj(prov_json))

        if isinstance(row.get("mp_doi"), str):
            all_dois.update(DOI_PATTERN.findall(row["mp_doi"]))
        if isinstance(row.get("literature_dois_found"), str):
            all_dois.update(DOI_PATTERN.findall(row["literature_dois_found"]))

        titles = []
        if isinstance(row.get("title_candidates_from_provenance"), str):
            titles.extend([x.strip() for x in row["title_candidates_from_provenance"].split("|") if x.strip()])
        titles.extend(flatten_strings_from_obj(prov_json, ("title", "citation", "reference")))

        poscar_path = find_poscar_for_mpid(str(mpid), CONFIG["MP_POSCAR_DIR"])

        e_above_hull = None
        if isinstance(summary, dict):
            e_above_hull = summary.get("energy_above_hull")
            if e_above_hull is not None:
                try:
                    e_above_hull = float(e_above_hull)
                except (TypeError, ValueError):
                    e_above_hull = None

        entries.append({
            "material_id": mpid,
            "formula_pretty": formula,
            "chemsys": get_chemsys(formula),
            "anonymous_formula": get_anonymous_formula(formula),
            "comp_frac": composition_to_fraction_dict(formula),
            "all_dois": sorted(all_dois),
            "titles": unique_keep_order(titles),
            "poscar_path": poscar_path,
            "summary_json_path": summary_path if isinstance(summary_path, str) else None,
            "doi_json_path": doi_path if isinstance(doi_path, str) else None,
            "provenance_json_path": prov_path if isinstance(prov_path, str) else None,
            "energy_above_hull": e_above_hull,
        })

        if (i + 1) % CONFIG["DEBUG_EVERY"] == 0:
            print(f"[DEBUG] loaded MP entries: {i + 1}", flush=True)

    return pd.DataFrame(entries)


# =========================================================
# Matching
# =========================================================
def candidate_subset_for_synth(s_formula: str | None, mp_df: pd.DataFrame) -> pd.DataFrame:
    if not s_formula:
        return mp_df.iloc[0:0].copy()

    # 1) 精确公式
    exact_formula = mp_df[mp_df["formula_pretty"] == s_formula]
    if len(exact_formula) > 0:
        return exact_formula.copy()

    # 2) 严格母相公式
    parent_formula = rounded_parent_formula(s_formula)
    if parent_formula and parent_formula != s_formula:
        parent_hits = mp_df[mp_df["formula_pretty"] == parent_formula]
        if len(parent_hits) > 0:
            return parent_hits.copy()

    # 3) 同 anonymous_formula + 同 chemsys
    s_chemsys = get_chemsys(s_formula)
    s_anon = get_anonymous_formula(s_formula)

    narrowed = mp_df[
        (mp_df["chemsys"] == s_chemsys) &
        (mp_df["anonymous_formula"] == s_anon)
    ]
    if len(narrowed) > 0:
        return narrowed.copy()

    # 4) 不再兜底到 same_chemsys
    return mp_df.iloc[0:0].copy()
def score_match(s_row: dict, mp_row: dict) -> tuple[float, str]:
    score = 0.0
    reasons = []

    s_formula = s_row.get("formula")
    m_formula = mp_row.get("formula_pretty")

    s_frac = s_row.get("comp_frac", {})
    m_frac = mp_row.get("comp_frac", {})

    # ---- exact formula ----
    if s_formula and m_formula and s_formula == m_formula:
        score += 80
        reasons.append("exact_formula")

    # ---- strict parent formula ----
    if s_formula and m_formula and s_formula != m_formula:
        if is_light_parent_match(s_formula, m_formula):
            score += 45
            reasons.append("parent_formula_strict")
        elif rounded_parent_formula(s_formula) == m_formula:
            reasons.append("parent_formula_weak")

    s_chemsys = s_row.get("chemsys")
    m_chemsys = mp_row.get("chemsys")
    if s_chemsys and m_chemsys and s_chemsys == m_chemsys:
        score += 8
        reasons.append("same_chemsys")

    s_anon = s_row.get("anonymous_formula")
    m_anon = mp_row.get("anonymous_formula")
    if s_anon and m_anon and s_anon == m_anon:
        score += 8
        reasons.append("same_anonymous_formula")

    if s_frac and m_frac:
        dist = comp_distance(s_frac, m_frac)
        if dist <= CONFIG["COMP_TOL"]:
            score += 20
            reasons.append(f"comp_dist<={CONFIG['COMP_TOL']}")
        elif dist <= 2 * CONFIG["COMP_TOL"]:
            score += 6
            reasons.append("comp_close")

        if same_element_set(s_frac, m_frac):
            score += 4
            reasons.append("same_element_set")

    s_dois = set(s_row.get("dois", []))
    m_dois = set(mp_row.get("all_dois", []))
    overlap = sorted(s_dois & m_dois)
    if overlap:
        score += 35
        reasons.append("doi_overlap")

    s_title = s_row.get("title")
    best_title_sim = 0.0
    for t in mp_row.get("titles", []):
        sim = title_similarity(s_title, t)
        if sim > best_title_sim:
            best_title_sim = sim

    if best_title_sim >= CONFIG["TITLE_SIM_THRESHOLD"]:
        score += 20
        reasons.append(f"title_sim={best_title_sim:.2f}")
    elif best_title_sim >= 0.60:
        score += 8
        reasons.append(f"title_sim={best_title_sim:.2f}")

    if mp_row.get("poscar_path"):
        score += 3
        reasons.append("has_poscar")

    e_hull = mp_row.get("energy_above_hull")
    if e_hull is not None and e_hull == 0.0:
        score += 2
        reasons.append("ground_state")
    elif e_hull is not None and e_hull <= 0.01:
        score += 1
        reasons.append("near_ground_state")

    return score, "|".join(reasons)

def is_reliable_match(s_row: dict, mp_row: dict, score: float, reason: str) -> bool:
    rs = set(reason.split("|")) if reason else set()

    has_exact = "exact_formula" in rs
    has_parent_strict = "parent_formula_strict" in rs
    has_doi = "doi_overlap" in rs
    has_same_chemsys = "same_chemsys" in rs
    has_same_anon = "same_anonymous_formula" in rs
    has_same_elemset = "same_element_set" in rs
    has_comp_tight = f"comp_dist<={CONFIG['COMP_TOL']}" in rs
    has_comp_close = "comp_close" in rs
    has_title_high = any(x.startswith("title_sim=") for x in rs)

    # 1) exact 一定通过
    if has_exact:
        return True

    # 2) strict parent 一定通过
    if has_parent_strict:
        return True

    # 3) DOI 必须同时有化学一致性，不再单独放行
    if has_doi and has_same_chemsys and (has_comp_tight or has_same_anon or has_parent_strict):
        return True

    # 4) 标题很像 + 很近的组成
    if has_title_high and has_comp_tight and has_same_chemsys:
        return True

    # 5) 同匿名式 + 同元素集 + 很近组成
    if has_same_anon and has_same_elemset and has_comp_tight:
        return True

    # 6) 其他一律不通过
    return False

# =========================================================
# Resume helpers
# =========================================================
def get_processed_synth_uids(output_dir: str | Path) -> set[str]:
    output_dir = Path(output_dir)
    processed = set()

    for name in ["direct_aligned_dataset.jsonl", "unmatched_records.jsonl"]:
        fp = output_dir / name
        if not fp.exists():
            continue
        for row in read_jsonl(fp):
            uid = row.get("synth_uid")
            if uid:
                processed.add(uid)

    return processed


# =========================================================
# Load synthesis datasets
# =========================================================
def load_synthesis_records() -> pd.DataFrame:
    rows = []

    for source_name, path in CONFIG["SYN_FILES"].items():
        print(f"[INFO] loading synthesis file: {path}", flush=True)

        data = read_json_or_jsonl(path)
        if CONFIG.get("PRINT_JSON_SHAPE", False):
            inspect_json_shape(data, max_depth=2)

        records = iter_records(data)
        print(f"[INFO] extracted {len(records)} records from {source_name}", flush=True)

        for idx, rec in enumerate(records):
            feats = build_synthesis_features(rec)

            rows.append({
                "synth_uid": f"{source_name}_{idx:08d}",
                "source_dataset": source_name,
                "record_index": idx,
                "raw_record": rec,
                **feats,
            })

            if (idx + 1) % CONFIG["DEBUG_EVERY"] == 0:
                print(f"[DEBUG] loaded {source_name}: {idx + 1}", flush=True)

    return pd.DataFrame(rows)


# =========================================================
# Main
# =========================================================
def main():
    safe_mkdir(CONFIG["OUTPUT_DIR"])
    out_dir = Path(CONFIG["OUTPUT_DIR"])

    aligned_jsonl_path = out_dir / "direct_aligned_dataset.jsonl"
    unmatched_jsonl_path = out_dir / "unmatched_records.jsonl"
    candidate_jsonl_path = out_dir / "all_candidates.jsonl"

    processed_uids = set()
    if CONFIG.get("RESUME", False):
        processed_uids = get_processed_synth_uids(out_dir)
        print(f"[INFO] resume mode: already processed {len(processed_uids)} synth records", flush=True)
    else:
        for fp in [aligned_jsonl_path, unmatched_jsonl_path, candidate_jsonl_path]:
            if fp.exists():
                fp.unlink()

    print("[INFO] loading MP entries...", flush=True)
    mp_df = build_mp_entries()
    print(f"[INFO] MP entries = {len(mp_df)}", flush=True)

    print("[INFO] loading synthesis records...", flush=True)
    synth_df = load_synthesis_records()
    print(f"[INFO] synthesis records = {len(synth_df)}", flush=True)

    for i, s in synth_df.iterrows():
        s_dict = s.to_dict()

        if s_dict["synth_uid"] in processed_uids:
            continue

        candidates = candidate_subset_for_synth(s_dict.get("formula"), mp_df)

        if len(candidates) == 0:
            unmatched_record = {
                "synth_uid": s_dict["synth_uid"],
                "source_dataset": s_dict["source_dataset"],
                "formula": s_dict.get("formula"),
                "title": s_dict.get("title"),
                "reason": "no_formula_candidate",
            }
            append_jsonl_record(unmatched_jsonl_path, unmatched_record)
            continue

        scored = []
        for _, m in candidates.iterrows():
            m_dict = m.to_dict()
            score, reason = score_match(s_dict, m_dict)
            if score < CONFIG["MIN_KEEP_SCORE"]:
                continue
            if not is_reliable_match(s_dict, m_dict, score, reason):
                continue
#            score, reason = score_match(s_dict, m_dict)
#            if score < CONFIG["MIN_KEEP_SCORE"]:
#                continue

            scored.append({
                "synth_uid": s_dict["synth_uid"],
                "source_dataset": s_dict["source_dataset"],
                "record_index": int(s_dict["record_index"]),
                "material_id": m_dict["material_id"],
                "mp_formula": m_dict["formula_pretty"],
                "synth_formula": s_dict.get("formula"),
                "score": score,
                "reason": reason,
                "poscar_path": m_dict.get("poscar_path"),
                "summary_json_path": m_dict.get("summary_json_path"),
                "doi_json_path": m_dict.get("doi_json_path"),
                "provenance_json_path": m_dict.get("provenance_json_path"),
            })

        if not scored:
            unmatched_record = {
                "synth_uid": s_dict["synth_uid"],
                "source_dataset": s_dict["source_dataset"],
                "formula": s_dict.get("formula"),
                "title": s_dict.get("title"),
                "reason": "all_candidates_below_threshold",
            }
            append_jsonl_record(unmatched_jsonl_path, unmatched_record)
            continue

        scored = sorted(scored, key=lambda x: (-x["score"], x["material_id"]))

        # Prefer original mp_id for solution_synthesis records (author-annotated polymorph)
        orig_mpid = None
        if s_dict.get("source_dataset") == "solution_synthesis":
            raw_rec = s_dict.get("raw_record") or {}
            target = raw_rec.get("target") or {}
            orig_mpid = target.get("mp_id") if isinstance(target, dict) else None

        if orig_mpid:
            orig_match = [x for x in scored if x["material_id"] == orig_mpid]
            if orig_match:
                scored = orig_match + [x for x in scored if x["material_id"] != orig_mpid]

        topk = scored[:CONFIG["TOPK_PER_RECORD"]]

        best = topk[0]
        if best["score"] < CONFIG["BEST_ONLY_MIN_SCORE"]:
            unmatched_record = {
                "synth_uid": s_dict["synth_uid"],
                "source_dataset": s_dict["source_dataset"],
                "formula": s_dict.get("formula"),
                "title": s_dict.get("title"),
                "reason": f"best_score_too_low:{best['score']}",
            }
            append_jsonl_record(unmatched_jsonl_path, unmatched_record)
            continue
        best_mp_row = candidates[candidates["material_id"] == best["material_id"]].iloc[0].to_dict()
        match_level = classify_match_level(s_dict, best_mp_row, best["reason"])

        aligned = {
            "id": f"{s_dict['synth_uid']}__{best['material_id']}",
            "synth_uid": s_dict["synth_uid"],
            "source_dataset": s_dict["source_dataset"],
            "record_index": int(s_dict["record_index"]),
            "parent_formula": s_dict.get("parent_formula"),
            # structure
            "material_id": best["material_id"],
            "mp_formula": best["mp_formula"],
            "poscar_path": best["poscar_path"],
            "summary_json_path": best["summary_json_path"],
            "doi_json_path": best["doi_json_path"],
            "provenance_json_path": best["provenance_json_path"],

            # matching
            "match_score": float(best["score"]),
            "match_reason": best["reason"],
            "match_level": match_level,

            # synthesis
            "synth_formula": s_dict.get("formula"),
            "title": s_dict.get("title"),
            "synthesis_text": s_dict.get("synthesis_text"),
            "dois": s_dict.get("dois", []),

            "precursors": s_dict.get("precursors", []),
            "steps": s_dict.get("steps", []),

            "temperatures_c": s_dict.get("temperatures_c", []),
            "times_h": s_dict.get("times_h", []),
            "max_temperature_c": s_dict.get("max_temperature_c"),
            "total_time_h": s_dict.get("total_time_h"),

            "atmosphere": s_dict.get("atmosphere"),
            "all_atmospheres": s_dict.get("all_atmospheres", []),

            "solvent": s_dict.get("solvent"),
            "all_solvents": s_dict.get("all_solvents", []),

            "ph": s_dict.get("ph"),

            "raw_synthesis_record": s_dict.get("raw_record"),
        }

        # 完整处理成功后再落盘，避免续算时候选重复
        for cand in topk:
            append_jsonl_record(candidate_jsonl_path, cand)
        append_jsonl_record(aligned_jsonl_path, aligned)

        if (i + 1) % CONFIG["DEBUG_EVERY"] == 0:
            print(
                f"[DEBUG] aligned {i + 1} | synth_uid={s_dict['synth_uid']} "
                f"| material_id={best['material_id']} | score={best['score']}",
                flush=True
            )

    # -------- rebuild outputs from jsonl --------
    all_candidates = read_jsonl(candidate_jsonl_path)
    best_records = read_jsonl(aligned_jsonl_path)
    unmatched = read_jsonl(unmatched_jsonl_path)

    all_df = pd.DataFrame(all_candidates)
    best_df = pd.DataFrame([
        {
            "id": r.get("id"),
            "synth_uid": r.get("synth_uid"),
            "source_dataset": r.get("source_dataset"),
            "material_id": r.get("material_id"),
            "mp_formula": r.get("mp_formula"),
            "synth_formula": r.get("synth_formula"),
            "poscar_path": r.get("poscar_path"),
            "match_score": r.get("match_score"),
            "match_reason": r.get("match_reason"),
            "match_level": r.get("match_level"),
            "num_precursors": len(r.get("precursors", [])),
            "num_steps": len(r.get("steps", [])),
            "max_temperature_c": r.get("max_temperature_c"),
            "total_time_h": r.get("total_time_h"),
            "atmosphere": r.get("atmosphere"),
            "solvent": r.get("solvent"),
            "ph": r.get("ph"),
        }
        for r in best_records
    ])
    unmatched_df = pd.DataFrame(unmatched)

    all_df.to_csv(out_dir / "all_candidates.csv", index=False, encoding="utf-8-sig")
    best_df.to_csv(out_dir / "aligned_summary.csv", index=False, encoding="utf-8-sig")
    unmatched_df.to_csv(out_dir / "unmatched.csv", index=False, encoding="utf-8-sig")
    write_json(best_records, out_dir / "direct_aligned_dataset.json")

    print(f"[INFO] saved: {out_dir / 'all_candidates.csv'}", flush=True)
    print(f"[INFO] saved: {out_dir / 'aligned_summary.csv'}", flush=True)
    print(f"[INFO] saved: {out_dir / 'unmatched.csv'}", flush=True)
    print(f"[INFO] saved: {out_dir / 'direct_aligned_dataset.jsonl'}", flush=True)
    print(f"[INFO] saved: {out_dir / 'direct_aligned_dataset.json'}", flush=True)
    print("[INFO] done.", flush=True)


if __name__ == "__main__":
    main()
