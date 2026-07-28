#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd

try:
    from pymatgen.core import Composition
except Exception:  # pragma: no cover
    Composition = None  # type: ignore


ELEMENT_RE = re.compile(r"([A-Z][a-z]?)")
NON_TARGET = {"H", "O", "C", "N"}
HALIDES = {"F", "Cl", "Br", "I"}
COMMON_OX = {
    "Li": [1], "Na": [1], "K": [1], "Rb": [1], "Cs": [1],
    "Mg": [2], "Ca": [2], "Sr": [2], "Ba": [2], "Zn": [2], "Cd": [2],
    "Al": [3], "Ga": [3], "In": [3], "Sc": [3], "Y": [3],
    "Ti": [4, 3], "Zr": [4], "Hf": [4], "V": [5, 4, 3], "Nb": [5, 4, 3], "Ta": [5, 4, 3],
    "Cr": [6, 3, 4], "Mo": [6, 4, 3], "W": [6, 4, 3],
    "Mn": [2, 3, 4], "Fe": [2, 3], "Co": [2, 3], "Ni": [2, 3], "Cu": [1, 2],
    "Bi": [3, 5], "Sb": [3, 5], "Sn": [2, 4], "Pb": [2, 4],
}
RARE_EARTH = {"La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"}


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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def elements(text: str) -> Set[str]:
    return set(ELEMENT_RE.findall(str(text)))


def clean_formula(label: str) -> str:
    s = str(label).split("·", 1)[0]
    s = s.replace("_", "")
    s = re.sub(r"^[αβγδT]-", "", s)
    s = re.sub(r"^\\d+(?=[A-Z])", "", s)
    s = re.sub(r"\\([0-9.]+\\s*[mM]?\\)$", "", s)
    s = re.sub(r"(?i)(product|charge|Nps|NPs)$", "", s)
    s = s.replace("[", "(").replace("]", ")")
    # Keep first component for parser diagnostics only.
    for sep in ["+", "/", ":"]:
        if sep in s:
            s = s.split(sep, 1)[0]
    if "-" in s and re.search(r"(?i)H2O|HCl|C[0-9]?H[0-9]+OH|CH3OH|dmf", s.split("-", 1)[1]):
        s = s.split("-", 1)[0]
    return s


def formula_key(label: str) -> tuple[str, str, str]:
    if Composition is None:
        return "", "failed", "pymatgen unavailable"
    f = clean_formula(label)
    try:
        return Composition(f).alphabetical_formula.replace(" ", ""), "ok", ""
    except Exception as exc:
        return "", "failed", str(exc)


def flags(label: str) -> Dict[str, bool]:
    s = str(label)
    elems = elements(s)
    return {
        "contains_carbonate": bool(re.search(r"CO3", s)),
        "contains_nitrate": bool(re.search(r"NO3", s, re.I)),
        "contains_hydroxide": bool(re.search(r"OH", s)),
        "contains_acetate": bool(re.search(r"CH3COO|CH3CO2|C2H3O2|\\(Ac\\)2|OAc", s)),
        "contains_sulfate": bool(re.search(r"SO4", s)),
        "contains_phosphate": bool(re.search(r"PO4|H3PO4|H2PO4|HPO4", s)),
        "contains_halide": bool(elems & HALIDES),
        "contains_hydrate": "·" in s or bool(re.search(r"H2O|nH2O", s, re.I)),
        "contains_organic": bool(re.search(r"CH3|C2H5|C3H7|C4H9|phen|Ph|OAc|Ac|COO|CN|dmf|Bu|Pr|Et|Me", s)),
    }


def family(label: str, elems: Set[str], fl: Dict[str, bool]) -> str:
    if len(elems) == 1 and re.fullmatch(r"[A-Z][a-z]?", str(label)):
        return "elemental"
    for fam, key in [
        ("carbonate", "contains_carbonate"), ("nitrate", "contains_nitrate"), ("hydroxide", "contains_hydroxide"),
        ("acetate", "contains_acetate"), ("sulfate", "contains_sulfate"), ("phosphate", "contains_phosphate"),
    ]:
        if fl[key]:
            return fam
    if fl["contains_halide"] and not any(fl[k] for k in ["contains_carbonate", "contains_nitrate", "contains_sulfate", "contains_phosphate"]):
        return "halide"
    if "O" in elems and not fl["contains_organic"]:
        return "oxide"
    if fl["contains_organic"]:
        return "organic"
    return "other_salt" if elems else "unknown"


def likely_oxidation_states(src_elems: Set[str]) -> Dict[str, List[int]]:
    out = {}
    for e in sorted(src_elems):
        if e in COMMON_OX:
            out[e] = COMMON_OX[e]
        elif e in RARE_EARTH:
            out[e] = [3]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build precursor ontology v4 with oxidation-state diagnostics.")
    ap.add_argument("--precursor_names", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [str(x) for x in load_json(Path(args.precursor_names))]
    rows = []
    for lab in names:
        key, status, err = formula_key(lab)
        elems = elements(lab)
        fl = flags(lab)
        fam = family(lab, elems, fl)
        src = sorted((elems - NON_TARGET) or elems)
        ox = likely_oxidation_states(set(src))
        rows.append({
            "canonical_precursor": lab,
            "canonical_formula": clean_formula(lab),
            "normalized_formula_key": key,
            "elements": json.dumps(sorted(elems), ensure_ascii=False),
            "target_source_elements": json.dumps(src, ensure_ascii=False),
            "precursor_family": fam,
            "anion_family": fam if fam in {"oxide", "carbonate", "nitrate", "hydroxide", "acetate", "sulfate", "phosphate", "halide"} else "none",
            "likely_oxidation_states": json.dumps(ox, ensure_ascii=False),
            "oxidation_state_parse_status": "ok" if ox else "unknown",
            "charge_balance_status": "not_checked",
            "contains_hydrate": fl["contains_hydrate"],
            "is_open_vocab": False,
            "parse_status": status,
            "parse_error": err,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "precursor_ontology.csv", index=False)
    write_json(out_dir / "precursor_ontology.json", df.to_dict(orient="records"))
    failed = df[df["parse_status"] != "ok"]
    failed.to_csv(out_dir / "remaining_failed_parse_v4.csv", index=False)
    report = ["# Precursor Ontology v4", "", f"- n_precursors: {len(df)}", f"- parse_failed: {len(failed)}", "", "## Family Distribution"]
    for fam, n in df["precursor_family"].value_counts().items():
        report.append(f"- {fam}: {n}")
    (out_dir / "precursor_ontology_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    summary = {
        "n_precursors": int(len(df)),
        "parse_failed": int(len(failed)),
        "family_distribution": df["precursor_family"].value_counts().to_dict(),
        "oxidation_state_parse_ok": int((df["oxidation_state_parse_status"] == "ok").sum()),
        "artifacts": {
            "csv": str((out_dir / "precursor_ontology.csv").resolve()),
            "remaining_failed_parse": str((out_dir / "remaining_failed_parse_v4.csv").resolve()),
        },
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
