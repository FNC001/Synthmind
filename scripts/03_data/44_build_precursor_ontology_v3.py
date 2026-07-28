#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd

try:
    from pymatgen.core import Composition
except Exception:  # pragma: no cover
    Composition = None  # type: ignore


ELEMENT_RE = re.compile(r"([A-Z][a-z]?)")
HALIDES = {"F", "Cl", "Br", "I"}
NON_TARGET_SOURCE = {"H", "O", "C", "N"}
SOLVENT_LIKE = {"H2O", "C2H5OH", "CH3OH", "(CH3)2CHOH", "NH3", "NH4OH"}


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(x) for x in obj]
    if isinstance(obj, set):
        return sorted(str(x) for x in obj)
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


def split_hydrate(label: str) -> tuple[str, List[str]]:
    parts = [p.strip() for p in str(label).split("·") if p.strip()]
    if not parts:
        return str(label), []
    return parts[0], parts[1:]


def formula_key(formula: str) -> tuple[str, str, str]:
    if Composition is None:
        return "", "failed", "pymatgen unavailable"
    try:
        comp = Composition(str(formula))
        return comp.alphabetical_formula.replace(" ", ""), "ok", ""
    except Exception as exc:
        return "", "failed", str(exc)


def elements_from_text(text: str) -> Set[str]:
    return set(ELEMENT_RE.findall(str(text)))


def flag_patterns(label: str) -> Dict[str, bool]:
    s = str(label)
    return {
        "contains_carbonate": bool(re.search(r"CO3", s)),
        "contains_nitrate": bool(re.search(r"NO3", s, re.I)),
        "contains_hydroxide": bool(re.search(r"OH", s)),
        "contains_acetate": bool(re.search(r"CH3COO|CH3CO2|C2H3O2|\\(Ac\\)2|Ac2|OAc", s)),
        "contains_sulfate": bool(re.search(r"SO4", s)),
        "contains_phosphate": bool(re.search(r"PO4|H3PO4|H2PO4|HPO4", s)),
        "contains_halide": bool(elements_from_text(s) & HALIDES),
        "contains_hydrate": "·" in s or bool(re.search(r"H2O|nH2O", s, re.I)),
        "contains_organic": bool(re.search(r"CH3|C2H5|C3H7|C4H9|phen|Ph|OAc|Ac|COO|CN|dmf|Bu|Pr|Et|Me", s)),
    }


def is_elemental(label: str, elements: Set[str], flags: Dict[str, bool]) -> bool:
    s = str(label).strip()
    if len(elements) != 1:
        return False
    if flags["contains_hydrate"] or flags["contains_organic"]:
        return False
    if any(ch.isdigit() for ch in s):
        return False
    return bool(re.fullmatch(r"[A-Z][a-z]?", s))


def family(label: str, elements: Set[str], flags: Dict[str, bool], elemental: bool) -> str:
    if elemental:
        return "elemental"
    if flags["contains_carbonate"]:
        return "carbonate"
    if flags["contains_nitrate"]:
        return "nitrate"
    if flags["contains_hydroxide"]:
        return "hydroxide"
    if flags["contains_acetate"]:
        return "acetate"
    if flags["contains_sulfate"]:
        return "sulfate"
    if flags["contains_phosphate"]:
        return "phosphate"
    if flags["contains_halide"] and not (flags["contains_carbonate"] or flags["contains_nitrate"] or flags["contains_sulfate"] or flags["contains_phosphate"]):
        return "halide"
    if "O" in elements and not flags["contains_organic"]:
        return "oxide"
    if flags["contains_organic"]:
        return "organic"
    if elements:
        return "other_salt"
    return "unknown"


def anion_family(label: str, precursor_family: str) -> str:
    if precursor_family in {"carbonate", "nitrate", "hydroxide", "acetate", "sulfate", "phosphate", "halide", "oxide"}:
        return precursor_family
    return "none"


def is_flux_or_salt(label: str, elements: Set[str], fam: str) -> bool:
    alkali = {"Li", "Na", "K", "Rb", "Cs"}
    alkaline = {"Mg", "Ca", "Sr", "Ba"}
    return fam in {"halide", "carbonate", "sulfate", "phosphate"} and bool(elements & (alkali | alkaline))


def parse_precursor(label: str) -> Dict[str, Any]:
    canonical, hydrate_parts = split_hydrate(label)
    key, status, err = formula_key(canonical)
    elems = elements_from_text(label)
    flags = flag_patterns(label)
    elemental = is_elemental(label, elems, flags)
    fam = family(label, elems, flags, elemental)
    target_source = sorted((elems - NON_TARGET_SOURCE) or elems)
    if str(label) in SOLVENT_LIKE:
        target_source = []
    return {
        "canonical_precursor": str(label),
        "canonical_formula": canonical,
        "normalized_formula_key": key,
        "elements": json.dumps(sorted(elems), ensure_ascii=False),
        "target_source_elements": json.dumps(target_source, ensure_ascii=False),
        "n_elements": int(len(elems)),
        "precursor_family": fam,
        "anion_family": anion_family(label, fam),
        "contains_oxide": bool(fam == "oxide"),
        "contains_carbonate": flags["contains_carbonate"],
        "contains_nitrate": flags["contains_nitrate"],
        "contains_hydroxide": flags["contains_hydroxide"],
        "contains_acetate": flags["contains_acetate"],
        "contains_sulfate": flags["contains_sulfate"],
        "contains_phosphate": flags["contains_phosphate"],
        "contains_halide": flags["contains_halide"],
        "contains_hydrate": flags["contains_hydrate"],
        "contains_organic": flags["contains_organic"],
        "is_elemental": elemental,
        "is_flux_or_salt": is_flux_or_salt(label, elems, fam),
        "is_solvent_like": str(label) in SOLVENT_LIKE or fam == "organic" and not target_source,
        "parse_status": status,
        "parse_error": err,
    }


def markdown_report(df: pd.DataFrame, out_dir: Path) -> str:
    fam_counts = df["precursor_family"].value_counts().to_dict()
    failed = df[df["parse_status"] != "ok"]
    lines = ["# Precursor Ontology v3", ""]
    lines.append(f"- n_precursors: {len(df)}")
    lines.append(f"- parse_failed: {len(failed)}")
    lines.append("")
    lines.append("## Family Distribution")
    for fam, n in fam_counts.items():
        lines.append(f"- {fam}: {n}")
    lines.append("")
    lines.append("## Family Examples")
    for fam in sorted(fam_counts):
        examples = df[df["precursor_family"] == fam]["canonical_precursor"].head(12).tolist()
        lines.append(f"- {fam}: {', '.join(examples)}")
    lines.append("")
    lines.append("## Parse Failures")
    if failed.empty:
        lines.append("- none")
    else:
        for x in failed["canonical_precursor"].head(80).tolist():
            lines.append(f"- {x}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build precursor ontology v3 from canonical precursor labels.")
    ap.add_argument("--precursor_names", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    names = [str(x) for x in load_json(Path(args.precursor_names))]
    df = pd.DataFrame([parse_precursor(x) for x in names])
    df.to_csv(output_dir / "precursor_ontology.csv", index=False)
    write_json(output_dir / "precursor_ontology.json", df.to_dict(orient="records"))
    (output_dir / "precursor_ontology_report.md").write_text(markdown_report(df, output_dir), encoding="utf-8")
    summary = {
        "n_precursors": int(len(df)),
        "parse_failed": int((df["parse_status"] != "ok").sum()),
        "family_distribution": df["precursor_family"].value_counts().to_dict(),
        "anion_distribution": df["anion_family"].value_counts().to_dict(),
        "artifacts": {
            "csv": str((output_dir / "precursor_ontology.csv").resolve()),
            "json": str((output_dir / "precursor_ontology.json").resolve()),
            "report": str((output_dir / "precursor_ontology_report.md").resolve()),
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
