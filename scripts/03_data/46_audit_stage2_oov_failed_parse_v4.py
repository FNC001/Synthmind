#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd

try:
    from pymatgen.core import Composition
except Exception:  # pragma: no cover
    Composition = None  # type: ignore


SUBSCRIPT = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
DOT_CHARS = "▪•∙⋅･．*"

NAME_ALIASES = {
    "alumina": "Al2O3",
    "magnesia": "MgO",
    "silica": "SiO2",
    "titania": "TiO2",
    "zirconia": "ZrO2",
    "lime": "CaO",
    "hematite": "Fe2O3",
    "magnetite": "Fe3O4",
    "manganese dioxide": "MnO2",
    "boric acid": "H3BO3",
    "ammonium metavanadate": "NH4VO3",
    "iron(iii) oxide": "Fe2O3",
    "iron(ii) oxide": "FeO",
    "manganese(iv) oxide": "MnO2",
    "cobalt(ii) nitrate": "Co(NO3)2",
    "nickel(ii) acetate": "Ni(CH3COO)2",
    "copper(ii) nitrate": "Cu(NO3)2",
    "aluminum nitrate nonahydrate": "Al(NO3)3·9H2O",
    "lithium carbonate": "Li2CO3",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    return {k: arr[k] for k in arr.files}


def get_y(pack: Dict[str, np.ndarray]) -> np.ndarray:
    for k in ["y_multi_hot", "y", "labels", "targets"]:
        if k in pack:
            return (np.asarray(pack[k]) > 0).astype(np.int8)
    raise KeyError(f"Missing y in keys={list(pack)}")


def parse_list(value: Any) -> List[str]:
    try:
        obj = json.loads(str(value))
        if isinstance(obj, list):
            return [str(x) for x in obj]
    except Exception:
        pass
    return []


def formula_key(s: str) -> str:
    if Composition is None:
        return ""
    try:
        return Composition(str(s)).alphabetical_formula.replace(" ", "")
    except Exception:
        return ""


def clean_label(label: str) -> tuple[str, str]:
    original = str(label or "").strip()
    s = unicodedata.normalize("NFKC", original).translate(SUBSCRIPT).strip()
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.replace("（", "(").replace("）", ")").replace("_", "")
    for ch in DOT_CHARS:
        s = s.replace(ch, "·")
    phrase = re.sub(r"[^a-z0-9()+]+", " ", s.lower()).strip()
    if phrase in NAME_ALIASES:
        return NAME_ALIASES[phrase], "common_alias"
    s = re.sub(r"(?i)H20", "H2O", s)
    s = re.sub(r"(?i)(mono)?hydrate$", "·H2O", s)
    for word, n in {
        "dihydrate": 2,
        "trihydrate": 3,
        "tetrahydrate": 4,
        "pentahydrate": 5,
        "hexahydrate": 6,
        "heptahydrate": 7,
        "octahydrate": 8,
        "nonahydrate": 9,
        "decahydrate": 10,
    }.items():
        s = re.sub(fr"(?i){word}$", f"·{n}H2O", s)
    s = re.sub(r"(?i)[\\.-]([0-9]+)H2O$", r"·\1H2O", s)
    s = re.sub(r"(?i)[\\.-]H2O$", "·H2O", s)
    s = re.sub(r"(?i)·1H2O$", "·H2O", s)
    s = re.sub(r"(?i)·nH2O$", "·H2O", s)
    s = re.sub(r"(?i)(?:\\((?:s|l|g|aq)\\)|solid|powder|aqueous|solution|anhydrous|amorphous|commercial)$", "", s)
    s = re.sub(r"\\s+", "", s)
    s = re.sub(r";+", ";", s).strip(";")
    reason = "text_normalize" if s != original else "unchanged"
    return s, reason


def candidate_patches(label: str) -> List[tuple[str, str, float]]:
    s, reason = clean_label(label)
    cands = [(s, reason, 0.80 if s != label else 0.0)]
    # Hydrate/solvate forms often need to merge to the base precursor when the
    # exact hydrate was never present in train.
    if "·" in s:
        cands.append((s.split("·", 1)[0], "hydrate_to_base", 0.88))
    # Solvent adduct suffix; keep the left reagent only for recognizable
    # solvent/mineral-acid adducts. Generic mixtures are left for review.
    if "-" in s:
        left, right = s.split("-", 1)
        if re.search(r"(?i)H2O|HCl|C[0-9]?H[0-9]+OH|CH3OH|C4H9OH|dmf", right):
            cands.append((left, "solvent_adduct_suffix_remove", 0.86))
    # Greek/polytype prefixes and isotope enrichment prefixes.
    cands.append((re.sub(r"^[αβγδ]-", "", s), "greek_phase_prefix_remove", 0.86))
    cands.append((re.sub(r"^[Tαβγδ]-", "", s), "phase_prefix_remove", 0.78))
    cands.append((re.sub(r"^\\d+(?=[A-Z])", "", s), "isotope_prefix_remove", 0.92))
    # Concentration or numeric annotation.
    cands.append((re.sub(r"\\([0-9.]+\\s*[mM]?\\)$", "", s), "concentration_suffix_remove", 0.82))
    cands.append((re.sub(r"product$", "", s, flags=re.I), "product_suffix_remove", 0.78))
    # Common shorthand organic ligands.
    cands.append((s.replace("iPr", "OCH(CH3)2").replace("OiPr", "OCH(CH3)2"), "ipr_expand", 0.72))
    # Broken hydrate missing oxygen: Ca(NO3)2.4H2 -> Ca(NO3)2·4H2O
    cands.append((re.sub(r"(?i)[\\.]([0-9]+)H2$", r"·\1H2O", s), "broken_hydrate_h2_to_h2o", 0.90))
    out = []
    seen = set()
    for cand, why, conf in cands:
        cand = cand.strip()
        if cand and cand not in seen:
            seen.add(cand)
            out.append((cand, why, conf))
    return out


def action_for(raw: str, suggested: str, train_labels: Set[str], failed: bool, parse_ok: bool, reason: str) -> tuple[str, float, str]:
    if suggested != raw and suggested in train_labels:
        if "hydrate" in reason or "H2O" in suggested:
            return "hydrate_normalize", 0.95, f"normalized to existing train label via {reason}"
        if reason == "common_alias":
            return "merge_alias", 0.95, "common-name alias maps to existing train label"
        return "merge_alias", 0.90, f"text cleanup maps to existing train label via {reason}"
    if suggested != raw and parse_ok:
        return "fix_formula_parse", 0.80 if failed else 0.70, f"cleaned formula parses after {reason}"
    if re.search(r"(?i)\\((?:s|l|g|aq)\\)|powder|solution|aqueous|anhydrous|amorphous|commercial", raw):
        return "phase_suffix_remove", 0.65, "contains removable descriptive suffix but no train-label match"
    if re.search(r"(?i)\\(ii\\)|\\(iii\\)|\\(iv\\)|\\(v\\)|\\(vi\\)", raw):
        return "oxidation_state_normalize", 0.55, "contains oxidation-state text; needs manual chemistry check"
    if failed:
        return "manual_review", 0.40, "parse failed and no high-confidence automatic patch"
    return "keep_true_oov", 0.50, "appears to be a valid unseen precursor"


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit Stage2 v4 OOV and failed-parse precursor labels.")
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--ontology_csv", required=True)
    ap.add_argument("--oov_rows", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [str(x) for x in load_json(dataset_dir / "precursor_names.json")]
    counts = {}
    for split in ["train", "val", "test"]:
        y = get_y(load_npz(dataset_dir / f"{split}.npz"))
        c = y.sum(axis=0).astype(int)
        counts[split] = {names[i]: int(v) for i, v in enumerate(c)}
    train_labels = {k for k, v in counts["train"].items() if v > 0}
    ont = pd.read_csv(args.ontology_csv)
    ont_map = ont.set_index("canonical_precursor").to_dict(orient="index")
    failed_labels = set(ont[ont["parse_status"] != "ok"]["canonical_precursor"].astype(str))
    oov_df = pd.read_csv(args.oov_rows)
    oov_counter: Counter = Counter()
    for labs in oov_df["oov_precursors"]:
        oov_counter.update(parse_list(labs))
    audit_labels = sorted(set(oov_counter) | failed_labels)
    rows = []
    for raw in audit_labels:
        candidates = candidate_patches(raw)
        suggested, clean_reason, base_conf = candidates[0]
        for cand, why, conf in candidates:
            if cand in train_labels:
                suggested, clean_reason, base_conf = cand, why, conf
                break
        parse_ok = bool(formula_key(suggested.split("·", 1)[0]))
        failed = raw in failed_labels
        action, conf, reason = action_for(raw, suggested, train_labels, failed, parse_ok, clean_reason)
        conf = max(conf, base_conf if suggested in train_labels else min(conf, base_conf))
        rec = ont_map.get(raw, {})
        rows.append({
            "raw_label": raw,
            "canonical_v2_label": raw,
            "formula_guess": suggested,
            "family_v3": rec.get("precursor_family", "unknown"),
            "parse_status_v3": rec.get("parse_status", "missing"),
            "train_count": counts["train"].get(raw, 0),
            "val_count": counts["val"].get(raw, 0),
            "test_count": counts["test"].get(raw, 0),
            "oov_count": int(oov_counter.get(raw, 0)),
            "appears_in_failed_parse": bool(failed),
            "suggested_action": action,
            "suggested_canonical_label": suggested,
            "confidence": conf,
            "reason": reason,
        })
    df = pd.DataFrame(rows).sort_values(["oov_count", "appears_in_failed_parse", "confidence"], ascending=[False, False, False])
    df.to_csv(out_dir / "oov_label_audit.csv", index=False)
    summary = {
        "n_audit_labels": int(len(df)),
        "n_oov_labels": int(len(oov_counter)),
        "n_failed_parse_labels": int(len(failed_labels)),
        "action_counts": df["suggested_action"].value_counts().to_dict(),
        "artifacts": {"audit_csv": str((out_dir / "oov_label_audit.csv").resolve())},
    }
    (out_dir / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
