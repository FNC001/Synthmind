#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set

import numpy as np
import pandas as pd


ELEMENT_RE = re.compile(r"([A-Z][a-z]?)")
NON_TARGET = {"O"}
IGNORED_PRECURSOR_ELEMENTS = {"H", "O", "C", "N"}
FLUX_LIKE = {"LiCl", "NaCl", "KCl", "RbCl", "CsCl", "LiF", "NaF", "KF", "KBr", "KI", "NaI", "SrCl2", "BaCl2"}
SOLVENT_LIKE_PATTERNS = re.compile(r"(?i)H2O|NH3|DMF|DMSO|EtOH|CH3OH|C2H5OH|ethylene|citric|urea")


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


def elements(text: str) -> Set[str]:
    return set(ELEMENT_RE.findall(str(text)))


def target_elements(formula: str) -> List[str]:
    elems = sorted(elements(formula) - NON_TARGET)
    return elems or sorted(elements(formula))


def true_sets(y: np.ndarray, names: Sequence[str]) -> List[List[str]]:
    out = []
    for i in range(y.shape[0]):
        out.append([str(names[j]) for j in np.where(y[i] > 0)[0]])
    return out


def source_elements(labels: Sequence[str]) -> List[str]:
    elems: Set[str] = set()
    for lab in labels:
        elems |= elements(lab) - IGNORED_PRECURSOR_ELEMENTS
    return sorted(elems)


def template_key(method: str, target: Sequence[str], families: Sequence[str], labels: Sequence[str]) -> str:
    return "|".join([
        str(method),
        ",".join(sorted(target)),
        ",".join(sorted(families)),
        json.dumps(sorted(labels), ensure_ascii=False, separators=(",", ":")),
    ])


def family_set(labels: Sequence[str], family_lookup: Dict[str, str]) -> List[str]:
    return sorted({family_lookup.get(x, "unknown") for x in labels})


def optional_flags(labels: Sequence[str]) -> Dict[str, Any]:
    flux = [x for x in labels if x in FLUX_LIKE or (set(elements(x)) & {"Li", "Na", "K", "Rb", "Cs"} and set(elements(x)) & {"F", "Cl", "Br", "I"})]
    solvent = [x for x in labels if SOLVENT_LIKE_PATTERNS.search(x)]
    return {
        "optional_flux_labels": sorted(flux),
        "optional_solvent_like_labels": sorted(solvent),
        "has_optional_flux": bool(flux),
        "has_solvent_like": bool(solvent),
    }


def split_templates(dataset_dir: Path, split: str, names: Sequence[str], family_lookup: Dict[str, str]) -> Counter:
    y = get_y(load_npz(dataset_dir / f"{split}.npz"))
    meta = pd.read_csv(dataset_dir / f"{split}_meta.csv")
    sets = true_sets(y, names)
    counts: Counter = Counter()
    for i, labels in enumerate(sets):
        method = str(meta.loc[i, "reaction_method"])
        target = target_elements(str(meta.loc[i, "formula"]))
        families = family_set(labels, family_lookup)
        key = template_key(method, target, families, labels)
        counts[key] += 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Build no-leakage method-specific Stage2 precursor template library from train split.")
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--ontology_csv", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [str(x) for x in load_json(dataset_dir / "precursor_names.json")]
    ont = pd.read_csv(args.ontology_csv)
    family_lookup = dict(zip(ont["canonical_precursor"].astype(str), ont["precursor_family"].astype(str)))
    train_y = get_y(load_npz(dataset_dir / "train.npz"))
    train_meta = pd.read_csv(dataset_dir / "train_meta.csv")
    train_sets = true_sets(train_y, names)
    val_counts = split_templates(dataset_dir, "val", names, family_lookup)

    agg: Dict[str, Dict[str, Any]] = {}
    for i, labels in enumerate(train_sets):
        method = str(train_meta.loc[i, "reaction_method"])
        formula = str(train_meta.loc[i, "formula"])
        target = target_elements(formula)
        families = family_set(labels, family_lookup)
        key = template_key(method, target, families, labels)
        if key not in agg:
            agg[key] = {
                "template_id": f"tpl_{len(agg):07d}",
                "reaction_method": method,
                "target_elements": sorted(target),
                "target_element_group": ",".join(sorted(target)),
                "precursor_families": families,
                "precursor_family_set": ",".join(families),
                "precursor_labels": sorted(labels),
                "source_elements": source_elements(labels),
                "train_frequency": 0,
                "val_frequency": int(val_counts.get(key, 0)),
                "coverage_count": 0,
                "set_size_sum": 0,
                **optional_flags(labels),
            }
        rec = agg[key]
        rec["train_frequency"] += 1
        rec["coverage_count"] += len(set(target) & set(rec["source_elements"]))
        rec["set_size_sum"] += len(labels)

    rows = []
    method_totals = Counter(train_meta["reaction_method"].astype(str))
    for rec in agg.values():
        n = max(int(rec["train_frequency"]), 1)
        avg_set_size = rec["set_size_sum"] / n
        avg_coverage = rec["coverage_count"] / n
        method_prior = rec["train_frequency"] / max(method_totals[rec["reaction_method"]], 1)
        cooccurrence_score = float(np.log1p(rec["train_frequency"]) + 0.5 * np.log1p(rec["val_frequency"]) + method_prior)
        row = {
            "template_id": rec["template_id"],
            "reaction_method": rec["reaction_method"],
            "target_elements": json.dumps(rec["target_elements"], ensure_ascii=False),
            "target_element_group": rec["target_element_group"],
            "precursor_families": json.dumps(rec["precursor_families"], ensure_ascii=False),
            "precursor_family_set": rec["precursor_family_set"],
            "precursor_labels": json.dumps(rec["precursor_labels"], ensure_ascii=False),
            "source_elements": json.dumps(rec["source_elements"], ensure_ascii=False),
            "optional_flux_labels": json.dumps(rec["optional_flux_labels"], ensure_ascii=False),
            "optional_solvent_like_labels": json.dumps(rec["optional_solvent_like_labels"], ensure_ascii=False),
            "has_optional_flux": rec["has_optional_flux"],
            "has_solvent_like": rec["has_solvent_like"],
            "train_frequency": int(rec["train_frequency"]),
            "val_frequency": int(rec["val_frequency"]),
            "coverage_count": float(avg_coverage),
            "average_set_size": float(avg_set_size),
            "cooccurrence_score": cooccurrence_score,
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["reaction_method", "train_frequency", "val_frequency"], ascending=[True, False, False])
    df.to_csv(out_dir / "method_precursor_templates.csv", index=False)
    write_json(out_dir / "method_precursor_templates.json", df.to_dict(orient="records"))
    summary = {
        "n_templates": int(len(df)),
        "n_train_rows": int(len(train_meta)),
        "by_method": df["reaction_method"].value_counts().to_dict(),
        "top_templates": df.head(25).to_dict(orient="records"),
        "artifacts": {
            "csv": str(out_dir / "method_precursor_templates.csv"),
            "json": str(out_dir / "method_precursor_templates.json"),
            "summary": str(out_dir / "summary.json"),
        },
    }
    write_json(out_dir / "summary.json", summary)
    report = ["# Stage2 v5 Method Precursor Templates", "", f"- n_templates: {len(df)}", "## By Method"]
    for method, count in df["reaction_method"].value_counts().items():
        report.append(f"- {method}: {count}")
    (out_dir / "method_precursor_templates_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
