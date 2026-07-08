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
NON_SOURCE = {"H", "O", "C", "N"}
ATMOSPHERE_FAMILY = "supplied_by_precursors_or_atmosphere"


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


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    return {k: arr[k] for k in arr.files}


def get_y(pack: Dict[str, np.ndarray]) -> np.ndarray:
    for k in ["y_multi_hot", "y", "labels", "targets"]:
        if k in pack:
            return (np.asarray(pack[k]) > 0).astype(np.int8)
    raise KeyError(f"Missing y in keys={list(pack)}")


def get_x(pack: Dict[str, np.ndarray]) -> np.ndarray:
    for k in ["x", "features", "X"]:
        if k in pack:
            return np.asarray(pack[k], dtype=np.float32)
    raise KeyError(f"Missing x in keys={list(pack)}")


def element_set(text: str) -> Set[str]:
    return set(ELEMENT_RE.findall(str(text)))


def parse_json_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    s = str(value or "").strip()
    if not s:
        return []
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(x) for x in obj]
    except Exception:
        pass
    return [x.strip() for x in s.split(",") if x.strip()]


def build_rows_for_split(
    split: str,
    dataset_dir: Path,
    output_dir: Path,
    ontology: pd.DataFrame,
    precursor_names: Sequence[str],
    family_vocab: Sequence[str],
    element_vocab: Sequence[str],
) -> Dict[str, Any]:
    pack = load_npz(dataset_dir / f"{split}.npz")
    meta = pd.read_csv(dataset_dir / f"{split}_meta.csv")
    y = get_y(pack)
    x = get_x(pack)
    ont = ontology.set_index("canonical_precursor").to_dict(orient="index")
    rows = []
    family_counter: Counter = Counter()
    sample_cover = []
    for i, row in meta.iterrows():
        formula = str(row["formula"])
        target_elements = sorted(element_set(formula) - {"O"})
        if not target_elements:
            target_elements = sorted(element_set(formula))
        active = np.where(y[i] > 0)[0].tolist()
        precs = [str(precursor_names[j]) for j in active]
        elem_to_families: Dict[str, Set[str]] = {e: set() for e in target_elements}
        elem_to_precursors: Dict[str, Set[str]] = {e: set() for e in target_elements}
        for p in precs:
            rec = ont.get(p, {})
            fam = str(rec.get("precursor_family", "unknown"))
            src_elems = set(parse_json_list(rec.get("target_source_elements", "[]")))
            if not src_elems:
                src_elems = element_set(p) - NON_SOURCE
            for elem in target_elements:
                if elem in src_elems:
                    elem_to_families[elem].add(fam)
                    elem_to_precursors[elem].add(p)
        covered = 0
        for elem in target_elements:
            fams = sorted(elem_to_families.get(elem, set()))
            if not fams and elem in {"O", "H", "C", "N"}:
                fams = [ATMOSPHERE_FAMILY]
            if fams:
                covered += 1
            for fam in fams:
                family_counter[fam] += 1
            rows.append({
                "sample_index": i,
                "id": row["id"],
                "material_id": row.get("material_id", ""),
                "formula": formula,
                "reaction_method": row.get("reaction_method", "other"),
                "source_dataset": row.get("source_dataset", ""),
                "target_elements": json.dumps(target_elements, ensure_ascii=False),
                "target_element": elem,
                "true_precursors": json.dumps(precs, ensure_ascii=False),
                "element_family_labels": json.dumps(fams, ensure_ascii=False),
                "element_source_precursors": json.dumps(sorted(elem_to_precursors.get(elem, set())), ensure_ascii=False),
                "has_family_label": bool(fams),
                "x_row_index": i,
            })
        sample_cover.append(covered / len(target_elements) if target_elements else 1.0)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / f"{split}_family_labels.csv", index=False)
    np.savez_compressed(output_dir / f"{split}_features.npz", x=x)
    return {
        "split": split,
        "n_samples": int(len(meta)),
        "n_element_rows": int(len(df)),
        "mean_target_elements": float(len(df) / max(len(meta), 1)),
        "mean_sample_family_coverage": float(np.mean(sample_cover)) if sample_cover else 0.0,
        "family_counts": dict(family_counter.most_common()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Stage2 element-family supervision dataset.")
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--ontology_csv", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ontology = pd.read_csv(args.ontology_csv)
    precursor_names = [str(x) for x in load_json(dataset_dir / "precursor_names.json")]

    families = sorted(set(ontology["precursor_family"].astype(str)) | {ATMOSPHERE_FAMILY, "unknown"})
    elements = sorted(set().union(*(element_set(x) for x in ontology["canonical_precursor"].astype(str))))
    write_json(output_dir / "family_vocab.json", families)
    write_json(output_dir / "element_vocab.json", elements)

    split_summaries = {}
    for split in ["train", "val", "test"]:
        split_summaries[split] = build_rows_for_split(
            split=split,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            ontology=ontology,
            precursor_names=precursor_names,
            family_vocab=families,
            element_vocab=elements,
        )
    summary = {
        "config": vars(args),
        "n_families": int(len(families)),
        "family_vocab": families,
        "n_elements": int(len(elements)),
        "splits": split_summaries,
        "artifacts": {
            "train": str((output_dir / "train_family_labels.csv").resolve()),
            "val": str((output_dir / "val_family_labels.csv").resolve()),
            "test": str((output_dir / "test_family_labels.csv").resolve()),
            "family_vocab": str((output_dir / "family_vocab.json").resolve()),
            "element_vocab": str((output_dir / "element_vocab.json").resolve()),
        },
    }
    write_json(output_dir / "family_dataset_summary.json", summary)
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
