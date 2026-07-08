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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    return {k: arr[k] for k in arr.files}


def get_y(pack: Dict[str, np.ndarray]) -> np.ndarray:
    for key in ["y_multi_hot", "y", "labels", "targets"]:
        if key in pack:
            return (np.asarray(pack[key]) > 0).astype(np.int8)
    raise KeyError(f"Missing y in keys={list(pack)}")


def family_of_precursor(label: str) -> str:
    s = str(label)
    if re.search(r"CO3", s):
        return "carbonate"
    if re.search(r"NO3", s, re.I):
        return "nitrate"
    if re.search(r"OH", s):
        return "hydroxide"
    if re.search(r"CH3COO|C2H3O2|Ac", s):
        return "acetate"
    if re.search(r"Cl|Br|I|F", s):
        return "halide"
    if re.search(r"SO4", s):
        return "sulfate"
    if re.search(r"PO4|H3PO4", s):
        return "phosphate"
    elems = set(ELEMENT_RE.findall(s))
    if len(elems - {"H", "O", "C", "N"}) == 1 and "O" in elems:
        return "oxide"
    if len(elems) == 1:
        return "elemental"
    return "other"


def y_to_sets(y: np.ndarray, names: Sequence[str]) -> List[Set[str]]:
    return [{str(names[j]) for j in np.where(y[i] > 0)[0]} for i in range(y.shape[0])]


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze Stage2 OOV precursor labels and exact-set train coverage.")
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--candidate_csv", default="")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [str(x) for x in load_json(dataset_dir / "precursor_names.json")]

    y_train = get_y(load_npz(dataset_dir / "train.npz"))
    y_eval = get_y(load_npz(dataset_dir / f"{args.split}.npz"))
    meta = pd.read_csv(dataset_dir / f"{args.split}_meta.csv")
    train_sets = y_to_sets(y_train, names)
    eval_sets = y_to_sets(y_eval, names)
    train_label_counts = y_train.sum(axis=0).astype(int)
    train_labels = {names[i] for i, c in enumerate(train_label_counts) if c > 0}
    train_exact_sets = {frozenset(s) for s in train_sets if s}

    top1_by_sample: Dict[int, bool] = {}
    top10_by_sample: Dict[int, bool] = {}
    if args.candidate_csv:
        cand = pd.read_csv(args.candidate_csv)
        for idx, group in cand.groupby("sample_index"):
            top1_by_sample[int(idx)] = bool(group[group["rank"] <= 1]["exact"].any())
            top10_by_sample[int(idx)] = bool(group[group["rank"] <= 10]["exact"].any())

    rows = []
    oov_label_counter: Counter = Counter()
    oov_method_counter: Counter = Counter()
    oov_family_counter: Counter = Counter()
    for i, true_set in enumerate(eval_sets):
        oov = sorted(true_set - train_labels)
        for lab in oov:
            oov_label_counter[lab] += 1
            oov_method_counter[str(meta.loc[i, "reaction_method"])] += 1
            oov_family_counter[family_of_precursor(lab)] += 1
        rows.append({
            "sample_index": i,
            "id": meta.loc[i, "id"],
            "formula": meta.loc[i, "formula"],
            "reaction_method": meta.loc[i, "reaction_method"],
            "true_precursors": json.dumps(sorted(true_set), ensure_ascii=False),
            "n_true": len(true_set),
            "oov_precursors": json.dumps(oov, ensure_ascii=False),
            "n_oov": len(oov),
            "has_oov": bool(oov),
            "exact_set_seen_in_train": frozenset(true_set) in train_exact_sets,
            "top1_exact": top1_by_sample.get(i),
            "top10_exact": top10_by_sample.get(i),
        })
    df = pd.DataFrame(rows)
    summary = {
        "config": vars(args),
        "data": {
            "n_train": int(y_train.shape[0]),
            "n_eval": int(y_eval.shape[0]),
            "n_labels": int(len(names)),
            "n_train_positive_labels": int(len(train_labels)),
        },
        "metrics": {
            "oov_true_label_occurrence_ratio": float(sum(len(set(s) - train_labels) for s in eval_sets) / max(sum(len(s) for s in eval_sets), 1)),
            "rows_with_oov_ratio": float(df["has_oov"].mean()),
            "exact_true_set_seen_in_train_ratio": float(df["exact_set_seen_in_train"].mean()),
            "top1_exact_has_oov": None if not top1_by_sample else float(df[df["has_oov"]]["top1_exact"].mean()),
            "top1_exact_no_oov": None if not top1_by_sample else float(df[~df["has_oov"]]["top1_exact"].mean()),
            "top10_exact_has_oov": None if not top10_by_sample else float(df[df["has_oov"]]["top10_exact"].mean()),
            "top10_exact_no_oov": None if not top10_by_sample else float(df[~df["has_oov"]]["top10_exact"].mean()),
        },
        "oov_by_reaction_method": dict(oov_method_counter.most_common()),
        "oov_by_family": dict(oov_family_counter.most_common()),
        "top_oov_labels": dict(oov_label_counter.most_common(50)),
        "artifacts": {
            "row_csv": str((out_dir / f"{args.split}_oov_rows.csv").resolve()),
            "summary_json": str((out_dir / f"{args.split}_oov_summary.json").resolve()),
        },
    }
    df.to_csv(out_dir / f"{args.split}_oov_rows.csv", index=False)
    write_json(out_dir / f"{args.split}_oov_summary.json", summary)
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
