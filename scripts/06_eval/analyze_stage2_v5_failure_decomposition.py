#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

import numpy as np
import pandas as pd


WEAK_METHODS = {"hydro_solvothermal", "precipitation", "flux_molten_salt", "other", "solution"}


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


def parse_list(text: Any) -> List[str]:
    try:
        obj = json.loads(str(text))
        if isinstance(obj, list):
            return [str(x) for x in obj]
    except Exception:
        pass
    return []


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    return {k: arr[k] for k in arr.files}


def get_y(pack: Dict[str, np.ndarray]) -> np.ndarray:
    for key in ["y_multi_hot", "y", "labels", "targets"]:
        if key in pack:
            return (np.asarray(pack[key]) > 0).astype(np.int8)
    raise KeyError(f"No label matrix found in npz keys={list(pack)}")


def split_label_counts(dataset_dir: Path, split: str, names: List[str]) -> Counter:
    y = get_y(load_npz(dataset_dir / f"{split}.npz"))
    counts: Counter = Counter()
    for j, name in enumerate(names):
        c = int(y[:, j].sum())
        if c:
            counts[str(name)] = c
    return counts


def set_metrics(true_set: Set[str], pred_set: Set[str]) -> Dict[str, float]:
    inter = len(true_set & pred_set)
    precision = inter / len(pred_set) if pred_set else 0.0
    recall = inter / len(true_set) if true_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = len(true_set | pred_set)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": inter / union if union else 1.0,
    }


def markdown(summary: Dict[str, Any]) -> str:
    lines = ["# Stage2 v5 Failure Decomposition", ""]
    lines.append("## Overall")
    for k, v in summary["overall"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Top500 Miss Failure Fractions")
    for k, v in summary["top500_miss_failure_fractions"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## OOV Miss Decomposition")
    for k, v in summary["oov_miss_decomposition"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Non-OOV Miss Decomposition")
    for k, v in summary["non_oov_miss_decomposition"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Missing Label Families")
    for k, v in summary["missing_label_family_distribution"].items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Decompose Stage2 v5 precursor-set failures from an evaluated candidate pool.")
    ap.add_argument("--candidate_csv", required=True, help="Evaluated test candidate CSV, usually v4 calibrated candidates.")
    ap.add_argument("--dataset_dir", required=True, help="Canonical Stage2 dataset directory with train/test npz and precursor_names.json.")
    ap.add_argument("--ontology_csv", required=True, help="Precursor ontology with family and parse_status.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--rank_col", default="calibrated_rank")
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_dir)
    names = [str(x) for x in load_json(dataset_dir / "precursor_names.json")]
    train_counts = split_label_counts(dataset_dir, "train", names)
    train_labels = set(train_counts)

    ont = pd.read_csv(args.ontology_csv)
    family_lookup = dict(zip(ont["canonical_precursor"].astype(str), ont["precursor_family"].astype(str)))
    parse_lookup = dict(zip(ont["canonical_precursor"].astype(str), ont["parse_status"].astype(str)))

    usecols = [
        "sample_index", "id", "formula", "reaction_method", "true_precursors", "pred_precursors",
        "exact", "f1", "jaccard", args.rank_col,
    ]
    cand = pd.read_csv(args.candidate_csv, usecols=lambda c: c in set(usecols))
    if args.rank_col not in cand.columns:
        raise KeyError(f"rank column {args.rank_col!r} not found in {args.candidate_csv}")
    cand["exact"] = cand["exact"].astype(str).str.lower().isin(["true", "1", "yes"])
    cand[args.rank_col] = pd.to_numeric(cand[args.rank_col], errors="coerce").fillna(10**9).astype(int)
    cand["f1"] = pd.to_numeric(cand["f1"], errors="coerce").fillna(0.0)
    cand["jaccard"] = pd.to_numeric(cand["jaccard"], errors="coerce").fillna(0.0)

    rows = []
    missing_family_counter: Counter = Counter()
    method_failure_counts: Dict[str, Counter] = defaultdict(Counter)

    for sample_index, group in cand.groupby("sample_index", sort=True):
        group = group.sort_values(args.rank_col)
        first = group.iloc[0]
        true_set = set(parse_list(first["true_precursors"]))
        pred_sets = [set(parse_list(x)) for x in group["pred_precursors"]]
        candidate_label_union = set().union(*pred_sets) if pred_sets else set()
        missing_true = sorted(true_set - candidate_label_union)
        all_true_present = len(missing_true) == 0
        exact_rows = group[group["exact"]]
        exact_present = not exact_rows.empty
        exact_rank = int(exact_rows[args.rank_col].min()) if exact_present else 0
        top1_exact = bool(exact_present and exact_rank <= 1)
        top10_exact = bool(exact_present and exact_rank <= 10)
        oov_labels = sorted([x for x in true_set if x not in train_labels])
        parse_failed_labels = sorted([x for x in true_set if parse_lookup.get(x, "failed") != "ok"])
        assembly_failure = bool(all_true_present and not exact_present)
        ranking_failure = bool(exact_present and exact_rank > 1)
        ranking_failure_top10 = bool(exact_present and exact_rank > 10)
        oov_failure = bool(oov_labels)
        parse_failure = bool(parse_failed_labels)
        method_template_failure = str(first["reaction_method"]) in WEAK_METHODS
        if not exact_present:
            for lab in missing_true:
                missing_family_counter[family_lookup.get(lab, "unknown")] += 1
        flags = {
            "top500_miss": not exact_present,
            "missing_label_failure": bool(missing_true),
            "assembly_failure": assembly_failure,
            "ranking_failure": ranking_failure,
            "ranking_failure_top10": ranking_failure_top10,
            "oov_failure": oov_failure,
            "parse_failure": parse_failure,
            "method_template_failure": method_template_failure,
        }
        for k, v in flags.items():
            if v:
                method_failure_counts[str(first["reaction_method"])][k] += 1
        best_idx = group["jaccard"].idxmax()
        best = group.loc[best_idx]
        rows.append({
            "sample_index": int(sample_index),
            "id": first["id"],
            "formula": first["formula"],
            "reaction_method": first["reaction_method"],
            "true_precursors": json.dumps(sorted(true_set), ensure_ascii=False),
            "n_true_precursors": len(true_set),
            "exact_set_present_top500": exact_present,
            "exact_rank": exact_rank,
            "top1_exact": top1_exact,
            "top10_exact": top10_exact,
            "all_true_labels_present_individually": all_true_present,
            "missing_true_labels": json.dumps(missing_true, ensure_ascii=False),
            "n_missing_true_labels": len(missing_true),
            "missing_label_failure": bool(missing_true),
            "assembly_failure": assembly_failure,
            "ranking_failure": ranking_failure,
            "ranking_failure_top10": ranking_failure_top10,
            "oov_failure": oov_failure,
            "oov_precursors": json.dumps(oov_labels, ensure_ascii=False),
            "n_oov_precursors": len(oov_labels),
            "parse_failure": parse_failure,
            "parse_failed_precursors": json.dumps(parse_failed_labels, ensure_ascii=False),
            "n_parse_failed_precursors": len(parse_failed_labels),
            "method_template_failure": method_template_failure,
            "top1_pred_precursors": first["pred_precursors"],
            "top1_f1": float(first["f1"]),
            "top1_jaccard": float(first["jaccard"]),
            "best_jaccard": float(best["jaccard"]),
            "best_f1": float(group["f1"].max()),
        })

    df = pd.DataFrame(rows)
    out_csv = out_dir / "test_failure_decomposition.csv"
    df.to_csv(out_csv, index=False)

    miss = df[~df["exact_set_present_top500"]]
    oov_miss = miss[miss["oov_failure"]]
    non_oov_miss = miss[~miss["oov_failure"]]

    def frac_table(sub: pd.DataFrame, cols: Iterable[str]) -> Dict[str, float]:
        if len(sub) == 0:
            return {c: 0.0 for c in cols}
        return {c: float(sub[c].mean()) for c in cols}

    failure_cols = ["missing_label_failure", "assembly_failure", "oov_failure", "parse_failure", "method_template_failure"]
    method_rows = []
    for method, sub in df.groupby("reaction_method", dropna=False):
        miss_sub = sub[~sub["exact_set_present_top500"]]
        row = {
            "reaction_method": method,
            "n_samples": int(len(sub)),
            "top1_exact": float(sub["top1_exact"].mean()),
            "top10_exact": float(sub["top10_exact"].mean()),
            "top500_exact": float(sub["exact_set_present_top500"].mean()),
            "n_top500_miss": int(len(miss_sub)),
        }
        row.update({f"miss_{k}": float(miss_sub[k].mean()) if len(miss_sub) else 0.0 for k in failure_cols})
        method_rows.append(row)
    method_df = pd.DataFrame(method_rows).sort_values("n_samples", ascending=False)
    method_df.to_csv(out_dir / "test_failure_by_reaction_method.csv", index=False)

    summary = {
        "config": vars(args),
        "overall": {
            "n_samples": int(len(df)),
            "top1_exact": float(df["top1_exact"].mean()),
            "top10_exact": float(df["top10_exact"].mean()),
            "top500_exact": float(df["exact_set_present_top500"].mean()),
            "n_top500_miss": int(len(miss)),
            "n_oov_samples": int(df["oov_failure"].sum()),
            "oov_top500_exact": float(df[df["oov_failure"]]["exact_set_present_top500"].mean()) if df["oov_failure"].any() else 0.0,
            "non_oov_top500_exact": float(df[~df["oov_failure"]]["exact_set_present_top500"].mean()) if (~df["oov_failure"]).any() else 0.0,
        },
        "top500_miss_failure_fractions": frac_table(miss, failure_cols),
        "oov_miss_decomposition": frac_table(oov_miss, ["missing_label_failure", "assembly_failure", "parse_failure", "method_template_failure"]),
        "non_oov_miss_decomposition": frac_table(non_oov_miss, ["missing_label_failure", "assembly_failure", "parse_failure", "method_template_failure"]),
        "missing_label_family_distribution": dict(missing_family_counter.most_common()),
        "artifacts": {
            "csv": str(out_csv),
            "method_csv": str(out_dir / "test_failure_by_reaction_method.csv"),
            "summary": str(out_dir / "failure_decomposition_summary.json"),
            "report": str(out_dir / "failure_decomposition_report.md"),
        },
    }
    write_json(out_dir / "failure_decomposition_summary.json", summary)
    (out_dir / "failure_decomposition_report.md").write_text(markdown(summary), encoding="utf-8")
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
