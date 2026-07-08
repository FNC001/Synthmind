#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

import numpy as np
import pandas as pd


ELEMENT_RE = re.compile(r"([A-Z][a-z]?)")
IGNORED = {"H", "O", "C", "N"}


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


def parse_list(text: Any) -> List[str]:
    try:
        obj = json.loads(str(text))
        if isinstance(obj, list):
            return [str(x) for x in obj]
    except Exception:
        pass
    return []


def elements(text: str) -> Set[str]:
    return set(ELEMENT_RE.findall(str(text)))


def target_elements(formula: str) -> List[str]:
    elems = sorted(elements(formula) - {"O"})
    return elems or sorted(elements(formula))


def label_elements(label: str) -> Set[str]:
    return elements(label) - IGNORED


def candidate_elements(labels: Sequence[str]) -> Set[str]:
    out: Set[str] = set()
    for lab in labels:
        out |= label_elements(lab)
    return out


def parse_bool(x: Any) -> bool:
    return str(x).lower() in {"true", "1", "yes"}


def set_metrics(true_set: Set[str], pred_set: Set[str]) -> Dict[str, Any]:
    inter = len(true_set & pred_set)
    precision = inter / len(pred_set) if pred_set else 0.0
    recall = inter / len(true_set) if true_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = len(true_set | pred_set)
    return {"precision": precision, "recall": recall, "f1": f1, "jaccard": inter / union if union else 1.0, "exact": pred_set == true_set}


def coverage_stats(labels: Sequence[str], formula: str) -> Dict[str, float]:
    target = set(target_elements(formula))
    present = candidate_elements(labels)
    return {
        "element_coverage": len(target & present) / len(target) if target else 1.0,
        "missing_element_count": float(len(target - present)),
        "extra_element_count": float(len(present - target)),
    }


def metrics_by_k(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in [1, 3, 5, 10, 50, 100, 200, 500]:
        sub = df[df["rank"] <= k]
        g = sub.groupby("sample_index", sort=False)
        out[f"top{k}_exact"] = float(g["exact"].any().mean()) if len(g) else 0.0
        out[f"top{k}_best_f1"] = float(g["f1"].max().mean()) if len(g) else 0.0
        out[f"top{k}_best_jaccard"] = float(g["jaccard"].max().mean()) if len(g) else 0.0
    return out


def repair_score(labels: Sequence[str], formula: str, base: float, label_score: float, source_bonus: float) -> Dict[str, float]:
    cov = coverage_stats(labels, formula)
    size = len(labels)
    target_size = max(1, len(target_elements(formula)))
    set_size_score = 1.0 / (1.0 + abs(size - target_size))
    total = (
        0.68 * base
        + 0.85 * label_score
        + 1.55 * cov["element_coverage"]
        + 0.55 * set_size_score
        + source_bonus
        - 1.10 * cov["missing_element_count"]
        - 0.36 * cov["extra_element_count"]
        - 0.05 * max(0, size - 4)
    )
    return {
        "method_template_score": 0.0,
        "family_score": 0.0,
        "original_v4_score": 0.0,
        "open_vocab_score": 0.0,
        "oov_risk_score": 0.25,
        "assembly_score": 1.0,
        "set_size_score": set_size_score,
        "cooccurrence_score": label_score,
        "method_prior_score": source_bonus,
        "mlp_score": 0.0,
        "retrieval_score": 0.0,
        **cov,
        "candidate_size": float(size),
        "total_score_v5": total,
    }


def add_repair(store: Dict[frozenset, Dict[str, Any]], labels: Sequence[str], formula: str, true_set: Set[str], base: float, label_score: float, source: str) -> None:
    labels = sorted(set(labels))
    if not labels:
        return
    feats = repair_score(labels, formula, base, label_score, 0.25 if source == "assembly_repair_combo" else 0.10)
    key = frozenset(labels)
    if key in store:
        return
    sm = set_metrics(true_set, set(labels))
    row = {
        "labels": labels,
        "candidate_source": source,
        "candidate_source_mix": source,
        "method_template_id": "",
        **feats,
        **sm,
    }
    store[key] = row


def main() -> None:
    ap = argparse.ArgumentParser(description="Repair Stage2 v5 candidate set assembly using no-leakage label recombination.")
    ap.add_argument("--candidate_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--summary_json", required=True)
    ap.add_argument("--top_n", type=int, default=500)
    ap.add_argument("--preserve_base_top", type=int, default=470)
    ap.add_argument("--label_pool_size", type=int, default=32)
    ap.add_argument("--combo_limit", type=int, default=900)
    args = ap.parse_args()

    cand = pd.read_csv(args.candidate_csv)
    cand["exact"] = cand["exact"].apply(parse_bool)
    for col in ["total_score_v5", "f1", "jaccard", "rank"]:
        cand[col] = pd.to_numeric(cand[col], errors="coerce").fillna(0.0)
    all_rows = []
    before_metrics = metrics_by_k(cand)

    for sample_index, group in cand.groupby("sample_index", sort=True):
        group = group.sort_values("rank").copy()
        first = group.iloc[0]
        formula = str(first["formula"])
        true_set = set(parse_list(first["true_precursors"]))
        store: Dict[frozenset, Dict[str, Any]] = {}
        label_scores: Dict[str, float] = defaultdict(float)
        label_sources: Dict[str, Set[str]] = defaultdict(set)
        for _, r in group.iterrows():
            labels = parse_list(r["pred_precursors"])
            key = frozenset(labels)
            rec = r.to_dict()
            rec["labels"] = sorted(set(labels))
            rec["exact"] = parse_bool(rec["exact"])
            store[key] = rec
            rank_bonus = 1.0 / math.log2(float(r["rank"]) + 2.0)
            score = rank_bonus + 0.06 * float(r["total_score_v5"])
            for lab in labels:
                if score > label_scores[lab]:
                    label_scores[lab] = score
                label_sources[lab].add(str(r.get("candidate_source", "")))
        label_pool = sorted(label_scores, key=lambda x: label_scores[x], reverse=True)[: int(args.label_pool_size)]
        target = set(target_elements(formula))
        # Size-aware combinations from high-confidence individual labels.
        sizes = sorted({max(1, min(5, len(target))), max(1, min(5, len(target) + 1)), max(1, min(5, len(target) - 1)), 2, 3})
        combo_count = 0
        for size in sizes:
            for combo in itertools.combinations(label_pool, size):
                present = candidate_elements(combo)
                if target and len(target & present) / len(target) < 0.75:
                    continue
                ls = float(np.mean([label_scores[x] for x in combo]))
                add_repair(store, combo, formula, true_set, 0.0, ls, "assembly_repair_combo")
                combo_count += 1
                if combo_count >= int(args.combo_limit):
                    break
            if combo_count >= int(args.combo_limit):
                break
        # Add/drop/swap around strong element-coverage candidates.
        strong = group.sort_values(["element_coverage", "total_score_v5"], ascending=[False, False]).head(80)
        for _, r in strong.iterrows():
            base_labels = sorted(set(parse_list(r["pred_precursors"])))
            if not base_labels:
                continue
            base_score = float(r["total_score_v5"])
            # add
            for lab in label_pool[:12]:
                if lab not in base_labels:
                    new = base_labels + [lab]
                    ls = float(np.mean([label_scores.get(x, 0.0) for x in new]))
                    add_repair(store, new, formula, true_set, base_score, ls, "assembly_repair_add")
            # drop
            if len(base_labels) > 1:
                for lab in base_labels:
                    new = [x for x in base_labels if x != lab]
                    ls = float(np.mean([label_scores.get(x, 0.0) for x in new])) if new else 0.0
                    add_repair(store, new, formula, true_set, base_score, ls, "assembly_repair_drop")
            # swap by overlapping source elements
            for old in base_labels:
                old_e = label_elements(old)
                for new_lab in label_pool[:18]:
                    if new_lab in base_labels:
                        continue
                    if old_e & label_elements(new_lab):
                        new = [new_lab if x == old else x for x in base_labels]
                        ls = float(np.mean([label_scores.get(x, 0.0) for x in new]))
                        add_repair(store, new, formula, true_set, base_score, ls, "assembly_repair_swap")

        rows = list(store.values())
        base_rows = [x for x in rows if not str(x.get("candidate_source", "")).startswith("assembly_repair")]
        repair_rows = [x for x in rows if str(x.get("candidate_source", "")).startswith("assembly_repair")]
        base_rows.sort(key=lambda x: float(x.get("rank", 10**9)))
        repair_rows.sort(key=lambda x: float(x.get("total_score_v5", 0.0)), reverse=True)
        selected = base_rows[: min(int(args.preserve_base_top), int(args.top_n))]
        seen = {frozenset(x.get("labels", parse_list(x.get("pred_precursors", "[]")))) for x in selected}
        for rec in repair_rows:
            if len(selected) >= int(args.top_n):
                break
            key = frozenset(rec.get("labels", parse_list(rec.get("pred_precursors", "[]"))))
            if key not in seen:
                selected.append(rec)
                seen.add(key)
        if len(selected) < int(args.top_n):
            for rec in base_rows[int(args.preserve_base_top):]:
                if len(selected) >= int(args.top_n):
                    break
                key = frozenset(rec.get("labels", parse_list(rec.get("pred_precursors", "[]"))))
                if key not in seen:
                    selected.append(rec)
                    seen.add(key)
        for rank, rec in enumerate(selected[: int(args.top_n)], start=1):
            labels = rec.pop("labels", parse_list(rec.get("pred_precursors", "[]")))
            row = dict(rec)
            row.update({
                "sample_index": int(sample_index),
                "rank": rank,
                "id": first["id"],
                "formula": formula,
                "reaction_method": first["reaction_method"],
                "true_precursors": json.dumps(sorted(true_set), ensure_ascii=False),
                "pred_precursors": json.dumps(sorted(labels), ensure_ascii=False),
                "candidate_set": json.dumps(sorted(labels), ensure_ascii=False),
            })
            # Recompute metrics for original rows as well, to avoid stale boolean/string state.
            row.update(set_metrics(true_set, set(labels)))
            all_rows.append(row)
        if (int(sample_index) + 1) % 500 == 0:
            print(f"[Progress] repaired {int(sample_index)+1}", flush=True)

    out = pd.DataFrame(all_rows).sort_values(["sample_index", "rank"])
    out_path = Path(args.output_csv).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    after_metrics = metrics_by_k(out)
    source_contrib = {}
    for src, sub in out.groupby("candidate_source"):
        source_contrib[src] = {
            "n_candidates": int(len(sub)),
            "exact_rows": int(sub["exact"].sum()),
            "unique_samples_with_exact": int(sub[sub["exact"]]["sample_index"].nunique()),
        }
    summary = {
        "config": vars(args),
        "data": {"n_rows": int(out["sample_index"].nunique()), "n_candidates": int(len(out))},
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "candidate_source_contribution": source_contrib,
        "artifacts": {"csv": str(out_path), "summary": str(Path(args.summary_json).resolve())},
    }
    write_json(Path(args.summary_json), summary)
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
