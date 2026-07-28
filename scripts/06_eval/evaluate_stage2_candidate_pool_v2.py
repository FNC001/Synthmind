#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


ELEMENT_RE = re.compile(r"([A-Z][a-z]?)")
IGNORED_EXTRA_ELEMENTS = {"H", "O", "C", "N"}


def load_base_module(project_root: Path) -> Any:
    path = project_root / "scripts/06_eval/evaluate_route_top10_with_setsize_rerank.py"
    spec = importlib.util.spec_from_file_location("route_top10_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import base evaluator from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def get_x(pack: Mapping[str, np.ndarray]) -> np.ndarray:
    for key in ["x", "features", "X"]:
        if key in pack:
            return np.asarray(pack[key], dtype=np.float32)
    raise KeyError(f"Missing x/features/X in npz keys={list(pack)}")


def get_y(pack: Mapping[str, np.ndarray]) -> np.ndarray:
    for key in ["y_multi_hot", "y", "labels", "targets"]:
        if key in pack:
            return (np.asarray(pack[key]) > 0).astype(np.int8)
    raise KeyError(f"Missing y labels in npz keys={list(pack)}")


def parse_json_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    s = str(value).strip()
    if not s:
        return []
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass
    return [x.strip() for x in s.replace(";", ",").split(",") if x.strip()]


def element_set(text: Any) -> Set[str]:
    return set(ELEMENT_RE.findall(str(text or "")))


def set_metrics(true_set: Set[str], pred_set: Set[str]) -> Dict[str, Any]:
    inter = len(true_set & pred_set)
    union = len(true_set | pred_set)
    precision = inter / len(pred_set) if pred_set else 0.0
    recall = inter / len(true_set) if true_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    jaccard = inter / union if union else 1.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "exact": pred_set == true_set,
        "any_overlap": inter > 0,
    }


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
    elems = element_set(s)
    if len(elems - {"H", "O", "C", "N"}) == 1 and "O" in elems:
        return "oxide"
    if len(elems) == 1:
        return "elemental"
    return "other"


FAMILIES = ["oxide", "carbonate", "nitrate", "hydroxide", "acetate", "halide", "sulfate", "phosphate", "elemental", "other"]


def formula_vector(formula: str, elements: Sequence[str]) -> np.ndarray:
    elems = element_set(formula)
    return np.asarray([1.0 if e in elems else 0.0 for e in elements], dtype=np.float32)


def chem_counts(labels: Sequence[str], target_formula: str) -> Dict[str, float]:
    target = element_set(target_formula) - {"H", "O"}
    present: Set[str] = set()
    for label in labels:
        present |= element_set(label) - IGNORED_EXTRA_ELEMENTS
    coverage = len(target & present) / len(target) if target else 1.0
    missing = len(target - present)
    extra = len(present - target)
    return {"coverage": float(coverage), "missing": float(missing), "extra": float(extra)}


def prob_features(labels: Sequence[str], label_to_idx: Mapping[str, int], probs: np.ndarray) -> Dict[str, float]:
    vals = []
    for label in labels:
        idx = label_to_idx.get(str(label))
        vals.append(float(probs[idx]) if idx is not None else 0.0)
    if not vals:
        vals = [0.0]
    arr = np.asarray(vals, dtype=np.float32)
    clipped = np.clip(arr, 1e-6, 1.0)
    return {
        "prob_sum": float(arr.sum()),
        "prob_mean": float(arr.mean()),
        "prob_min": float(arr.min()),
        "prob_max": float(arr.max()),
        "prob_log_mean": float(np.log(clipped).mean()),
    }


def family_counts(labels: Sequence[str]) -> Dict[str, int]:
    counts = Counter(family_of_precursor(x) for x in labels)
    return {f"fam_{fam}": int(counts.get(fam, 0)) for fam in FAMILIES}


def y_to_sets(y: np.ndarray, names: Sequence[str]) -> List[Set[str]]:
    return [{str(names[j]) for j in np.where(y[i] > 0)[0]} for i in range(y.shape[0])]


def build_template_indexes(train_meta: pd.DataFrame, train_sets: Sequence[Set[str]]) -> Dict[str, Any]:
    formula_to_sets: Dict[str, Counter] = defaultdict(Counter)
    material_to_sets: Dict[str, Counter] = defaultdict(Counter)
    method_to_sets: Dict[str, Counter] = defaultdict(Counter)
    global_sets: Counter = Counter()
    cooccur: Counter = Counter()
    label_freq: Counter = Counter()
    for i, row in train_meta.iterrows():
        s = frozenset(train_sets[i])
        if not s:
            continue
        formula_to_sets[str(row.get("formula", ""))][s] += 1
        material_to_sets[str(row.get("material_id", ""))][s] += 1
        method_to_sets[str(row.get("reaction_method", "other"))][s] += 1
        global_sets[s] += 1
        for lab in s:
            label_freq[lab] += 1
        labs = sorted(s)
        for a_idx, a in enumerate(labs):
            for b in labs[a_idx + 1:]:
                cooccur[(a, b)] += 1
    return {
        "formula_to_sets": formula_to_sets,
        "material_to_sets": material_to_sets,
        "method_to_sets": method_to_sets,
        "global_sets": global_sets,
        "cooccur": cooccur,
        "label_freq": label_freq,
    }


def template_frequency(candidate: Set[str], indexes: Mapping[str, Any]) -> int:
    return int(indexes["global_sets"].get(frozenset(candidate), 0))


def cooccur_mean(candidate: Sequence[str], indexes: Mapping[str, Any]) -> float:
    labs = sorted(candidate)
    if len(labs) < 2:
        return 0.0
    vals = []
    cooccur = indexes["cooccur"]
    for i, a in enumerate(labs):
        for b in labs[i + 1:]:
            vals.append(float(cooccur.get((a, b), 0)))
    return float(np.mean(vals)) if vals else 0.0


def candidate_base_score(
    labels: Sequence[str],
    probs: np.ndarray,
    label_to_idx: Mapping[str, int],
    target_formula: str,
    set_size_prob: float,
    pred_size: int,
    retrieval_similarity: float,
    template_freq: int,
    indexes: Mapping[str, Any],
) -> float:
    pf = prob_features(labels, label_to_idx, probs)
    chem = chem_counts(labels, target_formula)
    fams = Counter(family_of_precursor(x) for x in labels)
    common_family_bonus = sum(fams.get(x, 0) for x in ["oxide", "carbonate", "nitrate", "hydroxide", "acetate", "halide", "elemental"])
    return float(
        pf["prob_log_mean"]
        + 2.8 * chem["coverage"]
        - 0.45 * chem["extra"]
        - 0.65 * chem["missing"]
        - 0.12 * abs(len(labels) - int(pred_size))
        + 0.45 * math.log(max(float(set_size_prob), 1e-6))
        + 0.8 * float(retrieval_similarity)
        + 0.12 * math.log1p(int(template_freq))
        + 0.04 * float(common_family_bonus)
        + 0.02 * math.log1p(cooccur_mean(labels, indexes))
    )


def add_candidate(
    store: Dict[frozenset, Dict[str, Any]],
    labels: Iterable[str],
    source: str,
    score: float,
    retrieval_similarity: float = 0.0,
    route_frequency: int = 0,
) -> None:
    labs = [str(x) for x in labels if str(x)]
    key = frozenset(labs)
    if not key:
        return
    old = store.get(key)
    if old is None:
        store[key] = {
            "labels": sorted(key),
            "sources": {source},
            "score": float(score),
            "retrieval_similarity": float(retrieval_similarity),
            "route_frequency": int(route_frequency),
        }
    else:
        old["sources"].add(source)
        old["score"] = max(float(old["score"]), float(score))
        old["retrieval_similarity"] = max(float(old["retrieval_similarity"]), float(retrieval_similarity))
        old["route_frequency"] = max(int(old["route_frequency"]), int(route_frequency))


def make_hybrid_candidates_for_sample(
    base: Any,
    sample_index: int,
    meta_row: pd.Series,
    probs: np.ndarray,
    names: Sequence[str],
    label_to_idx: Mapping[str, int],
    pred_size: int,
    size_prob_by_k: Mapping[int, float],
    train_sets: Sequence[Set[str]],
    train_meta: pd.DataFrame,
    train_x: np.ndarray,
    train_formula_vec: np.ndarray,
    query_x: np.ndarray,
    query_formula_vec: np.ndarray,
    nn_model: NearestNeighbors | None,
    indexes: Mapping[str, Any],
    top_n: int,
    mlp_keep: int,
    retrieval_k: int,
    max_set_size: int,
) -> List[Dict[str, Any]]:
    formula = str(meta_row.get("formula", ""))
    method = str(meta_row.get("reaction_method", "other"))
    material_id = str(meta_row.get("material_id", ""))
    store: Dict[frozenset, Dict[str, Any]] = {}

    mlp_cands = base.generate_candidates(
        probs=probs,
        names=names,
        formula=formula,
        pred_size=int(pred_size),
        size_prob_by_k=size_prob_by_k,
        max_size=int(max_set_size),
        keep=int(mlp_keep),
    )
    for cand in mlp_cands:
        labels = cand["labels"]
        tf = template_frequency(set(labels), indexes)
        sp = float(size_prob_by_k.get(len(labels), 1e-6))
        score = candidate_base_score(labels, probs, label_to_idx, formula, sp, pred_size, 0.0, tf, indexes)
        add_candidate(store, labels, "mlp_beam", score, route_frequency=tf)

    # Historical templates by exact formula/material/method.
    for source, counter, limit, bonus in [
        ("formula_template", indexes["formula_to_sets"].get(formula, Counter()), 80, 0.45),
        ("material_template", indexes["material_to_sets"].get(material_id, Counter()), 60, 0.35),
        ("method_template", indexes["method_to_sets"].get(method, Counter()), 40, 0.10),
    ]:
        for labels_set, freq in counter.most_common(limit):
            labels = sorted(labels_set)
            if len(labels) > max_set_size:
                continue
            sp = float(size_prob_by_k.get(len(labels), 1e-6))
            score = candidate_base_score(labels, probs, label_to_idx, formula, sp, pred_size, bonus, int(freq), indexes)
            add_candidate(store, labels, source, score, retrieval_similarity=bonus, route_frequency=int(freq))

    # Descriptor + composition retrieval.
    if int(retrieval_k) > 0 and nn_model is not None:
        k_query = min(max(int(retrieval_k) * 5, int(retrieval_k)), len(train_sets))
        dists, inds = nn_model.kneighbors(query_x.reshape(1, -1), n_neighbors=k_query)
        seen_templates = 0
        for dist, idx in zip(dists[0], inds[0]):
            idx = int(idx)
            labels_set = set(train_sets[idx])
            if not labels_set or len(labels_set) > max_set_size:
                continue
            row = train_meta.iloc[idx]
            desc_sim = 1.0 - float(dist)
            elem_inter = float(np.minimum(query_formula_vec, train_formula_vec[idx]).sum())
            elem_union = float(np.maximum(query_formula_vec, train_formula_vec[idx]).sum())
            elem_sim = elem_inter / elem_union if elem_union else 1.0
            method_sim = 1.0 if str(row.get("reaction_method", "other")) == method else 0.0
            formula_sim = 1.0 if str(row.get("formula", "")) == formula else 0.0
            sim = 0.55 * desc_sim + 0.25 * elem_sim + 0.12 * method_sim + 0.08 * formula_sim
            labels = sorted(labels_set)
            tf = template_frequency(labels_set, indexes)
            sp = float(size_prob_by_k.get(len(labels), 1e-6))
            score = candidate_base_score(labels, probs, label_to_idx, formula, sp, pred_size, sim, tf, indexes)
            add_candidate(store, labels, "retrieval", score, retrieval_similarity=sim, route_frequency=tf)
            seen_templates += 1
            if seen_templates >= retrieval_k:
                break

    out = []
    for key, cand in store.items():
        labels = cand["labels"]
        sp = float(size_prob_by_k.get(len(labels), 1e-6))
        pf = prob_features(labels, label_to_idx, probs)
        chem = chem_counts(labels, formula)
        fam = family_counts(labels)
        tf = template_frequency(set(labels), indexes)
        out.append({
            "labels": labels,
            "label_set": set(labels),
            "sources": "+".join(sorted(cand["sources"])),
            "score": float(cand["score"]),
            "retrieval_similarity": float(cand["retrieval_similarity"]),
            "route_frequency": int(max(cand["route_frequency"], tf)),
            "candidate_size": int(len(labels)),
            "set_size_prob": sp,
            "template_seen_in_train": int(tf > 0),
            "cooccur_mean": cooccur_mean(labels, indexes),
            **pf,
            **chem,
            **fam,
        })
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:top_n]


def compute_metrics(df: pd.DataFrame, ks: Sequence[int]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    top1 = df[df["rank"] == 1]
    metrics["top1_exact"] = float(top1["exact"].mean())
    metrics["top1_f1"] = float(top1["f1"].mean())
    metrics["top1_jaccard"] = float(top1["jaccard"].mean())
    for k in ks:
        sub = df[df["rank"] <= int(k)]
        grouped = sub.groupby("sample_index", sort=False)
        metrics[f"top{k}_exact_recall"] = float(grouped["exact"].any().mean())
        metrics[f"top{k}_best_f1"] = float(grouped["f1"].max().mean())
        metrics[f"top{k}_best_jaccard"] = float(grouped["jaccard"].max().mean())
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate Stage2 hybrid candidate pool v2.")
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--stage2_run_dir", required=True)
    ap.add_argument("--set_size_model", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--top_n", type=int, default=500)
    ap.add_argument("--mlp_keep", type=int, default=500)
    ap.add_argument("--retrieval_k", type=int, default=80)
    ap.add_argument("--max_set_size", type=int, default=7)
    ap.add_argument("--batch_size", type=int, default=512)
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    dataset_dir = root / args.dataset_dir
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    base = load_base_module(root)

    names = [str(x) for x in load_json(dataset_dir / "precursor_names.json")]
    label_to_idx = {p: i for i, p in enumerate(names)}
    train_pack = load_npz(dataset_dir / "train.npz")
    train_meta = pd.read_csv(dataset_dir / "train_meta.csv")
    train_x = get_x(train_pack)
    train_y = get_y(train_pack)
    train_sets = y_to_sets(train_y, names)

    split_pack = load_npz(dataset_dir / f"{args.split}.npz")
    split_meta = pd.read_csv(dataset_dir / f"{args.split}_meta.csv")
    x = get_x(split_pack)
    y = get_y(split_pack)
    true_sets = y_to_sets(y, names)

    all_elements = sorted(set().union(*(element_set(f) for f in train_meta["formula"].astype(str).tolist() + split_meta["formula"].astype(str).tolist())))
    train_formula_vec = np.vstack([formula_vector(f, all_elements) for f in train_meta["formula"].astype(str)]).astype(np.float32)
    query_formula_vecs = np.vstack([formula_vector(f, all_elements) for f in split_meta["formula"].astype(str)]).astype(np.float32)

    stage2_model, model_names = base.load_stage2_model(root / args.stage2_run_dir, x.shape[1])
    if model_names != names:
        raise ValueError("Model precursor_names differ from dataset precursor_names.")
    probs = base.predict_probs(stage2_model, x, int(args.batch_size))

    size_pack = joblib.load(root / args.set_size_model)
    size_model = size_pack["model"] if isinstance(size_pack, dict) and "model" in size_pack else size_pack
    size_pred = size_model.predict(x).astype(int)
    size_proba = size_model.predict_proba(x)
    size_classes = [int(c) for c in size_model.classes_.tolist()]

    nn = None
    if int(args.retrieval_k) > 0:
        nn = NearestNeighbors(n_neighbors=min(max(args.retrieval_k * 5, args.retrieval_k), len(train_x)), metric="cosine", algorithm="brute")
        nn.fit(train_x)
    indexes = build_template_indexes(train_meta, train_sets)

    rows = []
    for i in range(len(x)):
        prob_by_k = {k: float(size_proba[i, j]) for j, k in enumerate(size_classes)}
        cands = make_hybrid_candidates_for_sample(
            base=base,
            sample_index=i,
            meta_row=split_meta.iloc[i],
            probs=probs[i],
            names=names,
            label_to_idx=label_to_idx,
            pred_size=int(size_pred[i]),
            size_prob_by_k=prob_by_k,
            train_sets=train_sets,
            train_meta=train_meta,
            train_x=train_x,
            train_formula_vec=train_formula_vec,
            query_x=x[i],
            query_formula_vec=query_formula_vecs[i],
            nn_model=nn,
            indexes=indexes,
            top_n=int(args.top_n),
            mlp_keep=int(args.mlp_keep),
            retrieval_k=int(args.retrieval_k),
            max_set_size=int(args.max_set_size),
        )
        true = true_sets[i]
        for rank, cand in enumerate(cands, start=1):
            labels = cand.pop("labels")
            pred_set = cand.pop("label_set")
            sm = set_metrics(true, pred_set)
            rows.append({
                "sample_index": i,
                "rank": rank,
                "id": split_meta.loc[i, "id"],
                "material_id": split_meta.loc[i, "material_id"],
                "formula": split_meta.loc[i, "formula"],
                "reaction_method": split_meta.loc[i, "reaction_method"],
                "true_precursors": json.dumps(sorted(true), ensure_ascii=False),
                "pred_precursors": json.dumps(labels, ensure_ascii=False),
                "predicted_size": int(size_pred[i]),
                "true_size": int(len(true)),
                "size_abs_error": abs(int(len(labels)) - int(size_pred[i])),
                **cand,
                **sm,
            })
        if (i + 1) % 500 == 0:
            print(f"[Progress] {args.split} {i + 1}/{len(x)}", flush=True)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["sample_index", "score"], ascending=[True, False]).copy()
        df["rank"] = df.groupby("sample_index").cumcount() + 1
    ks = [10, 20, 50, 100, 200, 500]
    ks = [k for k in ks if k <= int(args.top_n)]
    metrics = compute_metrics(df, ks)
    summary = {
        "config": vars(args),
        "data": {
            "n_rows": int(len(x)),
            "n_candidates": int(len(df)),
            "n_labels": int(len(names)),
            "mean_true_size": float(y.sum(axis=1).mean()),
        },
        "metrics": metrics,
        "artifacts": {
            "candidate_csv": str((out_dir / f"{args.split}_stage2_candidate_pool_v2.csv").resolve()),
            "summary_json": str((out_dir / f"{args.split}_stage2_candidate_pool_v2_summary.json").resolve()),
        },
    }
    df.to_csv(out_dir / f"{args.split}_stage2_candidate_pool_v2.csv", index=False)
    write_json(out_dir / f"{args.split}_stage2_candidate_pool_v2_summary.json", summary)
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
