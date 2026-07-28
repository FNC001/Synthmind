#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set

import numpy as np
import pandas as pd


ELEMENT_RE = re.compile(r"([A-Z][a-z]?)")
IGNORED_EXTRA_ELEMENTS = {"H", "O", "C", "N"}


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


def get_y(pack: Mapping[str, np.ndarray]) -> np.ndarray:
    for key in ["y_multi_hot", "y", "labels", "targets"]:
        if key in pack:
            return (np.asarray(pack[key]) > 0).astype(np.int8)
    raise KeyError(f"Missing y labels in npz keys={list(pack)}")


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


def family_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("prob_family__")]


def target_elements(formula: str) -> List[str]:
    elems = sorted(element_set(formula) - {"O"})
    return elems or sorted(element_set(formula))


def family_of_generated(label: str) -> str:
    s = str(label)
    if "CO3" in s:
        return "carbonate"
    if "NO3" in s:
        return "nitrate"
    if "OH" in s:
        return "hydroxide"
    if "CH3COO" in s or "C2H3O2" in s:
        return "acetate"
    if "SO4" in s:
        return "sulfate"
    if "PO4" in s:
        return "phosphate"
    if element_set(s) & {"F", "Cl", "Br", "I"}:
        return "halide"
    if "O" in element_set(s):
        return "oxide"
    if re.fullmatch(r"[A-Z][a-z]?", s):
        return "elemental"
    return "other_salt"


def rule_generate(element: str, family: str) -> List[str]:
    e = str(element)
    if family == "elemental":
        return [e]
    if family == "oxide":
        return [f"{e}O", f"{e}O2", f"{e}2O3", f"{e}3O4", f"{e}2O5"]
    if family == "carbonate":
        return [f"{e}CO3", f"{e}2CO3", f"{e}(CO3)2", f"{e}2(CO3)3"]
    if family == "nitrate":
        return [f"{e}NO3", f"{e}(NO3)2", f"{e}(NO3)3", f"{e}(NO3)2·6H2O", f"{e}(NO3)3·9H2O"]
    if family == "hydroxide":
        return [f"{e}OH", f"{e}(OH)2", f"{e}(OH)3"]
    if family == "acetate":
        return [f"{e}(CH3COO)", f"{e}(CH3COO)2", f"{e}(CH3COO)3", f"(CH3COO)2{e}", f"(CH3COO)3{e}"]
    if family == "sulfate":
        return [f"{e}SO4", f"{e}2(SO4)3", f"{e}SO4·H2O", f"{e}SO4·7H2O"]
    if family == "phosphate":
        return [f"{e}PO4", f"{e}3(PO4)2", f"{e}H2PO4", f"{e}2HPO4"]
    if family == "halide":
        return [f"{e}F", f"{e}F2", f"{e}Cl", f"{e}Cl2", f"{e}Cl3", f"{e}Br", f"{e}I"]
    return []


def y_to_sets(y: np.ndarray, names: Sequence[str]) -> List[Set[str]]:
    return [{str(names[j]) for j in np.where(y[i] > 0)[0]} for i in range(y.shape[0])]


def build_train_freq(dataset_dir: Path, names: Sequence[str]) -> Counter:
    y = get_y(load_npz(dataset_dir / "train.npz"))
    counts = y.sum(axis=0).astype(int)
    return Counter({str(names[i]): int(c) for i, c in enumerate(counts) if c > 0})


def build_ontology_indexes(ontology: pd.DataFrame, train_freq: Counter) -> Dict[str, Any]:
    by_elem_family: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    label_to_rec = {}
    for rec in ontology.to_dict(orient="records"):
        label = str(rec["canonical_precursor"])
        elems = set(parse_json_list(rec.get("target_source_elements", "[]"))) or (element_set(label) - IGNORED_EXTRA_ELEMENTS)
        fam = str(rec.get("precursor_family", "unknown"))
        row = {
            "label": label,
            "family": fam,
            "elements": elems,
            "train_freq": int(train_freq.get(label, 0)),
            "is_open_vocab": int(train_freq.get(label, 0) == 0),
        }
        label_to_rec[label] = row
        for e in elems:
            by_elem_family[(e, fam)].append(row)
    for key, rows in by_elem_family.items():
        rows.sort(key=lambda r: (r["train_freq"], -r["is_open_vocab"], r["label"]), reverse=True)
    return {"by_elem_family": by_elem_family, "label_to_rec": label_to_rec}


def family_predictions_for_sample(pred_df: pd.DataFrame, sample_index: int) -> Dict[str, List[tuple[str, float]]]:
    rows = pred_df[pred_df["sample_index"] == sample_index]
    fam_cols = family_columns(pred_df)
    out: Dict[str, List[tuple[str, float]]] = {}
    for _, row in rows.iterrows():
        elem = str(row["target_element"])
        vals = []
        for c in fam_cols:
            fam = c.replace("prob_family__", "")
            vals.append((fam, float(row[c])))
        vals.sort(key=lambda x: x[1], reverse=True)
        out[elem] = vals
    return out


def candidate_label_pool(
    sample_index: int,
    formula: str,
    family_pred: Dict[str, List[tuple[str, float]]],
    indexes: Mapping[str, Any],
    train_freq: Counter,
    seen_per_pair: int,
    open_per_pair: int,
    generated_per_pair: int,
) -> List[Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for elem in target_elements(formula):
        fams = family_pred.get(elem, [])[:4]
        if not fams:
            fams = [("oxide", 0.2), ("elemental", 0.1)]
        for fam, fam_score in fams:
            rows = indexes["by_elem_family"].get((elem, fam), [])
            seen = [r for r in rows if r["train_freq"] > 0][:seen_per_pair]
            open_rows = [r for r in rows if r["train_freq"] == 0][:open_per_pair]
            for r in seen + open_rows:
                label = r["label"]
                old = out.get(label, {})
                out[label] = {
                    "label": label,
                    "family": fam,
                    "family_score": max(float(fam_score), float(old.get("family_score", 0.0))),
                    "elements": set(r["elements"]),
                    "source": old.get("source", "") + ("+seen" if r["train_freq"] > 0 else "+ontology_open"),
                    "is_open_vocab": int(r["train_freq"] == 0),
                    "prior_frequency": int(r["train_freq"]),
                }
            for gen in rule_generate(elem, fam)[:generated_per_pair]:
                old = out.get(gen, {})
                out[gen] = {
                    "label": gen,
                    "family": family_of_generated(gen),
                    "family_score": max(float(fam_score) * 0.75, float(old.get("family_score", 0.0))),
                    "elements": element_set(gen) - IGNORED_EXTRA_ELEMENTS,
                    "source": old.get("source", "") + "+generated",
                    "is_open_vocab": int(train_freq.get(gen, 0) == 0),
                    "prior_frequency": int(train_freq.get(gen, 0)),
                }
    return list(out.values())


def score_set(
    labels: Sequence[str],
    label_meta: Mapping[str, Dict[str, Any]],
    formula: str,
    set_size: int,
    base_score: float = 0.0,
) -> Dict[str, float]:
    target = set(target_elements(formula))
    present: Set[str] = set()
    fam_scores = []
    freqs = []
    open_count = 0
    source_mix = Counter()
    for lab in labels:
        rec = label_meta.get(lab, {})
        elems = set(rec.get("elements", element_set(lab) - IGNORED_EXTRA_ELEMENTS))
        present |= elems
        fam_scores.append(float(rec.get("family_score", 0.0)))
        freqs.append(float(rec.get("prior_frequency", 0)))
        open_count += int(rec.get("is_open_vocab", 0))
        for src in str(rec.get("source", "")).split("+"):
            if src:
                source_mix[src] += 1
    coverage = len(target & present) / len(target) if target else 1.0
    missing = len(target - present)
    extra = len(present - target)
    family_score = float(np.mean(fam_scores)) if fam_scores else 0.0
    freq_score = float(np.log1p(np.mean(freqs))) if freqs else 0.0
    size_score = -abs(len(labels) - int(set_size))
    generated_count = source_mix.get("generated", 0)
    score = (
        float(base_score)
        + 1.5 * coverage
        + 1.0 * family_score
        + 0.35 * freq_score
        + 0.5 * size_score
        - 1.2 * missing
        - 0.8 * extra
        - 0.4 * max(0, len(labels) - max(int(set_size), 1))
        - 0.05 * generated_count
    )
    return {
        "total_score": float(score),
        "element_coverage": float(coverage),
        "missing_element_count": float(missing),
        "extra_element_count": float(extra),
        "family_score": float(family_score),
        "prior_frequency_score": float(freq_score),
        "set_size_score": float(size_score),
        "open_vocab_count": int(open_count),
        "generated_precursor_count": int(generated_count),
        "candidate_source_mix": "+".join(f"{k}:{v}" for k, v in sorted(source_mix.items())),
    }


def add_set(store: Dict[frozenset, Dict[str, Any]], labels: Iterable[str], meta: Dict[str, Any]) -> None:
    labs = sorted(set(str(x) for x in labels if str(x)))
    if not labs:
        return
    key = frozenset(labs)
    old = store.get(key)
    if old is None or float(meta["total_score"]) > float(old["total_score"]):
        store[key] = {"labels": labs, **meta}


def generate_sets_for_sample(
    sample_index: int,
    formula: str,
    true_set: Set[str],
    base_rows: pd.DataFrame,
    label_pool: List[Dict[str, Any]],
    set_size: int,
    top_n: int,
    max_set_size: int,
) -> List[Dict[str, Any]]:
    label_meta = {r["label"]: r for r in label_pool}
    store: Dict[frozenset, Dict[str, Any]] = {}
    if not base_rows.empty:
        for _, row in base_rows.iterrows():
            labels = parse_json_list(row["pred_precursors"])
            base_score = float(row.get("score", row.get("total_score", 0.0)))
            meta = score_set(labels, label_meta, formula, set_size, base_score=0.25 * base_score)
            meta["mlp_score"] = float(row.get("prob_log_mean", row.get("prob_score", 0.0)))
            meta["retrieval_score"] = float(row.get("retrieval_similarity", 0.0))
            add_set(store, labels, meta)

    # Beam from top family/open-vocab precursor labels.
    label_pool = sorted(label_pool, key=lambda r: (r.get("family_score", 0.0), math.log1p(r.get("prior_frequency", 0)), -r.get("is_open_vocab", 0)), reverse=True)
    top_labels = [r["label"] for r in label_pool[: min(80, len(label_pool))]]
    size_options = sorted(set([max(1, int(set_size) + d) for d in [-1, 0, 1, 2] if 1 <= int(set_size) + d <= max_set_size]))
    for k in size_options:
        n_pool = min(len(top_labels), {1: 60, 2: 50, 3: 34, 4: 22, 5: 16, 6: 12, 7: 10}.get(k, 10))
        count = 0
        for combo in itertools.combinations(top_labels[:n_pool], k):
            meta = score_set(combo, {r["label"]: r for r in label_pool}, formula, set_size, base_score=0.0)
            add_set(store, combo, meta)
            count += 1
            if count >= max(500, top_n * 3):
                break

    # Per-element product ensures each target element can contribute a family-matched source.
    per_elem = []
    for elem in target_elements(formula):
        candidates = [r["label"] for r in label_pool if elem in set(r.get("elements", set()))][:8]
        if candidates:
            per_elem.append(candidates)
    if per_elem:
        for n, combo in enumerate(itertools.product(*per_elem), start=1):
            if len(set(combo)) <= max_set_size:
                meta = score_set(combo, {r["label"]: r for r in label_pool}, formula, set_size, base_score=0.15)
                add_set(store, combo, meta)
            if n >= 5000:
                break

    out = list(store.values())
    out.sort(key=lambda x: x["total_score"], reverse=True)
    rows = []
    for rank, cand in enumerate(out[:top_n], start=1):
        pred_set = set(cand["labels"])
        sm = set_metrics(true_set, pred_set)
        rows.append({
            "sample_index": sample_index,
            "rank": rank,
            "pred_precursors": json.dumps(cand["labels"], ensure_ascii=False),
            **{k: v for k, v in cand.items() if k != "labels"},
            **sm,
        })
    return rows


def compute_metrics(df: pd.DataFrame, ks: Sequence[int]) -> Dict[str, float]:
    metrics = {}
    top1 = df[df["rank"] == 1]
    metrics["top1_exact"] = float(top1["exact"].mean())
    metrics["top1_f1"] = float(top1["f1"].mean())
    metrics["top1_jaccard"] = float(top1["jaccard"].mean())
    for k in ks:
        sub = df[df["rank"] <= k]
        g = sub.groupby("sample_index", sort=False)
        metrics[f"top{k}_exact"] = float(g["exact"].any().mean())
        metrics[f"top{k}_best_f1"] = float(g["f1"].max().mean())
        metrics[f"top{k}_best_jaccard"] = float(g["jaccard"].max().mean())
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Stage2 candidate pool v3 with family/open-vocab expansion.")
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--ontology_csv", required=True)
    ap.add_argument("--family_predictions", required=True)
    ap.add_argument("--base_candidate_csv", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--top_n", type=int, default=500)
    ap.add_argument("--seen_per_pair", type=int, default=18)
    ap.add_argument("--open_per_pair", type=int, default=12)
    ap.add_argument("--generated_per_pair", type=int, default=6)
    ap.add_argument("--max_set_size", type=int, default=7)
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [str(x) for x in load_json(dataset_dir / "precursor_names.json")]
    pack = load_npz(dataset_dir / f"{args.split}.npz")
    y = get_y(pack)
    meta = pd.read_csv(dataset_dir / f"{args.split}_meta.csv")
    true_sets = y_to_sets(y, names)
    fallback_set_len = y.sum(axis=1).astype(int)
    ontology = pd.read_csv(args.ontology_csv)
    train_freq = build_train_freq(dataset_dir, names)
    indexes = build_ontology_indexes(ontology, train_freq)
    family_pred = pd.read_csv(args.family_predictions)
    base = pd.read_csv(args.base_candidate_csv)
    base_groups = {int(i): g.copy() for i, g in base.groupby("sample_index", sort=False)}

    all_rows = []
    for i in range(len(meta)):
        formula = str(meta.loc[i, "formula"])
        fp = family_predictions_for_sample(family_pred, i)
        pool = candidate_label_pool(
            sample_index=i,
            formula=formula,
            family_pred=fp,
            indexes=indexes,
            train_freq=train_freq,
            seen_per_pair=int(args.seen_per_pair),
            open_per_pair=int(args.open_per_pair),
            generated_per_pair=int(args.generated_per_pair),
        )
        base_rows = base_groups.get(i, pd.DataFrame())
        if not base_rows.empty and "predicted_size" in base_rows.columns:
            pred_set_size = int(base_rows.iloc[0]["predicted_size"])
        elif not base_rows.empty and "candidate_size" in base_rows.columns:
            pred_set_size = int(base_rows.iloc[0]["candidate_size"])
        else:
            pred_set_size = int(fallback_set_len[i])
        rows = generate_sets_for_sample(
            sample_index=i,
            formula=formula,
            true_set=true_sets[i],
            base_rows=base_rows,
            label_pool=pool,
            set_size=pred_set_size,
            top_n=int(args.top_n),
            max_set_size=int(args.max_set_size),
        )
        for r in rows:
            r.update({
                "id": meta.loc[i, "id"],
                "material_id": meta.loc[i, "material_id"],
                "formula": formula,
                "reaction_method": meta.loc[i, "reaction_method"],
                "true_precursors": json.dumps(sorted(true_sets[i]), ensure_ascii=False),
            })
        all_rows.extend(rows)
        if (i + 1) % 500 == 0:
            print(f"[Progress] {args.split} {i + 1}/{len(meta)}", flush=True)

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["sample_index", "total_score"], ascending=[True, False]).copy()
    df["rank"] = df.groupby("sample_index").cumcount() + 1
    out_csv = out_dir / f"{args.split}_candidate_sets.csv"
    df.to_csv(out_csv, index=False)
    ks = [k for k in [1, 3, 5, 10, 20, 50, 100, 200, 500] if k <= int(args.top_n)]
    summary = {
        "config": vars(args),
        "data": {
            "n_rows": int(len(meta)),
            "n_candidates": int(len(df)),
            "mean_candidates_per_sample": float(len(df) / max(len(meta), 1)),
        },
        "metrics": compute_metrics(df, ks),
        "artifacts": {
            "candidate_csv": str(out_csv.resolve()),
            "summary_json": str((out_dir / f"{args.split}_candidate_pool_v3_summary.json").resolve()),
        },
    }
    write_json(out_dir / f"{args.split}_candidate_pool_v3_summary.json", summary)
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
