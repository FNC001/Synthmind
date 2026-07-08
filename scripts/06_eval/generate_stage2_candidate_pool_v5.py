#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

import numpy as np
import pandas as pd


ELEMENT_RE = re.compile(r"([A-Z][a-z]?)")
IGNORED = {"H", "O", "C", "N"}
WEAK_METHODS = {"hydro_solvothermal", "precipitation", "flux_molten_salt", "other", "solution"}
METHOD_FAMILY_PRIORS = {
    "hydro_solvothermal": {"nitrate", "halide", "acetate", "hydroxide", "oxide", "organic"},
    "precipitation": {"nitrate", "halide", "sulfate", "carbonate", "hydroxide", "organic"},
    "flux_molten_salt": {"halide", "oxide", "carbonate", "phosphate", "elemental"},
    "solid_state": {"oxide", "carbonate", "hydroxide", "nitrate", "elemental"},
    "solution": {"nitrate", "acetate", "halide", "organic", "hydroxide"},
    "other": {"oxide", "halide", "nitrate", "organic", "carbonate", "hydroxide"},
}


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


def patch_set(labels: Iterable[str], patch: Dict[str, str]) -> List[str]:
    return sorted({patch.get(str(x), str(x)) for x in labels if str(x)})


def set_metrics(true_set: Set[str], pred_set: Set[str]) -> Dict[str, Any]:
    inter = len(true_set & pred_set)
    precision = inter / len(pred_set) if pred_set else 0.0
    recall = inter / len(true_set) if true_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = len(true_set | pred_set)
    return {"precision": precision, "recall": recall, "f1": f1, "jaccard": inter / union if union else 1.0, "exact": pred_set == true_set}


def y_true_sets(dataset_dir: Path, split: str, names: Sequence[str], patch: Dict[str, str]) -> List[Set[str]]:
    y = get_y(load_npz(dataset_dir / f"{split}.npz"))
    out = []
    for i in range(y.shape[0]):
        labels = [str(names[j]) for j in np.where(y[i] > 0)[0]]
        out.append(set(patch_set(labels, patch)))
    return out


def label_source_elements(label: str) -> Set[str]:
    return elements(label) - IGNORED


def candidate_elements(labels: Sequence[str]) -> Set[str]:
    out: Set[str] = set()
    for lab in labels:
        out |= label_source_elements(lab)
    return out


def coverage_stats(labels: Sequence[str], formula: str) -> Dict[str, float]:
    target = set(target_elements(formula))
    present = candidate_elements(labels)
    coverage = len(target & present) / len(target) if target else 1.0
    return {
        "element_coverage": coverage,
        "missing_element_count": float(len(target - present)),
        "extra_element_count": float(len(present - target)),
    }


def family_score(labels: Sequence[str], method: str, family_lookup: Dict[str, str]) -> float:
    if not labels:
        return 0.0
    priors = METHOD_FAMILY_PRIORS.get(method, set())
    fams = [family_lookup.get(x, "unknown") for x in labels]
    return sum(1.0 for f in fams if f in priors) / len(fams)


def method_prior_score(method: str, labels: Sequence[str], family_lookup: Dict[str, str]) -> float:
    score = family_score(labels, method, family_lookup)
    if method == "flux_molten_salt" and any((elements(x) & {"Li", "Na", "K", "Rb", "Cs"}) and (elements(x) & {"F", "Cl", "Br", "I"}) for x in labels):
        score += 0.25
    if method in {"hydro_solvothermal", "solution"} and any(family_lookup.get(x, "") in {"nitrate", "halide", "acetate"} for x in labels):
        score += 0.15
    if method == "precipitation" and any(family_lookup.get(x, "") in {"carbonate", "hydroxide", "sulfate"} for x in labels):
        score += 0.15
    return float(min(score, 1.25))


def score_total(features: Dict[str, float], method: str) -> float:
    weak_bonus = 0.25 if method in WEAK_METHODS else 0.0
    return (
        0.75 * features.get("original_v4_score", 0.0)
        + 1.15 * features.get("method_template_score", 0.0)
        + 0.65 * features.get("family_score", 0.0)
        + 0.35 * features.get("open_vocab_score", 0.0)
        + 0.25 * features.get("oov_risk_score", 0.0)
        + 0.60 * features.get("assembly_score", 0.0)
        + 0.35 * features.get("set_size_score", 0.0)
        + 0.40 * features.get("cooccurrence_score", 0.0)
        + 0.45 * features.get("method_prior_score", 0.0)
        + 1.45 * features.get("element_coverage", 0.0)
        + weak_bonus
        - 1.10 * features.get("missing_element_count", 0.0)
        - 0.42 * features.get("extra_element_count", 0.0)
        - 0.05 * max(0.0, features.get("candidate_size", 0.0) - 4.0)
    )


def load_patch(path: str) -> Dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if not {"raw_label", "patched_label"}.issubset(df.columns):
        return {}
    return dict(zip(df["raw_label"].astype(str), df["patched_label"].astype(str)))


def load_templates(path: Path, patch: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
    df = pd.read_csv(path)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for _, r in df.iterrows():
        labels = patch_set(parse_list(r["precursor_labels"]), patch)
        source = set(parse_list(r["source_elements"])) | candidate_elements(labels)
        rec = {
            "template_id": str(r["template_id"]),
            "reaction_method": str(r["reaction_method"]),
            "labels": labels,
            "source_elements": source,
            "target_elements": set(parse_list(r["target_elements"])),
            "precursor_families": set(parse_list(r["precursor_families"])),
            "train_frequency": float(r.get("train_frequency", 0.0)),
            "val_frequency": float(r.get("val_frequency", 0.0)),
            "cooccurrence_score": float(r.get("cooccurrence_score", 0.0)),
            "average_set_size": float(r.get("average_set_size", len(labels))),
        }
        grouped[rec["reaction_method"]].append(rec)
    for method in grouped:
        grouped[method].sort(key=lambda x: (x["train_frequency"], x["val_frequency"], x["cooccurrence_score"]), reverse=True)
    return grouped


def template_score(template: Dict[str, Any], formula: str, method: str) -> float:
    tgt = set(target_elements(formula))
    src = set(template["source_elements"])
    cov = len(tgt & src) / len(tgt) if tgt else 1.0
    extra = len((src - tgt) - {"F", "Cl", "Br", "I"})
    target_j = len(tgt & set(template["target_elements"])) / len(tgt | set(template["target_elements"])) if (tgt | set(template["target_elements"])) else 0.0
    freq = math.log1p(float(template["train_frequency"])) + 0.35 * math.log1p(float(template["val_frequency"]))
    same_method = 1.0 if template["reaction_method"] == method else 0.0
    return 2.2 * cov + 0.7 * target_j + 0.35 * freq + 0.35 * same_method - 0.18 * extra


def build_label_index(dataset_dir: Path, names: Sequence[str], family_lookup: Dict[str, str]) -> Dict[str, Dict[str, List[tuple[str, float]]]]:
    train_y = get_y(load_npz(dataset_dir / "train.npz"))
    train_counts = train_y.sum(axis=0)
    index: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for j, lab in enumerate(names):
        c = float(train_counts[j])
        if c <= 0:
            continue
        fam = family_lookup.get(str(lab), "unknown")
        for e in label_source_elements(str(lab)):
            index[e][fam][str(lab)] += c
    out: Dict[str, Dict[str, List[tuple[str, float]]]] = {}
    for e, fam_map in index.items():
        out[e] = {}
        for fam, cnt in fam_map.items():
            out[e][fam] = [(lab, float(v)) for lab, v in cnt.most_common(12)]
    return out


def add_candidate(store: Dict[frozenset, Dict[str, Any]], labels: Sequence[str], formula: str, method: str, source: str, template_id: str, features: Dict[str, float]) -> None:
    labels = sorted(set(labels))
    if not labels:
        return
    cov = coverage_stats(labels, formula)
    fam = features.get("family_score", 0.0)
    all_feat = {
        "method_template_score": 0.0,
        "family_score": fam,
        "original_v4_score": 0.0,
        "open_vocab_score": 0.0,
        "oov_risk_score": 0.0,
        "assembly_score": 0.0,
        "set_size_score": 0.0,
        "cooccurrence_score": 0.0,
        "method_prior_score": 0.0,
        "mlp_score": 0.0,
        "retrieval_score": 0.0,
        **cov,
        "candidate_size": float(len(labels)),
        **features,
    }
    all_feat["total_score_v5"] = score_total(all_feat, method)
    key = frozenset(labels)
    if key not in store or all_feat["total_score_v5"] > store[key]["total_score_v5"]:
        store[key] = {
            "labels": labels,
            "candidate_source": source,
            "candidate_source_mix": source,
            "method_template_id": template_id,
            **all_feat,
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Stage2 v5 candidate pool with no-leakage method-specific templates.")
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--ontology_csv", required=True)
    ap.add_argument("--template_csv", required=True)
    ap.add_argument("--v4_candidate_csv", required=True)
    ap.add_argument("--patch_csv", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--top_n", type=int, default=500)
    ap.add_argument("--template_limit_weak", type=int, default=280)
    ap.add_argument("--template_limit_other", type=int, default=120)
    ap.add_argument("--assembly_combo_limit", type=int, default=500)
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_dir)
    patch = load_patch(args.patch_csv)
    names = [str(x) for x in load_json(dataset_dir / "precursor_names.json")]
    meta = pd.read_csv(dataset_dir / f"{args.split}_meta.csv")
    true_sets = y_true_sets(dataset_dir, args.split, names, patch)
    ont = pd.read_csv(args.ontology_csv)
    family_lookup = dict(zip(ont["canonical_precursor"].astype(str), ont["precursor_family"].astype(str)))
    templates_by_method = load_templates(Path(args.template_csv), patch)
    label_index = build_label_index(dataset_dir, names, family_lookup)
    v4 = pd.read_csv(args.v4_candidate_csv)
    v4_rank_col = "calibrated_rank" if "calibrated_rank" in v4.columns else "rank"
    groups = {int(i): g.copy() for i, g in v4.groupby("sample_index", sort=False)}
    all_rows = []

    for i in range(len(meta)):
        formula = str(meta.loc[i, "formula"])
        method = str(meta.loc[i, "reaction_method"])
        true = true_sets[i]
        store: Dict[frozenset, Dict[str, Any]] = {}
        group = groups.get(i, pd.DataFrame())
        for _, r in group.iterrows():
            labels = patch_set(parse_list(r["pred_precursors"]), patch)
            orig = float(r.get("calibrated_score", r.get("total_score_v5", r.get("total_score", r.get("calib_base_score", 0.0)))))
            feats = {
                "original_v4_score": orig,
                "family_score": float(r.get("family_score", family_score(labels, method, family_lookup)) if not pd.isna(r.get("family_score", 0.0)) else family_score(labels, method, family_lookup)),
                "open_vocab_score": 0.15 * float(r.get("open_vocab_count", 0.0) if "open_vocab_count" in r else 0.0),
                "oov_risk_score": 0.35 if method in WEAK_METHODS else 0.0,
                "assembly_score": 0.15,
                "set_size_score": 1.0 / (1.0 + abs(len(labels) - max(1, len(target_elements(formula))))),
                "cooccurrence_score": float(r.get("retrieval_score", 0.0) if not pd.isna(r.get("retrieval_score", 0.0)) else 0.0),
                "method_prior_score": method_prior_score(method, labels, family_lookup),
                "mlp_score": float(r.get("mlp_score", 0.0) if not pd.isna(r.get("mlp_score", 0.0)) else 0.0),
                "retrieval_score": float(r.get("retrieval_score", 0.0) if not pd.isna(r.get("retrieval_score", 0.0)) else 0.0),
            }
            add_candidate(store, labels, formula, method, "v4_base", "", feats)

        # Method-specific whole-route templates.
        candidates = templates_by_method.get(method, [])
        limit = args.template_limit_weak if method in WEAK_METHODS else args.template_limit_other
        scored_templates = []
        for tpl in candidates:
            sc = template_score(tpl, formula, method)
            if sc > 1.3:
                scored_templates.append((sc, tpl))
        scored_templates.sort(key=lambda x: x[0], reverse=True)
        for sc, tpl in scored_templates[:limit]:
            labels = tpl["labels"]
            feats = {
                "method_template_score": sc,
                "family_score": family_score(labels, method, family_lookup),
                "open_vocab_score": 0.0,
                "oov_risk_score": 0.65 if method in WEAK_METHODS else 0.15,
                "assembly_score": 0.45,
                "set_size_score": 1.0 / (1.0 + abs(len(labels) - tpl.get("average_set_size", len(labels)))),
                "cooccurrence_score": float(tpl.get("cooccurrence_score", 0.0)),
                "method_prior_score": method_prior_score(method, labels, family_lookup),
            }
            add_candidate(store, labels, formula, method, "method_template", tpl["template_id"], feats)

        # Train-label assembly candidates by element/family priors.
        per_elem: List[List[tuple[str, float]]] = []
        preferred = list(METHOD_FAMILY_PRIORS.get(method, {"oxide", "carbonate", "nitrate", "halide"}))
        for e in target_elements(formula):
            opts: List[tuple[str, float]] = []
            for fam in preferred:
                for lab, cnt in label_index.get(e, {}).get(fam, [])[:4]:
                    opts.append((lab, math.log1p(cnt) + (0.4 if fam in preferred else 0.0)))
            opts = sorted(dict(opts).items(), key=lambda x: x[1], reverse=True)[:5]
            if opts:
                per_elem.append(opts)
        combo_count = 0
        for combo in itertools.product(*per_elem) if per_elem else []:
            labels = [x[0] for x in combo]
            score = float(np.mean([x[1] for x in combo])) if combo else 0.0
            feats = {
                "method_template_score": 0.35,
                "family_score": family_score(labels, method, family_lookup),
                "open_vocab_score": 0.10,
                "oov_risk_score": 0.55 if method in WEAK_METHODS else 0.10,
                "assembly_score": 0.60,
                "set_size_score": 1.0 / (1.0 + abs(len(labels) - len(target_elements(formula)))),
                "cooccurrence_score": score,
                "method_prior_score": method_prior_score(method, labels, family_lookup),
            }
            add_candidate(store, labels, formula, method, "train_label_assembly", "", feats)
            combo_count += 1
            if combo_count >= int(args.assembly_combo_limit):
                break

        rows = list(store.values())
        rows.sort(key=lambda x: x["total_score_v5"], reverse=True)
        for rank, cand in enumerate(rows[: int(args.top_n)], start=1):
            labels = cand.pop("labels")
            pred = set(labels)
            sm = set_metrics(true, pred)
            all_rows.append({
                "sample_index": i,
                "rank": rank,
                "id": meta.loc[i, "id"],
                "formula": formula,
                "reaction_method": method,
                "true_precursors": json.dumps(sorted(true), ensure_ascii=False),
                "pred_precursors": json.dumps(labels, ensure_ascii=False),
                "candidate_set": json.dumps(labels, ensure_ascii=False),
                **cand,
                **sm,
            })
        if (i + 1) % 500 == 0:
            print(f"[Progress] {args.split} {i+1}/{len(meta)}", flush=True)

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["sample_index", "total_score_v5"], ascending=[True, False], kind="mergesort")
    df["rank"] = df.groupby("sample_index", sort=False).cumcount() + 1
    out_csv = out_dir / f"{args.split}_candidate_sets.csv"
    df.to_csv(out_csv, index=False)
    source_metrics = {}
    for src, sub in df.groupby("candidate_source"):
        source_metrics[src] = {
            "n_candidates": int(len(sub)),
            "exact_rows": int(sub["exact"].sum()),
            "unique_samples_with_exact": int(sub[sub["exact"]]["sample_index"].nunique()),
        }
    summary = {
        "config": vars(args),
        "data": {"n_rows": int(len(meta)), "n_candidates": int(len(df))},
        "metrics": metrics_by_k(df),
        "candidate_source_contribution": source_metrics,
        "artifacts": {"candidate_csv": str(out_csv), "summary": str(out_dir / f"{args.split}_candidate_pool_v5_summary.json")},
    }
    write_json(out_dir / f"{args.split}_candidate_pool_v5_summary.json", summary)
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
