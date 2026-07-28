#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

import numpy as np
import pandas as pd


ELEMENT_RE = __import__("re").compile(r"([A-Z][a-z]?)")
COMMON_OX = {
    "Li": [1], "Na": [1], "K": [1], "Rb": [1], "Cs": [1],
    "Mg": [2], "Ca": [2], "Sr": [2], "Ba": [2], "Zn": [2], "Cd": [2],
    "Al": [3], "Ga": [3], "In": [3], "Sc": [3], "Y": [3],
    "Ti": [4, 3], "Zr": [4], "Hf": [4], "V": [5, 4, 3], "Nb": [5, 4, 3], "Ta": [5, 4, 3],
    "Cr": [6, 3, 4], "Mo": [6, 4, 3], "W": [6, 4, 3],
    "Mn": [2, 3, 4], "Fe": [2, 3], "Co": [2, 3], "Ni": [2, 3], "Cu": [1, 2],
    "Bi": [3, 5], "Sb": [3, 5], "Sn": [2, 4], "Pb": [2, 4],
}
RARE_EARTH = {"La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"}
IGNORED = {"H", "O", "C", "N"}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    return {k: arr[k] for k in arr.files}


def get_y(pack: Dict[str, np.ndarray]) -> np.ndarray:
    for k in ["y_multi_hot", "y", "labels", "targets"]:
        if k in pack:
            return (np.asarray(pack[k]) > 0).astype(np.int8)
    raise KeyError(k)


def parse_list(s: Any) -> List[str]:
    try:
        obj = json.loads(str(s))
        if isinstance(obj, list):
            return [str(x) for x in obj]
    except Exception:
        pass
    return []


def elements(s: str) -> Set[str]:
    return set(ELEMENT_RE.findall(str(s)))


def target_elements(formula: str) -> List[str]:
    e = sorted(elements(formula) - {"O"})
    return e or sorted(elements(formula))


def patch_label(label: str, patch: Dict[str, str]) -> str:
    return patch.get(str(label), str(label))


def patched_set(labels: Iterable[str], patch: Dict[str, str]) -> List[str]:
    return sorted({patch_label(x, patch) for x in labels if str(x)})


def set_metrics(true_set: Set[str], pred_set: Set[str]) -> Dict[str, Any]:
    inter = len(true_set & pred_set)
    union = len(true_set | pred_set)
    precision = inter / len(pred_set) if pred_set else 0.0
    recall = inter / len(true_set) if true_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "jaccard": inter / union if union else 1.0, "exact": pred_set == true_set}


def oxidation_states(e: str) -> List[int]:
    if e in COMMON_OX:
        return COMMON_OX[e]
    if e in RARE_EARTH:
        return [3]
    return [1, 2, 3]


def charge_balanced_formulas(e: str, fam: str) -> List[tuple[str, int]]:
    out = []
    for ox in oxidation_states(e):
        if fam == "oxide":
            out += {1: [(f"{e}2O", ox)], 2: [(f"{e}O", ox)], 3: [(f"{e}2O3", ox)], 4: [(f"{e}O2", ox), (f"{e}3O4", ox)], 5: [(f"{e}2O5", ox)], 6: [(f"{e}O3", ox)]}.get(ox, [])
        elif fam == "carbonate":
            out += {1: [(f"{e}2CO3", ox)], 2: [(f"{e}CO3", ox)], 3: [(f"{e}2(CO3)3", ox)]}.get(ox, [])
        elif fam == "nitrate":
            out += {1: [(f"{e}NO3", ox)], 2: [(f"{e}(NO3)2", ox)], 3: [(f"{e}(NO3)3", ox)]}.get(ox, [])
        elif fam == "hydroxide":
            out += {1: [(f"{e}OH", ox)], 2: [(f"{e}(OH)2", ox)], 3: [(f"{e}(OH)3", ox)]}.get(ox, [])
        elif fam == "acetate":
            out += {1: [(f"{e}(CH3COO)", ox)], 2: [(f"{e}(CH3COO)2", ox)], 3: [(f"{e}(CH3COO)3", ox)]}.get(ox, [])
        elif fam == "sulfate":
            out += {1: [(f"{e}2SO4", ox)], 2: [(f"{e}SO4", ox)], 3: [(f"{e}2(SO4)3", ox)]}.get(ox, [])
        elif fam == "phosphate":
            out += {1: [(f"{e}3PO4", ox)], 2: [(f"{e}3(PO4)2", ox)], 3: [(f"{e}PO4", ox)]}.get(ox, [])
        elif fam == "halide":
            for x in ["F", "Cl", "Br", "I"]:
                out += {1: [(f"{e}{x}", ox)], 2: [(f"{e}{x}2", ox)], 3: [(f"{e}{x}3", ox)], 4: [(f"{e}{x}4", ox)]}.get(ox, [])
        elif fam == "elemental":
            out.append((e, ox))
    seen = set()
    uniq = []
    for f, ox in out:
        if f not in seen:
            seen.add(f); uniq.append((f, ox))
    return uniq


def family_predictions(pred: pd.DataFrame, sample_index: int) -> Dict[str, List[tuple[str, float]]]:
    rows = pred[pred.sample_index == sample_index]
    fam_cols = [c for c in pred.columns if c.startswith("prob_family__")]
    out = {}
    for _, r in rows.iterrows():
        vals = [(c.replace("prob_family__", ""), float(r[c])) for c in fam_cols]
        vals.sort(key=lambda x: x[1], reverse=True)
        out[str(r.target_element)] = vals[:4]
    return out


def score_candidate(labels: Sequence[str], formula: str, base: float, family_score: float = 0.0, ox_score: float = 0.0, source_prior: float = 0.0) -> Dict[str, Any]:
    target = set(target_elements(formula))
    present = set().union(*(elements(x) - IGNORED for x in labels)) if labels else set()
    coverage = len(target & present) / len(target) if target else 1.0
    missing = len(target - present)
    extra = len(present - target)
    score = base + 1.4 * coverage + 0.8 * family_score + 0.5 * ox_score + 0.25 * source_prior - 1.2 * missing - 0.75 * extra - 0.08 * max(0, len(labels) - 3)
    return {"calib_base_score": float(score), "element_coverage": coverage, "missing_element_count": missing, "extra_element_count": extra, "family_score": family_score, "oxidation_state_score": ox_score, "candidate_source_prior": source_prior}


def y_to_sets(y: np.ndarray, names: Sequence[str]) -> List[Set[str]]:
    return [{str(names[j]) for j in np.where(y[i] > 0)[0]} for i in range(y.shape[0])]


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Stage2 candidate pool v4 with alias patch and oxidation-state generation.")
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--base_candidate_csv", required=True)
    ap.add_argument("--patch_csv", required=True)
    ap.add_argument("--family_predictions", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--split", choices=["val", "test", "train"], default="test")
    ap.add_argument("--top_n", type=int, default=500)
    ap.add_argument("--per_element_family_n", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_dir)
    names = [str(x) for x in load_json(dataset_dir / "precursor_names.json")]
    y = get_y(load_npz(dataset_dir / f"{args.split}.npz"))
    meta = pd.read_csv(dataset_dir / f"{args.split}_meta.csv")
    true_sets = y_to_sets(y, names)
    patch_df = pd.read_csv(args.patch_csv)
    patch = dict(zip(patch_df.raw_label.astype(str), patch_df.patched_label.astype(str)))
    base = pd.read_csv(args.base_candidate_csv)
    fam_pred = pd.read_csv(args.family_predictions)
    groups = {int(i): g for i, g in base.groupby("sample_index", sort=False)}
    all_rows = []
    for i in range(len(meta)):
        formula = str(meta.loc[i, "formula"])
        true = true_sets[i]
        store: Dict[frozenset, Dict[str, Any]] = {}
        group = groups.get(i, pd.DataFrame())
        for _, r in group.iterrows():
            labels = patched_set(parse_list(r.pred_precursors), patch)
            base_score = float(r.get("total_score", r.get("score", 0.0)))
            met = score_candidate(labels, formula, base_score, float(r.get("family_score", 0.0)), float(r.get("oxidation_state_score", 0.0)), 1.0 if "seen" in str(r.get("candidate_source_mix", "")) else 0.0)
            key = frozenset(labels)
            if key and (key not in store or met["calib_base_score"] > store[key]["calib_base_score"]):
                store[key] = {"labels": labels, "candidate_source_mix": str(r.get("candidate_source_mix", "")) + "+patched_base", "open_vocab_count": int(r.get("open_vocab_count", 0)), "generated_precursor_count": int(r.get("generated_precursor_count", 0)), "mlp_score": float(r.get("mlp_score", 0.0) if not pd.isna(r.get("mlp_score", 0.0)) else 0.0), "retrieval_score": float(r.get("retrieval_score", 0.0) if not pd.isna(r.get("retrieval_score", 0.0)) else 0.0), **met}
        # Oxidation-state generated candidates.
        fp = family_predictions(fam_pred, i)
        per_elem = []
        label_meta = {}
        for e in target_elements(formula):
            labs = []
            for fam, fs in fp.get(e, [])[:4]:
                for f, ox in charge_balanced_formulas(e, fam)[: int(args.per_element_family_n)]:
                    labs.append(f)
                    label_meta[f] = {"family_score": fs, "oxidation_state": ox}
            if labs:
                per_elem.append(list(dict.fromkeys(labs))[:12])
        for n, combo in enumerate(itertools.product(*per_elem), start=1) if per_elem else []:
            labels = patched_set(combo, patch)
            fs = float(np.mean([label_meta.get(x, {}).get("family_score", 0.0) for x in combo])) if combo else 0.0
            oxs = [label_meta.get(x, {}).get("oxidation_state", 0) for x in combo]
            ox_score = 1.0 if all(oxs) else 0.0
            met = score_candidate(labels, formula, 0.0, fs, ox_score, 0.2)
            key = frozenset(labels)
            if key and (key not in store or met["calib_base_score"] > store[key]["calib_base_score"]):
                store[key] = {"labels": labels, "candidate_source_mix": "generated_oxidation_state", "open_vocab_count": len(labels), "generated_precursor_count": len(labels), "mlp_score": 0.0, "retrieval_score": 0.0, **met}
            if n >= 2500:
                break
        rows = list(store.values())
        rows.sort(key=lambda x: x["calib_base_score"], reverse=True)
        for rank, cand in enumerate(rows[: int(args.top_n)], start=1):
            pred = set(cand["labels"])
            sm = set_metrics(true, pred)
            all_rows.append({"sample_index": i, "rank": rank, "id": meta.loc[i, "id"], "formula": formula, "reaction_method": meta.loc[i, "reaction_method"], "true_precursors": json.dumps(sorted(true), ensure_ascii=False), "pred_precursors": json.dumps(cand.pop("labels"), ensure_ascii=False), **cand, **sm})
        if (i + 1) % 500 == 0:
            print(f"[Progress] {args.split} {i+1}/{len(meta)}", flush=True)
    df = pd.DataFrame(all_rows).sort_values(["sample_index", "calib_base_score"], ascending=[True, False])
    df["rank"] = df.groupby("sample_index").cumcount() + 1
    out_csv = out_dir / f"{args.split}_candidate_sets.csv"
    df.to_csv(out_csv, index=False)
    metrics = {}
    for k in [1,3,5,10,20,50,100,200,500]:
        if k > int(args.top_n): continue
        sub = df[df["rank"] <= k]; g = sub.groupby("sample_index")
        metrics[f"top{k}_exact"] = float(g.exact.any().mean())
        metrics[f"top{k}_best_f1"] = float(g.f1.max().mean())
        metrics[f"top{k}_best_jaccard"] = float(g.jaccard.max().mean())
    summary = {"config": vars(args), "data": {"n_rows": int(len(meta)), "n_candidates": int(len(df))}, "metrics": metrics, "artifacts": {"candidate_csv": str(out_csv.resolve())}}
    write_json(out_dir / f"{args.split}_candidate_pool_v4_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
