#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None


ELEMENT_RE = re.compile(r"([A-Z][a-z]?)")
LIGHT_OR_COUNTER = {"H", "O", "C", "N", "F", "Cl", "Br", "I", "S", "P"}
COMMON_OXIDES = {
    "Li": "Li2O", "Na": "Na2O", "K": "K2O", "Mg": "MgO", "Ca": "CaO", "Sr": "SrO", "Ba": "BaO",
    "Al": "Al2O3", "Ti": "TiO2", "Zr": "ZrO2", "Hf": "HfO2", "V": "V2O5", "Nb": "Nb2O5",
    "Ta": "Ta2O5", "Cr": "Cr2O3", "Mo": "MoO3", "W": "WO3", "Mn": "MnO2", "Fe": "Fe2O3",
    "Co": "Co3O4", "Ni": "NiO", "Cu": "CuO", "Zn": "ZnO", "Y": "Y2O3", "La": "La2O3",
    "Ce": "CeO2", "Pr": "Pr6O11", "Nd": "Nd2O3", "Sm": "Sm2O3", "Gd": "Gd2O3", "Bi": "Bi2O3",
    "Si": "SiO2", "Sn": "SnO2", "Pb": "PbO", "B": "B2O3", "P": "P2O5",
}


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_builtin(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def parse_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [str(x) for x in obj if str(x).strip()]
    except Exception:
        pass
    return [x.strip() for x in re.split(r"\s*\+\s*|;", text) if x.strip()]


def dump_list(items: Sequence[str]) -> str:
    return json.dumps([str(x) for x in items if str(x).strip()], ensure_ascii=False)


def elements(text: str) -> Set[str]:
    return set(ELEMENT_RE.findall(str(text)))


def target_source_elements(formula: str) -> Set[str]:
    elems = elements(formula) - {"O"}
    return elems or elements(formula)


def label_source_elements(label: str) -> Set[str]:
    return elements(label) - {"H", "O", "C", "N"}


def extra_forbidden(labels: Iterable[str], target: Set[str]) -> Set[str]:
    out: Set[str] = set()
    for lab in labels:
        out |= elements(str(lab)) - target - LIGHT_OR_COUNTER
    return out


def generated_formula(el: str, method: str, target: Set[str]) -> str:
    if method in {"melt_arc", "flux_molten_salt"} and "O" not in target:
        return el
    if el in {"Li", "Na", "K"}:
        return f"{el}2CO3" if "O" in target else el
    if el in {"Mg", "Ca", "Sr", "Ba"}:
        return f"{el}CO3" if "O" in target else el
    return COMMON_OXIDES.get(el, el)


def repair_set(labels: Sequence[str], formula: str, method: str) -> Tuple[List[str], List[str], Set[str], Set[str]]:
    target = target_source_elements(formula)
    kept: List[str] = []
    sources: List[str] = []
    covered: Set[str] = set()
    for lab in labels:
        src = elements(lab) & target
        if not src or extra_forbidden([lab], target):
            continue
        if src <= covered:
            continue
        kept.append(str(lab))
        sources.append("oof_retrieval_kept")
        covered |= src
    for el in sorted(target - covered):
        kept.append(generated_formula(el, method, target))
        sources.append("oof_repair_generated")
    missing = target - (set().union(*(elements(p) & target for p in kept)) if kept else set())
    extra = extra_forbidden(kept, target)
    return kept, sources, missing, extra


def set_metrics(true_labels: Sequence[str], pred_labels: Sequence[str]) -> Dict[str, float]:
    t = set(str(x) for x in true_labels if str(x).strip())
    p = set(str(x) for x in pred_labels if str(x).strip())
    inter = len(t & p)
    precision = inter / len(p) if p else 0.0
    recall = inter / len(t) if t else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = len(t | p)
    return {
        "precision": precision,
        "recall": recall,
        "precursor_f1_if_eval": f1,
        "precursor_jaccard_if_eval": inter / union if union else 1.0,
        "precursor_exact_if_eval": float(t == p),
    }


def fold_id(sample_id: str, n_folds: int) -> int:
    h = hashlib.md5(str(sample_id).encode("utf-8")).hexdigest()
    return int(h[:8], 16) % int(n_folds)


def feature_cols(df: pd.DataFrame, max_cols: int) -> List[str]:
    preferred = [
        c for c in df.columns
        if c.startswith("feat_frac_el__")
        or c in {
            "feat_n_elements_formula", "feat_total_atoms_formula", "feat_stoich_entropy", "feat_z_mean",
            "feat_z_std", "feat_frac_tm", "feat_frac_alkali", "feat_frac_alkaline", "feat_density",
            "feat_volume", "feat_nsites", "feat_nelements", "feat_spacegroup_number",
        }
        or c.startswith("feat_crystal_system__")
    ]
    return preferred[:max_cols]


def make_templates(train_other: pd.DataFrame) -> Dict[str, List[Tuple[Tuple[str, ...], int]]]:
    grouped: Dict[str, Counter] = defaultdict(Counter)
    for _, row in train_other.iterrows():
        labels = tuple(sorted(set(parse_list(row["true_precursor_set"]))))
        if labels:
            grouped[str(row["reaction_method"])][labels] += 1
            grouped["__all__"][labels] += 1
    return {k: v.most_common(300) for k, v in grouped.items()}


def coverage(labels: Sequence[str], formula: str) -> Tuple[float, int, int]:
    target = target_source_elements(formula)
    present: Set[str] = set()
    for lab in labels:
        present |= elements(lab) & target
    return (
        len(target & present) / len(target) if target else 1.0,
        len(target - present),
        len(extra_forbidden(labels, target)),
    )


def add_candidate(
    store: Dict[Tuple[str, ...], Dict[str, Any]],
    labels: Sequence[str],
    row: pd.Series,
    source: str,
    base_score: float,
    true_labels: Sequence[str],
    fold: int,
) -> None:
    formula = str(row["formula"])
    method = str(row["reaction_method"])
    repaired, sources, missing, extra = repair_set(labels, formula, method)
    if not repaired:
        return
    key = tuple(sorted(set(repaired)))
    cov, missing_n, extra_n = coverage(repaired, formula)
    m = set_metrics(true_labels, repaired)
    size_penalty = max(0, len(repaired) - 4) * 0.10
    score = float(base_score + 2.0 * cov - 1.2 * missing_n - 0.6 * extra_n - size_penalty)
    if key not in store or score > store[key]["precursor_score"]:
        mix = list(sources)
        if source not in mix:
            mix.append(source)
        store[key] = {
            "sample_id": row["sample_id"],
            "fold_id": fold,
            "formula": formula,
            "reaction_method": method,
            "true_precursors": dump_list(true_labels),
            "precursor_set": dump_list(key),
            "pred_precursors": dump_list(key),
            "candidate_set": dump_list(key),
            "precursor_score": score,
            "calibrated_score": score,
            "precursor_source_mix": "|".join(sorted(set(mix))),
            "contains_open_generated_precursor": int(any("generated" in x for x in mix)),
            "contains_repair_precursor": int(any("repair" in x for x in mix)),
            "chemistry_check_status": "ok" if missing_n == 0 and extra_n == 0 else "failed",
            "missing_source_elements": dump_list(sorted(missing)),
            "extra_forbidden_elements": dump_list(sorted(extra)),
            "element_coverage": cov,
            "missing_element_count": missing_n,
            "extra_element_count": extra_n,
            "candidate_size": len(key),
            **m,
        }


def summarize(df: pd.DataFrame) -> Dict[str, Any]:
    g = df.groupby("sample_id", sort=False)
    top1 = df[df["precursor_rank"] <= 1].groupby("sample_id", sort=False)["precursor_exact_if_eval"].max().mean()
    top10 = df[df["precursor_rank"] <= 10].groupby("sample_id", sort=False)["precursor_exact_if_eval"].max().mean()
    top20 = g["precursor_exact_if_eval"].max().mean()
    return {
        "rows": int(len(df)),
        "samples": int(df["sample_id"].nunique()),
        "top1_exact": float(top1),
        "top10_exact": float(top10),
        "top20_exact": float(top20),
        "mean_f1_top1": float(df[df["precursor_rank"] <= 1]["precursor_f1_if_eval"].mean()),
        "mean_jaccard_top1": float(df[df["precursor_rank"] <= 1]["precursor_jaccard_if_eval"].mean()),
        "mean_best_f1_top20": float(g["precursor_f1_if_eval"].max().mean()),
        "mean_best_jaccard_top20": float(g["precursor_jaccard_if_eval"].max().mean()),
        "chemistry_ok_rate": float((df["chemistry_check_status"] == "ok").mean()),
        "open_generated_rate": float(pd.to_numeric(df["contains_open_generated_precursor"], errors="coerce").fillna(0).mean()),
        "repair_rate": float(pd.to_numeric(df["contains_repair_precursor"], errors="coerce").fillna(0).mean()),
    }


def rerank_oof(df: pd.DataFrame, n_folds: int, seed: int) -> pd.DataFrame:
    out = df.copy()
    num_cols = [
        "precursor_score", "element_coverage", "missing_element_count", "extra_element_count",
        "candidate_size", "contains_open_generated_precursor", "contains_repair_precursor",
    ]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cats = pd.get_dummies(out[["reaction_method", "precursor_source_mix"]].astype(str), dtype=np.float32)
    x = pd.concat([out[num_cols].reset_index(drop=True), cats.reset_index(drop=True)], axis=1)
    out["oof_exact_probability"] = 0.0
    out["oof_f1_prediction"] = 0.0
    for fold in range(n_folds):
        train_mask = out["fold_id"].astype(int) != fold
        hold_mask = ~train_mask
        y_exact = pd.to_numeric(out.loc[train_mask, "precursor_exact_if_eval"], errors="coerce").fillna(0).astype(int)
        y_f1 = pd.to_numeric(out.loc[train_mask, "precursor_f1_if_eval"], errors="coerce").fillna(0.0)
        if lgb is not None and y_exact.nunique() > 1:
            clf = lgb.LGBMClassifier(
                objective="binary", n_estimators=180, learning_rate=0.05, num_leaves=31,
                min_child_samples=25, subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
                random_state=seed + fold, n_jobs=4, verbose=-1,
            )
            reg = lgb.LGBMRegressor(
                objective="regression", n_estimators=160, learning_rate=0.05, num_leaves=31,
                min_child_samples=25, subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
                random_state=seed + 100 + fold, n_jobs=4, verbose=-1,
            )
            clf.fit(x.loc[train_mask].to_numpy(np.float32), y_exact)
            reg.fit(x.loc[train_mask].to_numpy(np.float32), y_f1)
            out.loc[hold_mask, "oof_exact_probability"] = clf.predict_proba(x.loc[hold_mask].to_numpy(np.float32))[:, 1]
            out.loc[hold_mask, "oof_f1_prediction"] = np.clip(reg.predict(x.loc[hold_mask].to_numpy(np.float32)), 0, 1)
        else:
            base_prob = float(y_exact.mean()) if len(y_exact) else 0.0
            base_f1 = float(y_f1.mean()) if len(y_f1) else 0.0
            out.loc[hold_mask, "oof_exact_probability"] = base_prob
            out.loc[hold_mask, "oof_f1_prediction"] = base_f1
    out["calibrated_score"] = (
        4.0 * pd.to_numeric(out["oof_exact_probability"], errors="coerce").fillna(0.0)
        + 1.5 * pd.to_numeric(out["oof_f1_prediction"], errors="coerce").fillna(0.0)
        + 0.05 * pd.to_numeric(out["precursor_score"], errors="coerce").fillna(0.0)
        - 0.10 * pd.to_numeric(out["contains_open_generated_precursor"], errors="coerce").fillna(0.0)
        - 0.05 * pd.to_numeric(out["contains_repair_precursor"], errors="coerce").fillna(0.0)
    )
    out = out.sort_values(["sample_id", "calibrated_score", "precursor_score"], ascending=[True, False, False], kind="mergesort")
    out["precursor_rank"] = out.groupby("sample_id", sort=False).cumcount() + 1
    return out[out["precursor_rank"] <= 20].copy().reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate fold-safe Stage2 train OOF top20 precursor candidates v4.")
    ap.add_argument("--train_csv", default="data/interim/generative/stage3_condition_targets_v3_20260610/train.csv")
    ap.add_argument("--output_dir", default="outputs/evaluation/stage2_train_oof_top20_candidates_v4_20260612")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--n_neighbors", type=int, default=35)
    ap.add_argument("--max_feature_cols", type=int, default=96)
    ap.add_argument("--seed", type=int, default=20260612)
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.train_csv)
    train["fold_id"] = [fold_id(x, args.n_folds) for x in train["sample_id"].astype(str)]
    fcols = feature_cols(train, args.max_feature_cols)
    x_all = train[fcols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    rows: List[Dict[str, Any]] = []

    for fold in range(args.n_folds):
        hold = train[train["fold_id"] == fold].copy()
        other = train[train["fold_id"] != fold].copy()
        scaler = StandardScaler().fit(x_all[other.index])
        x_other = scaler.transform(x_all[other.index])
        x_hold = scaler.transform(x_all[hold.index])
        templates = make_templates(other)
        method_indices: Dict[str, Tuple[np.ndarray, NearestNeighbors]] = {}
        for method, idx_values in other.groupby("reaction_method", sort=False).groups.items():
            pos = np.array([other.index.get_loc(i) for i in idx_values], dtype=int)
            if len(pos) < 2:
                continue
            nn = NearestNeighbors(n_neighbors=min(args.n_neighbors, len(pos)), metric="euclidean")
            nn.fit(x_other[pos])
            method_indices[str(method)] = (np.asarray(list(idx_values), dtype=int), nn)

        for local_i, (_, row) in enumerate(hold.iterrows()):
            method = str(row["reaction_method"])
            true_labels = parse_list(row["true_precursor_set"])
            store: Dict[Tuple[str, ...], Dict[str, Any]] = {}
            if method in method_indices:
                idx_values, nn = method_indices[method]
                dists, nbrs = nn.kneighbors(x_hold[local_i:local_i + 1])
                for rank, (dist, nbr_local) in enumerate(zip(dists[0], nbrs[0]), start=1):
                    src_row = train.loc[int(idx_values[int(nbr_local)])]
                    labels = parse_list(src_row["true_precursor_set"])
                    sim = 1.0 / (1.0 + float(dist))
                    add_candidate(store, labels, row, "oof_retrieval", 2.0 * sim + 0.05 * (args.n_neighbors - rank), true_labels, fold)
            for labels, freq in templates.get(method, [])[:120] + templates.get("__all__", [])[:80]:
                cov, missing_n, extra_n = coverage(labels, str(row["formula"]))
                if cov <= 0:
                    continue
                add_candidate(store, labels, row, "oof_method_template", math.log1p(freq) + cov - 0.3 * missing_n - 0.2 * extra_n, true_labels, fold)
                if len(store) >= args.top_k * 5:
                    break
            target = target_source_elements(str(row["formula"]))
            generated = [generated_formula(el, method, target) for el in sorted(target)]
            add_candidate(store, generated, row, "oof_rule_generated", 0.4, true_labels, fold)
            cand = sorted(store.values(), key=lambda r: r["precursor_score"], reverse=True)[: args.top_k]
            for rank, rec in enumerate(cand, start=1):
                rec["precursor_rank"] = rank
                rows.append(rec)

    out = pd.DataFrame(rows)
    pre_rerank_summary = summarize(out.sort_values(["sample_id", "precursor_rank"], kind="mergesort"))
    out = rerank_oof(out, args.n_folds, args.seed)
    csv_path = outdir / "train_oof_top20_precursor_candidates.csv"
    out.to_csv(csv_path, index=False)
    try:
        out.to_parquet(outdir / "train_oof_top20_precursor_candidates.parquet", index=False)
    except Exception as exc:
        (outdir / "train_oof_top20_precursor_candidates.parquet.SKIPPED.txt").write_text(str(exc), encoding="utf-8")
    summary = {
        "config": vars(args),
        "generation_method": "fold-safe retrieval/template/rule approximation; neural Stage2 scores are not fully OOF",
        "feature_cols": fcols,
        "pre_rerank_metrics": pre_rerank_summary,
        "metrics": summarize(out),
        "fold_counts": {str(k): int(v) for k, v in train["fold_id"].value_counts().sort_index().items()},
    }
    write_json(outdir / "train_oof_generation_summary.json", summary)
    report = [
        "# Stage2 Train OOF Top20 Precursor Candidates v4",
        "",
        "Generation method: fold-safe approximation. Retrieval/template/cooccurrence sources exclude the target fold. No sample uses its own true precursor set as a direct candidate source. Existing global neural Stage2 scores are not retrained per fold and therefore are not used as fully OOF neural scores.",
        "",
        "```json",
        json.dumps(to_builtin(summary), ensure_ascii=False, indent=2),
        "```",
    ]
    (outdir / "train_oof_generation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
