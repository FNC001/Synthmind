#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set

import numpy as np
import pandas as pd


FEATURES = [
    "original_v4_score",
    "total_score_v5",
    "method_template_score",
    "family_score",
    "mlp_score",
    "retrieval_score",
    "set_size_score",
    "cooccurrence_score",
    "open_vocab_score",
    "oov_risk_score",
    "assembly_score",
    "missing_element_count",
    "extra_element_count",
    "method_prior_score",
    "v3_reranker_score",
    "v3_total_score",
    "external_score_present",
    "template_source_flag",
    "repair_source_flag",
    "train_assembly_source_flag",
    "solid_state_flag",
    "solution_flag",
    "melt_arc_flag",
    "solid_template_score",
    "solution_template_score",
    "melt_template_score",
]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def parse_list(text: Any) -> list[str]:
    try:
        obj = json.loads(str(text))
        if isinstance(obj, list):
            return [str(x) for x in obj]
    except Exception:
        pass
    return []


def load_patch(path: str) -> Dict[str, str]:
    p = Path(path)
    if not path or not p.exists():
        return {}
    df = pd.read_csv(p)
    if not {"raw_label", "patched_label"}.issubset(df.columns):
        return {}
    return dict(zip(df["raw_label"].astype(str), df["patched_label"].astype(str)))


def set_key(sample_index: Any, labels_text: Any, patch: Dict[str, str]) -> str:
    labels = sorted({patch.get(x, x) for x in parse_list(labels_text) if x})
    return f"{int(sample_index)}\t{json.dumps(labels, ensure_ascii=False, separators=(',', ':'))}"


def add_external_scores(df: pd.DataFrame, external_csv: str, patch: Dict[str, str]) -> pd.DataFrame:
    df["v3_reranker_score"] = 0.0
    df["v3_total_score"] = 0.0
    df["external_score_present"] = 0.0
    if not external_csv:
        return df
    p = Path(external_csv)
    if not p.exists():
        return df
    ext = pd.read_csv(p)
    if not {"sample_index", "pred_precursors"}.issubset(ext.columns):
        return df
    for col in ["rerank_score", "total_score"]:
        if col not in ext.columns:
            ext[col] = 0.0
        ext[col] = pd.to_numeric(ext[col], errors="coerce").fillna(0.0)
    ext["_key"] = [set_key(i, pset, patch) for i, pset in zip(ext["sample_index"], ext["pred_precursors"])]
    ext = ext.sort_values("rerank_score", ascending=False).drop_duplicates("_key")
    r_lookup = ext.set_index("_key")["rerank_score"].to_dict()
    t_lookup = ext.set_index("_key")["total_score"].to_dict()
    sample_col = "original_sample_index" if "original_sample_index" in df.columns else "sample_index"
    keys = [set_key(i, pset, patch) for i, pset in zip(df[sample_col], df["pred_precursors"])]
    df["v3_reranker_score"] = [float(r_lookup.get(k, 0.0)) for k in keys]
    df["v3_total_score"] = [float(t_lookup.get(k, 0.0)) for k in keys]
    df["external_score_present"] = [1.0 if k in r_lookup else 0.0 for k in keys]
    return df


def load_train_labels(dataset_dir: str, patch: Dict[str, str]) -> Set[str]:
    root = Path(dataset_dir)
    if not root.exists():
        return set()
    names = json.loads((root / "precursor_names.json").read_text(encoding="utf-8"))
    arr = np.load(root / "train.npz", allow_pickle=True)
    y = arr["y_multi_hot"] if "y_multi_hot" in arr.files else arr[arr.files[0]]
    return {patch.get(str(names[int(j)]), str(names[int(j)])) for j in np.where(y.sum(axis=0) > 0)[0]}


def load_candidates(path: Path, external_csv: str, patch: Dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = add_external_scores(df, external_csv, patch)
    df["exact"] = df["exact"].astype(str).str.lower().isin(["true", "1", "yes"])
    for col in ["f1", "jaccard"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    src = df.get("candidate_source", pd.Series([""] * len(df))).astype(str)
    method = df["reaction_method"].astype(str)
    df["template_source_flag"] = src.str.contains("method_template").astype(float)
    df["repair_source_flag"] = src.str.contains("assembly_repair").astype(float)
    df["train_assembly_source_flag"] = src.str.contains("train_label_assembly").astype(float)
    df["solid_state_flag"] = method.eq("solid_state").astype(float)
    df["solution_flag"] = method.eq("solution").astype(float)
    df["melt_arc_flag"] = method.eq("melt_arc").astype(float)
    mts = pd.to_numeric(df.get("method_template_score", 0.0), errors="coerce").fillna(0.0)
    df["solid_template_score"] = mts * df["solid_state_flag"]
    df["solution_template_score"] = mts * df["solution_flag"]
    df["melt_template_score"] = mts * df["melt_arc_flag"]
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def oov_ids(df: pd.DataFrame, train_labels: Set[str]) -> Set[int]:
    ids = set()
    if not train_labels:
        return ids
    for _, r in df.drop_duplicates("sample_index").iterrows():
        true = set(parse_list(r["true_precursors"]))
        if any(x not in train_labels for x in true):
            ids.add(int(r["sample_index"]))
    return ids


def score_array(df: pd.DataFrame, weights: Dict[str, float]) -> np.ndarray:
    out = np.zeros(len(df), dtype=np.float64)
    for col in FEATURES:
        out += float(weights.get(col, 0.0)) * df[col].to_numpy(np.float64)
    return out


def ranked_frame(df: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    keep = [
        "sample_index", "original_sample_index", "id", "formula", "reaction_method",
        "true_precursors", "pred_precursors", "candidate_source", "candidate_source_mix", "exact", "f1", "jaccard",
    ]
    out = df[[c for c in keep if c in df.columns]].copy()
    out["core_calibrated_score"] = score
    out = out.sort_values(["sample_index", "core_calibrated_score"], ascending=[True, False], kind="mergesort")
    out["core_calibrated_rank"] = out.groupby("sample_index", sort=False).cumcount() + 1
    return out


def metrics(ranked: pd.DataFrame, subset_ids: Optional[Set[int]] = None, prefix: str = "") -> Dict[str, float]:
    if subset_ids is not None:
        ranked = ranked[ranked["sample_index"].isin(subset_ids)]
    out: Dict[str, float] = {}
    n = ranked["sample_index"].nunique()
    for k in [1, 3, 5, 10, 50, 100, 200, 500, 1000]:
        sub = ranked[ranked["core_calibrated_rank"] <= k]
        if sub.empty:
            continue
        g = sub.groupby("sample_index", sort=False)
        out[f"{prefix}top{k}_exact"] = float(g["exact"].any().sum() / max(n, 1))
        out[f"{prefix}top{k}_best_f1"] = float(g["f1"].max().mean())
        out[f"{prefix}top{k}_best_jaccard"] = float(g["jaccard"].max().mean())
    top1 = ranked[ranked["core_calibrated_rank"] == 1]
    out[f"{prefix}top1_f1"] = float(top1["f1"].mean()) if len(top1) else 0.0
    out[f"{prefix}top1_jaccard"] = float(top1["jaccard"].mean()) if len(top1) else 0.0
    return out


def all_metrics(ranked: pd.DataFrame, oov: Set[int]) -> Dict[str, float]:
    out = metrics(ranked)
    if oov:
        out.update(metrics(ranked, oov, "oov_"))
        non = set(ranked["sample_index"].unique()) - oov
        out.update(metrics(ranked, non, "non_oov_"))
    return out


def objective(m: Dict[str, float]) -> float:
    return 0.35 * m.get("top1_exact", 0.0) + 0.30 * m.get("top10_exact", 0.0) + 0.20 * m.get("top200_exact", 0.0) + 0.15 * m.get("top500_exact", 0.0)


def trial_weights(rng: np.random.Generator, n_trials: int) -> Iterable[Dict[str, float]]:
    base = {k: 0.0 for k in FEATURES}
    base["total_score_v5"] = 1.0
    yield base
    mix = dict(base)
    mix.update({"v3_reranker_score": 1.8, "v3_total_score": 0.55, "template_source_flag": 0.18, "repair_source_flag": -0.5, "open_vocab_score": -0.2})
    yield mix
    for _ in range(max(0, n_trials - 2)):
        yield {
            "original_v4_score": float(rng.uniform(0.0, 0.7)),
            "total_score_v5": float(rng.uniform(0.3, 1.3)),
            "method_template_score": float(rng.uniform(0.0, 0.8)),
            "family_score": float(rng.uniform(0.0, 0.8)),
            "mlp_score": float(rng.uniform(0.0, 0.4)),
            "retrieval_score": float(rng.uniform(0.0, 0.7)),
            "set_size_score": float(rng.uniform(0.0, 0.8)),
            "cooccurrence_score": float(rng.uniform(0.0, 0.6)),
            "open_vocab_score": float(rng.uniform(-0.4, 0.05)),
            "oov_risk_score": float(rng.uniform(-0.15, 0.25)),
            "assembly_score": float(rng.uniform(0.0, 0.5)),
            "missing_element_count": float(rng.uniform(-1.6, -0.3)),
            "extra_element_count": float(rng.uniform(-1.0, -0.05)),
            "method_prior_score": float(rng.uniform(0.0, 0.9)),
            "v3_reranker_score": float(rng.uniform(0.0, 2.3)),
            "v3_total_score": float(rng.uniform(0.0, 0.9)),
            "external_score_present": float(rng.uniform(-0.2, 0.4)),
            "template_source_flag": float(rng.uniform(-0.1, 0.5)),
            "repair_source_flag": float(rng.uniform(-0.8, -0.05)),
            "train_assembly_source_flag": float(rng.uniform(-0.25, 0.2)),
            "solid_state_flag": float(rng.uniform(-0.15, 0.15)),
            "solution_flag": float(rng.uniform(-0.15, 0.15)),
            "melt_arc_flag": float(rng.uniform(-0.15, 0.15)),
            "solid_template_score": float(rng.uniform(0.0, 0.4)),
            "solution_template_score": float(rng.uniform(0.0, 0.4)),
            "melt_template_score": float(rng.uniform(0.0, 0.5)),
        }


def by_method(ranked: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, sub in ranked.groupby("reaction_method", dropna=False):
        m = metrics(sub)
        rows.append({
            "reaction_method": method,
            "n_samples": int(sub["sample_index"].nunique()),
            "top1_exact": m.get("top1_exact", 0.0),
            "top10_exact": m.get("top10_exact", 0.0),
            "top200_exact": m.get("top200_exact", 0.0),
            "top500_exact": m.get("top500_exact", 0.0),
            "top1000_exact": m.get("top1000_exact", 0.0),
        })
    return pd.DataFrame(rows).sort_values("n_samples", ascending=False)


def search_best_weights(val: pd.DataFrame, val_oov: Set[int], n_trials: int, seed: int, label: str) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    best: Optional[Dict[str, Any]] = None
    records = []
    for i, weights in enumerate(trial_weights(rng, int(n_trials)), start=1):
        ranked = ranked_frame(val, score_array(val, weights))
        met = all_metrics(ranked, val_oov)
        obj = objective(met)
        rec = {"trial": i, "label": label, "objective": obj, **weights, **met}
        records.append(rec)
        if best is None or obj > best["objective"]:
            best = {"trial": i, "label": label, "objective": obj, "weights": weights, "metrics": met}
        if i % 10 == 0:
            print(f"[Search {label}] {i}/{n_trials} best={best['objective']:.6f}", flush=True)
    assert best is not None
    best["records"] = records
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate Stage2 core-method candidate scores on core val and evaluate core test once.")
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--patch_csv", default="")
    ap.add_argument("--val_external_rerank_csv", default="")
    ap.add_argument("--test_external_rerank_csv", default="")
    ap.add_argument("--n_trials", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260611)
    ap.add_argument("--per_method", action="store_true", help="Fit separate calibration weights for each core reaction_method.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    patch = load_patch(args.patch_csv)
    train_labels = load_train_labels(args.dataset_dir, patch)
    val = load_candidates(Path(args.val_csv), args.val_external_rerank_csv, patch)
    test = load_candidates(Path(args.test_csv), args.test_external_rerank_csv, patch)
    val_oov = oov_ids(val, train_labels)
    test_oov = oov_ids(test, train_labels)
    if args.per_method:
        all_records = []
        best_by_method = {}
        ranked_parts = []
        for offset, method in enumerate(["solid_state", "solution", "melt_arc"], start=1):
            val_m = val[val["reaction_method"].astype(str) == method].copy()
            test_m = test[test["reaction_method"].astype(str) == method].copy()
            method_oov = {i for i in val_oov if i in set(val_m["sample_index"].unique())}
            best_m = search_best_weights(val_m, method_oov, int(args.n_trials), int(args.seed) + offset * 1000, method)
            all_records.extend(best_m.pop("records"))
            best_by_method[method] = best_m
            ranked_parts.append(ranked_frame(test_m, score_array(test_m, best_m["weights"])))
        best = {"mode": "per_method", "by_method": best_by_method}
        pd.DataFrame(all_records).sort_values(["label", "objective"], ascending=[True, False]).to_csv(out_dir / "val_weight_search.csv", index=False)
        write_json(out_dir / "best_weights.json", best)
        test_ranked = pd.concat(ranked_parts, ignore_index=True)
    else:
        best_global = search_best_weights(val, val_oov, int(args.n_trials), int(args.seed), "global")
        records = best_global.pop("records")
        best = best_global
        pd.DataFrame(records).sort_values("objective", ascending=False).to_csv(out_dir / "val_weight_search.csv", index=False)
        write_json(out_dir / "best_weights.json", best)
        test_ranked = ranked_frame(test, score_array(test, best["weights"]))
    test_ranked.to_csv(out_dir / "test_core_candidate_sets_calibrated.csv", index=False)
    method_df = by_method(test_ranked)
    method_df.to_csv(out_dir / "test_core_by_reaction_method.csv", index=False)
    test_metrics = all_metrics(test_ranked, test_oov)
    summary = {
        "config": vars(args),
        "n_val_oov_rows": int(len(val_oov)),
        "n_test_oov_rows": int(len(test_oov)),
        "best_val": best,
        "test_metrics": test_metrics,
        "by_reaction_method": method_df.to_dict(orient="records"),
        "artifacts": {
            "search": str(out_dir / "val_weight_search.csv"),
            "best_weights": str(out_dir / "best_weights.json"),
            "test_calibrated": str(out_dir / "test_core_candidate_sets_calibrated.csv"),
            "metrics": str(out_dir / "test_core_calibrated_metrics.json"),
        },
    }
    write_json(out_dir / "test_core_calibrated_metrics.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
