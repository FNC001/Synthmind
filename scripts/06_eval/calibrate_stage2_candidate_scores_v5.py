#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set

import numpy as np
import pandas as pd


BASE_FEATURES = [
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
    keys = [set_key(i, pset, patch) for i, pset in zip(df["sample_index"], df["pred_precursors"])]
    df["v3_reranker_score"] = [float(r_lookup.get(k, 0.0)) for k in keys]
    df["v3_total_score"] = [float(t_lookup.get(k, 0.0)) for k in keys]
    df["external_score_present"] = [1.0 if k in r_lookup else 0.0 for k in keys]
    return df


def load_train_labels(dataset_dir: str, patch: Dict[str, str]) -> Set[str]:
    if not dataset_dir:
        return set()
    root = Path(dataset_dir)
    names = json.loads((root / "precursor_names.json").read_text(encoding="utf-8"))
    arr = np.load(root / "train.npz", allow_pickle=True)
    y = arr["y_multi_hot"] if "y_multi_hot" in arr.files else arr[arr.files[0]]
    pos = np.where(y.sum(axis=0) > 0)[0]
    return {patch.get(str(names[int(j)]), str(names[int(j)])) for j in pos}


def load_candidate_csv(path: Path, external_csv: str, patch: Dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = add_external_scores(df, external_csv, patch)
    df["exact"] = df["exact"].astype(str).str.lower().isin(["true", "1", "yes"])
    for col in ["f1", "jaccard"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    src = df.get("candidate_source", pd.Series([""] * len(df))).astype(str)
    df["template_source_flag"] = src.eq("method_template").astype(float)
    df["repair_source_flag"] = src.str.startswith("assembly_repair").astype(float)
    df["train_assembly_source_flag"] = src.eq("train_label_assembly").astype(float)
    for col in BASE_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def oov_ids_from_candidates(df: pd.DataFrame, train_labels: Set[str]) -> Set[int]:
    if not train_labels:
        return set()
    ids = set()
    first = df.drop_duplicates("sample_index")
    for _, r in first.iterrows():
        true = set(parse_list(r["true_precursors"]))
        if any(x not in train_labels for x in true):
            ids.add(int(r["sample_index"]))
    return ids


def score_array(df: pd.DataFrame, weights: Dict[str, float]) -> np.ndarray:
    score = np.zeros(len(df), dtype=np.float64)
    for col in BASE_FEATURES:
        score += float(weights.get(col, 0.0)) * df[col].to_numpy(dtype=np.float64)
    return score


def ranked_frame(df: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    keep = [
        "sample_index", "id", "formula", "reaction_method", "true_precursors", "pred_precursors",
        "candidate_source", "candidate_source_mix", "exact", "f1", "jaccard",
    ]
    out = df[[c for c in keep if c in df.columns]].copy()
    out["calibrated_score_v5"] = score
    out = out.sort_values(["sample_index", "calibrated_score_v5"], ascending=[True, False], kind="mergesort")
    out["calibrated_rank_v5"] = out.groupby("sample_index", sort=False).cumcount() + 1
    return out


def metrics_from_ranked(ranked: pd.DataFrame, oov_ids: Optional[Set[int]] = None, prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    n = ranked["sample_index"].nunique()
    for k in [1, 3, 5, 10, 50, 100, 200, 500]:
        sub = ranked[ranked["calibrated_rank_v5"] <= k]
        g = sub.groupby("sample_index", sort=False)
        out[f"{prefix}top{k}_exact"] = float(g["exact"].any().sum() / max(n, 1)) if len(g) else 0.0
        out[f"{prefix}top{k}_best_f1"] = float(g["f1"].max().mean()) if len(g) else 0.0
        out[f"{prefix}top{k}_best_jaccard"] = float(g["jaccard"].max().mean()) if len(g) else 0.0
    top1 = ranked[ranked["calibrated_rank_v5"] == 1]
    out[f"{prefix}top1_f1"] = float(top1["f1"].mean()) if len(top1) else 0.0
    out[f"{prefix}top1_jaccard"] = float(top1["jaccard"].mean()) if len(top1) else 0.0
    if oov_ids:
        oov = ranked[ranked["sample_index"].isin(oov_ids)]
        non = ranked[~ranked["sample_index"].isin(oov_ids)]
        if len(oov):
            out.update(metrics_from_ranked(oov, None, f"{prefix}oov_"))
        if len(non):
            out.update(metrics_from_ranked(non, None, f"{prefix}non_oov_"))
    return out


def objective(m: Dict[str, float]) -> float:
    return (
        0.30 * m.get("top1_exact", 0.0)
        + 0.25 * m.get("top10_exact", 0.0)
        + 0.20 * m.get("top200_exact", 0.0)
        + 0.15 * m.get("top500_exact", 0.0)
        + 0.10 * m.get("oov_top500_exact", m.get("top500_exact", 0.0))
    )


def trial_weights(rng: np.random.Generator, n_trials: int) -> Iterable[Dict[str, float]]:
    base = {k: 0.0 for k in BASE_FEATURES}
    base["total_score_v5"] = 1.0
    yield base
    ext = dict(base)
    ext.update({"v3_reranker_score": 1.5, "v3_total_score": 0.55, "template_source_flag": 0.15, "repair_source_flag": -0.25})
    yield ext
    for _ in range(max(0, n_trials - 2)):
        yield {
            "original_v4_score": float(rng.uniform(0.0, 0.8)),
            "total_score_v5": float(rng.uniform(0.4, 1.4)),
            "method_template_score": float(rng.uniform(0.0, 0.9)),
            "family_score": float(rng.uniform(0.0, 0.8)),
            "mlp_score": float(rng.uniform(0.0, 0.5)),
            "retrieval_score": float(rng.uniform(0.0, 0.6)),
            "set_size_score": float(rng.uniform(0.0, 0.7)),
            "cooccurrence_score": float(rng.uniform(0.0, 0.8)),
            "open_vocab_score": float(rng.uniform(-0.2, 0.2)),
            "oov_risk_score": float(rng.uniform(0.0, 0.5)),
            "assembly_score": float(rng.uniform(0.0, 0.6)),
            "missing_element_count": float(rng.uniform(-1.5, -0.2)),
            "extra_element_count": float(rng.uniform(-0.9, -0.05)),
            "method_prior_score": float(rng.uniform(0.0, 0.8)),
            "v3_reranker_score": float(rng.uniform(0.0, 2.0)),
            "v3_total_score": float(rng.uniform(0.0, 0.9)),
            "external_score_present": float(rng.uniform(-0.2, 0.4)),
            "template_source_flag": float(rng.uniform(-0.15, 0.45)),
            "repair_source_flag": float(rng.uniform(-0.6, 0.05)),
            "train_assembly_source_flag": float(rng.uniform(-0.25, 0.25)),
        }


def by_group_metrics(ranked: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for key, g in ranked.groupby(group_col, dropna=False):
        m = metrics_from_ranked(g)
        rows.append({
            group_col: key,
            "n_samples": int(g["sample_index"].nunique()),
            "top1_exact": m.get("top1_exact", 0.0),
            "top10_exact": m.get("top10_exact", 0.0),
            "top200_exact": m.get("top200_exact", 0.0),
            "top500_exact": m.get("top500_exact", 0.0),
        })
    return pd.DataFrame(rows).sort_values("n_samples", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate Stage2 v5 candidate scores on val and evaluate once on test.")
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--dataset_dir", default="")
    ap.add_argument("--patch_csv", default="")
    ap.add_argument("--val_external_rerank_csv", default="")
    ap.add_argument("--test_external_rerank_csv", default="")
    ap.add_argument("--n_trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260610)
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    patch = load_patch(args.patch_csv)
    train_labels = load_train_labels(args.dataset_dir, patch)
    val = load_candidate_csv(Path(args.val_csv), args.val_external_rerank_csv, patch)
    test = load_candidate_csv(Path(args.test_csv), args.test_external_rerank_csv, patch)
    val_oov = oov_ids_from_candidates(val, train_labels)
    test_oov = oov_ids_from_candidates(test, train_labels)
    rng = np.random.default_rng(args.seed)
    records = []
    best: Optional[Dict[str, Any]] = None
    for i, weights in enumerate(trial_weights(rng, int(args.n_trials)), start=1):
        ranked = ranked_frame(val, score_array(val, weights))
        met = metrics_from_ranked(ranked, val_oov)
        obj = objective(met)
        rec = {"trial": i, "objective": obj, **weights, **met}
        records.append(rec)
        if best is None or obj > best["objective"]:
            best = {"trial": i, "objective": obj, "weights": weights, "metrics": met}
        if i % 10 == 0:
            print(f"[Search] {i}/{args.n_trials} best={best['objective']:.6f}", flush=True)
    assert best is not None
    pd.DataFrame(records).sort_values("objective", ascending=False).to_csv(out_dir / "val_weight_search.csv", index=False)
    write_json(out_dir / "best_weights.json", best)
    test_ranked = ranked_frame(test, score_array(test, best["weights"]))
    test_ranked.to_csv(out_dir / "test_candidate_sets_calibrated_v5.csv", index=False)
    test_metrics = metrics_from_ranked(test_ranked, test_oov)
    by_group_metrics(test_ranked, "reaction_method").to_csv(out_dir / "test_by_reaction_method.csv", index=False)
    by_group_metrics(test_ranked, "candidate_source").to_csv(out_dir / "test_by_candidate_source.csv", index=False)
    summary = {
        "config": vars(args),
        "n_val_oov_rows": int(len(val_oov)),
        "n_test_oov_rows": int(len(test_oov)),
        "best_val": best,
        "test_metrics": test_metrics,
        "artifacts": {
            "val_weight_search": str(out_dir / "val_weight_search.csv"),
            "best_weights": str(out_dir / "best_weights.json"),
            "test_calibrated_csv": str(out_dir / "test_candidate_sets_calibrated_v5.csv"),
            "test_metrics": str(out_dir / "test_calibrated_metrics.json"),
        },
    }
    write_json(out_dir / "test_calibrated_metrics.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
