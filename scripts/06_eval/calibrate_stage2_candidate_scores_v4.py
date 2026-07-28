#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set

import numpy as np
import pandas as pd


FEATURES = [
    "calib_base_score",
    "element_coverage",
    "family_score",
    "oxidation_state_score",
    "candidate_source_prior",
    "mlp_score",
    "retrieval_score",
    "external_rerank_score",
    "external_total_score",
    "external_score_present",
    "open_vocab_count",
    "generated_precursor_count",
    "missing_element_count",
    "extra_element_count",
]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def parse_list(s: Any) -> list[str]:
    try:
        obj = json.loads(str(s))
        if isinstance(obj, list):
            return [str(x) for x in obj]
    except Exception:
        pass
    return []


def load_patch(path: str) -> Dict[str, str]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if not {"raw_label", "patched_label"}.issubset(df.columns):
        return {}
    return dict(zip(df["raw_label"].astype(str), df["patched_label"].astype(str)))


def set_key(sample_index: Any, labels_text: Any, patch: Dict[str, str]) -> str:
    labels = sorted({patch.get(x, x) for x in parse_list(labels_text) if x})
    return f"{int(sample_index)}\t{json.dumps(labels, ensure_ascii=False, separators=(',', ':'))}"


def add_external_scores(df: pd.DataFrame, external_csv: str, patch: Dict[str, str]) -> pd.DataFrame:
    df["external_rerank_score"] = 0.0
    df["external_total_score"] = 0.0
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
    lookup_rerank = ext.set_index("_key")["rerank_score"].to_dict()
    lookup_total = ext.set_index("_key")["total_score"].to_dict()
    keys = [set_key(i, pset, patch) for i, pset in zip(df["sample_index"], df["pred_precursors"])]
    df["external_rerank_score"] = [float(lookup_rerank.get(k, 0.0)) for k in keys]
    df["external_total_score"] = [float(lookup_total.get(k, 0.0)) for k in keys]
    df["external_score_present"] = [1.0 if k in lookup_rerank else 0.0 for k in keys]
    return df


def load_candidate_csv(path: Path, external_csv: str = "", patch: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = add_external_scores(df, external_csv, patch or {})
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["exact"] = df["exact"].astype(str).str.lower().isin(["true", "1", "yes"])
    for col in ["f1", "jaccard"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def load_oov_ids(path: Optional[str]) -> Optional[Set[int]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "has_oov" in df.columns:
        df = df[df["has_oov"].astype(str).str.lower().isin(["true", "1", "yes"])]
    if "sample_index" not in df.columns:
        return None
    return {int(x) for x in df["sample_index"].dropna().tolist()}


def score_array(df: pd.DataFrame, weights: Dict[str, float]) -> np.ndarray:
    score = np.zeros(len(df), dtype=np.float64)
    for col in FEATURES:
        score += float(weights.get(col, 0.0)) * df[col].to_numpy(dtype=np.float64)
    return score


def ranked_frame(df: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    ranked = df[[
        "sample_index",
        "id",
        "formula",
        "reaction_method",
        "true_precursors",
        "pred_precursors",
        "candidate_source_mix",
        "exact",
        "f1",
        "jaccard",
    ]].copy()
    ranked["calibrated_score"] = score
    ranked = ranked.sort_values(["sample_index", "calibrated_score"], ascending=[True, False], kind="mergesort")
    ranked["calibrated_rank"] = ranked.groupby("sample_index", sort=False).cumcount() + 1
    return ranked


def ndcg_exact_at_10(ranked: pd.DataFrame) -> float:
    top = ranked[ranked["calibrated_rank"] <= 10]
    exact_rows = top[top["exact"]]
    if exact_rows.empty:
        return 0.0
    best_rank = exact_rows.groupby("sample_index")["calibrated_rank"].min()
    gains = 1.0 / np.log2(best_rank.to_numpy(dtype=np.float64) + 1.0)
    n_samples = ranked["sample_index"].nunique()
    return float(gains.sum() / max(n_samples, 1))


def metrics_from_ranked(ranked: pd.DataFrame, oov_ids: Optional[Set[int]] = None, prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    n_samples = ranked["sample_index"].nunique()
    for k in [1, 3, 5, 10, 20, 50, 100, 200, 500]:
        sub = ranked[ranked["calibrated_rank"] <= k]
        if sub.empty:
            continue
        g = sub.groupby("sample_index", sort=False)
        out[f"{prefix}top{k}_exact"] = float(g["exact"].any().sum() / max(n_samples, 1))
        out[f"{prefix}top{k}_best_f1"] = float(g["f1"].max().mean())
        out[f"{prefix}top{k}_best_jaccard"] = float(g["jaccard"].max().mean())
    top1 = ranked[ranked["calibrated_rank"] == 1]
    out[f"{prefix}top1_f1"] = float(top1["f1"].mean()) if not top1.empty else 0.0
    out[f"{prefix}top1_jaccard"] = float(top1["jaccard"].mean()) if not top1.empty else 0.0
    out[f"{prefix}ndcg10_exact"] = ndcg_exact_at_10(ranked)
    if oov_ids:
        oov_ranked = ranked[ranked["sample_index"].isin(oov_ids)]
        non_oov_ranked = ranked[~ranked["sample_index"].isin(oov_ids)]
        if not oov_ranked.empty:
            out.update(metrics_from_ranked(oov_ranked, None, prefix=f"{prefix}oov_"))
        if not non_oov_ranked.empty:
            out.update(metrics_from_ranked(non_oov_ranked, None, prefix=f"{prefix}non_oov_"))
    return out


def objective(metrics: Dict[str, float]) -> float:
    return (
        0.35 * metrics.get("top1_exact", 0.0)
        + 0.25 * metrics.get("top10_exact", 0.0)
        + 0.20 * metrics.get("top200_exact", 0.0)
        + 0.10 * metrics.get("oov_top500_exact", metrics.get("top500_exact", 0.0))
        + 0.10 * metrics.get("ndcg10_exact", 0.0)
    )


def trial_weights(rng: np.random.Generator, n_trials: int) -> Iterable[Dict[str, float]]:
    baseline = {
        "calib_base_score": 1.0,
        "element_coverage": 0.0,
        "family_score": 0.0,
        "oxidation_state_score": 0.0,
        "candidate_source_prior": 0.0,
        "mlp_score": 0.0,
        "retrieval_score": 0.0,
        "open_vocab_count": 0.0,
        "generated_precursor_count": 0.0,
        "missing_element_count": 0.0,
        "extra_element_count": 0.0,
    }
    yield baseline
    hand_tuned = dict(baseline)
    hand_tuned.update({
        "element_coverage": 1.0,
        "family_score": 0.6,
        "oxidation_state_score": 0.3,
        "candidate_source_prior": 0.25,
        "mlp_score": 0.25,
        "retrieval_score": 0.20,
        "open_vocab_count": -0.08,
        "generated_precursor_count": -0.06,
        "missing_element_count": -1.0,
        "extra_element_count": -0.6,
    })
    yield hand_tuned
    for _ in range(max(0, n_trials - 2)):
        yield {
            "calib_base_score": float(rng.uniform(0.4, 1.4)),
            "element_coverage": float(rng.uniform(0.0, 1.4)),
            "family_score": float(rng.uniform(0.0, 1.2)),
            "oxidation_state_score": float(rng.uniform(0.0, 0.8)),
            "candidate_source_prior": float(rng.uniform(0.0, 0.5)),
            "mlp_score": float(rng.uniform(0.0, 0.7)),
            "retrieval_score": float(rng.uniform(0.0, 0.7)),
            "external_rerank_score": float(rng.uniform(0.0, 1.8)),
            "external_total_score": float(rng.uniform(0.0, 0.8)),
            "external_score_present": float(rng.uniform(-0.3, 0.4)),
            "open_vocab_count": float(rng.uniform(-0.25, 0.05)),
            "generated_precursor_count": float(rng.uniform(-0.25, 0.05)),
            "missing_element_count": float(rng.uniform(-1.8, -0.3)),
            "extra_element_count": float(rng.uniform(-1.2, -0.1)),
        }


def by_method_metrics(ranked: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, g in ranked.groupby("reaction_method", dropna=False):
        m = metrics_from_ranked(g, None, prefix="")
        rows.append({
            "reaction_method": method,
            "n_samples": int(g["sample_index"].nunique()),
            "top1_exact": m.get("top1_exact", 0.0),
            "top10_exact": m.get("top10_exact", 0.0),
            "top200_exact": m.get("top200_exact", 0.0),
            "top500_exact": m.get("top500_exact", 0.0),
            "top1_f1": m.get("top1_f1", 0.0),
            "top1_jaccard": m.get("top1_jaccard", 0.0),
        })
    return pd.DataFrame(rows).sort_values(["n_samples", "reaction_method"], ascending=[False, True])


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate Stage2 v4 candidate scores on val and evaluate fixed weights on test.")
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--val_oov_rows", default="")
    ap.add_argument("--test_oov_rows", default="")
    ap.add_argument("--val_external_rerank_csv", default="")
    ap.add_argument("--test_external_rerank_csv", default="")
    ap.add_argument("--patch_csv", default="")
    ap.add_argument("--n_trials", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20260610)
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    patch = load_patch(args.patch_csv)
    val = load_candidate_csv(Path(args.val_csv), args.val_external_rerank_csv, patch)
    test = load_candidate_csv(Path(args.test_csv), args.test_external_rerank_csv, patch)
    val_oov = load_oov_ids(args.val_oov_rows)
    test_oov = load_oov_ids(args.test_oov_rows)

    rng = np.random.default_rng(args.seed)
    records = []
    best: Optional[Dict[str, Any]] = None
    for i, weights in enumerate(trial_weights(rng, int(args.n_trials)), start=1):
        ranked = ranked_frame(val, score_array(val, weights))
        met = metrics_from_ranked(ranked, val_oov, prefix="")
        obj = objective(met)
        rec = {"trial": i, "objective": obj, **weights, **met}
        records.append(rec)
        if best is None or obj > best["objective"]:
            best = {"trial": i, "objective": obj, "weights": weights, "metrics": met}
        if i % 10 == 0:
            print(f"[Search] {i}/{args.n_trials} best={best['objective']:.6f}", flush=True)

    if best is None:
        raise RuntimeError("No calibration trial was evaluated.")

    search_df = pd.DataFrame(records).sort_values("objective", ascending=False)
    search_df.to_csv(out_dir / "val_weight_search_v4.csv", index=False)
    write_json(out_dir / "best_calibration_weights_v4.json", best)

    test_ranked = ranked_frame(test, score_array(test, best["weights"]))
    test_metrics = metrics_from_ranked(test_ranked, test_oov, prefix="")
    test_ranked.to_csv(out_dir / "test_candidate_sets_calibrated.csv", index=False)
    by_method_metrics(test_ranked).to_csv(out_dir / "test_by_reaction_method_calibrated.csv", index=False)
    write_json(out_dir / "test_calibrated_metrics_v4.json", {
        "best_val": best,
        "test_metrics": test_metrics,
        "artifacts": {
            "calibrated_test_csv": str((out_dir / "test_candidate_sets_calibrated.csv").resolve()),
            "by_method_csv": str((out_dir / "test_by_reaction_method_calibrated.csv").resolve()),
            "search_csv": str((out_dir / "val_weight_search_v4.csv").resolve()),
        },
    })
    print(json.dumps({"best_val": best, "test_metrics": test_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
