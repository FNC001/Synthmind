#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from stage35_feature_repair_utils import repair_precursor_chem_features


CORE_METHODS = {"solid_state", "solution", "melt_arc"}
STRICT_ATM_UNKNOWN = {"<UNK_OR_MISSING>", "", "nan", "none", "null"}


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv_topk(path: Path, rank_cols: List[str], top_k: int, usecols: List[str] | None = None) -> pd.DataFrame:
    chunks: List[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=lambda c: usecols is None or c in usecols, chunksize=250_000):
        rank_col = next((c for c in rank_cols if c in chunk.columns), None)
        if rank_col is None:
            raise ValueError(f"No rank column among {rank_cols} in {path}")
        ranks = pd.to_numeric(chunk[rank_col], errors="coerce").fillna(10**9)
        chunks.append(chunk[ranks <= top_k].copy())
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def minmax(s: pd.Series) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    lo, hi = vals.quantile(0.01), vals.quantile(0.99)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return vals.fillna(0.0)
    return ((vals - lo) / (hi - lo)).clip(0, 1).fillna(0.0)


def first_present(df: pd.DataFrame, names: List[str], default: Any = np.nan) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series(default, index=df.index)


def load_precursors(path: Path, top_k: int) -> pd.DataFrame:
    probe = pd.read_csv(path, nrows=1)
    wanted = [
        "sample_index", "sample_id", "id", "fold_id", "formula", "reaction_method", "true_precursors",
        "pred_precursors", "candidate_set", "precursor_set", "candidate_source", "candidate_source_mix",
        "precursor_source_mix", "rank", "calibrated_rank_v5", "precursor_rank", "total_score_v5",
        "calibrated_score_v5", "calibrated_score", "precursor_score", "oof_exact_probability",
        "oof_f1_prediction", "method_template_score", "family_score", "original_v4_score",
        "open_vocab_score", "oov_risk_score", "assembly_score", "set_size_score", "cooccurrence_score",
        "method_prior_score", "mlp_score", "retrieval_score", "element_coverage", "missing_element_count",
        "extra_element_count", "candidate_size", "precision", "recall", "precursor_f1_if_eval",
        "precursor_jaccard_if_eval", "precursor_exact_if_eval", "f1", "jaccard", "exact",
        "contains_open_generated_precursor", "contains_repair_precursor", "chemistry_check_status",
    ]
    df = read_csv_topk(path, ["precursor_rank", "calibrated_rank_v5", "rank"], top_k, [c for c in wanted if c in probe.columns])
    original_cols = set(df.columns)
    if "id" in df.columns and "sample_id" not in df.columns:
        df = df.rename(columns={"id": "sample_id"})
    if "sample_id" not in df.columns:
        raise ValueError(f"Missing sample_id/id in {path}")
    if "pred_precursors" not in df.columns:
        df["pred_precursors"] = first_present(df, ["candidate_set", "precursor_set"], "")
    if "candidate_set" not in df.columns:
        df["candidate_set"] = df["pred_precursors"]
    df["sample_id"] = df["sample_id"].astype(str)
    df["reaction_method"] = df["reaction_method"].astype(str)
    df["precursor_rank"] = pd.to_numeric(first_present(df, ["precursor_rank", "calibrated_rank_v5", "rank"], 999), errors="coerce").fillna(999).astype(int)
    df["precursor_score"] = pd.to_numeric(
        first_present(df, ["calibrated_score_v5", "calibrated_score", "total_score_v5", "precursor_score", "oof_exact_probability"], 0.0),
        errors="coerce",
    ).fillna(0.0)
    df["precursor_exact_if_eval"] = first_present(df, ["precursor_exact_if_eval", "exact"], False).astype(str).str.lower().isin(["true", "1", "1.0"])
    df["precursor_f1_if_eval"] = pd.to_numeric(first_present(df, ["precursor_f1_if_eval", "f1"], 0.0), errors="coerce").fillna(0.0)
    df["precursor_jaccard_if_eval"] = pd.to_numeric(first_present(df, ["precursor_jaccard_if_eval", "jaccard"], 0.0), errors="coerce").fillna(0.0)
    for c, default in [
        ("element_coverage", 0.0), ("missing_element_count", 0.0), ("extra_element_count", 0.0),
        ("candidate_size", 0.0), ("contains_open_generated_precursor", 0), ("contains_repair_precursor", 0),
        ("retrieval_score", 0.0), ("method_template_score", 0.0), ("family_score", 0.0),
        ("cooccurrence_score", 0.0), ("mlp_score", 0.0), ("set_size_score", 0.0),
        ("oof_exact_probability", 0.0), ("oof_f1_prediction", 0.0),
    ]:
        if c not in df.columns:
            df[c] = default
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(default)
    missing_chem_cols = {
        c
        for c in ["element_coverage", "missing_element_count", "extra_element_count", "candidate_size"]
        if c not in original_cols
    }
    df = repair_precursor_chem_features(df, force_columns=missing_chem_cols)
    if "candidate_source_mix" not in df.columns:
        df["candidate_source_mix"] = first_present(df, ["precursor_source_mix", "candidate_source"], "").astype(str)
    if "candidate_source" not in df.columns:
        df["candidate_source"] = first_present(df, ["precursor_source_mix", "candidate_source_mix"], "unknown").astype(str)
    mix = df["candidate_source_mix"].fillna("").astype(str) + " " + df["candidate_source"].fillna("").astype(str)
    lower = mix.str.lower()
    df["contains_open_generated_precursor"] = np.maximum(
        pd.to_numeric(df["contains_open_generated_precursor"], errors="coerce").fillna(0).astype(int),
        lower.str.contains("open_generated|generated|open_vocab", regex=True).astype(int),
    )
    df["contains_repair_precursor"] = np.maximum(
        pd.to_numeric(df["contains_repair_precursor"], errors="coerce").fillna(0).astype(int),
        lower.str.contains("repair|known_vocab_repair", regex=True).astype(int),
    )
    if "chemistry_check_status" not in df.columns:
        ok = (pd.to_numeric(df["missing_element_count"], errors="coerce").fillna(0) <= 0) & (
            pd.to_numeric(df["extra_element_count"], errors="coerce").fillna(0) <= 0
        )
        df["chemistry_check_status"] = np.where(ok, "ok", "failed")
    return df[df["sample_id"].ne("")].copy()


def load_conditions(path: Path, top_k: int) -> pd.DataFrame:
    probe = pd.read_csv(path, nrows=1)
    cols = [
        "sample_id", "condition_candidate_id", "reaction_method", "temperature_c", "temperature_low_c",
        "temperature_high_c", "time_h", "time_low_h", "time_high_h", "atmosphere", "solvent",
        "condition_source", "condition_rank_calibrated_v4", "condition_calibrated_score_v4",
        "condition_rank_missing_aware_v4", "condition_rank_strict_comparable_v4", "total_score_raw",
        "temperature_point_score", "temperature_bin_score", "time_point_score", "time_bin_score",
        "atmosphere_probability", "solvent_probability", "atmosphere_known_probability",
        "solvent_known_probability", "atmosphere_class_probability", "solvent_class_probability",
        "precursor_uncertainty_score", "method_expert_score", "template_score", "multimodal_score",
        "missing_aware_score", "strict_comparable_score", "retrieval_score", "method_template_score",
        "multimodal_template_score", "condition_prior_score", "precursor_confidence_score",
        "open_generated_penalty", "repair_penalty", "true_temperature_c", "true_time_h", "true_atmosphere",
        "atmosphere_known_mask", "strict_hit_if_eval", "relaxed_hit_if_eval", "temp_error", "time_error",
        "atmosphere_correct", "condition_rank_raw",
    ]
    df = read_csv_topk(path, ["condition_rank_calibrated_v4", "condition_rank_missing_aware_v4", "condition_rank_raw"], top_k, [c for c in cols if c in probe.columns])
    df["sample_id"] = df["sample_id"].astype(str)
    df["reaction_method"] = df["reaction_method"].astype(str)
    df["condition_rank"] = pd.to_numeric(first_present(df, ["condition_rank_calibrated_v4", "condition_rank_raw"], 999), errors="coerce").fillna(999).astype(int)
    df["condition_score"] = pd.to_numeric(first_present(df, ["condition_calibrated_score_v4", "total_score_raw"], 0.0), errors="coerce").fillna(0.0)
    for c, default in [
        ("atmosphere_known_probability", 0.0), ("solvent_known_probability", 0.0),
        ("atmosphere_class_probability", 0.0), ("solvent_class_probability", 0.0),
        ("precursor_uncertainty_score", 0.0), ("method_expert_score", 0.0), ("template_score", 0.0),
        ("multimodal_score", 0.0), ("missing_aware_score", 0.0), ("strict_comparable_score", 0.0),
        ("retrieval_score", 0.0), ("method_template_score", 0.0), ("multimodal_template_score", 0.0),
        ("condition_prior_score", 0.0), ("open_generated_penalty", 0.0), ("repair_penalty", 0.0),
    ]:
        if c not in df.columns:
            df[c] = default
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(default)
    return df


def add_eval_flags(df: pd.DataFrame) -> pd.DataFrame:
    df["strict_condition_hit_if_eval"] = pd.to_numeric(df["strict_hit_if_eval"], errors="coerce").fillna(0).astype(int)
    df["relaxed_condition_hit_if_eval"] = pd.to_numeric(df["relaxed_hit_if_eval"], errors="coerce").fillna(0).astype(int)
    known = pd.to_numeric(df.get("atmosphere_known_mask", 0), errors="coerce").fillna(0).astype(int) == 1
    pred_atm = df.get("atmosphere", "").astype(str).str.lower()
    true_atm = df.get("true_atmosphere", "").astype(str).str.lower()
    true_known = ~true_atm.isin(STRICT_ATM_UNKNOWN)
    atm_ok = known & true_known & pred_atm.eq(true_atm)
    temp_err = pd.to_numeric(df.get("temp_error", np.inf), errors="coerce").fillna(np.inf)
    time_err = pd.to_numeric(df.get("time_error", np.inf), errors="coerce").fillna(np.inf)
    df["strict_comparable_condition_strict_hit_if_eval"] = ((temp_err <= 100) & (time_err <= 24) & atm_ok).astype(int)
    df["strict_comparable_condition_relaxed_hit_if_eval"] = ((temp_err <= 200) & (time_err <= 48) & atm_ok).astype(int)
    exact = df["precursor_exact_if_eval"].astype(bool)
    jac = pd.to_numeric(df["precursor_jaccard_if_eval"], errors="coerce").fillna(0.0)
    df["strict_route_hit_if_eval"] = (exact & (df["strict_condition_hit_if_eval"] > 0)).astype(int)
    df["relaxed_route_hit_if_eval"] = (exact & (df["relaxed_condition_hit_if_eval"] > 0)).astype(int)
    df["usable_relaxed_route_hit_if_eval"] = ((jac >= 0.5) & (df["relaxed_condition_hit_if_eval"] > 0)).astype(int)
    df["strict_comparable_strict_route_hit_if_eval"] = (exact & (df["strict_comparable_condition_strict_hit_if_eval"] > 0)).astype(int)
    df["strict_comparable_relaxed_route_hit_if_eval"] = (exact & (df["strict_comparable_condition_relaxed_hit_if_eval"] > 0)).astype(int)
    df["strict_comparable_usable_relaxed_route_hit_if_eval"] = ((jac >= 0.5) & (df["strict_comparable_condition_relaxed_hit_if_eval"] > 0)).astype(int)
    return df


def route_metrics(df: pd.DataFrame, rank_col: str, protocol: str = "missing_aware") -> Dict[str, float]:
    if protocol == "strict_comparable":
        strict_col = "strict_comparable_strict_route_hit_if_eval"
        relaxed_col = "strict_comparable_relaxed_route_hit_if_eval"
        usable_col = "strict_comparable_usable_relaxed_route_hit_if_eval"
    else:
        strict_col = "strict_route_hit_if_eval"
        relaxed_col = "relaxed_route_hit_if_eval"
        usable_col = "usable_relaxed_route_hit_if_eval"
    out: Dict[str, float] = {"n_samples": int(df["sample_id"].nunique()), "n_candidates": int(len(df))}
    for k in [1, 3, 5, 10, 20, 50, 100, 200, 400]:
        sub = df[df[rank_col] <= k]
        g = sub.groupby("sample_id", sort=False)
        if len(g):
            out[f"top{k}_strict_route"] = float(g[strict_col].max().mean())
            out[f"top{k}_relaxed_route"] = float(g[relaxed_col].max().mean())
            out[f"top{k}_usable_relaxed_route"] = float(g[usable_col].max().mean())
            out[f"top{k}_precursor_exact"] = float(g["precursor_exact_if_eval"].max().mean())
            out[f"top{k}_condition_relaxed"] = float(g["relaxed_condition_hit_if_eval"].max().mean())
    for method, sub in df.groupby("reaction_method", sort=False):
        if len(sub):
            g = sub[sub[rank_col] <= 10].groupby("sample_id", sort=False)
            if len(g):
                out[f"method_{method}_top10_relaxed_route"] = float(g[relaxed_col].max().mean())
    core = df[df["reaction_method"].isin(CORE_METHODS)]
    cg = core[core[rank_col] <= 10].groupby("sample_id", sort=False)
    out["core_top10_relaxed_route"] = float(cg[relaxed_col].max().mean()) if len(cg) else 0.0
    return out


def build_split(split: str, prec_path: Path, cond_path: Path, outdir: Path, top_precursors: int, top_conditions: int) -> pd.DataFrame:
    p = load_precursors(prec_path, top_precursors)
    c = load_conditions(cond_path, top_conditions)
    p["precursor_score_norm"] = minmax(p["precursor_score"])
    c["condition_score_norm"] = minmax(c["condition_score"])
    df = p.merge(c, on=["sample_id", "reaction_method"], how="inner", suffixes=("_precursor", "_condition"))
    if df.empty:
        raise ValueError(f"No merged route candidates for split={split}")
    df["route_candidate_id"] = (
        df["sample_id"].astype(str)
        + "::p"
        + pd.to_numeric(df["precursor_rank"], errors="coerce").fillna(999).astype(int).astype(str)
        + "::c"
        + pd.to_numeric(df["condition_rank"], errors="coerce").fillna(999).astype(int).astype(str)
    )
    df["precursor_confidence"] = df["precursor_score_norm"] * (
        1.0 - 0.08 * pd.to_numeric(df.get("missing_element_count", 0), errors="coerce").fillna(0)
    ).clip(lower=0.0)
    df["condition_confidence"] = df["condition_score_norm"]
    df["precursor_condition_compatibility_score"] = pd.to_numeric(df.get("element_coverage", 0), errors="coerce").fillna(0.0)
    df["reaction_method_prior_score"] = df["reaction_method"].isin(CORE_METHODS).astype(float) * 0.2
    df["candidate_source_mix"] = first_present(df, ["candidate_source_mix", "precursor_source_mix"], "").fillna("").astype(str)
    df["contains_open_generated_precursor"] = pd.to_numeric(df.get("contains_open_generated_precursor", 0), errors="coerce").fillna(0).astype(int)
    df["contains_repair_precursor"] = pd.to_numeric(df.get("contains_repair_precursor", 0), errors="coerce").fillna(0).astype(int)
    df["route_total_score_raw"] = (
        1.15 * df["precursor_score_norm"]
        + 1.05 * df["condition_score_norm"]
        + 0.35 * pd.to_numeric(df.get("element_coverage", 0), errors="coerce").fillna(0)
        + 0.18 * pd.to_numeric(df.get("method_expert_score", 0), errors="coerce").fillna(0)
        + 0.12 * pd.to_numeric(df.get("missing_aware_score", 0), errors="coerce").fillna(0)
        + 0.08 * pd.to_numeric(df.get("template_score", 0), errors="coerce").fillna(0)
        + 0.06 * pd.to_numeric(df.get("oof_exact_probability", 0), errors="coerce").fillna(0)
        + 0.04 * pd.to_numeric(df.get("oof_f1_prediction", 0), errors="coerce").fillna(0)
        + 0.15 * df["reaction_method_prior_score"]
        - 0.16 * np.log1p(pd.to_numeric(df["precursor_rank"], errors="coerce").fillna(top_precursors))
        - 0.12 * np.log1p(pd.to_numeric(df["condition_rank"], errors="coerce").fillna(top_conditions))
        - 0.10 * df["contains_open_generated_precursor"]
        - 0.06 * df["contains_repair_precursor"]
        - 0.05 * pd.to_numeric(df.get("precursor_uncertainty_score", 0), errors="coerce").fillna(0)
        - pd.to_numeric(df.get("open_generated_penalty", 0), errors="coerce").fillna(0.0)
        - pd.to_numeric(df.get("repair_penalty", 0), errors="coerce").fillna(0.0)
    )
    df = add_eval_flags(df)
    df = df.sort_values(["sample_id", "route_total_score_raw"], ascending=[True, False], kind="mergesort")
    df["route_rank_raw"] = df.groupby("sample_id", sort=False).cumcount() + 1
    output = outdir / f"{split}_route_candidates.csv"
    df.to_csv(output, index=False)
    metrics = {
        "missing_aware": route_metrics(df, "route_rank_raw", "missing_aware"),
        "strict_comparable": route_metrics(df, "route_rank_raw", "strict_comparable"),
    }
    write_json(outdir / f"{split}_route_candidate_metrics.json", metrics)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Stage35 route candidates v4 from Stage2 OOF/v5 precursors and Stage3 v4 calibrated conditions.")
    ap.add_argument("--train_precursors", default="outputs/evaluation/stage2_train_oof_top20_candidates_v4_20260612/train_oof_top20_precursor_candidates.csv")
    ap.add_argument("--val_precursors", default="outputs/evaluation/stage2_candidate_pool_v5_20260610/val_candidate_sets_repaired.csv")
    ap.add_argument("--test_precursors", default="outputs/evaluation/stage2_score_calibration_v5_20260610/test_candidate_sets_calibrated_v5.csv")
    ap.add_argument("--condition_dir", default="outputs/evaluation/stage3_condition_calibration_v4_20260612")
    ap.add_argument("--output_dir", default="outputs/evaluation/stage35_route_candidates_v4_20260612")
    ap.add_argument("--top_precursors", type=int, default=20)
    ap.add_argument("--top_conditions", type=int, default=20)
    ap.add_argument("--splits", default="train,val,test", help="Comma-separated splits to build.")
    args = ap.parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    split_to_prec = {"train": Path(args.train_precursors), "val": Path(args.val_precursors), "test": Path(args.test_precursors)}
    summaries: Dict[str, Any] = {}
    for split in [s.strip() for s in str(args.splits).split(",") if s.strip()]:
        df = build_split(
            split,
            split_to_prec[split],
            Path(args.condition_dir) / f"{split}_condition_candidates_calibrated.csv",
            outdir,
            int(args.top_precursors),
            int(args.top_conditions),
        )
        summaries[split] = {
            "rows": int(len(df)),
            "samples": int(df["sample_id"].nunique()),
            "missing_aware": route_metrics(df, "route_rank_raw", "missing_aware"),
            "strict_comparable": route_metrics(df, "route_rank_raw", "strict_comparable"),
        }
    schema = {
        "config": vars(args),
        "note": "Train split uses fold-safe Stage2 v4 OOF top20 precursor candidates. Val/test use Stage2 v5 repaired/calibrated candidates. Conditions are Stage3 v4 calibrated candidates.",
        "splits": summaries,
    }
    write_json(outdir / "stage35_route_candidate_v4_summary.json", schema)
    write_json(outdir / "route_candidate_schema.json", {"columns_note": "CSV columns are merged precursor, condition, v4 score, source flag, and evaluation fields.", **schema})
    report = ["# Stage35 Route Candidate v4 Build Report", "", json.dumps(to_builtin(schema), ensure_ascii=False, indent=2)]
    (outdir / "route_candidate_build_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(to_builtin(summaries), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
