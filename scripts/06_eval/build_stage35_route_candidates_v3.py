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
    chunks = []
    for chunk in pd.read_csv(path, usecols=lambda c: usecols is None or c in usecols, chunksize=250_000, low_memory=False):
        rank_col = next((c for c in rank_cols if c in chunk.columns), None)
        if rank_col is None:
            raise ValueError(f"No rank column in {path}")
        chunks.append(chunk[pd.to_numeric(chunk[rank_col], errors="coerce").fillna(10**9) <= top_k].copy())
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def first_present(df: pd.DataFrame, names: List[str], default: Any = np.nan) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series(default, index=df.index)


def load_precursors(path: Path, top_k: int) -> pd.DataFrame:
    probe = pd.read_csv(path, nrows=1)
    wanted = [
        "sample_index", "id", "sample_id", "formula", "reaction_method", "true_precursors", "pred_precursors", "candidate_set",
        "candidate_source", "candidate_source_mix", "precursor_source_mix", "rank", "calibrated_rank_v5", "total_score_v5",
        "calibrated_score_v5", "calibrated_score", "precursor_rank", "precursor_score", "element_coverage",
        "missing_element_count", "extra_element_count", "candidate_size", "exact", "f1", "jaccard",
        "precursor_exact_if_eval", "precursor_f1_if_eval", "precursor_jaccard_if_eval",
        "contains_open_generated_precursor", "contains_repair_precursor", "chemistry_check_status",
    ]
    df = read_csv_topk(path, ["calibrated_rank_v5", "rank", "precursor_rank"], top_k, [c for c in wanted if c in probe.columns])
    original_cols = set(df.columns)
    if "id" in df.columns and "sample_id" not in df.columns:
        df = df.rename(columns={"id": "sample_id"})
    if "pred_precursors" not in df.columns and "candidate_set" in df.columns:
        df["pred_precursors"] = df["candidate_set"]
    if "precursor_rank" not in df.columns:
        df["precursor_rank"] = pd.to_numeric(df.get("calibrated_rank_v5", df.get("rank")), errors="coerce")
    else:
        df["precursor_rank"] = pd.to_numeric(df["precursor_rank"], errors="coerce")
    if "precursor_score" not in df.columns:
        df["precursor_score"] = pd.to_numeric(
            first_present(df, ["calibrated_score_v5", "calibrated_score", "total_score_v5"], -df["precursor_rank"]),
            errors="coerce",
        )
    else:
        df["precursor_score"] = pd.to_numeric(df["precursor_score"], errors="coerce")
    for c, default in [("exact", False), ("f1", 0.0), ("jaccard", 0.0), ("element_coverage", 0.0), ("missing_element_count", 0.0), ("extra_element_count", 0.0), ("candidate_size", 0.0), ("candidate_source_mix", "")]:
        if c not in df.columns:
            df[c] = default
    missing_chem_cols = {
        c
        for c in ["element_coverage", "missing_element_count", "extra_element_count", "candidate_size"]
        if c not in original_cols
    }
    df = repair_precursor_chem_features(df, force_columns=missing_chem_cols)
    df["precursor_exact_if_eval"] = first_present(df, ["precursor_exact_if_eval", "exact"], False).astype(str).str.lower().isin(["true", "1", "1.0"])
    df["precursor_jaccard_if_eval"] = pd.to_numeric(
        first_present(df, ["precursor_jaccard_if_eval", "jaccard"], 0.0), errors="coerce"
    ).fillna(0.0)
    if "candidate_source_mix" not in df.columns or df["candidate_source_mix"].fillna("").astype(str).eq("").all():
        df["candidate_source_mix"] = first_present(df, ["precursor_source_mix", "candidate_source"], "").astype(str)
    mix = df.get("candidate_source_mix", pd.Series("", index=df.index)).fillna("").astype(str) + " " + df.get("candidate_source", pd.Series("", index=df.index)).fillna("").astype(str)
    lower = mix.str.lower()
    df["contains_open_generated_precursor"] = np.maximum(
        pd.to_numeric(first_present(df, ["contains_open_generated_precursor"], 0), errors="coerce").fillna(0).astype(int),
        lower.str.contains("open_generated|generated|open_vocab", regex=True).astype(int),
    )
    df["contains_repair_precursor"] = np.maximum(
        pd.to_numeric(first_present(df, ["contains_repair_precursor"], 0), errors="coerce").fillna(0).astype(int),
        lower.str.contains("repair|known_vocab_repair", regex=True).astype(int),
    )
    if "chemistry_check_status" not in df.columns:
        df["chemistry_check_status"] = np.where((pd.to_numeric(df["missing_element_count"], errors="coerce").fillna(0) <= 0) & (pd.to_numeric(df["extra_element_count"], errors="coerce").fillna(0) <= 0), "ok", "failed")
    return df


def load_train_precursors_from_predprec(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame({
        "sample_index": df.get("sample_index", np.arange(len(df))),
        "sample_id": df["sample_id"].astype(str),
        "formula": df["formula"].astype(str),
        "reaction_method": df["reaction_method"].astype(str),
        "true_precursors": df["true_precursor_set"].astype(str),
        "pred_precursors": df["predicted_precursor_set_chem_checked"].astype(str),
        "candidate_source": "pseudo_predicted_train",
        "candidate_source_mix": df.get("precursor_source_mix", "").astype(str),
        "precursor_rank": 1,
        "precursor_score": pd.to_numeric(df.get("precursor_confidence_score", 1.0), errors="coerce").fillna(1.0),
        "element_coverage": 1.0,
        "missing_element_count": 0.0,
        "extra_element_count": 0.0,
        "candidate_size": pd.to_numeric(df.get("precursor_set_size", 1), errors="coerce").fillna(1),
        "precursor_exact_if_eval": pd.to_numeric(df.get("precursor_exact", 0), errors="coerce").fillna(0).astype(float) > 0.5,
        "precursor_jaccard_if_eval": pd.to_numeric(df.get("precursor_jaccard_to_true", df.get("precursor_jaccard", 0)), errors="coerce").fillna(0.0),
        "f1": pd.to_numeric(df.get("precursor_f1_to_true", df.get("precursor_f1", 0)), errors="coerce").fillna(0.0),
        "jaccard": pd.to_numeric(df.get("precursor_jaccard_to_true", df.get("precursor_jaccard", 0)), errors="coerce").fillna(0.0),
        "exact": pd.to_numeric(df.get("precursor_exact", 0), errors="coerce").fillna(0.0) > 0.5,
        "contains_open_generated_precursor": pd.to_numeric(df.get("contains_open_generated_precursor", 0), errors="coerce").fillna(0).astype(int),
        "contains_repair_precursor": pd.to_numeric(df.get("contains_repair_precursor", 0), errors="coerce").fillna(0).astype(int),
        "chemistry_check_status": df.get("precursor_check_status", "ok").astype(str),
    })
    return out[out["chemistry_check_status"].eq("ok")].copy()


def load_conditions(path: Path, top_k: int) -> pd.DataFrame:
    cols = [
        "sample_id", "reaction_method", "temperature_c", "temperature_low_c", "temperature_high_c", "time_h",
        "time_low_h", "time_high_h", "atmosphere", "solvent", "condition_source", "condition_rank_calibrated_v3",
        "condition_calibrated_score_v3", "total_score_raw", "temperature_point_score", "temperature_bin_score",
        "time_point_score", "time_bin_score", "atmosphere_probability", "solvent_probability", "retrieval_score",
        "method_template_score", "multimodal_template_score", "condition_prior_score", "precursor_confidence_score",
        "open_generated_penalty", "repair_penalty", "strict_hit_if_eval", "relaxed_hit_if_eval",
        "temp_error", "time_error", "atmosphere_correct", "true_atmosphere", "atmosphere_known_mask",
    ]
    df = read_csv_topk(path, ["condition_rank_calibrated_v3", "condition_rank_raw"], top_k, cols)
    df["condition_rank"] = pd.to_numeric(df.get("condition_rank_calibrated_v3", df.get("condition_rank_raw")), errors="coerce")
    df["condition_score"] = pd.to_numeric(df.get("condition_calibrated_score_v3", df.get("total_score_raw")), errors="coerce")
    return df


def minmax(s: pd.Series) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    lo, hi = vals.quantile(0.01), vals.quantile(0.99)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return vals.fillna(0.0)
    return ((vals - lo) / (hi - lo)).clip(0, 1).fillna(0.0)


def route_metrics(df: pd.DataFrame, rank_col: str) -> Dict[str, float]:
    out = {"n_samples": int(df["sample_id"].nunique()), "n_candidates": int(len(df))}
    for k in [1, 3, 5, 10, 50, 100, 200, 400]:
        sub = df[df[rank_col] <= k]
        g = sub.groupby("sample_id", sort=False)
        if not len(g):
            continue
        out[f"top{k}_strict_route"] = float(g["strict_route_hit_if_eval"].max().mean())
        out[f"top{k}_relaxed_route"] = float(g["relaxed_route_hit_if_eval"].max().mean())
        out[f"top{k}_usable_relaxed_route"] = float(g["usable_relaxed_route_hit_if_eval"].max().mean())
        out[f"top{k}_precursor_exact"] = float(g["precursor_exact_if_eval"].max().mean())
        out[f"top{k}_condition_relaxed"] = float(g["relaxed_condition_hit_if_eval"].max().mean())
    core = df[df["reaction_method"].isin(CORE_METHODS)]
    if len(core):
        cg = core[core[rank_col] <= 10].groupby("sample_id", sort=False)
        out["core_top10_relaxed_route"] = float(cg["relaxed_route_hit_if_eval"].max().mean()) if len(cg) else 0.0
    return out


def build_split(split: str, prec_path: Path, cond_path: Path, outdir: Path, top_precursors: int, top_conditions: int) -> pd.DataFrame:
    p = load_precursors(prec_path, top_precursors)
    c = load_conditions(cond_path, top_conditions)
    p["precursor_score_norm"] = minmax(p["precursor_score"])
    c["condition_score_norm"] = minmax(c["condition_score"])
    df = p.merge(c, on=["sample_id", "reaction_method"], how="inner", suffixes=("_precursor", "_condition"))
    df["route_candidate_id"] = df["sample_id"].astype(str) + "::p" + df["precursor_rank"].astype(int).astype(str) + "::c" + df["condition_rank"].astype(int).astype(str)
    df["precursor_confidence"] = df["precursor_score_norm"] * (1.0 - 0.1 * pd.to_numeric(df.get("missing_element_count", 0), errors="coerce").fillna(0))
    df["condition_confidence"] = df["condition_score_norm"]
    df["precursor_condition_compatibility_score"] = pd.to_numeric(df.get("element_coverage", 0), errors="coerce").fillna(0.0)
    df["reaction_method_prior_score"] = df["reaction_method"].isin(CORE_METHODS).astype(float) * 0.2
    df["contains_open_generated_precursor"] = pd.to_numeric(df.get("contains_open_generated_precursor", 0), errors="coerce").fillna(0).astype(int)
    df["contains_repair_precursor"] = pd.to_numeric(df.get("contains_repair_precursor", 0), errors="coerce").fillna(0).astype(int)
    df["route_total_score_raw"] = (
        1.2 * df["precursor_score_norm"]
        + 1.1 * df["condition_score_norm"]
        + 0.4 * df["precursor_condition_compatibility_score"]
        + 0.2 * df["reaction_method_prior_score"]
        - 0.18 * np.log1p(pd.to_numeric(df["precursor_rank"], errors="coerce").fillna(top_precursors))
        - 0.12 * np.log1p(pd.to_numeric(df["condition_rank"], errors="coerce").fillna(top_conditions))
        - 0.15 * df["contains_open_generated_precursor"]
        - 0.10 * df["contains_repair_precursor"]
        - pd.to_numeric(df.get("open_generated_penalty", 0), errors="coerce").fillna(0.0)
        - pd.to_numeric(df.get("repair_penalty", 0), errors="coerce").fillna(0.0)
    )
    df["strict_condition_hit_if_eval"] = pd.to_numeric(df["strict_hit_if_eval"], errors="coerce").fillna(0).astype(int)
    df["relaxed_condition_hit_if_eval"] = pd.to_numeric(df["relaxed_hit_if_eval"], errors="coerce").fillna(0).astype(int)
    df["strict_route_hit_if_eval"] = (df["precursor_exact_if_eval"] & (df["strict_condition_hit_if_eval"] > 0)).astype(int)
    df["relaxed_route_hit_if_eval"] = (df["precursor_exact_if_eval"] & (df["relaxed_condition_hit_if_eval"] > 0)).astype(int)
    df["usable_relaxed_route_hit_if_eval"] = ((df["precursor_jaccard_if_eval"] >= 0.5) & (df["relaxed_condition_hit_if_eval"] > 0)).astype(int)
    df = df.sort_values(["sample_id", "route_total_score_raw"], ascending=[True, False], kind="mergesort")
    df["route_rank_raw"] = df.groupby("sample_id", sort=False).cumcount() + 1
    df.to_csv(outdir / f"{split}_route_candidates.csv", index=False)
    mm = route_metrics(df, "route_rank_raw")
    write_json(outdir / f"{split}_route_candidate_metrics.json", mm)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Stage35 route candidates v3 from Stage2 v5 precursors and Stage3 v3 calibrated conditions.")
    ap.add_argument("--train_precursors", default="data/interim/generative/stage3_condition_dataset_predprec_oof_v3_20260610/train.csv")
    ap.add_argument("--val_precursors", default="outputs/evaluation/stage2_candidate_pool_v5_20260610/val_candidate_sets_repaired.csv")
    ap.add_argument("--test_precursors", default="outputs/evaluation/stage2_score_calibration_v5_20260610/test_candidate_sets_calibrated_v5.csv")
    ap.add_argument("--condition_dir", default="outputs/evaluation/stage3_condition_calibration_v3_final_20260612")
    ap.add_argument("--output_dir", default="outputs/evaluation/stage35_route_candidates_v3_final_20260612")
    ap.add_argument("--top_precursors", type=int, default=20)
    ap.add_argument("--top_conditions", type=int, default=20)
    ap.add_argument("--splits", default="train,val,test", help="Comma-separated splits to build.")
    ap.add_argument("--train_use_topk_precursors", type=int, default=0, help="Use train_precursors as a ranked top-k candidate file instead of pseudo single-candidate train mode.")
    args = ap.parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    split_to_prec = {"train": Path(args.train_precursors), "val": Path(args.val_precursors), "test": Path(args.test_precursors)}
    for split in [s.strip() for s in str(args.splits).split(",") if s.strip()]:
        pp = split_to_prec[split]
        cp = Path(args.condition_dir) / f"{split}_condition_candidates_calibrated.csv"
        if split == "train" and not int(args.train_use_topk_precursors):
            # The project does not currently have no-leakage Stage2 v5 train top20 candidates.
            # Use the pseudo-predicted chemistry-checked train precursor as a single candidate.
            p = load_train_precursors_from_predprec(pp)
            tmp = outdir / "_tmp_train_precursors.csv"
            p.to_csv(tmp, index=False)
            df = build_split(split, tmp, cp, outdir, 1, args.top_conditions)
            tmp.unlink(missing_ok=True)
        else:
            df = build_split(split, pp, cp, outdir, args.top_precursors, args.top_conditions)
        summaries[split] = route_metrics(df, "route_rank_raw")
    schema = {"config": vars(args), "note": "Train uses one pseudo-predicted no-leakage precursor candidate because Stage2 v5 train top20 candidates are unavailable.", "splits": summaries}
    write_json(outdir / "stage35_route_candidate_v3_summary.json", schema)
    write_json(outdir / "route_candidate_schema.json", {"columns_note": "CSV columns are union of precursor, condition, score, source flag, and evaluation fields.", **schema})
    report = ["# Stage35 Route Candidate v3 Final Build Report", "", json.dumps(to_builtin(schema), ensure_ascii=False, indent=2)]
    (outdir / "route_candidate_build_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(to_builtin(summaries), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
