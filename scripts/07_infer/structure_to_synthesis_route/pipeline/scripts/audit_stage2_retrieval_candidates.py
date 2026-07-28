#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval_csv", required=True)
    ap.add_argument("--merged_csv", required=True)
    ap.add_argument("--reranked_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--output_md", required=True)
    ap.add_argument("--summary_json", required=True)
    ap.add_argument("--top_n", type=int, default=50)
    return ap.parse_args()


def safe_read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def safe_num(s, default=0.0):
    try:
        return pd.to_numeric(s, errors="coerce").fillna(default)
    except Exception:
        return pd.Series(dtype=float)


def has_col(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns


def retrieval_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)

    if "is_retrieval_candidate" in df.columns:
        s = df["is_retrieval_candidate"].astype(str).str.lower()
        return s.isin(["true", "1", "yes"])

    if "candidate_source" in df.columns:
        return df["candidate_source"].astype(str).str.contains("retrieval", case=False, na=False)

    return pd.Series([False] * len(df), index=df.index)


def summarize_table(df: pd.DataFrame, name: str, top_n: int) -> dict:
    if df.empty:
        return {
            f"{name}_exists": False,
            f"{name}_n_rows": 0,
            f"{name}_n_retrieval_rows": 0,
            f"{name}_retrieval_frac": 0.0,
            f"{name}_topn_retrieval_rows": 0,
            f"{name}_topn_retrieval_frac": 0.0,
        }

    mask = retrieval_mask(df)
    top = df.head(top_n).copy()
    top_mask = retrieval_mask(top)

    out = {
        f"{name}_exists": True,
        f"{name}_n_rows": int(len(df)),
        f"{name}_n_retrieval_rows": int(mask.sum()),
        f"{name}_retrieval_frac": float(mask.mean()) if len(mask) else 0.0,
        f"{name}_topn_retrieval_rows": int(top_mask.sum()),
        f"{name}_topn_retrieval_frac": float(top_mask.mean()) if len(top_mask) else 0.0,
    }

    if "retrieval_similarity" in df.columns:
        sim = pd.to_numeric(df.loc[mask, "retrieval_similarity"], errors="coerce")
        out[f"{name}_retrieval_similarity_mean"] = float(sim.mean()) if sim.notna().any() else None
        out[f"{name}_retrieval_similarity_max"] = float(sim.max()) if sim.notna().any() else None

    if "retrieval_element_coverage" in df.columns:
        cov = pd.to_numeric(df.loc[mask, "retrieval_element_coverage"], errors="coerce")
        out[f"{name}_retrieval_element_coverage_mean"] = float(cov.mean()) if cov.notna().any() else None
        out[f"{name}_retrieval_element_coverage_max"] = float(cov.max()) if cov.notna().any() else None

    if "element_coverage" in df.columns:
        cov2 = pd.to_numeric(df.loc[mask, "element_coverage"], errors="coerce")
        out[f"{name}_retrieval_final_element_coverage_mean"] = float(cov2.mean()) if cov2.notna().any() else None
        out[f"{name}_retrieval_final_element_coverage_max"] = float(cov2.max()) if cov2.notna().any() else None

    if "element_missing" in df.columns:
        miss = df.loc[mask, "element_missing"].fillna("").astype(str).str.strip()
        out[f"{name}_retrieval_rows_with_missing_elements"] = int(miss.ne("").sum())

    if "extra_element_penalty" in df.columns:
        penalty = pd.to_numeric(df.loc[mask, "extra_element_penalty"], errors="coerce").fillna(0.0)
        out[f"{name}_retrieval_rows_with_extra_penalty"] = int((penalty > 0).sum())

    return out


def build_retrieval_rows(df: pd.DataFrame, table_name: str, top_n: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out = df.copy()
    out["_table_name"] = table_name
    out["_row_index"] = range(1, len(out) + 1)
    out["_is_top_n"] = out["_row_index"] <= top_n
    out["_is_retrieval"] = retrieval_mask(out)

    keep = [
        "_table_name",
        "_row_index",
        "_is_top_n",
        "_is_retrieval",
        "candidate_source",
        "precursor_set",
        "source_score",
        "stage2_score",
        "element_rerank_score",
        "source_rank",
        "element_coverage",
        "element_hit",
        "element_missing",
        "extra_element_penalty",
        "is_retrieval_candidate",
        "retrieval_source_formula",
        "retrieval_source_elements",
        "retrieval_similarity",
        "retrieval_element_coverage",
        "retrieval_source_split",
        "retrieval_label_key",
        "retrieval_source_index",
        "retrieval_source_file",
    ]
    keep = [c for c in keep if c in out.columns]
    return out[keep]


def main():
    args = parse_args()

    retrieval = safe_read_csv(args.retrieval_csv)
    merged = safe_read_csv(args.merged_csv)
    reranked = safe_read_csv(args.reranked_csv)

    audit_parts = [
        build_retrieval_rows(retrieval, "retrieval_raw", args.top_n),
        build_retrieval_rows(merged, "merged_pool", args.top_n),
        build_retrieval_rows(reranked, "element_reranked_pool", args.top_n),
    ]
    audit = pd.concat([x for x in audit_parts if len(x)], ignore_index=True) if audit_parts else pd.DataFrame()

    summary = {}
    summary.update(summarize_table(retrieval, "retrieval_raw", args.top_n))
    summary.update(summarize_table(merged, "merged_pool", args.top_n))
    summary.update(summarize_table(reranked, "element_reranked_pool", args.top_n))

    raw_n = summary.get("retrieval_raw_n_retrieval_rows", 0)
    merged_n = summary.get("merged_pool_n_retrieval_rows", 0)
    reranked_n = summary.get("element_reranked_pool_n_retrieval_rows", 0)
    reranked_topn = summary.get("element_reranked_pool_topn_retrieval_rows", 0)

    #if raw_n == 0:
    #    audit_level = "major_warning"
    #    interpretation = "No retrieval candidates were produced."
    #elif merged_n == 0:
    #    audit_level = "major_warning"
    #    interpretation = "Retrieval candidates were produced but did not survive into the merged Stage2 pool."
    #elif reranked_n == 0:
    #    audit_level = "major_warning"
    #    interpretation = "Retrieval candidates entered the merged pool but did not survive element reranking."
    #elif reranked_topn == 0:
    #    audit_level = "minor_warning"
    #    interpretation = "Retrieval candidates survive in the reranked pool, but none appear in the top-N candidates."
    #else:
    #    audit_level = "pass"
    #    interpretation = "Retrieval candidates are present and contribute to the top-N reranked Stage2 pool."
    topn_frac = float(summary.get("element_reranked_pool_topn_retrieval_frac", 0.0) or 0.0)
    topn_rows = int(summary.get("element_reranked_pool_topn_retrieval_rows", 0) or 0)

    reranked_frac = float(summary.get("element_reranked_pool_retrieval_frac", 0.0) or 0.0)
    merged_frac = float(summary.get("merged_pool_retrieval_frac", 0.0) or 0.0)

    raw_similarity_mean = summary.get("retrieval_raw_retrieval_similarity_mean", None)
    raw_similarity_max = summary.get("retrieval_raw_retrieval_similarity_max", None)
    raw_cov_mean = summary.get("retrieval_raw_retrieval_element_coverage_mean", None)
    raw_cov_max = summary.get("retrieval_raw_retrieval_element_coverage_max", None)

    retrieval_cov_mean = summary.get("element_reranked_pool_retrieval_final_element_coverage_mean", None)
    retrieval_cov_max = summary.get("element_reranked_pool_retrieval_final_element_coverage_max", None)
    retrieval_missing_n = int(summary.get("element_reranked_pool_retrieval_rows_with_missing_elements", 0) or 0)
    retrieval_extra_n = int(summary.get("element_reranked_pool_retrieval_rows_with_extra_penalty", 0) or 0)

    weak_reasons = []

    # Retrieval exists but only weakly contributes to the final top-N pool.
    if topn_rows > 0 and topn_frac < 0.10:
        weak_reasons.append("low_topn_retrieval_fraction")

    # Retrieval candidates are present, but chemically incomplete after element reranking.
    if retrieval_cov_mean is not None and float(retrieval_cov_mean) < 0.999:
        weak_reasons.append("retrieval_element_coverage_below_full")

    if retrieval_missing_n > 0:
        weak_reasons.append("retrieval_has_missing_target_elements")

    if retrieval_extra_n > 0:
        weak_reasons.append("retrieval_has_extra_element_penalty")

    # Retrieval raw similarity/coverage is weak before reranking.
    if raw_similarity_max is not None and float(raw_similarity_max) < 0.50:
        weak_reasons.append("low_raw_retrieval_similarity")

    if raw_cov_max is not None and float(raw_cov_max) < 0.999:
        weak_reasons.append("low_raw_retrieval_element_coverage")

    # Survival path: where retrieval candidates are lost.
    if raw_n == 0:
        retrieval_survival_path = "not_generated"
    elif merged_n == 0:
        retrieval_survival_path = "generated_but_not_merged"
    elif reranked_n == 0:
        retrieval_survival_path = "merged_but_removed_by_element_rerank"
    elif reranked_topn == 0:
        retrieval_survival_path = "survived_rerank_but_not_topn"
    else:
        retrieval_survival_path = "survived_to_topn"

    # Support strength: separated from audit level so downstream summaries can use it directly.
    if raw_n == 0:
        retrieval_support_strength = "none"
    elif reranked_topn == 0:
        retrieval_support_strength = "available_but_not_selected"
    elif topn_frac < 0.10:
        retrieval_support_strength = "weak_topn_support"
    elif weak_reasons:
        retrieval_support_strength = "chemically_weak_support"
    elif topn_frac >= 0.30:
        retrieval_support_strength = "strong_topn_support"
    else:
        retrieval_support_strength = "moderate_topn_support"

    # Final audit level.
    if raw_n == 0:
        audit_level = "major_warning"
        interpretation = "No retrieval candidates were produced."
    elif merged_n == 0:
        audit_level = "major_warning"
        interpretation = "Retrieval candidates were produced but did not survive into the merged Stage2 pool."
    elif reranked_n == 0:
        audit_level = "major_warning"
        interpretation = "Retrieval candidates entered the merged pool but did not survive element reranking."
    elif reranked_topn == 0:
        audit_level = "minor_warning"
        interpretation = "Retrieval candidates survive in the reranked pool, but none appear in the top-N candidates."
    elif weak_reasons:
        audit_level = "minor_warning"
        interpretation = (
            "Retrieval candidates are present in the top-N reranked Stage2 pool, "
            "but their contribution is weak or chemically incomplete: "
            + ";".join(weak_reasons)
        )
    else:
        audit_level = "pass"
        interpretation = "Retrieval candidates are present and make a strong contribution to the top-N reranked Stage2 pool."

    retrieval_support_interpretation = (
        f"retrieval_survival_path={retrieval_survival_path}; "
        f"retrieval_support_strength={retrieval_support_strength}; "
        f"raw_retrieval_rows={raw_n}; merged_retrieval_rows={merged_n}; "
        f"reranked_retrieval_rows={reranked_n}; topn_retrieval_rows={reranked_topn}; "
        f"topn_retrieval_frac={topn_frac:.4f}"
    )

    summary["weak_retrieval_reasons"] = weak_reasons
    summary["retrieval_survival_path"] = retrieval_survival_path
    summary["retrieval_support_strength"] = retrieval_support_strength
    summary["retrieval_support_interpretation"] = retrieval_support_interpretation
    summary["retrieval_raw_similarity_mean"] = raw_similarity_mean
    summary["retrieval_raw_similarity_max"] = raw_similarity_max
    summary["retrieval_raw_element_coverage_mean"] = raw_cov_mean
    summary["retrieval_raw_element_coverage_max"] = raw_cov_max
    summary["retrieval_final_element_coverage_mean"] = retrieval_cov_mean
    summary["retrieval_final_element_coverage_max"] = retrieval_cov_max
    summary.update({
        "top_n": int(args.top_n),
        "audit_level": audit_level,
        "interpretation": interpretation,
        "claim_boundary": "stage2_retrieval_audit_is_internal_candidate_pool_diagnostic_not_experimental_validation",
        "input_retrieval_csv": str(args.retrieval_csv),
        "input_merged_csv": str(args.merged_csv),
        "input_reranked_csv": str(args.reranked_csv),
        "output_csv": str(args.output_csv),
        "output_md": str(args.output_md),
    })

    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    summary_json = Path(args.summary_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    audit.to_csv(output_csv, index=False)

    preview_cols = [c for c in [
        "_table_name",
        "_row_index",
        "_is_top_n",
        "_is_retrieval",
        "candidate_source",
        "precursor_set",
        "element_coverage",
        "element_missing",
        "extra_element_penalty",
        "retrieval_source_formula",
        "retrieval_source_elements",
        "retrieval_similarity",
        "retrieval_element_coverage",
    ] if c in audit.columns]

    if len(audit) and preview_cols:
        audit[preview_cols].head(120).to_markdown(output_md, index=False)
    else:
        output_md.write_text("# Stage2 Retrieval Audit\n\nNo audit rows available.\n", encoding="utf-8")

    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[SAVE] {output_csv}")
    print(f"[SAVE] {output_md}")
    print(f"[SAVE] {summary_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
