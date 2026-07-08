#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path("/Users/wyc/SynPred")
ROUTE_SUBDIR = "routes_flow_fallback_retrieval_baseline_element_reranked"

# 水合物、硝酸盐、碳酸盐、铵盐、有机残基等常见非目标元素
COMMON_NON_TARGET_OK = {"H", "O", "N", "C"}
ELEMENT_RE = re.compile(r"([A-Z][a-z]?)")


def read_json_if_exists(path):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {
            "_json_read_error": str(e),
            "_json_path": str(path),
        }


def dict_to_json_str(x) -> str:
    if isinstance(x, dict):
        return json.dumps(x, ensure_ascii=False)
    if isinstance(x, list):
        return ";".join(str(v) for v in x)
    if x is None:
        return ""
    return str(x)


def first_existing_value(d: dict, keys: list[str], default=""):
    """
    Read the first non-empty/non-null value from a json dict.
    This is used to keep backward compatibility across V29/V30/V31/V32 outputs.
    """
    for k in keys:
        if k in d:
            v = d.get(k)
            if v is not None and str(v) != "nan" and str(v) != "":
                return v
    return default


def parse_formula_elements(text: str) -> set[str]:
    if not isinstance(text, str) or not text.strip():
        return set()
    return set(ELEMENT_RE.findall(text))


def parse_precursor_set_elements(precursor_set: str) -> set[str]:
    if not isinstance(precursor_set, str) or not precursor_set.strip():
        return set()
    return set(ELEMENT_RE.findall(precursor_set))


def read_first_existing(paths: list[Path]) -> tuple[pd.DataFrame | None, Path | None]:
    """
    Read the first existing CSV.

    Priority is controlled by candidate_files in summarize_one().
    pipeline_v3 now writes final_recommended_routes.csv after:
      v43 template ranker -> safe-strict gate -> condition/QC finalizer.
    """
    for p in paths:
        if not p.exists():
            continue
        try:
            return pd.read_csv(p), p
        except Exception as e:
            print(f"[WARN] failed to read {p}: {e}")
            continue
    return None, None


def get_col(row: pd.Series, names: list[str], default=""):
    for name in names:
        if name in row.index:
            val = row[name]
            if pd.notna(val):
                return val
    return default


def sort_by_best_available_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the first row is the best available route.

    Priority:
      1. final recommendation rank
      2. v43 safe-strict rank
      3. v43 template-aware chem-only rank
      4. v3 learned rank
      5. v3 joint rank
      6. final route rank
      7. stage35 v21 rank
      8. best-per-precursor rank
    """
    rank_cols = [
        "final_recommendation_rank",
        "stage35_v43_safe_strict_rank",
        "stage35_v43_template_chemonly_rank",
        "v3_learned_rank",
        "v3_joint_rank",
        "final_route_rank",
        "stage35_v21_rank",
        "best_route_rank",
    ]

    out = df.copy()

    for c in rank_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            return out.sort_values(c, ascending=True, na_position="last")

    return out


def classify_route(row: pd.Series) -> dict:
    formula = str(
        get_col(
            row,
            ["formula", "formula_x", "formula_y", "material_id", "sample_id"],
            "",
        )
    )
    precursor_set = str(get_col(row, ["precursor_set"], ""))

    target_elements_all = parse_formula_elements(formula)
    target_core = target_elements_all - {"H", "O"}

    precursor_elements_all = parse_precursor_set_elements(precursor_set)

    missing_core = sorted(target_core - precursor_elements_all)
    extra_all = sorted(precursor_elements_all - target_elements_all)
    extra_strict = sorted([e for e in extra_all if e not in COMMON_NON_TARGET_OK])

    n_extra_strict = len(extra_strict)
    n_missing_core = len(missing_core)

    temp = pd.to_numeric(get_col(row, ["temperature_c"], None), errors="coerce")
    time_h = pd.to_numeric(get_col(row, ["time_h"], None), errors="coerce")

    temp_warning = pd.notna(temp) and (temp < 300 or temp > 1600)
    time_warning = pd.notna(time_h) and (time_h < 0.1 or time_h > 240)

    precursor_qc_level = str(get_col(row, ["precursor_qc_level"], ""))

    reasons = []

    if n_missing_core > 0:
        reasons.append(f"missing_target_core_elements={';'.join(missing_core)}")
    if n_extra_strict > 0:
        reasons.append(f"extra_strict_elements={';'.join(extra_strict)}")
    if temp_warning:
        reasons.append(f"temperature_out_of_range={temp}")
    if time_warning:
        reasons.append(f"time_out_of_range={time_h}")
    if precursor_qc_level and precursor_qc_level not in {"pass", "nan", "None"}:
        reasons.append(f"precursor_qc_level={precursor_qc_level}")

    if n_missing_core > 0:
        qc_level = "fail_missing_target"
        status = "review_required"
    elif n_extra_strict > 0:
        qc_level = "major_warning_extra_element"
        status = "review_required"
    elif temp_warning or time_warning:
        qc_level = "minor_warning_condition"
        status = "recommended_with_validation"
    elif precursor_qc_level in {"major_warning", "review_required"}:
        qc_level = "major_warning_precursor"
        status = "review_required"
    elif precursor_qc_level in {"minor_warning"}:
        qc_level = "minor_warning_precursor"
        status = "recommended_with_validation"
    else:
        qc_level = "pass"
        status = "recommended"

    return {
        "target_elements_all": ";".join(sorted(target_elements_all)),
        "target_core_elements": ";".join(sorted(target_core)),
        "precursor_elements_all": ";".join(sorted(precursor_elements_all)),
        "missing_core_elements": ";".join(missing_core),
        "extra_elements_all": ";".join(extra_all),
        "extra_elements_strict": ";".join(extra_strict),
        "n_missing_core": n_missing_core,
        "n_extra_strict": n_extra_strict,
        "has_extra_strict_elements": n_extra_strict > 0,
        "has_missing_core_elements": n_missing_core > 0,
        "qc_level_auto": qc_level,
        "recommendation_auto": status,
        "qc_reason_auto": " | ".join(reasons),
    }


def summarize_one(infer_name: str) -> dict:
    route_dir = PROJECT_ROOT / "outputs" / "inference" / infer_name / ROUTE_SUBDIR

    candidate_files = [
        route_dir / "final_recommended_routes.csv",
        route_dir / "final_top_routes_v43_safe_strict_reranked.csv",
        route_dir / "final_top_routes_v43_template_chemonly_reranked.csv",
        route_dir / "final_top_routes_v3_learned_reranked.csv",
        route_dir / "final_top_routes_v3_joint_reranked.csv",
        route_dir / "final_top_routes_with_joint_features.csv",
        route_dir / "final_top_routes_with_confidence.csv",
        route_dir / "final_top_routes_with_metadata_stage3_reference.csv",
        route_dir / "final_top_routes_with_stage3_condition_reference.csv",
        route_dir / "final_top_routes_with_condition_confidence.csv",
        route_dir / "final_top_routes_with_precursor_qc.csv",
        route_dir / "final_top_routes.csv",
    ]

    df, used_path = read_first_existing(candidate_files)

    if df is None or used_path is None:
        return {
            "infer_name": infer_name,
            "status": "missing",
            "csv_path": "",
            "route_source_file": "",
        }

    if len(df) == 0:
        return {
            "infer_name": infer_name,
            "status": "empty",
            "csv_path": str(used_path),
            "route_source_file": used_path.name,
        }

    df = sort_by_best_available_rank(df)
    row = df.iloc[0]
    qc = classify_route(row)

    out = {
        "infer_name": infer_name,
        "status": "ok",
        "sample_id": get_col(row, ["sample_id"], ""),
        "material_id": get_col(row, ["material_id"], ""),
        "formula": get_col(row, ["formula", "formula_x", "formula_y"], ""),
        "precursor_set": get_col(row, ["precursor_set"], ""),
        "temperature_c": get_col(row, ["temperature_c"], ""),
        "time_h": get_col(row, ["time_h"], ""),

        # Stage3 / Stage35 legacy scores.
        "stage3_score": get_col(row, ["stage3_score", "condition_score"], ""),
        "stage35_v21_score": get_col(row, ["stage35_v21_score"], ""),
        "stage35_v2_prob": get_col(row, ["stage35_v2_prob"], ""),
        "v3_joint_feature_score": get_col(row, ["v3_joint_feature_score"], ""),
        "v3_learned_ranker_score": get_col(row, ["v3_learned_ranker_score"], ""),

        # Final recommendation outputs.
        "final_recommendation_rank": get_col(row, ["final_recommendation_rank"], ""),
        "final_recommendation_score": get_col(row, ["final_recommendation_score"], ""),
        "final_recommendation_status": get_col(row, ["final_recommendation_status"], ""),
        "final_recommendation_source": get_col(row, ["final_recommendation_source"], ""),
        "final_recommendation_penalty": get_col(row, ["final_recommendation_penalty"], ""),
        "final_recommendation_penalty_reason": get_col(row, ["final_recommendation_penalty_reason"], ""),
        "final_recommendation_base_score": get_col(row, ["final_recommendation_base_score"], ""),
        "final_recommendation_condition_support": get_col(row, ["final_recommendation_condition_support"], ""),
        "final_recommendation_condition_support_col": get_col(row, ["final_recommendation_condition_support_col"], ""),
        "final_recommendation_base_score_col": get_col(row, ["final_recommendation_base_score_col"], ""),
        "final_recommendation_condition_warning_col": get_col(row, ["final_recommendation_condition_warning_col"], ""),
        "final_recommendation_condition_status_col": get_col(row, ["final_recommendation_condition_status_col"], ""),
        "final_recommendation_adjusted_confidence": get_col(row, ["final_recommendation_adjusted_confidence"], ""),
        "final_recommendation_support_factor": get_col(row, ["final_recommendation_support_factor"], ""),

        # v4.3 template-aware ranker outputs.
        "stage35_v43_template_chemonly_rank": get_col(row, ["stage35_v43_template_chemonly_rank"], ""),
        "stage35_v43_template_chemonly_score": get_col(row, ["stage35_v43_template_chemonly_score"], ""),
        "stage35_v43_template_chemonly_mean_prob": get_col(row, ["stage35_v43_template_chemonly_mean_prob"], ""),
        "stage35_v43_template_chemonly_win_rate": get_col(row, ["stage35_v43_template_chemonly_win_rate"], ""),
        "stage35_v43_template_chemonly_wins": get_col(row, ["stage35_v43_template_chemonly_wins"], ""),
        "stage35_v43_template_chemonly_losses": get_col(row, ["stage35_v43_template_chemonly_losses"], ""),

        # v4.3 safe-strict outputs.
        "stage35_v43_safe_strict_rank": get_col(row, ["stage35_v43_safe_strict_rank"], ""),
        "stage35_v43_safe_strict_score": get_col(row, ["stage35_v43_safe_strict_score"], ""),
        "stage35_v43_safe_bucket": get_col(row, ["stage35_v43_safe_bucket"], ""),
        "stage35_v43_safe_reason": get_col(row, ["stage35_v43_safe_reason"], ""),

        # v4.3 route-template features.
        "route_template_primary": get_col(row, ["route_template_primary"], ""),
        "route_template_secondary": get_col(row, ["route_template_secondary"], ""),
        "route_template_type_signature": get_col(row, ["route_template_type_signature"], ""),
        "route_template_confidence": get_col(row, ["route_template_confidence"], ""),
        "route_template_matches_target_anion": get_col(row, ["route_template_matches_target_anion"], ""),
        "route_template_is_common_solid_state": get_col(row, ["route_template_is_common_solid_state"], ""),
        "route_template_is_overly_elemental": get_col(row, ["route_template_is_overly_elemental"], ""),
        "route_template_elemental_ratio": get_col(row, ["route_template_elemental_ratio"], ""),
        "route_template_n_types": get_col(row, ["route_template_n_types"], ""),

        # Reliability / QC outputs.
        "confidence_level": get_col(row, ["confidence_level"], ""),
        "recommendation_status": get_col(row, ["recommendation_status"], ""),
        "warning_level": get_col(row, ["warning_level"], ""),
        "precursor_qc_level": get_col(row, ["precursor_qc_level"], ""),
        "precursor_qc_status": get_col(row, ["precursor_qc_status"], ""),
        "element_coverage": get_col(row, ["element_coverage"], ""),
        "element_hit": get_col(row, ["element_hit"], ""),
        "element_missing": get_col(row, ["element_missing"], ""),

        # Internal condition distribution confidence.
        "condition_distribution_support_score": get_col(row, ["condition_distribution_support_score"], ""),
        "condition_distribution_confidence_level": get_col(row, ["condition_distribution_confidence_level"], ""),
        "condition_distribution_warning_level": get_col(row, ["condition_distribution_warning_level"], ""),
        "condition_distribution_recommendation_status": get_col(row, ["condition_distribution_recommendation_status"], ""),
        "condition_adjusted_confidence_score": get_col(row, ["condition_adjusted_confidence_score"], ""),

        # Real Stage3 MDN/Flow condition reference support.
        "real_stage3_condition_reference_support_score": get_col(row, ["real_stage3_condition_reference_support_score"], ""),
        "real_stage3_condition_reference_level": get_col(row, ["real_stage3_condition_reference_level"], ""),
        "real_stage3_condition_reference_warning_level": get_col(row, ["real_stage3_condition_reference_warning_level"], ""),
        "real_stage3_condition_reference_recommendation_status": get_col(row, ["real_stage3_condition_reference_recommendation_status"], ""),
        "real_stage3_condition_reference_warning_reason": get_col(row, ["real_stage3_condition_reference_warning_reason"], ""),

        # V32 metadata-aware Stage3 reference support.
        "metadata_aware_stage3_reference_support_score": get_col(row, ["metadata_aware_stage3_reference_support_score"], ""),
        "metadata_aware_stage3_reference_level": get_col(row, ["metadata_aware_stage3_reference_level"], ""),
        "metadata_aware_stage3_reference_warning_level": get_col(row, ["metadata_aware_stage3_reference_warning_level"], ""),
        "metadata_aware_stage3_reference_recommendation_status": get_col(row, ["metadata_aware_stage3_reference_recommendation_status"], ""),
        "metadata_aware_stage3_mp_id": get_col(row, ["metadata_aware_stage3_mp_id"], ""),
        "metadata_aware_stage3_mp_formula": get_col(row, ["metadata_aware_stage3_mp_formula"], ""),
        "metadata_aware_stage3_element_jaccard": get_col(row, ["metadata_aware_stage3_element_jaccard"], ""),
        "metadata_aware_stage3_family_compatibility": get_col(row, ["metadata_aware_stage3_family_compatibility"], ""),
        "metadata_aware_stage3_formula_exact_match": get_col(row, ["metadata_aware_stage3_formula_exact_match"], ""),

        # Provenance.
        "route_source_file": used_path.name,
        "csv_path": str(used_path),
    }

    out.update(qc)

    # Final recommendation summary.
    final_summary_json = route_dir / "final_recommended_routes_summary.json"
    final_summary = read_json_if_exists(final_summary_json)

    final_summary_top1_condition_support = first_existing_value(
        final_summary,
        [
            "top1_condition_support_score",
            "top1_metadata_aware_stage3_reference_support_score",
            "top1_real_stage3_condition_reference_support_score",
            "top1_condition_distribution_support_score",
        ],
        "",
    )

    final_summary_top1_v43_score = first_existing_value(
        final_summary,
        [
            "top1_stage35_v43_safe_strict_score",
            "top1_stage35_v43_template_chemonly_score",
        ],
        "",
    )

    out.update({
        "final_recommended_routes_summary_json": str(final_summary_json),
        "final_recommended_routes_summary_exists": bool(final_summary_json.exists()),
        "final_summary_input_csv": final_summary.get("input_csv", ""),
        "final_summary_n_routes": final_summary.get("n_routes", ""),
        "final_summary_top_n": final_summary.get("top_n", ""),
        "final_summary_finalizer": final_summary.get("finalizer", ""),
        "final_summary_top1_precursor_set": final_summary.get("top1_precursor_set", ""),
        "final_summary_top1_score": final_summary.get("top1_final_recommendation_score", ""),
        "final_summary_top1_status": final_summary.get("top1_final_recommendation_status", ""),
        "final_summary_top1_penalty_reason": final_summary.get("top1_final_recommendation_penalty_reason", ""),
        "final_summary_top1_v43_score": final_summary_top1_v43_score,

        # Corrected condition support summary:
        # Prefer actual finalizer support, then metadata-aware, then real Stage3, then old internal distribution.
        "final_summary_top1_condition_support": final_summary_top1_condition_support,
        "final_summary_condition_support_col_used_top1": final_summary.get("condition_support_col_used_top1", ""),
        "final_summary_base_score_col_used_top1": final_summary.get("base_score_col_used_top1", ""),
        "final_summary_condition_warning_col_used_top1": final_summary.get("condition_warning_col_used_top1", ""),
        "final_summary_condition_status_col_used_top1": final_summary.get("condition_status_col_used_top1", ""),

        # Explicit support channels.
        "final_summary_top1_metadata_aware_stage3_reference_support": final_summary.get("top1_metadata_aware_stage3_reference_support_score", ""),
        "final_summary_top1_metadata_aware_stage3_mp_id": final_summary.get("top1_metadata_aware_stage3_mp_id", ""),
        "final_summary_top1_metadata_aware_stage3_mp_formula": final_summary.get("top1_metadata_aware_stage3_mp_formula", ""),
        "final_summary_top1_real_stage3_condition_reference_support": final_summary.get("top1_real_stage3_condition_reference_support_score", ""),
        "final_summary_top1_condition_distribution_support": final_summary.get("top1_condition_distribution_support_score", ""),

        "final_summary_condition_support_col_used_counts": dict_to_json_str(final_summary.get("condition_support_col_used_counts", {})),
        "final_summary_base_score_col_used_counts": dict_to_json_str(final_summary.get("base_score_col_used_counts", {})),
        "final_summary_status_counts": dict_to_json_str(final_summary.get("final_recommendation_status_counts", {})),
        "final_summary_n_penalized": final_summary.get("n_penalized", ""),
        "final_summary_claim_boundary": final_summary.get("claim_boundary", ""),
        "final_summary_interpretation": final_summary.get("interpretation", ""),
    })

    # Final recommendation audit summary.
    final_audit_json = route_dir / "final_recommended_routes_audit_summary.json"
    final_audit = read_json_if_exists(final_audit_json)

    out.update({
        "final_recommended_routes_audit_json": str(final_audit_json),
        "final_recommended_routes_audit_exists": bool(final_audit_json.exists()),
        "final_audit_level_counts": dict_to_json_str(final_audit.get("final_audit_level_counts", {})),
        "final_audit_flag_counts": dict_to_json_str(final_audit.get("final_audit_flag_counts", {})),
        "final_audit_n_major_warning": final_audit.get("n_major_warning", ""),
        "final_audit_n_minor_warning": final_audit.get("n_minor_warning", ""),
        "final_audit_n_pass": final_audit.get("n_pass", ""),
        "final_audit_n_condition_sanity_warning": final_audit.get("n_condition_sanity_warning", ""),
        "final_audit_n_condition_reference_used": final_audit.get("n_condition_reference_used", ""),
        "final_audit_n_real_stage3_condition_warning": final_audit.get("n_real_stage3_condition_warning", ""),
        "final_audit_condition_support_col_counts": dict_to_json_str(final_audit.get("condition_support_col_counts", {})),
        "final_audit_condition_warning_col_counts": dict_to_json_str(final_audit.get("condition_warning_col_counts", {})),
        "final_audit_status_counts": dict_to_json_str(final_audit.get("final_status_counts", {})),
        "final_audit_top1_precursor_set": final_audit.get("top1_precursor_set", ""),
        "final_audit_top1_score": final_audit.get("top1_final_recommendation_score", ""),
        "final_audit_top1_level": final_audit.get("top1_final_audit_level", ""),
        "final_audit_top1_flags": final_audit.get("top1_final_audit_flags", ""),
        "final_audit_top1_condition_support_col": final_audit.get("top1_condition_support_col", ""),
        "final_audit_top1_condition_support": final_audit.get("top1_condition_support", ""),
        "final_audit_top1_condition_warning_col": final_audit.get("top1_condition_warning_col", ""),
        "final_audit_top1_condition_warning_level": final_audit.get("top1_condition_warning_level", ""),
        "final_audit_claim_boundary": final_audit.get("claim_boundary", ""),
        "final_audit_interpretation": final_audit.get("interpretation", ""),
    })

    # Stage2 retrieval-conditioned generation audit.
    retrieval_audit_json = (
        PROJECT_ROOT
        / "outputs"
        / "inference"
        / infer_name
        / "stage2_audit"
        / "stage2_retrieval_audit_summary.json"
    )
    retrieval_audit = read_json_if_exists(retrieval_audit_json)

    weak_reasons = retrieval_audit.get("weak_retrieval_reasons", "")
    if isinstance(weak_reasons, list):
        weak_reasons = ";".join(str(x) for x in weak_reasons)
    elif weak_reasons is None:
        weak_reasons = ""
    else:
        weak_reasons = str(weak_reasons)

    out.update({
        "stage2_retrieval_audit_json": str(retrieval_audit_json),
        "stage2_retrieval_audit_exists": bool(retrieval_audit_json.exists()),
        "stage2_retrieval_audit_level": retrieval_audit.get("audit_level", "missing"),
        "stage2_retrieval_survival_path": retrieval_audit.get("retrieval_survival_path", ""),
        "stage2_retrieval_support_strength": retrieval_audit.get("retrieval_support_strength", ""),
        "stage2_retrieval_support_interpretation": retrieval_audit.get("retrieval_support_interpretation", ""),
        "stage2_retrieval_weak_reasons": weak_reasons,
        "stage2_retrieval_interpretation": retrieval_audit.get("interpretation", ""),

        "stage2_retrieval_raw_exists": retrieval_audit.get("retrieval_raw_exists"),
        "stage2_retrieval_raw_n_rows": retrieval_audit.get("retrieval_raw_n_rows"),
        "stage2_retrieval_raw_retrieval_frac": retrieval_audit.get("retrieval_raw_retrieval_frac"),
        "stage2_retrieval_raw_topn_frac": retrieval_audit.get("retrieval_raw_topn_retrieval_frac"),
        "stage2_retrieval_raw_similarity_mean": retrieval_audit.get("retrieval_raw_retrieval_similarity_mean"),
        "stage2_retrieval_raw_similarity_max": retrieval_audit.get("retrieval_raw_retrieval_similarity_max"),
        "stage2_retrieval_raw_element_coverage_mean": retrieval_audit.get("retrieval_raw_retrieval_element_coverage_mean"),
        "stage2_retrieval_raw_element_coverage_max": retrieval_audit.get("retrieval_raw_retrieval_element_coverage_max"),

        "stage2_merged_pool_exists": retrieval_audit.get("merged_pool_exists"),
        "stage2_merged_pool_n_rows": retrieval_audit.get("merged_pool_n_rows"),
        "stage2_merged_pool_retrieval_frac": retrieval_audit.get("merged_pool_retrieval_frac"),
        "stage2_merged_pool_topn_retrieval_frac": retrieval_audit.get("merged_pool_topn_retrieval_frac"),
        "stage2_merged_pool_retrieval_similarity_mean": retrieval_audit.get("merged_pool_retrieval_similarity_mean"),
        "stage2_merged_pool_retrieval_similarity_max": retrieval_audit.get("merged_pool_retrieval_similarity_max"),
        "stage2_merged_pool_retrieval_element_coverage_mean": retrieval_audit.get("merged_pool_retrieval_element_coverage_mean"),
        "stage2_merged_pool_retrieval_element_coverage_max": retrieval_audit.get("merged_pool_retrieval_element_coverage_max"),

        "stage2_element_reranked_pool_exists": retrieval_audit.get("element_reranked_pool_exists"),
        "stage2_element_reranked_pool_n_rows": retrieval_audit.get("element_reranked_pool_n_rows"),
        "stage2_element_reranked_pool_retrieval_frac": retrieval_audit.get("element_reranked_pool_retrieval_frac"),
        "stage2_element_reranked_pool_topn_retrieval_frac": retrieval_audit.get("element_reranked_pool_topn_retrieval_frac"),
        "stage2_element_reranked_pool_retrieval_similarity_mean": retrieval_audit.get("element_reranked_pool_retrieval_similarity_mean"),
        "stage2_element_reranked_pool_retrieval_similarity_max": retrieval_audit.get("element_reranked_pool_retrieval_similarity_max"),
        "stage2_element_reranked_pool_retrieval_element_coverage_mean": retrieval_audit.get("element_reranked_pool_retrieval_element_coverage_mean"),
        "stage2_element_reranked_pool_retrieval_element_coverage_max": retrieval_audit.get("element_reranked_pool_retrieval_element_coverage_max"),
        "stage2_element_reranked_pool_retrieval_final_element_coverage_mean": retrieval_audit.get("element_reranked_pool_retrieval_final_element_coverage_mean"),
        "stage2_element_reranked_pool_retrieval_final_element_coverage_max": retrieval_audit.get("element_reranked_pool_retrieval_final_element_coverage_max"),
        "stage2_element_reranked_pool_retrieval_rows_with_missing_elements": retrieval_audit.get("element_reranked_pool_retrieval_rows_with_missing_elements"),
        "stage2_element_reranked_pool_retrieval_rows_with_extra_penalty": retrieval_audit.get("element_reranked_pool_retrieval_rows_with_extra_penalty"),
    })

    # Condition diversity audit summary.
    condition_diversity_json = route_dir / "condition_diversity_audit_summary.json"
    condition_diversity = read_json_if_exists(condition_diversity_json)

    out.update({
        "condition_diversity_audit_json": str(condition_diversity_json),
        "condition_diversity_audit_exists": bool(condition_diversity_json.exists()),
        "condition_diversity_n_groups": condition_diversity.get("n_groups", ""),
        "condition_diversity_status_counts": dict_to_json_str(condition_diversity.get("condition_diversity_status_counts", {})),
        "condition_diversity_audit_level_counts": dict_to_json_str(condition_diversity.get("condition_diversity_audit_level_counts", {})),
        "condition_diversity_n_major_warning_groups": condition_diversity.get("n_major_warning_groups", ""),
        "condition_diversity_n_minor_warning_groups": condition_diversity.get("n_minor_warning_groups", ""),
        "condition_diversity_n_pass_groups": condition_diversity.get("n_pass_groups", ""),
        "condition_diversity_n_baseline_seed_groups": condition_diversity.get("n_baseline_seed_groups", ""),
        "condition_diversity_claim_boundary": condition_diversity.get("claim_boundary", ""),
        "condition_diversity_interpretation": condition_diversity.get("interpretation", ""),
    })

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--prefix", default="benchmark_")
    ap.add_argument("--width", type=int, default=3)
    ap.add_argument("--output_csv", default=None)
    args = ap.parse_args()

    rows = []
    for i in range(args.start, args.end + 1):
        infer_name = f"{args.prefix}{i:0{args.width}d}"
        rows.append(summarize_one(infer_name))

    df = pd.DataFrame(rows)

    pd.set_option("display.max_columns", 280)
    pd.set_option("display.width", 380)
    print(df.to_string(index=False))

    if args.output_csv:
        out = Path(args.output_csv)
    else:
        out = (
            PROJECT_ROOT
            / "outputs"
            / "inference"
            / f"_benchmark_top1_qc_summary_{args.start:03d}_{args.end:03d}.csv"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n[SAVE] {out}")

    summary = {
        "n_total": len(df),
        "status_counts": df["status"].value_counts(dropna=False).to_dict()
        if "status" in df
        else {},

        "route_source_file_counts": df["route_source_file"].value_counts(dropna=False).to_dict()
        if "route_source_file" in df
        else {},

        "qc_level_auto_counts": df["qc_level_auto"].value_counts(dropna=False).to_dict()
        if "qc_level_auto" in df
        else {},

        "recommendation_auto_counts": df["recommendation_auto"].value_counts(dropna=False).to_dict()
        if "recommendation_auto" in df
        else {},

        "final_recommendation_status_counts": df["final_recommendation_status"].value_counts(dropna=False).to_dict()
        if "final_recommendation_status" in df
        else {},

        "final_recommendation_condition_support_col_counts": df["final_recommendation_condition_support_col"].value_counts(dropna=False).to_dict()
        if "final_recommendation_condition_support_col" in df
        else {},

        "final_summary_condition_support_col_used_top1_counts": df["final_summary_condition_support_col_used_top1"].value_counts(dropna=False).to_dict()
        if "final_summary_condition_support_col_used_top1" in df
        else {},

        "final_audit_top1_level_counts": df["final_audit_top1_level"].value_counts(dropna=False).to_dict()
        if "final_audit_top1_level" in df
        else {},

        "final_audit_top1_condition_support_col_counts": df["final_audit_top1_condition_support_col"].value_counts(dropna=False).to_dict()
        if "final_audit_top1_condition_support_col" in df
        else {},

        "stage2_retrieval_audit_level_counts": df["stage2_retrieval_audit_level"].value_counts(dropna=False).to_dict()
        if "stage2_retrieval_audit_level" in df
        else {},

        "stage2_retrieval_support_strength_counts": df["stage2_retrieval_support_strength"].value_counts(dropna=False).to_dict()
        if "stage2_retrieval_support_strength" in df
        else {},

        "condition_diversity_audit_exists_counts": df["condition_diversity_audit_exists"].value_counts(dropna=False).to_dict()
        if "condition_diversity_audit_exists" in df
        else {},

        "n_final_summary_missing": int((df["final_recommended_routes_summary_exists"] == False).sum())
        if "final_recommended_routes_summary_exists" in df
        else None,

        "n_final_audit_missing": int((df["final_recommended_routes_audit_exists"] == False).sum())
        if "final_recommended_routes_audit_exists" in df
        else None,

        "n_stage2_retrieval_audit_missing": int((df["stage2_retrieval_audit_level"] == "missing").sum())
        if "stage2_retrieval_audit_level" in df
        else None,

        "n_condition_diversity_audit_missing": int((df["condition_diversity_audit_exists"] == False).sum())
        if "condition_diversity_audit_exists" in df
        else None,
    }

    print("\n[SUMMARY]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
