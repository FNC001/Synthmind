from __future__ import annotations

def get_active_poscar_elements(poscar: Path, ignore_elements: set[str] | None = None) -> tuple[list[str], list[str]]:
    """Parse VASP5 POSCAR element symbols and return all/active target elements."""
    if ignore_elements is None:
        ignore_elements = {"H", "O"}

    try:
        lines = poscar.read_text(errors="ignore").splitlines()
    except Exception:
        return [], []

    if len(lines) < 6:
        return [], []

    import re
    elems = []
    for x in lines[5].strip().split():
        if re.fullmatch(r"[A-Z][a-z]?", x):
            elems.append(x)

    active = [e for e in elems if e not in ignore_elements]
    return elems, active


def make_unsupported_target_status(
    case_id: str,
    input_poscar: Path,
    out_root: Path,
    pipeline_log: Path | None = None,
) -> dict:
    return {
        "case_id": case_id,
        "audit_time": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_poscar": str(input_poscar),
        "out_dir": str(out_root),
        "route_dir": "",
        "final_csv": "",
        "final_md": "",
        "has_final_recommended_routes": False,
        "has_final_recommended_md": False,
        "has_stage2_final_csv": False,
        "has_stage3_flow_flat_csv": False,
        "problem_type": "target_has_only_ignored_elements",
        "recommended_action": "skip_route_prediction",
        "final_case_status": "unsupported_target_elements",
        "refined_case_status": "unsupported_target_elements",
        "refined_case_reason": "active_target_elements_empty_after_ignoring_H_O",
        "n_stage2_candidates": 0,
        "has_stage2_candidates": False,
        "n_stage3_conditions": 0,
        "has_stage3_conditions": False,
        "n_final_routes": 0,
        "has_final_recommendation": False,
        "top1_precursor_set": "",
        "top1_score": None,
        "top1_condition_support_score": None,
        "top1_status": "",
        "top1_audit_level": "",
        "pipeline_log": str(pipeline_log) if pipeline_log else "",
    }


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adaptive batch structure-to-synthesis pipeline.

Layer 1:
    Run existing pipeline_v3 case by case.

Layer 1.5:
    Immediately audit each case output.

Layer 2/3 placeholder:
    Mark cases that need recovery. Real recovery modules can be plugged in later.

This script does not modify pipeline_v3. It only orchestrates it.
"""


import argparse
import json
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

import pandas as pd
import yaml


ROUTE_SUBDIR = "routes_flow_fallback_retrieval_baseline_element_reranked"


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_poscar_files(batch_poscar_dir: Path) -> List[Path]:
    """
    Find POSCAR-like files in a batch directory.
    Supports:
      - direct files under batch_poscar_dir
      - files in subdirectories
    """
    if not batch_poscar_dir.exists():
        raise FileNotFoundError(f"batch_poscar_dir does not exist: {batch_poscar_dir}")

    candidates = []
    patterns = [
        "POSCAR",
        "CONTCAR",
        "*.vasp",
        "*.poscar",
        "*.POSCAR",
        "*.cif",
    ]

    for pat in patterns:
        candidates.extend(batch_poscar_dir.rglob(pat))

    # Remove duplicates and keep files only
    uniq = []
    seen = set()
    for p in candidates:
        p = p.resolve()
        if p.is_file() and str(p) not in seen:
            seen.add(str(p))
            uniq.append(p)

    return sorted(uniq)


def make_case_id(poscar_path: Path, idx: int) -> str:
    """
    Make a stable case id.
    """
    stem = poscar_path.stem
    parent = poscar_path.parent.name

    raw = f"{idx:06d}_{parent}_{stem}"
    keep = []
    for ch in raw:
        if ch.isalnum() or ch in ["_", "-", "."]:
            keep.append(ch)
        else:
            keep.append("_")
    case_id = "".join(keep)
    return f"case_{case_id}"


def prepare_case_input(
    poscar_path: Path,
    case_id: str,
    case_input_root: Path,
) -> Path:
    """
    pipeline_v3 expects:
        data/infer/<infer_name>/poscars/<POSCAR files>
    """
    case_poscar_dir = case_input_root / case_id / "poscars"
    safe_mkdir(case_poscar_dir)

    # Copy as POSCAR if original is POSCAR-like.
    # Keep original suffix/name if needed, but POSCAR is safest.
    target = case_poscar_dir / "POSCAR"
    shutil.copy2(poscar_path, target)

    return case_poscar_dir


def run_pipeline_v3_for_case(
    pipeline_v3_dir: Path,
    pipeline_v3_config: Path,
    case_id: str,
    start_from: str,
    log_file: Path,
) -> int:
    cmd = [
        "python",
        str(pipeline_v3_dir / "run_pipeline.py"),
        "--config",
        str(pipeline_v3_config),
        "--infer_name",
        case_id,
        "--start_from",
        start_from,
    ]

    safe_mkdir(log_file.parent)
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"[START] {now()}\n")
        f.write("[CMD] " + " ".join(cmd) + "\n\n")
        f.flush()

        proc = subprocess.run(
            cmd,
            cwd=str(pipeline_v3_dir),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )

        f.write(f"\n[END] {now()}\n")
        f.write(f"[RETURN_CODE] {proc.returncode}\n")

    return proc.returncode


def read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _safe_float(v, default=None):
    try:
        if v is None:
            return default
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def _has_suspicious_precursor_text(text: str) -> bool:
    """
    Lightweight precursor string sanity check.
    This is not a chemistry parser; it only catches obvious suspicious tokens
    that should be manually reviewed in large-scale screening.
    """
    if not text:
        return False

    suspicious_tokens = [
        "P4P4",
        "Ca2CO3",
        "Co2O3",
        "CuCO3-Cu(OH)2",
    ]

    return any(tok in text for tok in suspicious_tokens)


def assign_refined_case_status(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a second-level screening status without overwriting final_case_status.

    final_case_status:
        Original adaptive status, close to pipeline/finalizer output.

    refined_case_status:
        More useful for large-scale screening and prioritization.
    """
    final_status = str(result.get("final_case_status", "unknown"))
    problem_type = str(result.get("problem_type", ""))

    precursor = str(
        result.get("top1_precursor_set")
        or result.get("summary_top1_precursor_set")
        or ""
    )

    cond_support = _safe_float(
        result.get("top1_condition_support_score"),
        default=None,
    )
    if cond_support is None:
        cond_support = _safe_float(
            result.get("summary_top1_condition_support_score"),
            default=None,
        )
    if cond_support is None:
        cond_support = _safe_float(
            result.get("summary_top1_real_stage3_condition_reference_support_score"),
            default=None,
        )

    final_score = _safe_float(
        result.get("top1_score"),
        default=None,
    )
    if final_score is None:
        final_score = _safe_float(
            result.get("summary_top1_final_recommendation_score"),
            default=None,
        )

    audit_level = str(
        result.get("top1_audit_level")
        or result.get("audit_top1_final_audit_level")
        or ""
    ).lower()

    reasons = []

    suspicious_precursor = _has_suspicious_precursor_text(precursor)
    if suspicious_precursor:
        reasons.append("suspicious_precursor_formula")

    if "major" in audit_level:
        reasons.append("major_audit_warning")

    if cond_support is None:
        reasons.append("missing_condition_support")
    elif cond_support < 0.50:
        reasons.append("very_low_condition_support")
    elif cond_support < 0.75:
        reasons.append("low_condition_support")
    elif cond_support < 0.90:
        reasons.append("moderate_condition_support")

    if final_score is None:
        reasons.append("missing_final_score")
    elif final_score < 0.40:
        reasons.append("low_final_score")
    elif final_score < 0.55:
        reasons.append("moderate_final_score")

    # Hard failure/recovery states keep their meaning.
    if final_status in {
        "pipeline_failed",
        "needs_stage2_recovery",
        "needs_stage3_recovery",
        "needs_condition_reexport",
        "needs_route_finalization_recovery",
        "needs_manual_or_rule_recovery",
    }:
        refined = final_status

    # Strong warning overrides score.
    elif suspicious_precursor or "major" in audit_level:
        refined = "needs_manual_check"

    # Review-required cases are refined into useful screening levels.
    elif final_status == "review_required_only" or "review" in final_status:
        if cond_support is not None and cond_support >= 0.85 and final_score is not None and final_score >= 0.40:
            refined = "high_confidence_review"
        elif cond_support is not None and cond_support >= 0.70 and final_score is not None and final_score >= 0.25:
            refined = "medium_confidence_review"
        else:
            refined = "low_confidence_review"

    # Pass with validation: tiered like pass but flagged for extra checks.
    elif final_status == "pass_with_validation":
        if cond_support is not None and cond_support >= 0.85:
            refined = "pass_high_confidence_validation"
        elif cond_support is not None and cond_support >= 0.70:
            refined = "pass_medium_confidence_validation"
        else:
            refined = "pass_low_confidence_validation"

    # Pass remains pass unless the secondary checks reveal weak confidence.
    elif final_status == "pass":
        if cond_support is not None and cond_support >= 0.85:
            refined = "pass_high_confidence"
        elif cond_support is not None and cond_support >= 0.70:
            refined = "pass_medium_confidence"
        else:
            refined = "pass_low_confidence"

    else:
        refined = final_status or "unknown"

    result["refined_case_status"] = refined
    result["refined_case_reason"] = ";".join(reasons) if reasons else "no_secondary_warning"
    result["refined_condition_support_score"] = cond_support
    result["refined_final_score"] = final_score
    result["refined_has_suspicious_precursor"] = bool(suspicious_precursor)

    return result


def audit_case_outputs(
    project_root: Path,
    case_id: str,
    audit_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Lightweight immediate audit for one case.

    It reads the existing pipeline_v3 outputs and classifies the case.
    """
    out_dir = project_root / "outputs" / "inference" / case_id
    route_dir = out_dir / ROUTE_SUBDIR

    final_csv = route_dir / "final_recommended_routes.csv"
    final_md = route_dir / "final_recommended_routes.md"
    final_summary_json = route_dir / "final_recommended_routes_summary.json"
    final_audit_csv = route_dir / "final_recommended_routes_audit.csv"
    final_audit_summary_json = route_dir / "final_recommended_routes_audit_summary.json"

    stage2_final_csv = (
        project_root
        / "data"
        / "interim"
        / "infer"
        / case_id
        / "stage2_summary"
        / "unique_sets_ranked_with_fallback_retrieval_baseline_element_reranked.csv"
    )

    flow_flat_csv = (
        out_dir
        / "stage3_condition_predictions_flow_fallback_retrieval_baseline_element_reranked"
        / "test_candidates_flat.csv"
    )

    result: Dict[str, Any] = {
        "case_id": case_id,
        "audit_time": now(),
        "out_dir": str(out_dir),
        "route_dir": str(route_dir),
        "final_csv": str(final_csv),
        "final_md": str(final_md),
        "has_final_recommended_routes": final_csv.exists(),
        "has_final_recommended_md": final_md.exists(),
        "has_stage2_final_csv": stage2_final_csv.exists(),
        "has_stage3_flow_flat_csv": flow_flat_csv.exists(),
        "problem_type": "",
        "recommended_action": "",
        "final_case_status": "unknown",
    }

    # Stage2 status
    stage2_df = read_csv_if_exists(stage2_final_csv)
    if stage2_df is None:
        result["n_stage2_candidates"] = 0
        result["has_stage2_candidates"] = False
    else:
        result["n_stage2_candidates"] = int(len(stage2_df))
        result["has_stage2_candidates"] = len(stage2_df) > 0

    # Stage3 status
    flow_df = read_csv_if_exists(flow_flat_csv)
    if flow_df is None:
        result["n_stage3_conditions"] = 0
        result["has_stage3_conditions"] = False
    else:
        result["n_stage3_conditions"] = int(len(flow_df))
        result["has_stage3_conditions"] = len(flow_df) > 0

    # Final route status
    final_df = read_csv_if_exists(final_csv)
    summary = read_json_if_exists(final_summary_json)
    audit_summary = read_json_if_exists(final_audit_summary_json)

    result["summary_json_exists"] = bool(summary)
    result["audit_summary_json_exists"] = bool(audit_summary)

    if final_df is None or len(final_df) == 0:
        result["n_final_routes"] = 0
        result["has_final_recommendation"] = False
    else:
        result["n_final_routes"] = int(len(final_df))
        result["has_final_recommendation"] = True

        top = final_df.iloc[0].to_dict()

        precursor_col = pick_col(
            final_df,
            [
                "precursor_set",
                "precursors",
                "pred_precursor_set",
                "top1_precursor_set",
            ],
        )
        score_col = pick_col(
            final_df,
            [
                "final_recommendation_score",
                "stage35_v43_safe_strict_score",
                "stage35_v43_template_chemonly_score",
                "v3_joint_score",
            ],
        )
        support_col = pick_col(
            final_df,
            [
                "real_stage3_condition_reference_support_score",
                "condition_support_score",
                "condition_distribution_support_score",
            ],
        )
        status_col = pick_col(
            final_df,
            [
                "final_recommendation_status",
                "real_stage3_condition_reference_recommendation_status",
                "recommendation_status",
            ],
        )
        audit_col = pick_col(
            final_df,
            [
                "final_audit_level",
                "real_stage3_condition_reference_warning_level",
                "warning_level",
            ],
        )

        result["top1_precursor_set"] = str(top.get(precursor_col, "")) if precursor_col else ""
        result["top1_score"] = float(top.get(score_col)) if score_col and pd.notna(top.get(score_col)) else None
        result["top1_condition_support_score"] = (
            float(top.get(support_col)) if support_col and pd.notna(top.get(support_col)) else None
        )
        result["top1_status"] = str(top.get(status_col, "")) if status_col else ""
        result["top1_audit_level"] = str(top.get(audit_col, "")) if audit_col else ""

    # Pull useful summary fields if available
    for k in [
        "top1_precursor_set",
        "top1_final_recommendation_score",
        "top1_final_recommendation_status",
        "top1_condition_support_score",
        "top1_real_stage3_condition_reference_support_score",
        "final_recommendation_status_counts",
        "n_penalized",
    ]:
        if k in summary:
            result[f"summary_{k}"] = summary[k]

    for k in [
        "top1_final_audit_level",
        "top1_final_audit_flags",
        "n_major_warning",
        "n_minor_warning",
        "n_pass",
        "n_condition_sanity_warning",
    ]:
        if k in audit_summary:
            result[f"audit_{k}"] = audit_summary[k]

    # Decision rules
    min_support = float(audit_cfg.get("min_condition_support_score", 0.70))
    low_support = float(audit_cfg.get("low_condition_support_score", 0.50))

    has_stage2 = bool(result.get("has_stage2_candidates"))
    has_stage3 = bool(result.get("has_stage3_conditions"))
    has_final = bool(result.get("has_final_recommendation"))

    cond_support = result.get("top1_condition_support_score")
    if cond_support is None:
        cond_support = result.get("summary_top1_condition_support_score")
    if cond_support is None:
        cond_support = result.get("summary_top1_real_stage3_condition_reference_support_score")

    status = str(result.get("top1_status") or result.get("summary_top1_final_recommendation_status") or "")
    audit_level = str(result.get("top1_audit_level") or result.get("audit_top1_final_audit_level") or "")

    if not has_stage2:
        result["final_case_status"] = "needs_stage2_recovery"
        result["problem_type"] = "stage2_no_candidate"
        result["recommended_action"] = "rerun_stage2_with_more_sampling_fallback_retrieval_baseline"

    elif has_stage2 and not has_stage3:
        result["final_case_status"] = "needs_stage3_recovery"
        result["problem_type"] = "stage3_no_condition"
        result["recommended_action"] = "run_stage3_feature_source_audit_then_regenerate_or_reexport"

    elif not has_final:
        result["final_case_status"] = "needs_route_finalization_recovery"
        result["problem_type"] = "no_final_route"
        result["recommended_action"] = "inspect_route_summarization_and_finalizer"

    elif cond_support is not None and float(cond_support) < low_support:
        result["final_case_status"] = "needs_condition_reexport"
        result["problem_type"] = "condition_support_too_low"
        result["recommended_action"] = "run_condition_reexport_or_clipping_diagnostic"

    elif "major" in audit_level.lower():
        result["final_case_status"] = "needs_manual_or_rule_recovery"
        result["problem_type"] = "major_audit_warning"
        result["recommended_action"] = "inspect_precursor_and_condition_qc"

    elif cond_support is not None and float(cond_support) < min_support:
        result["final_case_status"] = "review_required_low_condition_support"
        result["problem_type"] = "condition_support_moderate"
        result["recommended_action"] = "keep_result_but_mark_review_required"

    elif status == "recommended_with_validation":
        result["final_case_status"] = "pass_with_validation"
        result["problem_type"] = "minor_validation_needed"
        result["recommended_action"] = "accept_with_validation"

    elif "review" in status.lower():
        result["final_case_status"] = "review_required_only"
        result["problem_type"] = "minor_or_route_review_required"
        result["recommended_action"] = "keep_result_no_heavy_recovery"

    else:
        result["final_case_status"] = "pass"
        result["problem_type"] = "none"
        result["recommended_action"] = "accept"

    result = assign_refined_case_status(result)
    return result



def build_standard_case_status(status: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert verbose internal status into a stable, human-readable schema.

    This is the schema used by batch reports and downstream recovery dispatch.
    It keeps the original final_case_status but exposes key fields with
    simpler names.
    """
    final_recommendation_status = (
        status.get("top1_status")
        or status.get("summary_top1_final_recommendation_status")
        or ""
    )

    condition_support_score = (
        status.get("top1_condition_support_score")
        if status.get("top1_condition_support_score") is not None
        else status.get("summary_top1_condition_support_score")
    )
    if condition_support_score is None:
        condition_support_score = status.get("summary_top1_real_stage3_condition_reference_support_score")

    audit_level = (
        status.get("top1_audit_level")
        or status.get("audit_top1_final_audit_level")
        or ""
    )

    # Keep recovery_action simple and operational.
    final_case_status = str(status.get("final_case_status", "unknown"))
    if final_case_status in ["pass", "pass_with_validation", "review_required_only"]:
        recovery_action = "none"
    elif final_case_status in ["high_confidence_review", "medium_confidence_review", "low_confidence_review"]:
        recovery_action = "none"
    elif final_case_status in ["needs_stage2_recovery"]:
        recovery_action = "stage2_recovery"
    elif final_case_status in ["needs_stage3_recovery"]:
        recovery_action = "stage3_recovery"
    elif final_case_status in ["needs_condition_reexport"]:
        recovery_action = "condition_reexport"
    elif final_case_status in ["needs_route_finalization_recovery"]:
        recovery_action = "route_finalizer_recovery"
    elif final_case_status in ["pipeline_failed"]:
        recovery_action = "inspect_pipeline_log"
    else:
        recovery_action = status.get("recommended_action", "")

    # Layer 1 means the standard pipeline run for this case.
    if final_case_status == "pipeline_failed":
        layer1_status = "failed"
    elif status.get("has_final_recommendation") or status.get("has_stage2_candidates") or status.get("has_stage3_conditions"):
        layer1_status = "finished"
    else:
        layer1_status = "unknown"

    return {
        "case_id": status.get("case_id", ""),
        "input_poscar": status.get("input_poscar", ""),
        "layer1_status": layer1_status,

        "has_stage2_candidates": bool(status.get("has_stage2_candidates", False)),
        "n_stage2_candidates": int(status.get("n_stage2_candidates", 0) or 0),

        "has_stage3_conditions": bool(status.get("has_stage3_conditions", False)),
        "n_stage3_conditions": int(status.get("n_stage3_conditions", 0) or 0),

        "has_final_recommendation": bool(status.get("has_final_recommendation", False)),
        "n_final_routes": int(status.get("n_final_routes", 0) or 0),

        "final_recommendation_status": final_recommendation_status,
        "condition_support_score": condition_support_score,
        "audit_level": audit_level,

        "top1_precursor_set": status.get("top1_precursor_set") or status.get("summary_top1_precursor_set", ""),
        "top1_final_score": status.get("top1_score") or status.get("summary_top1_final_recommendation_score"),

        "problem_type": status.get("problem_type", ""),
        "recovery_action": recovery_action,
        "recommended_action": status.get("recommended_action", ""),

        "final_case_status": status.get("final_case_status", ""),
        "refined_case_status": status.get("refined_case_status", ""),
        "refined_case_reason": status.get("refined_case_reason", ""),
        "refined_condition_support_score": status.get("refined_condition_support_score"),
        "refined_final_score": status.get("refined_final_score"),

        "pipeline_log": status.get("pipeline_log", ""),
        "audit_time": status.get("audit_time", ""),
    }


def write_case_status(case_status_path: Path, status: Dict[str, Any]) -> None:
    safe_mkdir(case_status_path.parent)
    case_status_path.write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_batch_report(master_df: pd.DataFrame, report_path: Path) -> None:
    safe_mkdir(report_path.parent)

    lines = []
    lines.append("# Adaptive Batch Pipeline Report")
    lines.append("")
    lines.append(f"Generated at: {now()}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total cases: {len(master_df)}")

    if "final_case_status" in master_df.columns:
        lines.append("")
        lines.append("### Case status counts")
        lines.append("")
        counts = master_df["final_case_status"].fillna("unknown").value_counts()
        lines.append("| status | count |")
        lines.append("|---|---:|")
        for k, v in counts.items():
            lines.append(f"| {k} | {int(v)} |")

    if "refined_case_status" in master_df.columns:
        lines.append("")
        lines.append("### Refined case status counts")
        lines.append("")
        counts = master_df["refined_case_status"].fillna("unknown").value_counts()
        lines.append("| refined_case_status | count |")
        lines.append("|---|---:|")
        for k, v in counts.items():
            lines.append(f"| {k} | {int(v)} |")

    if "problem_type" in master_df.columns:
        lines.append("")
        lines.append("### Problem type counts")
        lines.append("")
        counts = master_df["problem_type"].fillna("unknown").value_counts()
        lines.append("| problem_type | count |")
        lines.append("|---|---:|")
        for k, v in counts.items():
            lines.append(f"| {k} | {int(v)} |")

    lines.append("")
    lines.append("## Cases needing recovery")
    lines.append("")
    need = master_df[
        ~master_df["final_case_status"].fillna("").isin(["pass", "review_required_only"])
    ].copy()

    if len(need) == 0:
        lines.append("No cases require heavy recovery.")
    else:
        cols = [
            "case_id",
            "final_case_status",
            "problem_type",
            "recommended_action",
            "n_stage2_candidates",
            "n_stage3_conditions",
            "n_final_routes",
        ]
        cols = [c for c in cols if c in need.columns]
        lines.append(need[cols].to_markdown(index=False))

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _process_one_case(
    idx: int,
    poscar: Path,
    case_id: str,
    n_total: int,
    cases_root: Path,
    logs_root: Path,
    case_input_root: Path,
    pipeline_v3_dir: Path,
    pipeline_v3_config: Path,
    start_from: str,
    project_root: Path,
    audit_cfg: Dict[str, Any],
    mode: str,
) -> Dict[str, Any]:
    """Process a single case: prepare input, run pipeline, audit outputs."""
    case_dir = cases_root / case_id
    status_path = case_dir / "case_status.json"
    log_file = logs_root / f"{case_id}.log"

    safe_mkdir(case_dir)

    elems, active_elems = get_active_poscar_elements(poscar, ignore_elements={"H", "O"})
    if elems and not active_elems:
        status = make_unsupported_target_status(
            case_id=case_id,
            input_poscar=poscar,
            out_root=case_dir,
            pipeline_log=log_file,
        )
        write_case_status(status_path, status)
        standard_status = build_standard_case_status(status)
        write_case_status(case_dir / "case_status_standard.json", standard_status)
        return status

    if mode == "run_and_audit":
        case_poscar_dir = prepare_case_input(
            poscar_path=poscar,
            case_id=case_id,
            case_input_root=case_input_root,
        )

        rc = run_pipeline_v3_for_case(
            pipeline_v3_dir=pipeline_v3_dir,
            pipeline_v3_config=pipeline_v3_config,
            case_id=case_id,
            start_from=start_from,
            log_file=log_file,
        )

        if rc != 0:
            status = {
                "case_id": case_id,
                "input_poscar": str(poscar),
                "audit_time": now(),
                "pipeline_return_code": rc,
                "pipeline_log": str(log_file),
                "final_case_status": "pipeline_failed",
                "problem_type": "pipeline_failed",
                "recommended_action": "inspect_pipeline_log",
            }
            write_case_status(status_path, status)
            standard_status = build_standard_case_status(status)
            write_case_status(case_dir / "case_status_standard.json", standard_status)
            return status

    status = audit_case_outputs(
        project_root=project_root,
        case_id=case_id,
        audit_cfg=audit_cfg,
    )
    status["input_poscar"] = str(poscar)
    status["pipeline_log"] = str(log_file)

    write_case_status(status_path, status)
    standard_status = build_standard_case_status(status)
    write_case_status(case_dir / "case_status_standard.json", standard_status)
    return status


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", default="run_and_audit", choices=["run_and_audit", "audit_only"])
    ap.add_argument("--limit", type=int, default=0, help="Limit number of cases for testing.")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=1, help="Number of parallel workers (default 1 = sequential).")
    ap.add_argument("--recovery_mode", action="store_true", help="Only process cases marked as needing recovery in master_status.")
    ap.add_argument(
        "--case_id_filter",
        default="",
        help="Comma-separated case IDs to process/audit. Example: case_000001_x,case_000002_y",
    )
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))

    project_root = Path(cfg["project_root"]).expanduser().resolve()
    pipeline_v3_dir = Path(cfg["pipeline_v3_dir"]).expanduser().resolve()
    pipeline_v3_config = Path(cfg["pipeline_v3_config"]).expanduser().resolve()
    batch_poscar_dir = Path(cfg["batch_poscar_dir"]).expanduser().resolve()
    case_input_root = Path(cfg["case_input_root"]).expanduser().resolve()
    batch_output_root = Path(cfg["batch_output_root"]).expanduser().resolve()
    batch_name = str(cfg.get("batch_name", "batch_001"))
    start_from = str(cfg.get("pipeline_start_from", "make_infer_split"))
    resume = bool(cfg.get("resume", True)) and not args.force

    # Recovery start_from mapping (status -> pipeline step to restart from).
    recovery_start_from_map = {
        "needs_condition_reexport": "run_stage3_flow",
        "needs_stage3_recovery": "build_stage3_features",
        "needs_stage2_recovery": "sample_stage2_gflownet",
        "needs_route_finalization_recovery": "summarize_routes",
        "pipeline_failed": "make_infer_split",
    }

    out_root = batch_output_root / batch_name
    cases_root = out_root / "cases"
    logs_root = out_root / "logs"
    safe_mkdir(out_root)
    safe_mkdir(cases_root)
    safe_mkdir(logs_root)

    poscars = find_poscar_files(batch_poscar_dir)
    if args.limit and args.limit > 0:
        poscars = poscars[: args.limit]

    n_workers = max(1, args.workers)

    print("=" * 60)
    print("Adaptive batch pipeline")
    print(f"project_root      = {project_root}")
    print(f"batch_name        = {batch_name}")
    print(f"batch_poscar_dir  = {batch_poscar_dir}")
    print(f"n_poscars         = {len(poscars)}")
    print(f"mode              = {args.mode}")
    print(f"resume            = {resume}")
    print(f"workers           = {n_workers}")
    print(f"recovery_mode     = {args.recovery_mode}")
    print(f"case_id_filter    = {args.case_id_filter}")
    print(f"out_root          = {out_root}")
    print("=" * 60)

    all_status: List[Dict[str, Any]] = []

    case_id_filter = {x.strip() for x in str(args.case_id_filter).split(",") if x.strip()}

    # In recovery_mode, load existing master_status and filter to recovery-needed cases.
    recovery_case_ids: set = set()
    if args.recovery_mode:
        existing_master = out_root / "master_status.csv"
        if existing_master.exists():
            edf = pd.read_csv(existing_master)
            recovery_statuses = {
                "needs_stage2_recovery",
                "needs_stage3_recovery",
                "needs_condition_reexport",
                "needs_route_finalization_recovery",
                "needs_manual_or_rule_recovery",
                "pipeline_failed",
            }
            recovery_case_ids = set(
                edf[edf["final_case_status"].astype(str).isin(recovery_statuses)]["case_id"].tolist()
            )
            print(f"[RECOVERY] {len(recovery_case_ids)} cases need recovery")
        else:
            print("[RECOVERY] No existing master_status.csv found, running all cases.")

    # Build task list: (idx, poscar, case_id) tuples to process.
    tasks_to_run = []
    skipped_status = []

    for idx, poscar in enumerate(poscars, start=1):
        case_id = make_case_id(poscar, idx)

        if case_id_filter and case_id not in case_id_filter:
            continue

        if args.recovery_mode and recovery_case_ids and case_id not in recovery_case_ids:
            # In recovery mode, skip cases that don't need recovery but collect their status.
            status_path = cases_root / case_id / "case_status.json"
            if status_path.exists():
                old = read_json_if_exists(status_path)
                if old:
                    skipped_status.append(old)
            continue

        case_dir = cases_root / case_id
        status_path = case_dir / "case_status.json"

        # In recovery mode, don't skip cases that need recovery.
        is_recovery_target = args.recovery_mode and case_id in recovery_case_ids

        if resume and not is_recovery_target and status_path.exists():
            old = read_json_if_exists(status_path)
            if old.get("final_case_status") in [
                "pass",
                "review_required_only",
                "needs_stage2_recovery",
                "needs_stage3_recovery",
                "needs_condition_reexport",
                "needs_route_finalization_recovery",
                "needs_manual_or_rule_recovery",
                "review_required_low_condition_support",
                "high_confidence_review",
                "medium_confidence_review",
                "low_confidence_review",
                "needs_manual_check",
                "pass_high_confidence",
                "pass_medium_confidence",
                "pass_low_confidence",
                "pipeline_failed",
                "unsupported_target_elements",
            ]:
                skipped_status.append(old)
                continue

        tasks_to_run.append((idx, poscar, case_id))

    # Determine per-case start_from (recovery mode uses status-specific restart points).
    def get_start_from_for_case(case_id: str) -> str:
        if not args.recovery_mode:
            return start_from
        status_path = cases_root / case_id / "case_status.json"
        if status_path.exists():
            old = read_json_if_exists(status_path)
            old_status = old.get("final_case_status", "")
            return recovery_start_from_map.get(old_status, start_from)
        return start_from

    print(f"\n[PLAN] {len(tasks_to_run)} cases to process, {len(skipped_status)} already done/skipped")

    # Process cases (parallel or sequential).
    processed_status: List[Dict[str, Any]] = []

    if n_workers <= 1 or len(tasks_to_run) <= 1:
        # Sequential mode (original behavior).
        for idx, poscar, case_id in tasks_to_run:
            status = _process_one_case(
                idx=idx,
                poscar=poscar,
                case_id=case_id,
                n_total=len(poscars),
                cases_root=cases_root,
                logs_root=logs_root,
                case_input_root=case_input_root,
                pipeline_v3_dir=pipeline_v3_dir,
                pipeline_v3_config=pipeline_v3_config,
                start_from=get_start_from_for_case(case_id),
                project_root=project_root,
                audit_cfg=cfg.get("audit", {}),
                mode=args.mode,
            )
            processed_status.append(status)
    else:
        # Parallel mode.
        print(f"\n[PARALLEL] Launching {n_workers} workers for {len(tasks_to_run)} cases...")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_case = {}
            for idx, poscar, case_id in tasks_to_run:
                fut = executor.submit(
                    _process_one_case,
                    idx=idx,
                    poscar=poscar,
                    case_id=case_id,
                    n_total=len(poscars),
                    cases_root=cases_root,
                    logs_root=logs_root,
                    case_input_root=case_input_root,
                    pipeline_v3_dir=pipeline_v3_dir,
                    pipeline_v3_config=pipeline_v3_config,
                    start_from=get_start_from_for_case(case_id),
                    project_root=project_root,
                    audit_cfg=cfg.get("audit", {}),
                    mode=args.mode,
                )
                future_to_case[fut] = case_id

            n_done = 0
            for fut in as_completed(future_to_case):
                case_id = future_to_case[fut]
                n_done += 1
                try:
                    status = fut.result()
                    processed_status.append(status)
                    print(f"[{n_done}/{len(tasks_to_run)}] {case_id} -> {status.get('final_case_status')}")
                except Exception as e:
                    print(f"[{n_done}/{len(tasks_to_run)}] {case_id} -> EXCEPTION: {e}")
                    processed_status.append({
                        "case_id": case_id,
                        "audit_time": now(),
                        "final_case_status": "pipeline_failed",
                        "problem_type": "worker_exception",
                        "recommended_action": "inspect_exception",
                        "pipeline_log": str(e),
                    })

    all_status = skipped_status + processed_status
    master_df = pd.DataFrame(all_status)
    master_csv = out_root / "master_status.csv"
    master_json = out_root / "master_status.json"

    standard_status_list = [build_standard_case_status(x) for x in all_status]
    master_standard_df = pd.DataFrame(standard_status_list)
    master_standard_csv = out_root / "master_status_standard.csv"
    master_standard_json = out_root / "master_status_standard.json"

    report_md = out_root / "batch_report.md"

    master_df.to_csv(master_csv, index=False)
    master_json.write_text(
        json.dumps(all_status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    master_standard_df.to_csv(master_standard_csv, index=False)
    master_standard_json.write_text(
        json.dumps(standard_status_list, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    build_batch_report(master_df, report_md)

    print()
    print("=" * 60)
    print("[DONE] Adaptive batch pipeline")
    print(f"[SAVE] {master_csv}")
    print(f"[SAVE] {master_json}")
    print(f"[SAVE] {master_standard_csv}")
    print(f"[SAVE] {master_standard_json}")
    print(f"[SAVE] {report_md}")
    print("=" * 60)


if __name__ == "__main__":
    main()
