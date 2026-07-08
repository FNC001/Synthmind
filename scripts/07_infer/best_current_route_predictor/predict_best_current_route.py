#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


CORE_METHODS = {"solid_state", "solution", "melt_arc"}


CORE_ARTIFACTS = {
    "stage2_dataset": "data/interim/generative/stage2_setpred_dataset/descriptor/route_method_stratified_canonical_v4_20260610_relaxed_only",
    "stage2_candidate_pool": "outputs/evaluation/stage2_score_calibration_v5_20260610",
    "stage2_candidate_pool_repaired": "outputs/evaluation/stage2_candidate_pool_v5_20260610",
    "stage3_dataset_chem_checked": "data/interim/generative/stage3_condition_dataset_chem_checked/method_stratified_v5_20260610",
    "stage3_predicted_precursor_dataset": "data/interim/generative/stage3_condition_dataset_predprec_oof_v3_20260610",
    "stage3_condition_targets": "data/interim/generative/stage3_condition_targets_v3_20260610",
    "stage3_model": "runs/stage3/distributional_condition_v3_20260610/stage3_distributional_condition_v3.joblib",
    "stage3_condition_candidates": "outputs/evaluation/stage3_condition_calibration_v3_final_20260612",
    "stage3_condition_candidates_fallback_v2": "outputs/evaluation/stage3_condition_calibration_v2_20260610",
    "stage35_route_candidates": "outputs/evaluation/stage35_route_candidates_v3_final_20260612",
    "stage35_route_score_calibration": "outputs/evaluation/stage35_route_score_calibration_v3_final_20260612",
    "stage35_reranker": "runs/stage35/route_reranker_v3_final_20260612/stage35_route_reranker_v3_final.joblib",
    "stage35_raw_score_fallback": "route_calibrated_score_v3_final",
}

FALLBACK_ARTIFACTS = {
    **CORE_ARTIFACTS,
}


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Select the current best SynPred route-prediction artifacts for core/non-core methods.")
    ap.add_argument("--formula", default="")
    ap.add_argument("--poscar", default="")
    ap.add_argument("--reaction_method", default="", help="solid_state, solution, melt_arc, or a non-core method.")
    ap.add_argument("--output_json", default="")
    args = ap.parse_args()

    method = str(args.reaction_method).strip()
    if method in CORE_METHODS:
        mode = "stage2_v5_stage3_v3_stage35_v3_core"
        confidence = "normal"
        artifacts = CORE_ARTIFACTS
        notes = [
            "Using Stage2 v5 chemistry-checked precursor candidates.",
            "Using Stage3 v3 distributional condition candidates with missing-aware atmosphere/solvent labels.",
            "Recommended temperature and time should be emitted as intervals, not as unique exact values.",
            "Using final Stage35 reranker v3 blend by default because it improved held-out test top1/top10 under both missing-aware and strict-comparable protocols.",
        ]
    elif method:
        mode = "stage2_v5_stage3_v3_all_method_fallback"
        confidence = "medium_low"
        artifacts = FALLBACK_ARTIFACTS
        notes = [
            "Reaction method is outside the core set; use all-method fallback and lower route confidence.",
            "If atmosphere or solvent is missing/low confidence, mark the field as low confidence in user-facing output.",
            "Open-generated or repaired precursor candidates should lower route confidence.",
        ]
    else:
        mode = "stage2_v5_stage3_v3_core_method_sweep"
        confidence = "medium_low"
        artifacts = CORE_ARTIFACTS
        notes = ["No reaction_method was supplied; output solid_state, solution, and melt_arc routes and compare Stage35 v3 scores."]

    result = {
        "formula": args.formula,
        "poscar": args.poscar,
        "reaction_method": method,
        "mode": mode,
        "confidence": confidence,
        "core_methods": sorted(CORE_METHODS),
        "artifacts": artifacts,
        "default_ranking": "stage35_reranker_v3_final_blend_score",
        "fallback_ranking": "route_calibrated_score_v3_final",
        "default_ranking_note": "Blend final Stage35 LambdaRank score with raw route score using validation-selected alpha=0.70.",
        "condition_output_style": "temperature/time intervals plus atmosphere/solvent confidence",
        "requires_chemistry_checked_precursors": True,
        "uses_missing_aware_condition_labels": True,
        "notes": notes,
    }
    if args.output_json:
        write_json(Path(args.output_json), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
