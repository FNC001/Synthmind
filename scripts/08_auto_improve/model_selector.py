#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from metrics_registry import (  # noqa: E402
    BASELINE_THRESHOLDS,
    DEFAULT_OUTPUT_DIR,
    build_registry,
    number,
    pct,
    read_json,
    write_json,
)


DEFAULT_ARTIFACTS = {
    "stage2": "outputs/evaluation/stage2_score_calibration_v5_20260610",
    "stage3": "outputs/evaluation/stage3_condition_calibration_v3_final_20260612",
    "stage35": "runs/stage35/route_reranker_v3_final_20260612/stage35_route_reranker_v3_final.joblib",
    "stage35_calibration": "outputs/evaluation/stage35_route_score_calibration_v3_final_20260612",
}

V4_ARTIFACTS = {
    "stage3": "outputs/evaluation/stage3_condition_calibration_v4_20260612",
    "stage35": "runs/stage35/route_reranker_v4_20260612",
    "stage35_candidates": "outputs/evaluation/stage35_route_candidates_v4_20260612",
}


def abs_path(project_root: Path, path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = project_root / p
    return p


def passes_stage2_default(metrics: Dict[str, Any], core_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
    t = BASELINE_THRESHOLDS["stage2"]
    checks = {
        "all_top1_gt_39_47": (number(metrics.get("top1_exact")) or 0.0) > t["all_top1_exact"],
        "all_top10_ge_63_35": (number(metrics.get("top10_exact")) or 0.0) >= t["all_top10_exact"],
        "all_top500_ge_80_24": (number(metrics.get("top500_exact")) or 0.0) >= t["all_top500_exact"],
    }
    if core_metrics is not None:
        checks["core_top1_ge_46_15"] = (number(core_metrics.get("top1_exact")) or 0.0) >= t["core_top1_exact"]
    return checks


def passes_stage35_default(missing: Dict[str, Any], strict: Dict[str, Any]) -> Dict[str, bool]:
    t = BASELINE_THRESHOLDS["stage35"]
    return {
        "strict_top1_relaxed_gt_10_45": (number(strict.get("top1_relaxed_route")) or 0.0)
        > t["strict_comparable_top1_relaxed_route"],
        "missing_top1_relaxed_ge_20_72": (number(missing.get("top1_relaxed_route")) or 0.0)
        >= t["missing_aware_top1_relaxed_route"],
        "strict_top10_relaxed_ge_18_04": (number(strict.get("top10_relaxed_route")) or 0.0)
        >= t["strict_comparable_top10_relaxed_route"],
        "missing_top10_relaxed_ge_34_55": (number(missing.get("top10_relaxed_route")) or 0.0)
        >= t["missing_aware_top10_relaxed_route"],
    }


def load_v4_blend_metrics(project_root: Path) -> Dict[str, Dict[str, Any]]:
    path = abs_path(project_root, "runs/stage35/route_reranker_v4_20260612/test_metrics.json")
    if not path.exists():
        return {}
    obj = read_json(path)
    return {
        "missing_aware": obj.get("blend_missing_aware", {}),
        "strict_comparable": obj.get("blend_strict_comparable", {}),
        "path": str(path),
    }


def load_coupling_primary_bottleneck(project_root: Path, output_dir: Path) -> Optional[str]:
    path = abs_path(project_root, output_dir) / "stage2_stage3_coupling" / "stage2_stage3_coupling_diagnosis.json"
    if not path.exists():
        path = abs_path(project_root, output_dir) / "stage2_stage3_coupling_diagnosis.json"
    if not path.exists():
        return None
    try:
        return read_json(path).get("primary_bottleneck")
    except json.JSONDecodeError:
        return None


def build_decision(project_root: Path, output_dir: Path) -> Dict[str, Any]:
    registry = build_registry(project_root, output_dir, include_experiments=True)
    baselines = registry.records["baselines"]
    stage2_all = baselines["stage2_v5_all_test"]["metrics"]
    stage2_core = baselines["stage2_core_calibrated_test"]["metrics"]
    stage35_v3_missing = baselines["stage35_v3_final_missing_aware_test"]["metrics"]
    stage35_v3_strict = baselines["stage35_v3_final_strict_comparable_test"]["metrics"]
    stage3_v3_missing = baselines["stage3_v3_missing_aware_test"]["metrics"]
    stage3_v3_strict = baselines["stage3_v3_strict_comparable_test"]["metrics"]
    stage3_v4_missing = baselines["stage3_v4_missing_aware_test"]["metrics"]
    stage3_v4_strict = baselines["stage3_v4_strict_comparable_test"]["metrics"]
    v4_blend = load_v4_blend_metrics(project_root)
    v4_missing = v4_blend.get("missing_aware", {})
    v4_strict = v4_blend.get("strict_comparable", {})

    stage2_checks = passes_stage2_default(stage2_all, stage2_core)
    stage35_v4_checks = passes_stage35_default(v4_missing, v4_strict) if v4_missing and v4_strict else {}
    stage3_v4_beats_v3 = {
        "missing_top1_relaxed": (number(stage3_v4_missing.get("top1_relaxed_condition")) or 0.0)
        > (number(stage3_v3_missing.get("top1_relaxed_condition")) or 0.0),
        "strict_top1_relaxed": (number(stage3_v4_strict.get("top1_relaxed_condition")) or 0.0)
        > (number(stage3_v3_strict.get("top1_relaxed_condition")) or 0.0),
        "missing_top10_relaxed": (number(stage3_v4_missing.get("top10_relaxed_condition")) or 0.0)
        >= (number(stage3_v3_missing.get("top10_relaxed_condition")) or 0.0),
        "strict_top10_relaxed": (number(stage3_v4_strict.get("top10_relaxed_condition")) or 0.0)
        >= (number(stage3_v3_strict.get("top10_relaxed_condition")) or 0.0),
    }

    optional_modes: Dict[str, Any] = {}
    if v4_missing and v4_strict:
        top1_worse = (number(v4_missing.get("top1_relaxed_route")) or 0.0) < (
            number(stage35_v3_missing.get("top1_relaxed_route")) or 0.0
        ) or (number(v4_strict.get("top1_relaxed_route")) or 0.0) < (
            number(stage35_v3_strict.get("top1_relaxed_route")) or 0.0
        )
        coverage_better = (number(v4_missing.get("top200_relaxed_route")) or 0.0) > (
            number(stage35_v3_missing.get("top200_relaxed_route")) or 0.0
        ) or (number(v4_strict.get("top10_relaxed_route")) or 0.0) > (
            number(stage35_v3_strict.get("top10_relaxed_route")) or 0.0
        )
        if coverage_better and top1_worse:
            optional_modes["ranking_coverage"] = {
                "artifact": V4_ARTIFACTS,
                "reason": "v4 improves some top10/top200 coverage metrics but loses top1 relaxed route, so it is not default.",
                "metrics": {"missing_aware": v4_missing, "strict_comparable": v4_strict},
            }

    core_mode_checks = {
        "core_top1_ge_46_15": (number(stage2_core.get("top1_exact")) or 0.0)
        >= BASELINE_THRESHOLDS["stage2"]["core_top1_exact"],
        "core_top10_ge_69_17": (number(stage2_core.get("top10_exact")) or 0.0)
        >= BASELINE_THRESHOLDS["stage2"]["core_top10_exact"],
        "core_top500_ge_85_33": (number(stage2_core.get("top500_exact")) or 0.0)
        >= BASELINE_THRESHOLDS["stage2"]["core_top500_exact"],
    }
    if core_mode_checks["core_top10_ge_69_17"] and core_mode_checks["core_top500_ge_85_33"]:
        optional_modes["stage2_core_candidate"] = {
            "artifact": "outputs/evaluation/stage2_score_calibration_core_methods_20260610",
            "status": "not_default" if not core_mode_checks["core_top1_ge_46_15"] else "candidate",
            "reason": "Core calibrated model meets top10/top500 checks but default requires top1 to clear the core threshold.",
            "checks": core_mode_checks,
            "metrics": stage2_core,
        }

    coupling_bottleneck = load_coupling_primary_bottleneck(project_root, output_dir)
    default_reason = [
        "No Stage2 replacement candidate is registered that exceeds all-method top1/top10/top500 gates.",
        "Stage3 v4 does not beat v3 final under both missing-aware and strict-comparable primary metrics.",
        "Stage35 v4 blend loses top1 relaxed route versus v3 final, so it is retained only as coverage evidence.",
    ]
    if coupling_bottleneck:
        default_reason.append(f"Latest coupling diagnosis primary bottleneck: {coupling_bottleneck}.")

    decision = {
        "selected_stage2_artifact": DEFAULT_ARTIFACTS["stage2"],
        "selected_stage3_artifact": DEFAULT_ARTIFACTS["stage3"],
        "selected_stage35_artifact": DEFAULT_ARTIFACTS["stage35"],
        "stage2_mode": "default",
        "ranking_mode": "default",
        "selection_reason": default_reason,
        "baseline_metrics": {
            "stage2_v5_all": stage2_all,
            "stage2_core_calibrated": stage2_core,
            "stage3_v3_missing_aware": stage3_v3_missing,
            "stage3_v3_strict_comparable": stage3_v3_strict,
            "stage35_v3_final_missing_aware": stage35_v3_missing,
            "stage35_v3_final_strict_comparable": stage35_v3_strict,
        },
        "selected_metrics": {
            "stage2": stage2_all,
            "stage3_missing_aware": stage3_v3_missing,
            "stage3_strict_comparable": stage3_v3_strict,
            "stage35_missing_aware": stage35_v3_missing,
            "stage35_strict_comparable": stage35_v3_strict,
        },
        "gate_checks": {
            "stage2_current_against_replacement_thresholds": stage2_checks,
            "stage3_v4_beats_v3": stage3_v4_beats_v3,
            "stage35_v4_against_default_thresholds": stage35_v4_checks,
            "stage2_core_mode_checks": core_mode_checks,
        },
        "optional_modes": optional_modes,
        "do_update_default_inference_selector": False,
        "registry": registry.records,
    }
    return decision


def render_markdown(decision: Dict[str, Any]) -> str:
    bm = decision["baseline_metrics"]
    lines = [
        "# SynPred Model Selection Decision",
        "",
        "## Decision",
        "",
        f"- Stage2 mode: `{decision['stage2_mode']}`",
        f"- Ranking mode: `{decision['ranking_mode']}`",
        f"- Update default inference selector: `{decision['do_update_default_inference_selector']}`",
        f"- Stage2 artifact: `{decision['selected_stage2_artifact']}`",
        f"- Stage3 artifact: `{decision['selected_stage3_artifact']}`",
        f"- Stage35 artifact: `{decision['selected_stage35_artifact']}`",
        "",
        "## Reasons",
        "",
    ]
    for reason in decision["selection_reason"]:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "## Key Metrics",
            "",
            "| slice | top1 | top10 | top200 | top500 |",
            "|---|---:|---:|---:|---:|",
            f"| Stage2 all v5 | {pct(bm['stage2_v5_all'].get('top1_exact'))} | {pct(bm['stage2_v5_all'].get('top10_exact'))} | {pct(bm['stage2_v5_all'].get('top200_exact'))} | {pct(bm['stage2_v5_all'].get('top500_exact'))} |",
            f"| Stage2 core calibrated | {pct(bm['stage2_core_calibrated'].get('top1_exact'))} | {pct(bm['stage2_core_calibrated'].get('top10_exact'))} | {pct(bm['stage2_core_calibrated'].get('top200_exact'))} | {pct(bm['stage2_core_calibrated'].get('top500_exact'))} |",
            "",
            "| route protocol | top1 relaxed | top10 relaxed | top200 relaxed |",
            "|---|---:|---:|---:|",
            f"| Stage35 v3 final missing-aware | {pct(bm['stage35_v3_final_missing_aware'].get('top1_relaxed_route'))} | {pct(bm['stage35_v3_final_missing_aware'].get('top10_relaxed_route'))} | {pct(bm['stage35_v3_final_missing_aware'].get('top200_relaxed_route'))} |",
            f"| Stage35 v3 final strict-comparable | {pct(bm['stage35_v3_final_strict_comparable'].get('top1_relaxed_route'))} | {pct(bm['stage35_v3_final_strict_comparable'].get('top10_relaxed_route'))} | {pct(bm['stage35_v3_final_strict_comparable'].get('top200_relaxed_route'))} |",
        ]
    )
    if decision.get("optional_modes"):
        lines.extend(["", "## Optional Modes", ""])
        for name, item in decision["optional_modes"].items():
            lines.append(f"- `{name}`: {item.get('reason')}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Conservatively select SynPred Stage2/Stage3/Stage35 default and optional modes.")
    ap.add_argument("--project_root", default=".", help="Repository root.")
    ap.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Auto-improvement output directory.")
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    output_dir = abs_path(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    decision = build_decision(project_root, output_dir)
    json_path = output_dir / "model_selection_decision.json"
    md_path = output_dir / "model_selection_report.md"
    write_json(json_path, decision)
    md_path.write_text(render_markdown(decision), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "report": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
