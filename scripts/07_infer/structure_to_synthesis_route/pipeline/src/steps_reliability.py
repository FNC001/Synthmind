#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _get(cfg: dict, dotted_key: str, default=None):
    cur = cfg
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

def _get_bool(cfg: dict, key: str, default: bool = True) -> bool:
    cur = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return bool(cur)


def _get_int(cfg: dict, key: str, default: int) -> int:
    cur = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    try:
        return int(cur)
    except Exception:
        return default


def _get_str(cfg: dict, dotted_key: str, default: str = "") -> str:
    """
    Read a dotted config key as string.

    Example:
      _get_str(cfg, "reliability.stage3_condition_reference.reference_csv", "")
    """
    cur = cfg
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]

    if cur is None:
        return default

    return str(cur)

def _bool_enabled(x, default=False) -> bool:
    """Safely parse enabled-like config values."""
    if x is None:
        return bool(default)
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _pipeline_project_root(cfg: dict) -> Path:
    root = cfg.get("project_root")
    if not root:
        raise ValueError("project_root must be set in pipeline config")
    return Path(str(root)).expanduser().resolve()


def _pipeline_dir_from_cfg(cfg: dict) -> Path:
    """
    Return pipeline_v3 directory.

    This file lives in:
      pipeline_v3/src/steps_reliability.py

    so parent.parent should be:
      pipeline_v3
    """
    return Path(__file__).resolve().parents[1]


def _registry_path_from_cfg(cfg: dict) -> Path:
    pipeline_dir = _pipeline_dir_from_cfg(cfg)
    return pipeline_dir / "configs" / "stage3_reference_registry.json"


def _load_stage3_reference_registry(cfg: dict) -> dict:
    p = _registry_path_from_cfg(cfg)
    if not p.exists():
        print(f"[WARN] missing stage3 reference registry: {p}")
        return {}

    import json
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] failed to read stage3 reference registry: {p} | {e}")
        return {}


def _resolve_project_path(project_root: Path, x) -> Path:
    p = Path(str(x))
    return p if p.is_absolute() else project_root / p


def _run_shell_script_if_needed(
    *,
    project_root: Path,
    script_path: Path,
    summary_json: Path,
    step_name: str,
    force: bool = False,
) -> dict:
    """
    Run one benchmark shell script only if needed.

    Returns a small status dict for pipeline manifest/debug.
    """
    if not script_path.exists():
        print(f"[SKIP missing] {step_name}: {script_path}")
        return {
            "step": step_name,
            "status": "missing_script",
            "script": str(script_path),
            "summary_json": str(summary_json),
        }

    if summary_json.exists() and not force:
        print(f"[KEEP existing] {step_name}: {summary_json}")
        return {
            "step": step_name,
            "status": "kept_existing_summary",
            "script": str(script_path),
            "summary_json": str(summary_json),
        }

    cmd = ["bash", str(script_path), str(project_root)]
    print(f"===== GAP-RECOVERY: {step_name} =====")
    print("[RUN]", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[WARN] {step_name} failed with returncode={e.returncode}")
        return {
            "step": step_name,
            "status": "failed",
            "returncode": int(e.returncode),
            "script": str(script_path),
            "summary_json": str(summary_json),
        }

    if summary_json.exists():
        return {
            "step": step_name,
            "status": "pass",
            "script": str(script_path),
            "summary_json": str(summary_json),
        }

    return {
        "step": step_name,
        "status": "completed_but_missing_summary",
        "script": str(script_path),
        "summary_json": str(summary_json),
    }
def run_stage3_gap_recovery_if_enabled(r, cfg: dict) -> None:
    """
    Run V41/V42/V43/V43b/V44 gap-recovery chain from pipeline_v3.

    V41: audit remaining V33 targets for Stage3 feature regeneration readiness
    V42: inspect local MP JSON files for structure-like payloads
    V43: reconstruct POSCAR files from MP JSON payloads
    V43b: validate recovered POSCAR files
    V44: regenerate Stage3-compatible NPZ input extension from validated POSCAR files

    This layer prepares structure-backed Stage3 inputs.
    It does not fabricate MDN/Flow condition candidates and does not claim experimental validation.
    """
    refresh_cfg = cfg.get("stage3_reference_refresh", {}) or {}
    gap_cfg = refresh_cfg.get("gap_recovery", {}) or {}

    if not _bool_enabled(gap_cfg.get("enabled", False)):
        print("[SKIP disabled] stage3 gap recovery")
        return

    project_root = _pipeline_project_root(cfg)
    registry = _load_stage3_reference_registry(cfg)
    force = _bool_enabled(gap_cfg.get("force", False))

    print("===== AUX: Stage3 gap recovery V41/V42/V43/V43b/V44 =====")
    print(f"[INFO] project_root = {project_root}")
    print(f"[INFO] force        = {force}")

    jobs = [
        (
            "run_v41_feature_regeneration_readiness",
            "v41_feature_regeneration_readiness",
            "v41",
        ),
        (
            "run_v42_recover_structure_sources_from_mp_json",
            "v42_recover_structure_sources_from_mp_json",
            "v42",
        ),
        (
            "run_v43_reconstruct_poscar_from_mp_json",
            "v43_reconstruct_poscar_from_mp_json",
            "v43",
        ),
        (
            "run_v43b_validate_recovered_poscar",
            "v43b_validate_recovered_poscar",
            "v43b",
        ),
        (
            "run_v44_stage3_feature_npz_regeneration",
            "v44_stage3_feature_npz_regeneration",
            "v44",
        ),
    ]

    results = []

    for cfg_key, step_name, registry_key in jobs:
        step_cfg = gap_cfg.get(cfg_key, {}) or {}

        if not _bool_enabled(step_cfg.get("enabled", False)):
            print(f"[SKIP disabled] {step_name}")
            continue

        reg = registry.get(registry_key, {}) or {}
        script_value = reg.get("run_script", "")
        summary_value = reg.get("summary_json", "")

        if not script_value:
            print(f"[SKIP missing registry] {step_name}: registry.{registry_key}.run_script is empty")
            results.append({
                "step": step_name,
                "status": "missing_registry_run_script",
                "registry_key": registry_key,
            })
            continue

        if not summary_value:
            print(f"[SKIP missing registry] {step_name}: registry.{registry_key}.summary_json is empty")
            results.append({
                "step": step_name,
                "status": "missing_registry_summary_json",
                "registry_key": registry_key,
            })
            continue

        script = _resolve_project_path(project_root, script_value)
        summary = _resolve_project_path(project_root, summary_value)

        result = _run_shell_script_if_needed(
            project_root=project_root,
            script_path=script,
            summary_json=summary,
            step_name=step_name,
            force=force,
        )
        results.append(result)

        # 如果某一步真正失败，后面的步骤通常依赖它，建议停止继续跑。
        if result.get("status") == "failed":
            print(f"[STOP] {step_name} failed; downstream gap-recovery steps are skipped.")
            break

    import json

    out_dir = project_root / "outputs" / "stage3_gap_recovery"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_json = out_dir / "stage3_gap_recovery_summary.json"

    good_status = {"pass", "kept_existing_summary"}
    overall_status = (
        "pass"
        if results and all(x.get("status") in good_status for x in results)
        else "pass_with_warnings"
    )

    summary = {
        "status": overall_status,
        "project_root": str(project_root),
        "force": bool(force),
        "results": results,
        "claim_boundary": "stage3_gap_recovery_prepares_structure_backed_stage3_inputs_not_experimental_validation",
        "interpretation": (
            "This pipeline layer runs V41/V42/V43/V43b/V44 to recover structures and regenerate "
            "Stage3-compatible input features for remaining V33 gaps. It does not fabricate or validate "
            "experimental synthesis routes."
        ),
    }

    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[SAVE] {summary_json}")

    try:
        r.outputs["stage3_gap_recovery_summary_json"] = str(summary_json)

        v44 = registry.get("v44", {}) or {}
        if v44.get("stage3_input_dir"):
            r.outputs["v44_stage3_input_extension_dir"] = str(
                _resolve_project_path(project_root, v44["stage3_input_dir"])
            )
        if v44.get("stage2_candidates_dir"):
            r.outputs["v44_stage2_candidate_subset_dir"] = str(
                _resolve_project_path(project_root, v44["stage2_candidates_dir"])
            )
        if v44.get("summary_json"):
            r.outputs["v44_stage3_feature_npz_regeneration_summary_json"] = str(
                _resolve_project_path(project_root, v44["summary_json"])
            )
    except Exception as e:
        print(f"[WARN] gap recovery output registration failed: {e}")

def run_stage3_gap_closure_if_enabled(r, cfg: dict) -> None:
    """
    Run V44/V45/V45b/V48/V49/V50 gap-closure chain from pipeline_v3.

    This layer orchestrates existing benchmark scripts. It does not fabricate
    Stage3 candidates. Each sub-step is controlled by config:

      stage3_reference_refresh.gap_closure.*

    Recommended main path:
      V44 -> V45 -> V45b -> V48 -> V49 -> V50

    Diagnostic / optional:
      V45b_register can stay disabled unless a fast patch registration script
      is available and desired.
    """
    refresh_cfg = cfg.get("stage3_reference_refresh", {}) or {}
    gap_cfg = refresh_cfg.get("gap_closure", {}) or {}

    if not _bool_enabled(gap_cfg.get("enabled", False)):
        print("[SKIP disabled] stage3 gap closure")
        return

    project_root = _pipeline_project_root(cfg)
    registry = _load_stage3_reference_registry(cfg)
    force = _bool_enabled(gap_cfg.get("force", False))

    print("===== AUX: Stage3 gap closure V44/V45/V45b/V48/V49/V50 =====")
    print(f"[INFO] project_root = {project_root}")
    print(f"[INFO] force        = {force}")

    results = []

    def run_registered_step(config_key: str, registry_key: str, step_name: str):
        block = gap_cfg.get(config_key, {}) or {}

        if not _bool_enabled(block.get("enabled", False)):
            print(f"[SKIP disabled] {step_name}")
            return

        reg = registry.get(registry_key, {}) or {}
        script = _resolve_project_path(project_root, reg.get("run_script", ""))
        summary = _resolve_project_path(project_root, reg.get("summary_json", ""))

        result = _run_shell_script_if_needed(
            project_root=project_root,
            script_path=script,
            summary_json=summary,
            step_name=step_name,
            force=force,
        )
        results.append(result)

        # Store useful outputs when present.
        try:
            r.outputs[f"{registry_key}_summary_json"] = str(summary)
            for k, v in reg.items():
                if isinstance(v, str) and (
                    k.endswith("_csv")
                    or k.endswith("_json")
                    or k.endswith("_md")
                    or k.endswith("_dir")
                ):
                    r.outputs[f"{registry_key}_{k}"] = str(_resolve_project_path(project_root, v))
        except Exception as e:
            print(f"[WARN] gap closure output registration for {registry_key} failed: {e}")

    run_registered_step(
        "run_v44_stage3_feature_npz_regeneration",
        "v44",
        "v44_stage3_feature_npz_regeneration",
    )

    run_registered_step(
        "run_v45_real_stage3_export_from_v44",
        "v45",
        "v45_real_stage3_export_from_v44",
    )

    run_registered_step(
        "run_v45b_normalize_v45_candidates",
        "v45b",
        "v45b_normalize_v45_candidates_and_realign",
    )

    run_registered_step(
        "run_v45b_register_patch",
        "v45b_register",
        "v45b_register_normalized_patch",
    )

    run_registered_step(
        "run_v48_global_clip_flow_export",
        "v48",
        "v48_global_clip_flow_export",
    )

    run_registered_step(
        "run_v49_merge_and_realign_all_gap_cases",
        "v49",
        "v49_merge_v40c_v48_and_realign_all_gap_cases",
    )

    run_registered_step(
        "run_v50_final_gap_closure_report",
        "v50",
        "v50_final_gap_closure_report",
    )

    import json

    out_dir = project_root / "outputs" / "stage3_gap_closure"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_json = out_dir / "stage3_gap_closure_summary.json"

    ok_status = {"pass", "kept_existing_summary"}

    summary = {
        "status": "pass" if results and all(x.get("status") in ok_status for x in results) else "pass_with_warnings",
        "project_root": str(project_root),
        "force": bool(force),
        "results": results,
        "claim_boundary": "stage3_gap_closure_orchestrates_internal_benchmark_outputs_not_experimental_validation",
        "interpretation": (
            "This pipeline layer runs the V44/V45/V45b/V48/V49/V50 gap-closure chain. "
            "It reuses existing benchmark scripts and records their summaries. "
            "It does not fabricate Stage3 candidates or claim experimental validation."
        ),
    }

    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[SAVE] {summary_json}")

    r.outputs["stage3_gap_closure_summary_json"] = str(summary_json)

def _project_root(r, cfg: dict) -> Path:
    return Path(cfg.get("project_root", getattr(r, "project_root", "")))


def _first_existing(paths: list[str | Path]) -> Path | None:
    for p in paths:
        if not p:
            continue
        pp = Path(p)
        if pp.exists():
            return pp
    return None


def _set_current_csv(r, csv_path: Path, md_path: Path | None = None) -> None:
    """
    Reliability-layer current route table.

    This key is used to avoid losing newly attached columns when later steps
    still read historical names such as final_top_routes_with_confidence.csv.
    """
    r.outputs["final_top_routes_current_csv"] = str(csv_path)
    if md_path is not None:
        r.outputs["final_top_routes_current_md"] = str(md_path)


def _get_current_csv(r, route_out_dir: Path) -> Path:
    candidates = [
        r.outputs.get("final_top_routes_current_csv", ""),
        r.outputs.get("final_top_routes_with_metadata_stage3_reference_csv", ""),
        str(route_out_dir / "final_top_routes_with_metadata_stage3_reference.csv"),
        r.outputs.get("final_top_routes_with_stage3_condition_reference_csv", ""),
        str(route_out_dir / "final_top_routes_with_stage3_condition_reference.csv"),
        r.outputs.get("final_top_routes_with_condition_confidence_csv", ""),
        str(route_out_dir / "final_top_routes_with_condition_confidence.csv"),
        r.outputs.get("final_top_routes_with_confidence_csv", ""),
        str(route_out_dir / "final_top_routes_with_confidence.csv"),
        r.outputs.get("final_top_routes_with_precursor_qc_csv", ""),
        str(route_out_dir / "final_top_routes_with_precursor_qc.csv"),
        r.outputs.get("final_top_routes_csv", ""),
        str(route_out_dir / "final_top_routes.csv"),
    ]
    p = _first_existing(candidates)
    return p if p is not None else route_out_dir / "final_top_routes.csv"


def run_precursor_qc_if_enabled(r, cfg: dict) -> None:
    if not _get_bool(cfg, "reliability.precursor_qc.enabled", True):
        print("[SKIP disabled] precursor-level QC")
        return

    route_out_dir = Path(r.outputs.get("route_out_dir", ""))
    if not route_out_dir.exists():
        print("[SKIP] precursor QC; route_out_dir missing.")
        return

    input_csv = Path(r.outputs.get("final_top_routes_csv", ""))
    if not input_csv.exists():
        input_csv = route_out_dir / "final_top_routes.csv"

    qc_script = ROOT / "scripts" / "qc_route_precursors.py"
    qc_csv = route_out_dir / "final_top_routes_with_precursor_qc.csv"
    qc_md = route_out_dir / "final_top_routes_with_precursor_qc.md"
    qc_summary_json = route_out_dir / "final_top_routes_with_precursor_qc_summary.json"

    if not input_csv.exists() or not qc_script.exists():
        print("[SKIP] route precursor QC; missing final_top_routes_csv or qc script.")
        return

    print("===== FINAL+0.5: route precursor QC =====")
    r.run([
        "python",
        str(qc_script),
        "--input_csv", str(input_csv),
        "--output_csv", str(qc_csv),
        "--output_md", str(qc_md),
        "--summary_json", str(qc_summary_json),
        "--top_n", str(_get_int(cfg, "final.top_n", 30)),
    ])

    r.outputs["final_top_routes_with_precursor_qc_csv"] = str(qc_csv)
    r.outputs["final_top_routes_with_precursor_qc_md"] = str(qc_md)
    r.outputs["final_top_routes_with_precursor_qc_summary_json"] = str(qc_summary_json)
    r.outputs["final_top_routes_csv_for_confidence"] = str(qc_csv)
    _set_current_csv(r, qc_csv, qc_md)


def attach_route_confidence_if_enabled(r, cfg: dict) -> None:
    if not _get_bool(cfg, "reliability.route_confidence.enabled", True):
        print("[SKIP disabled] attach route confidence")
        return

    route_out_dir = Path(r.outputs.get("route_out_dir", ""))
    if not route_out_dir.exists():
        print("[SKIP] route confidence; route_out_dir missing.")
        return

    confidence_script = ROOT / "scripts" / "attach_route_confidence.py"

    input_csv = _first_existing([
        r.outputs.get("final_top_routes_current_csv", ""),
        r.outputs.get("final_top_routes_csv_for_confidence", ""),
        r.outputs.get("final_top_routes_with_precursor_qc_csv", ""),
        str(route_out_dir / "final_top_routes_with_precursor_qc.csv"),
        r.outputs.get("final_top_routes_csv", ""),
        str(route_out_dir / "final_top_routes.csv"),
    ])

    if input_csv is None:
        input_csv = route_out_dir / "final_top_routes.csv"

    output_csv = route_out_dir / "final_top_routes_with_confidence.csv"
    output_md = route_out_dir / "final_top_routes_with_confidence.md"
    summary_json = route_out_dir / "final_top_routes_with_confidence_summary.json"

    if not input_csv.exists() or not confidence_script.exists():
        print("[SKIP] route confidence; missing input csv or confidence script.")
        return

    print("===== FINAL+1: attach route confidence =====")
    print(f"[INFO] confidence input csv: {input_csv}")
    r.run([
        "python",
        str(confidence_script),
        "--input_csv", str(input_csv),
        "--output_csv", str(output_csv),
        "--output_md", str(output_md),
        "--summary_json", str(summary_json),
    ])

    r.outputs["final_top_routes_with_confidence_csv"] = str(output_csv)
    r.outputs["final_top_routes_with_confidence_md"] = str(output_md)
    r.outputs["final_top_routes_with_confidence_summary_json"] = str(summary_json)
    _set_current_csv(r, output_csv, output_md)


def attach_stage3_condition_reference_support_if_enabled(r, cfg: dict) -> None:
    """
    FINAL+1.1:
    Attach real Stage3 condition-reference support score.

    This uses a pre-built real Stage3 MDN/Flow condition candidate library
    as the reference distribution for temperature/time plausibility.
    """
    if not _get_bool(cfg, "reliability.attach_stage3_condition_reference_support.enabled", True):
        print("[SKIP disabled] attach_stage3_condition_reference_support")
        return

    if "route_out_dir" not in r.outputs:
        print("[SKIP] attach stage3 condition reference support; missing route_out_dir.")
        return

    route_out_dir = Path(r.outputs["route_out_dir"])
    script = ROOT / "scripts" / "attach_stage3_condition_reference_support.py"

    input_csv = _get_current_csv(r, route_out_dir)

    reference_csv = Path(
        _get_str(
            cfg,
            "reliability.attach_stage3_condition_reference_support.reference_csv",
            str(
                Path(r.project_root)
                / "data"
                / "interim"
                / "references"
                / "stage3_condition_reference"
                / "current"
                / "stage3_condition_reference.csv"
            ),
        )
    )
    if not reference_csv.is_absolute():
        reference_csv = Path(r.project_root) / reference_csv

    output_csv = route_out_dir / "final_top_routes_with_stage3_condition_reference.csv"
    output_md = route_out_dir / "final_top_routes_with_stage3_condition_reference.md"
    summary_json = route_out_dir / "final_top_routes_with_stage3_condition_reference_summary.json"

    if not script.exists():
        print(f"[SKIP] attach stage3 condition reference support; missing script: {script}")
        return

    if not input_csv.exists():
        print(f"[SKIP] attach stage3 condition reference support; missing input csv: {input_csv}")
        return

    if not reference_csv.exists():
        print(f"[SKIP] attach stage3 condition reference support; missing reference csv: {reference_csv}")
        return

    print("===== FINAL+1.1: attach real Stage3 condition reference support =====")
    print(f"[INFO] condition reference input csv: {input_csv}")
    print(f"[INFO] condition reference csv: {reference_csv}")

    r.run([
        "python",
        str(script),
        "--input_csv", str(input_csv),
        "--reference_csv", str(reference_csv),
        "--output_csv", str(output_csv),
        "--output_md", str(output_md),
        "--summary_json", str(summary_json),
        "--top_n", str(_get_int(cfg, "final.top_n", 30)),
    ])

    r.outputs["final_top_routes_with_stage3_condition_reference_csv"] = str(output_csv)
    r.outputs["final_top_routes_with_stage3_condition_reference_md"] = str(output_md)
    r.outputs["final_top_routes_with_stage3_condition_reference_summary_json"] = str(summary_json)

    # Keep backward-compatible confidence key, but also update canonical current csv.
    r.outputs["final_top_routes_with_confidence_csv"] = str(output_csv)
    r.outputs["final_top_routes_with_confidence_md"] = str(output_md)
    _set_current_csv(r, output_csv, output_md)


def attach_metadata_aware_stage3_reference_support_if_enabled(r, cfg: dict) -> None:
    """
    FINAL+1.15:
    Attach V32 metadata-aware Stage3 reference support.

    This step should run after real Stage3 condition-reference support and before
    condition_distribution_confidence, so all downstream tables can preserve:
      metadata_aware_stage3_reference_support_score
      metadata_aware_stage3_reference_level
      metadata_aware_stage3_reference_warning_level
      metadata_aware_stage3_reference_recommendation_status
      metadata_aware_stage3_mp_id
      metadata_aware_stage3_mp_formula

    Script expected:
      pipeline_v3/scripts/attach_metadata_aware_stage3_reference_support.py
    """
    if not _get_bool(cfg, "reliability.attach_metadata_aware_stage3_reference_support.enabled", True):
        print("[SKIP disabled] attach_metadata_aware_stage3_reference_support")
        return

    if "route_out_dir" not in r.outputs:
        print("[SKIP] attach metadata-aware stage3 reference support; missing route_out_dir.")
        return

    project_root = _project_root(r, cfg)
    route_out_dir = Path(r.outputs["route_out_dir"])

    script = ROOT / "scripts" / "attach_metadata_aware_stage3_reference_support.py"

    input_csv = _get_current_csv(r, route_out_dir)

    metadata_alignment_csv = Path(
        _get_str(
            cfg,
            "reliability.attach_metadata_aware_stage3_reference_support.metadata_alignment_csv",
            str(
                project_root
                / "outputs"
                / "benchmark_100_v32_mp_metadata_aware_stage3_alignment"
                / "metadata_aware_alignment_v32"
                / "v32_metadata_aware_external_stage3_alignment_summary.csv"
            ),
        )
    )

    output_csv = route_out_dir / "final_top_routes_with_metadata_stage3_reference.csv"
    output_md = route_out_dir / "final_top_routes_with_metadata_stage3_reference.md"
    summary_json = route_out_dir / "final_top_routes_with_metadata_stage3_reference_summary.json"

    if not script.exists():
        print(f"[SKIP] attach metadata-aware stage3 reference support; missing script: {script}")
        return

    if not input_csv.exists():
        print(f"[SKIP] attach metadata-aware stage3 reference support; missing input csv: {input_csv}")
        return

    if not metadata_alignment_csv.exists():
        print(f"[SKIP] attach metadata-aware stage3 reference support; missing metadata alignment csv: {metadata_alignment_csv}")
        return

    print("===== FINAL+1.15: attach metadata-aware Stage3 reference support =====")
    print(f"[INFO] metadata-aware reference input csv: {input_csv}")
    print(f"[INFO] metadata-aware alignment csv: {metadata_alignment_csv}")

    r.run([
        "python",
        str(script),
        "--input_csv", str(input_csv),
        "--metadata_alignment_csv", str(metadata_alignment_csv),
        "--output_csv", str(output_csv),
        "--output_md", str(output_md),
        "--summary_json", str(summary_json),
        "--top_n", str(_get_int(cfg, "final.top_n", 30)),
    ])

    r.outputs["final_top_routes_with_metadata_stage3_reference_csv"] = str(output_csv)
    r.outputs["final_top_routes_with_metadata_stage3_reference_md"] = str(output_md)
    r.outputs["final_top_routes_with_metadata_stage3_reference_summary_json"] = str(summary_json)

    # Important: downstream steps must consume this enriched table.
    r.outputs["final_top_routes_with_confidence_csv"] = str(output_csv)
    r.outputs["final_top_routes_with_confidence_md"] = str(output_md)
    _set_current_csv(r, output_csv, output_md)


def add_condition_distribution_confidence_if_enabled(r, cfg: dict) -> None:
    """
    FINAL+1.2:
    Add Stage3 condition-distribution confidence.

    Input preference:
      final_top_routes_current_csv
      final_top_routes_with_metadata_stage3_reference.csv
      final_top_routes_with_stage3_condition_reference.csv
      final_top_routes_with_confidence.csv
      final_top_routes_with_precursor_qc.csv
      final_top_routes.csv
    """
    if not r.step_enabled("add_condition_distribution_confidence"):
        print("[SKIP disabled] add_condition_distribution_confidence")
        return

    route_out_dir = Path(r.outputs.get("route_out_dir", ""))
    if not route_out_dir.exists():
        print("[SKIP] condition distribution confidence; route_out_dir missing.")
        return

    script = ROOT / "scripts" / "add_condition_distribution_confidence.py"

    input_csv = _get_current_csv(r, route_out_dir)

    output_csv = route_out_dir / "final_top_routes_with_condition_confidence.csv"
    output_md = route_out_dir / "final_top_routes_with_condition_confidence.md"
    summary_json = route_out_dir / "final_top_routes_with_condition_confidence_summary.json"

    if input_csv is None or not input_csv.exists():
        print("[SKIP] condition distribution confidence; missing input csv.")
        return

    if not script.exists():
        print(f"[SKIP] condition distribution confidence; missing script: {script}")
        return

    print("===== FINAL+1.2: add condition distribution confidence =====")
    print(f"[INFO] condition distribution input csv: {input_csv}")

    r.run([
        "python",
        str(script),
        "--input_csv", str(input_csv),
        "--output_csv", str(output_csv),
        "--output_md", str(output_md),
        "--summary_json", str(summary_json),
        "--top_n", str(_get_int(cfg, "final.top_n", 30)),
    ])

    r.outputs["final_top_routes_with_condition_confidence_csv"] = str(output_csv)
    r.outputs["final_top_routes_with_condition_confidence_md"] = str(output_md)
    r.outputs["final_top_routes_with_condition_confidence_summary_json"] = str(summary_json)

    r.outputs["final_top_routes_with_confidence_csv"] = str(output_csv)
    r.outputs["final_top_routes_with_confidence_md"] = str(output_md)
    _set_current_csv(r, output_csv, output_md)


def audit_condition_diversity_if_enabled(r, cfg: dict) -> None:
    """
    FINAL+1.3:
    Audit Stage3 condition diversity and clipped/baseline condition outputs.
    """
    if not _get_bool(cfg, "reliability.condition_diversity_audit.enabled", True):
        print("[SKIP disabled] condition diversity audit")
        return

    route_out_dir = Path(r.outputs.get("route_out_dir", ""))
    if not route_out_dir.exists():
        print("[SKIP] condition diversity audit; route_out_dir missing.")
        return

    script = ROOT / "scripts" / "audit_condition_diversity.py"
    input_csv = _get_current_csv(r, route_out_dir)

    output_csv = route_out_dir / "condition_diversity_audit.csv"
    output_md = route_out_dir / "condition_diversity_audit.md"
    summary_json = route_out_dir / "condition_diversity_audit_summary.json"

    if input_csv is None or not input_csv.exists():
        print("[SKIP] condition diversity audit; missing input csv.")
        return

    if not script.exists():
        print(f"[SKIP] condition diversity audit; missing script: {script}")
        return

    print("===== FINAL+1.3: audit condition diversity =====")
    print(f"[INFO] condition diversity audit input csv: {input_csv}")

    r.run([
        "python",
        str(script),
        "--input_csv", str(input_csv),
        "--output_csv", str(output_csv),
        "--output_md", str(output_md),
        "--summary_json", str(summary_json),
        "--top_n", str(_get_int(cfg, "final.top_n", 30)),
    ])

    r.outputs["condition_diversity_audit_csv"] = str(output_csv)
    r.outputs["condition_diversity_audit_md"] = str(output_md)
    r.outputs["condition_diversity_audit_summary_json"] = str(summary_json)


def postprocess_confidence_with_precursor_qc_if_enabled(r, cfg: dict) -> None:
    if not _get_bool(cfg, "reliability.qc_confidence_postprocess.enabled", True):
        print("[SKIP disabled] QC-aware confidence postprocess")
        return

    route_out_dir = Path(r.outputs.get("route_out_dir", ""))
    if not route_out_dir.exists():
        print("[SKIP] QC-aware confidence postprocess; route_out_dir missing.")
        return

    post_script = ROOT / "scripts" / "postprocess_confidence_with_precursor_qc.py"

    input_csv = _get_current_csv(r, route_out_dir)

    output_csv = route_out_dir / "final_top_routes_with_confidence.csv"
    output_md = route_out_dir / "final_top_routes_with_confidence.md"

    if not input_csv.exists() or not post_script.exists():
        print("[SKIP] QC-aware confidence postprocess; missing confidence csv or postprocess script.")
        return

    print("===== FINAL+1.5: postprocess confidence with precursor QC =====")
    print(f"[INFO] QC-aware confidence input csv: {input_csv}")

    r.run([
        "python",
        str(post_script),
        "--input_csv", str(input_csv),
        "--output_csv", str(output_csv),
        "--output_md", str(output_md),
    ])

    r.outputs["final_top_routes_with_confidence_csv"] = str(output_csv)
    r.outputs["final_top_routes_with_confidence_md"] = str(output_md)
    _set_current_csv(r, output_csv, output_md)


def export_final_report_if_enabled(r, cfg: dict) -> None:
    if not _get_bool(cfg, "reliability.final_report.enabled", True):
        print("[SKIP disabled] export final report")
        return

    route_out_dir = Path(r.outputs.get("route_out_dir", ""))
    if not route_out_dir.exists():
        print("[SKIP] final report; route_out_dir missing.")
        return

    report_script = ROOT / "scripts" / "export_final_report_v21.py"
    output_md = route_out_dir / "final_report.md"

    if not report_script.exists():
        print("[SKIP] final report; missing export_final_report_v21.py")
        return

    print("===== FINAL+2: export final report =====")
    r.run([
        "python",
        str(report_script),
        "--route_out_dir", str(route_out_dir),
        "--output_md", str(output_md),
        "--top_n", str(_get_int(cfg, "final.report_top_n", 10)),
    ])

    r.outputs["final_report_md"] = str(output_md)


def build_joint_route_features_if_enabled(r, cfg: dict) -> None:
    if not r.step_enabled("build_joint_route_features"):
        print("[SKIP disabled] build_joint_route_features")
        return

    if "route_out_dir" not in r.outputs:
        print("[SKIP] build joint route features; missing route_out_dir in outputs.")
        return

    route_out_dir = Path(r.outputs["route_out_dir"])
    input_csv = _get_current_csv(r, route_out_dir)

    script = ROOT / "scripts" / "build_joint_route_features.py"

    output_csv = route_out_dir / "final_top_routes_with_joint_features.csv"
    output_md = route_out_dir / "final_top_routes_with_joint_features.md"
    summary_json = route_out_dir / "final_top_routes_with_joint_features_summary.json"

    if not input_csv.exists():
        print(f"[SKIP] build joint route features; missing input csv: {input_csv}")
        return

    if not script.exists():
        print(f"[SKIP] build joint route features; missing script: {script}")
        return

    print("===== FINAL+2.5: build joint route features =====")
    print(f"[INFO] joint route feature input csv: {input_csv}")

    r.run([
        "python",
        str(script),
        "--input_csv", str(input_csv),
        "--output_csv", str(output_csv),
        "--output_md", str(output_md),
        "--summary_json", str(summary_json),
        "--top_n", str(_get_int(cfg, "final.top_n", 30)),
    ])

    r.outputs["final_top_routes_with_joint_features_csv"] = str(output_csv)
    r.outputs["final_top_routes_with_joint_features_md"] = str(output_md)
    r.outputs["final_top_routes_with_joint_features_summary_json"] = str(summary_json)
    _set_current_csv(r, output_csv, output_md)


def apply_v3_joint_route_rerank_if_enabled(r, cfg: dict) -> None:
    if not r.step_enabled("apply_v3_joint_route_rerank"):
        print("[SKIP disabled] apply_v3_joint_route_rerank")
        return

    if "route_out_dir" not in r.outputs:
        print("[SKIP] apply v3 joint route rerank; missing route_out_dir in outputs.")
        return

    route_out_dir = Path(r.outputs["route_out_dir"])

    input_csv = _first_existing([
        r.outputs.get("final_top_routes_with_joint_features_csv", ""),
        str(route_out_dir / "final_top_routes_with_joint_features.csv"),
        r.outputs.get("final_top_routes_current_csv", ""),
    ])

    if input_csv is None:
        input_csv = route_out_dir / "final_top_routes_with_joint_features.csv"

    script = ROOT / "scripts" / "apply_v3_joint_route_rerank.py"

    output_csv = route_out_dir / "final_top_routes_v3_joint_reranked.csv"
    output_md = route_out_dir / "final_top_routes_v3_joint_reranked.md"
    summary_json = route_out_dir / "final_top_routes_v3_joint_reranked_summary.json"

    if not input_csv.exists():
        print(f"[SKIP] apply v3 joint route rerank; missing input csv: {input_csv}")
        return

    if not script.exists():
        print(f"[SKIP] apply v3 joint route rerank; missing script: {script}")
        return

    print("===== FINAL+3: apply v3 joint route rerank =====")
    r.run([
        "python",
        str(script),
        "--input_csv", str(input_csv),
        "--output_csv", str(output_csv),
        "--output_md", str(output_md),
        "--summary_json", str(summary_json),
        "--top_n", str(_get_int(cfg, "final.top_n", 30)),
    ])

    r.outputs["final_top_routes_v3_joint_reranked_csv"] = str(output_csv)
    r.outputs["final_top_routes_v3_joint_reranked_md"] = str(output_md)
    r.outputs["final_top_routes_v3_joint_reranked_summary_json"] = str(summary_json)
    _set_current_csv(r, output_csv, output_md)


def export_final_report_v3_if_enabled(r, cfg: dict) -> None:
    if not r.step_enabled("export_final_report_v3"):
        print("[SKIP disabled] export_final_report_v3")
        return

    if "route_out_dir" not in r.outputs:
        print("[SKIP] export final report v3; missing route_out_dir in outputs.")
        return

    route_out_dir = Path(r.outputs["route_out_dir"])
    script = ROOT / "scripts" / "export_final_report_v3.py"
    output_md = route_out_dir / "final_report_v3.md"

    if not script.exists():
        print(f"[SKIP] export final report v3; missing script: {script}")
        return

    print("===== FINAL+3.5: export final report v3 =====")
    r.run([
        "python",
        str(script),
        "--route_out_dir", str(route_out_dir),
        "--output_md", str(output_md),
        "--top_n", str(_get_int(cfg, "final.report_top_n", 10)),
    ])

    r.outputs["final_report_v3_md"] = str(output_md)


def build_v3_learned_ranker_dataset_if_enabled(r, cfg: dict) -> None:
    if not r.step_enabled("build_v3_learned_ranker_dataset"):
        print("[SKIP disabled] build_v3_learned_ranker_dataset")
        return

    if "route_out_dir" not in r.outputs:
        print("[SKIP] build v3 learned ranker dataset; missing route_out_dir in outputs.")
        return

    route_out_dir = Path(r.outputs["route_out_dir"])
    script = ROOT / "scripts" / "build_v3_learned_ranker_dataset.py"

    input_csv = _first_existing([
        r.outputs.get("final_top_routes_v3_joint_reranked_csv", ""),
        str(route_out_dir / "final_top_routes_v3_joint_reranked.csv"),
        r.outputs.get("final_top_routes_current_csv", ""),
    ])

    if input_csv is None:
        input_csv = route_out_dir / "final_top_routes_v3_joint_reranked.csv"

    output_csv = route_out_dir / "v3_learned_ranker_dataset.csv"
    feature_cols_json = route_out_dir / "v3_learned_ranker_feature_cols.json"
    summary_json = route_out_dir / "v3_learned_ranker_dataset_summary.json"

    if not script.exists():
        print(f"[SKIP] build v3 learned ranker dataset; missing script: {script}")
        return

    if not input_csv.exists():
        print(f"[SKIP] build v3 learned ranker dataset; missing input csv: {input_csv}")
        return

    print("===== FINAL+4: build v3 learned ranker dataset =====")
    r.run([
        "python",
        str(script),
        "--input_csv", str(input_csv),
        "--output_csv", str(output_csv),
        "--feature_cols_json", str(feature_cols_json),
        "--summary_json", str(summary_json),
    ])

    r.outputs["v3_learned_ranker_dataset_csv"] = str(output_csv)
    r.outputs["v3_learned_ranker_feature_cols_json"] = str(feature_cols_json)
    r.outputs["v3_learned_ranker_dataset_summary_json"] = str(summary_json)


def apply_v3_learned_ranker_if_enabled(r, cfg: dict) -> None:
    if not r.step_enabled("apply_v3_learned_ranker"):
        print("[SKIP disabled] apply_v3_learned_ranker")
        return

    if "route_out_dir" not in r.outputs:
        print("[SKIP] apply v3 learned ranker; missing route_out_dir in outputs.")
        return

    route_out_dir = Path(r.outputs["route_out_dir"])
    script = ROOT / "scripts" / "apply_v3_learned_ranker.py"

    input_csv = _first_existing([
        r.outputs.get("final_top_routes_v3_joint_reranked_csv", ""),
        str(route_out_dir / "final_top_routes_v3_joint_reranked.csv"),
        r.outputs.get("final_top_routes_current_csv", ""),
    ])

    if input_csv is None:
        input_csv = route_out_dir / "final_top_routes_v3_joint_reranked.csv"

    model_path = Path(cfg.get("v3_learned_ranker_model_path", ""))
    feature_cols_json = Path(cfg.get("v3_learned_ranker_feature_cols_json", ""))

    project_root = _project_root(r, cfg)
    if not model_path.is_absolute():
        model_path = project_root / model_path
    if not feature_cols_json.is_absolute():
        feature_cols_json = project_root / feature_cols_json

    output_csv = route_out_dir / "final_top_routes_v3_learned_reranked.csv"
    output_md = route_out_dir / "final_top_routes_v3_learned_reranked.md"
    summary_json = route_out_dir / "final_top_routes_v3_learned_reranked_summary.json"

    if not script.exists():
        print(f"[SKIP] apply v3 learned ranker; missing script: {script}")
        return

    if not input_csv.exists():
        print(f"[SKIP] apply v3 learned ranker; missing input csv: {input_csv}")
        return

    if not model_path.exists():
        print(f"[SKIP] apply v3 learned ranker; missing model_path: {model_path}")
        return

    if not feature_cols_json.exists():
        print(f"[SKIP] apply v3 learned ranker; missing feature_cols_json: {feature_cols_json}")
        return

    print("===== FINAL+5: apply v3 learned ranker =====")
    r.run([
        "python",
        str(script),
        "--input_csv", str(input_csv),
        "--model_path", str(model_path),
        "--feature_cols_json", str(feature_cols_json),
        "--output_csv", str(output_csv),
        "--output_md", str(output_md),
        "--summary_json", str(summary_json),
        "--top_n", str(_get_int(cfg, "final.top_n", 30)),
    ])

    r.outputs["final_top_routes_v3_learned_reranked_csv"] = str(output_csv)
    r.outputs["final_top_routes_v3_learned_reranked_md"] = str(output_md)
    r.outputs["final_top_routes_v3_learned_reranked_summary_json"] = str(summary_json)
    _set_current_csv(r, output_csv, output_md)

def build_v36_target_aware_stage2_candidates_if_enabled(r, cfg: dict) -> None:
    """
    AUX/V36:
    Build target-aware Stage2-candidate-compatible tables for V33 formula-exact MP targets.

    Important:
      This is an expansion/compatibility utility, not normal Stage2 model inference.
      It should stay disabled by default.
    """
    if not _get_bool(
        cfg,
        #"stage3_reference_expansion.build_v36_target_aware_stage2_candidates.enabled",
        "stage3_reference_refresh.build_v36_target_aware_stage2_candidates.enabled",
        False,
    ):
        print("[SKIP disabled] build_v36_target_aware_stage2_candidates")
        return

    project_root = _project_root(r, cfg)

    script = (
        project_root
        / "scripts/07_infer/structure_to_synthesis_route/benchmark/"
        / "benchmark_100_v36_v33_target_aware_stage2_candidate_construction/"
        / "run_full_v36_v33_target_aware_stage2_candidate_construction.sh"
    )

    if not script.exists():
        print(f"[SKIP] V36 target-aware Stage2 candidate construction; missing script: {script}")
        return

    print("===== AUX+V36: build target-aware Stage2 candidates =====")
    print(f"[INFO] V36 script: {script}")

    r.run([
        "bash",
        str(script),
        str(project_root),
    ])

    out_root = (
        project_root
        / "outputs/benchmark_100_v36_v33_target_aware_stage2_candidate_construction"
    )
    out_dir = out_root / "target_aware_stage2_candidates_v36"
    final_report = out_root / "FINAL_REPORT_V36" / "FINAL_BENCHMARK_100_V36_REPORT.md"

    r.outputs["v36_target_aware_stage2_candidate_dir"] = str(out_dir)
    r.outputs["v36_target_aware_stage2_test_candidates_csv"] = str(out_dir / "test_candidates.csv")
    r.outputs["v36_target_aware_stage2_val_candidates_csv"] = str(out_dir / "val_candidates.csv")
    r.outputs["v36_target_aware_stage2_all_candidates_csv"] = str(
        out_dir / "v36_target_aware_stage2_candidates.csv"
    )
    r.outputs["v36_target_aware_stage2_summary_json"] = str(
        out_dir / "v36_target_aware_stage2_candidates_summary.json"
    )
    r.outputs["v36_final_report_md"] = str(final_report)


def run_v37_stage3_input_preflight_if_enabled(r, cfg: dict) -> None:
    """
    AUX/V37-preflight:
    Check whether V36 target-aware Stage2 candidates can be consumed by existing Stage3 input NPZ files.

    This does not generate real Stage3 candidates. It only audits compatibility.
    """
    if not _get_bool(
        cfg,
        #"stage3_reference_expansion.run_v37_stage3_input_preflight.enabled",
        "stage3_reference_refresh.run_v37_stage3_input_preflight.enabled",
        False,
    ):
        print("[SKIP disabled] run_v37_stage3_input_preflight")
        return

    project_root = _project_root(r, cfg)

    script = (
        project_root
        / "scripts/07_infer/structure_to_synthesis_route/benchmark/"
        / "benchmark_100_v37_real_stage3_generation_from_v36_targetaware_stage2/"
        / "run_v37_preflight_stage3_input_compatibility_audit.sh"
    )

    if not script.exists():
        print(f"[SKIP] V37 Stage3 input preflight; missing script: {script}")
        return

    v36_test_csv = (
        project_root
        / "outputs/benchmark_100_v36_v33_target_aware_stage2_candidate_construction/"
        / "target_aware_stage2_candidates_v36/test_candidates.csv"
    )

    if not v36_test_csv.exists():
        print(f"[SKIP] V37 Stage3 input preflight; missing V36 candidates: {v36_test_csv}")
        print("       Enable build_v36_target_aware_stage2_candidates first.")
        return

    print("===== AUX+V37.1: Stage3 input compatibility preflight =====")
    print(f"[INFO] V37 preflight script: {script}")

    r.run([
        "bash",
        str(script),
        str(project_root),
    ])

    out_root = (
        project_root
        / "outputs/benchmark_100_v37_real_stage3_generation_from_v36_targetaware_stage2"
    )
    preflight_dir = out_root / "stage3_input_preflight_v37"
    final_dir = out_root / "FINAL_REPORT_V37"

    r.outputs["v37_stage3_input_preflight_dir"] = str(preflight_dir)
    r.outputs["v37_stage3_input_payload_compatibility_csv"] = str(
        preflight_dir / "v37_stage3_input_payload_compatibility.csv"
    )
    r.outputs["v37_stage3_input_payload_compatibility_summary_json"] = str(
        preflight_dir / "v37_stage3_input_payload_compatibility_summary.json"
    )
    r.outputs["v37_preflight_blocked_report_md"] = str(
        final_dir / "FINAL_BENCHMARK_100_V37_PREFLIGHT_BLOCKED_REPORT.md"
    )


def run_v37_stage3_export_interface_plan_if_enabled(r, cfg: dict) -> None:
    """
    AUX/V37-plan:
    Build command-level interface plans for real Stage3 MDN/Flow export from V36 target-aware candidates.

    This is still an interface plan, not a claim that real Stage3 candidates have been generated.
    """
    if not _get_bool(
        cfg,
        #"stage3_reference_expansion.run_v37_stage3_export_interface_plan.enabled",
        "stage3_reference_refresh.run_v37_stage3_export_interface_plan.enabled",
        False,
    ):
        print("[SKIP disabled] run_v37_stage3_export_interface_plan")
        return

    project_root = _project_root(r, cfg)

    script = (
        project_root
        / "scripts/07_infer/structure_to_synthesis_route/benchmark/"
        / "benchmark_100_v37_real_stage3_generation_from_v36_targetaware_stage2/"
        / "run_v37_stage3_export_interface_plan.sh"
    )

    if not script.exists():
        print(f"[SKIP] V37 Stage3 export interface plan; missing script: {script}")
        return

    v36_dir = (
        project_root
        / "outputs/benchmark_100_v36_v33_target_aware_stage2_candidate_construction/"
        / "target_aware_stage2_candidates_v36"
    )

    if not (v36_dir / "test_candidates.csv").exists():
        print(f"[SKIP] V37 Stage3 export interface plan; missing V36 candidate dir: {v36_dir}")
        print("       Enable build_v36_target_aware_stage2_candidates first.")
        return

    print("===== AUX+V37.2: Stage3 export interface plan =====")
    print(f"[INFO] V37 interface-plan script: {script}")

    r.run([
        "bash",
        str(script),
        str(project_root),
    ])

    out_root = (
        project_root
        / "outputs/benchmark_100_v37_real_stage3_generation_from_v36_targetaware_stage2"
    )
    plan_dir = out_root / "stage3_export_interface_plan_v37"
    final_dir = out_root / "FINAL_REPORT_V37"

    r.outputs["v37_stage3_export_interface_plan_dir"] = str(plan_dir)
    r.outputs["v37_stage3_export_inventory_csv"] = str(
        plan_dir / "v37_stage3_export_interface_inventory.csv"
    )
    r.outputs["v37_stage3_export_inventory_summary_json"] = str(
        plan_dir / "v37_stage3_export_interface_inventory_summary.json"
    )
    r.outputs["v37_mdn_export_plan_sh"] = str(plan_dir / "run_v37_mdn_export.sh")
    r.outputs["v37_flow_export_plan_sh"] = str(plan_dir / "run_v37_flow_export.sh")
    r.outputs["v37_possible_stage3_sampler_scripts_txt"] = str(
        plan_dir / "v37_possible_stage3_sampler_or_export_scripts.txt"
    )
    r.outputs["v37_interface_plan_report_md"] = str(
        final_dir / "FINAL_BENCHMARK_100_V37_INTERFACE_PLAN_REPORT.md"
    )
def run_v38_stage3_feature_source_audit_if_enabled(r, cfg: dict) -> None:
    block = cfg.get("run_v38_stage3_feature_source_audit", {}) or {}
    if not block.get("enabled", False):
        print("[SKIP disabled] run_v38_stage3_feature_source_audit")
        return

    project_root = Path(cfg["project_root"])
    script = (
        project_root
        / "scripts/07_infer/structure_to_synthesis_route/benchmark"
        / "benchmark_100_v38_stage3_input_feature_extension_for_v33_targets"
        / "run_v38_stage3_feature_source_audit.sh"
    )

    if not script.exists():
        print(f"[SKIP] run_v38_stage3_feature_source_audit; missing script: {script}")
        return

    print("===== AUX: V38 Stage3 feature source audit =====")
    #run_cmd(["bash", str(script), str(project_root)])
    r.run(["bash", str(script), str(project_root)])

def run_v39_stage3_input_feature_extension_construction_if_enabled(r, cfg: dict) -> None:
    block = cfg.get("run_v39_stage3_input_feature_extension_construction", {}) or {}
    if not block.get("enabled", False):
        print("[SKIP disabled] run_v39_stage3_input_feature_extension_construction")
        return

    project_root = Path(cfg["project_root"])
    script = (
        project_root
        / "scripts/07_infer/structure_to_synthesis_route/benchmark"
        / "benchmark_100_v39_stage3_input_feature_extension_construction"
        / "run_v39_stage3_input_feature_extension_construction.sh"
    )

    if not script.exists():
        print(f"[SKIP] run_v39_stage3_input_feature_extension_construction; missing script: {script}")
        return

    print("===== AUX: V39 Stage3 input feature extension construction =====")
    #run_cmd(["bash", str(script), str(project_root)])
    r.run(["bash", str(script), str(project_root)])

    out_root = project_root / "outputs/benchmark_100_v39_stage3_input_feature_extension_construction"
    summary_json = out_root / "audit_v39/v39_stage3_input_extension_summary.json"
    ext_dir = out_root / "stage3_input_extension_v39"
    cand_dir = out_root / "stage2_candidates_for_v39_partial_stage3_export"

    if summary_json.exists():
        r.outputs["v39_stage3_input_extension_summary_json"] = str(summary_json)
    if ext_dir.exists():
        r.outputs["v39_stage3_input_extension_dir"] = str(ext_dir)
    if cand_dir.exists():
        r.outputs["v39_stage2_candidate_subset_dir"] = str(cand_dir)

def audit_stage2_retrieval_candidates_if_enabled(r, cfg: dict) -> None:
    """
    Audit Stage2 retrieval-conditioned candidate contribution.
    """
    if not _get_bool(cfg, "reliability.stage2_retrieval_audit.enabled", True):
        print("[SKIP disabled] stage2 retrieval audit")
        return

    work_dir = Path(getattr(r, "work_dir", ""))
    out_dir = Path(getattr(r, "out_dir", ""))

    stage2_summary_dir = work_dir / "stage2_summary"

    reranked_csv = Path(
        r.outputs.get(
            "stage2_final_csv",
            stage2_summary_dir / "unique_sets_ranked_with_fallback_retrieval_baseline_element_reranked.csv",
        )
    )
    retrieval_csv = Path(
        r.outputs.get(
            "stage2_retrieval_csv",
            stage2_summary_dir / "retrieval_npz_candidates.csv",
        )
    )
    merged_csv = Path(
        r.outputs.get(
            "stage2_merged_csv",
            stage2_summary_dir / "unique_sets_ranked_with_fallback_retrieval_baseline.csv",
        )
    )

    audit_dir = out_dir / "stage2_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    output_csv = audit_dir / "stage2_retrieval_audit.csv"
    output_md = audit_dir / "stage2_retrieval_audit.md"
    summary_json = audit_dir / "stage2_retrieval_audit_summary.json"

    script = ROOT / "scripts" / "audit_stage2_retrieval_candidates.py"

    if not script.exists():
        print(f"[SKIP] stage2 retrieval audit; missing script: {script}")
        return

    missing = [p for p in [retrieval_csv, merged_csv, reranked_csv] if not p.exists()]
    if missing:
        print("[SKIP] stage2 retrieval audit; missing inputs:")
        for p in missing:
            print(f"  - {p}")
        return

    print("===== FINAL+0.2: audit stage2 retrieval candidates =====")
    r.run([
        "python",
        str(script),
        "--retrieval_csv", str(retrieval_csv),
        "--merged_csv", str(merged_csv),
        "--reranked_csv", str(reranked_csv),
        "--output_csv", str(output_csv),
        "--output_md", str(output_md),
        "--summary_json", str(summary_json),
        "--top_n", str(_get_int(cfg, "final.top_n", 30)),
    ])

    r.outputs["stage2_retrieval_audit_csv"] = str(output_csv)
    r.outputs["stage2_retrieval_audit_md"] = str(output_md)
    r.outputs["stage2_retrieval_audit_summary_json"] = str(summary_json)


def add_v43_route_template_features_if_enabled(r, cfg: dict) -> None:
    """
    FINAL+6:
    Add route-template features to the newest available v3 route table.
    """
    if not r.step_enabled("add_v43_route_template_features"):
        print("[SKIP disabled] add_v43_route_template_features")
        return

    if "route_out_dir" not in r.outputs:
        print("[SKIP] add v43 template features; missing route_out_dir in outputs.")
        return

    project_root = _project_root(r, cfg)
    route_out_dir = Path(r.outputs["route_out_dir"])

    script = (
        project_root
        / "scripts/07_infer/structure_to_synthesis_route/route_ranker/v43_template_aware/01_add_route_template_features.py"
    )

    input_csv = _first_existing([
        r.outputs.get("final_top_routes_v3_learned_reranked_csv", ""),
        str(route_out_dir / "final_top_routes_v3_learned_reranked.csv"),
        r.outputs.get("final_top_routes_v3_joint_reranked_csv", ""),
        str(route_out_dir / "final_top_routes_v3_joint_reranked.csv"),
        r.outputs.get("final_top_routes_with_joint_features_csv", ""),
        str(route_out_dir / "final_top_routes_with_joint_features.csv"),
        r.outputs.get("final_top_routes_current_csv", ""),
        r.outputs.get("final_top_routes_with_confidence_csv", ""),
        str(route_out_dir / "final_top_routes_with_confidence.csv"),
        r.outputs.get("final_top_routes_csv", ""),
        str(route_out_dir / "final_top_routes.csv"),
    ])

    output_csv = route_out_dir / "synthesis_routes_stage35_v43_template_features.csv"
    output_md = route_out_dir / "synthesis_routes_stage35_v43_template_features.md"
    summary_json = route_out_dir / "synthesis_routes_stage35_v43_template_features_summary.json"

    if input_csv is None:
        print("[SKIP] add v43 template features; no usable input csv.")
        return

    if not script.exists():
        print(f"[SKIP] add v43 template features; missing script: {script}")
        return

    print("===== FINAL+6: add v4.3 route-template features =====")
    print(f"[INFO] v43 feature input csv: {input_csv}")

    r.run([
        "python",
        str(script),
        "--input_csv", str(input_csv),
        "--output_csv", str(output_csv),
        "--output_md", str(output_md),
        "--summary_json", str(summary_json),
        "--top_n", str(_get_int(cfg, "final.top_n", 30)),
    ])

    r.outputs["stage35_v43_template_features_csv"] = str(output_csv)
    r.outputs["stage35_v43_template_features_md"] = str(output_md)
    r.outputs["stage35_v43_template_features_summary_json"] = str(summary_json)
    _set_current_csv(r, output_csv, output_md)


def apply_v43_template_ranker_if_enabled(r, cfg: dict) -> None:
    """
    FINAL+7:
    Apply v4.3 template-aware chem-only pairwise ranker.
    """
    if not r.step_enabled("apply_v43_template_ranker"):
        print("[SKIP disabled] apply_v43_template_ranker")
        return

    if "route_out_dir" not in r.outputs:
        print("[SKIP] apply v43 template ranker; missing route_out_dir in outputs.")
        return

    project_root = _project_root(r, cfg)
    route_out_dir = Path(r.outputs["route_out_dir"])

    script = (
        project_root
        / "scripts/07_infer/structure_to_synthesis_route/route_ranker/v43_template_aware/04_apply_v43_template_pairwise_ranker_chemonly.py"
    )

    input_csv = _first_existing([
        r.outputs.get("stage35_v43_template_features_csv", ""),
        str(route_out_dir / "synthesis_routes_stage35_v43_template_features.csv"),
        r.outputs.get("final_top_routes_current_csv", ""),
    ])

    if input_csv is None:
        input_csv = route_out_dir / "synthesis_routes_stage35_v43_template_features.csv"

    model_path = Path(
        cfg.get(
            "v43_template_ranker_model_path",
            project_root
            / "runs/stage35/route_ranker_v43_template_aware/stage35_v43_template_pairwise_chemonly_extratrees.joblib",
        )
    )

    feature_cols_json = Path(
        cfg.get(
            "v43_template_ranker_feature_cols_json",
            project_root
            / "runs/stage35/route_ranker_v43_template_aware/stage35_v43_template_pairwise_chemonly_feature_cols.json",
        )
    )

    output_csv = route_out_dir / "final_top_routes_v43_template_chemonly_reranked.csv"
    output_md = route_out_dir / "final_top_routes_v43_template_chemonly_reranked.md"
    summary_json = route_out_dir / "final_top_routes_v43_template_chemonly_reranked_summary.json"

    if not script.exists():
        print(f"[SKIP] apply v43 template ranker; missing script: {script}")
        return

    if not input_csv.exists():
        print(f"[SKIP] apply v43 template ranker; missing input csv: {input_csv}")
        return

    if not model_path.exists():
        print(f"[SKIP] apply v43 template ranker; missing model_path: {model_path}")
        return

    if not feature_cols_json.exists():
        print(f"[SKIP] apply v43 template ranker; missing feature_cols_json: {feature_cols_json}")
        return

    print("===== FINAL+7: apply v4.3 template-aware chem-only ranker =====")

    r.run([
        "python",
        str(script),
        "--input_csv", str(input_csv),
        "--model_path", str(model_path),
        "--feature_cols_json", str(feature_cols_json),
        "--output_csv", str(output_csv),
        "--output_md", str(output_md),
        "--summary_json", str(summary_json),
        "--top_n", str(_get_int(cfg, "final.top_n", 30)),
    ])

    r.outputs["final_top_routes_v43_template_chemonly_reranked_csv"] = str(output_csv)
    r.outputs["final_top_routes_v43_template_chemonly_reranked_md"] = str(output_md)
    r.outputs["final_top_routes_v43_template_chemonly_reranked_summary_json"] = str(summary_json)
    _set_current_csv(r, output_csv, output_md)


def apply_v43_safe_strict_gate_if_enabled(r, cfg: dict) -> None:
    """
    FINAL+8:
    Apply a post-v43 safe-strict gate.
    """
    if not r.step_enabled("apply_v43_safe_strict_gate"):
        print("[SKIP disabled] apply_v43_safe_strict_gate")
        return

    if "route_out_dir" not in r.outputs:
        print("[SKIP] apply v43 safe-strict gate; missing route_out_dir in outputs.")
        return

    route_out_dir = Path(r.outputs["route_out_dir"])
    script = ROOT / "scripts" / "apply_v43_safe_strict_gate.py"

    input_csv = _first_existing([
        r.outputs.get("final_top_routes_v43_template_chemonly_reranked_csv", ""),
        str(route_out_dir / "final_top_routes_v43_template_chemonly_reranked.csv"),
        r.outputs.get("final_top_routes_current_csv", ""),
    ])

    if input_csv is None:
        input_csv = route_out_dir / "final_top_routes_v43_template_chemonly_reranked.csv"

    output_csv = route_out_dir / "final_top_routes_v43_safe_strict_reranked.csv"
    output_md = route_out_dir / "final_top_routes_v43_safe_strict_reranked.md"
    summary_json = route_out_dir / "final_top_routes_v43_safe_strict_reranked_summary.json"

    if not script.exists():
        print(f"[SKIP] apply v43 safe-strict gate; missing script: {script}")
        return

    if not input_csv.exists():
        print(f"[SKIP] apply v43 safe-strict gate; missing input csv: {input_csv}")
        return

    print("===== FINAL+8: apply v4.3 safe-strict extra-element gate =====")

    r.run([
        "python",
        str(script),
        "--input_csv", str(input_csv),
        "--output_csv", str(output_csv),
        "--output_md", str(output_md),
        "--summary_json", str(summary_json),
        "--top_n", str(_get_int(cfg, "final.top_n", 30)),
    ])

    r.outputs["final_top_routes_v43_safe_strict_reranked_csv"] = str(output_csv)
    r.outputs["final_top_routes_v43_safe_strict_reranked_md"] = str(output_md)
    r.outputs["final_top_routes_v43_safe_strict_reranked_summary_json"] = str(summary_json)
    _set_current_csv(r, output_csv, output_md)


def finalize_recommended_routes_if_enabled(r, cfg: dict) -> None:
    """
    FINAL+9:
    Final recommendation layer.

    Prefer the latest v43 safe-strict table, but because current_csv is updated
    throughout reliability steps, all metadata/reference columns should already
    be carried into v43/safe-strict outputs.
    """
    if not _get_bool(cfg, "reliability.finalize_recommended_routes.enabled", True):
        print("[SKIP disabled] finalize_recommended_routes")
        return

    route_out_dir = Path(r.outputs.get("route_out_dir", ""))
    if not route_out_dir.exists():
        print("[SKIP] finalize recommended routes; route_out_dir missing.")
        return

    script = ROOT / "scripts" / "finalize_recommended_routes.py"

    candidate_inputs = [
        r.outputs.get("final_top_routes_v43_safe_strict_reranked_csv", ""),
        str(route_out_dir / "final_top_routes_v43_safe_strict_reranked.csv"),
        r.outputs.get("final_top_routes_v43_template_chemonly_reranked_csv", ""),
        str(route_out_dir / "final_top_routes_v43_template_chemonly_reranked.csv"),
        r.outputs.get("final_top_routes_v3_learned_reranked_csv", ""),
        str(route_out_dir / "final_top_routes_v3_learned_reranked.csv"),
        r.outputs.get("final_top_routes_v3_joint_reranked_csv", ""),
        str(route_out_dir / "final_top_routes_v3_joint_reranked.csv"),
        r.outputs.get("final_top_routes_current_csv", ""),
        r.outputs.get("final_top_routes_with_condition_confidence_csv", ""),
        str(route_out_dir / "final_top_routes_with_condition_confidence.csv"),
        r.outputs.get("final_top_routes_with_confidence_csv", ""),
        str(route_out_dir / "final_top_routes_with_confidence.csv"),
        r.outputs.get("final_top_routes_csv", ""),
        str(route_out_dir / "final_top_routes.csv"),
    ]

    input_csv = None
    input_source = ""

    for candidate in candidate_inputs:
        if candidate and Path(candidate).exists():
            input_csv = Path(candidate)
            input_source = input_csv.name
            break

    output_csv = route_out_dir / "final_recommended_routes.csv"
    output_md = route_out_dir / "final_recommended_routes.md"
    summary_json = route_out_dir / "final_recommended_routes_summary.json"

    if input_csv is None or not input_csv.exists():
        print("[SKIP] finalize recommended routes; missing input csv.")
        return

    if not script.exists():
        print(f"[SKIP] finalize recommended routes; missing script: {script}")
        return

    print("===== FINAL+9: finalize recommended routes =====")
    print(f"[INFO] final recommendation input csv: {input_csv}")

    r.run([
        "python",
        str(script),
        "--input_csv", str(input_csv),
        "--output_csv", str(output_csv),
        "--output_md", str(output_md),
        "--summary_json", str(summary_json),
        "--top_n", str(_get_int(cfg, "final.top_n", 30)),
    ])

    r.outputs["final_recommended_routes_csv"] = str(output_csv)
    r.outputs["final_recommended_routes_md"] = str(output_md)
    r.outputs["final_recommended_routes_summary_json"] = str(summary_json)
    r.outputs["final_recommended_routes_source"] = f"finalizer_from_{input_source}"
    _set_current_csv(r, output_csv, output_md)


def audit_final_recommended_routes_if_enabled(r, cfg: dict) -> None:
    """
    FINAL+10:
    Audit final_recommended_routes.csv for score/rank/status consistency.
    """
    if not _get_bool(cfg, "reliability.audit_final_recommended_routes.enabled", True):
        print("[SKIP disabled] audit_final_recommended_routes")
        return

    route_out_dir = Path(r.outputs.get("route_out_dir", ""))
    if not route_out_dir.exists():
        print("[SKIP] audit final recommended routes; route_out_dir missing.")
        return

    script = ROOT / "scripts" / "audit_final_recommended_routes.py"

    input_csv = Path(
        r.outputs.get(
            "final_recommended_routes_csv",
            route_out_dir / "final_recommended_routes.csv",
        )
    )

    output_csv = route_out_dir / "final_recommended_routes_audit.csv"
    output_md = route_out_dir / "final_recommended_routes_audit.md"
    summary_json = route_out_dir / "final_recommended_routes_audit_summary.json"

    if not input_csv.exists():
        print(f"[SKIP] audit final recommended routes; missing input csv: {input_csv}")
        return

    if not script.exists():
        print(f"[SKIP] audit final recommended routes; missing script: {script}")
        return

    print("===== FINAL+10: audit final recommended routes =====")
    print(f"[INFO] final recommendation audit input csv: {input_csv}")

    r.run([
        "python",
        str(script),
        "--input_csv", str(input_csv),
        "--output_csv", str(output_csv),
        "--output_md", str(output_md),
        "--summary_json", str(summary_json),
        "--top_n", str(_get_int(cfg, "final.top_n", 30)),
    ])

    r.outputs["final_recommended_routes_audit_csv"] = str(output_csv)
    r.outputs["final_recommended_routes_audit_md"] = str(output_md)
    r.outputs["final_recommended_routes_audit_summary_json"] = str(summary_json)

def refresh_stage3_reference_library_if_enabled(r, cfg: dict) -> None:
    """
    AUX: Refresh active Stage3 condition reference library before reliability scoring.

    This wrapper calls:
      pipeline_v3/scripts/refresh_stage3_reference_library.py

    It merges current / V32 / V34-discovered real Stage3-style condition candidates
    into:
      data/interim/references/stage3_condition_reference/current/stage3_condition_reference.csv
    """
    if not _get_bool(cfg, "stage3_reference_refresh.enabled", False):
        print("[SKIP disabled] refresh_stage3_reference_library")
        return

    project_root = _project_root(r, cfg)

    script = (
        project_root
        / "scripts/07_infer/structure_to_synthesis_route/pipeline/scripts/"
        / "refresh_stage3_reference_library.py"
    )

    if not script.exists():
        print(f"[SKIP] refresh_stage3_reference_library; missing script: {script}")
        return

    args = [
        "python",
        str(script),
        "--project_root",
        str(project_root),
    ]

    if _get_bool(cfg, "stage3_reference_refresh.run_v34_discovery", False):
        args.append("--run_v34_discovery")

    if _get_bool(cfg, "stage3_reference_refresh.backup_existing", False):
        args.append("--backup_existing")

    if _get_bool(cfg, "stage3_reference_refresh.dry_run", False):
        args.append("--dry_run")
    if _get_bool(cfg, "stage3_reference_refresh.no_v32", False):
        args.append("--no_v32")
    if _get_bool(cfg, "stage3_reference_refresh.no_v34_discovered", False):
        args.append("--no_v34_discovered")
    registry_json = _get(cfg, "stage3_reference_refresh.registry_json", "")
    if registry_json:
        args.extend(["--registry_json", str(registry_json)])

    output_csv = _get(cfg, "stage3_reference_refresh.output_csv", "")
    if output_csv:
        args.extend(["--output_csv", str(output_csv)])

    output_summary_json = _get(cfg, "stage3_reference_refresh.output_summary_json", "")
    if output_summary_json:
        args.extend(["--output_summary_json", str(output_summary_json)])

    output_md = _get(cfg, "stage3_reference_refresh.output_md", "")
    if output_md:
        args.extend(["--output_md", str(output_md)])

    print("===== AUX: refresh Stage3 reference library =====")
    print(f"[INFO] refresh script: {script}")

    r.run(args)

    refresh_out_dir = project_root / "outputs/stage3_reference_refresh"
    current_ref_csv = (
        project_root
        / "data/interim/references/stage3_condition_reference/current/stage3_condition_reference.csv"
    )

    r.outputs["stage3_reference_refresh_summary_json"] = str(
        refresh_out_dir / "stage3_reference_refresh_summary.json"
    )
    r.outputs["stage3_reference_refresh_report_md"] = str(
        refresh_out_dir / "stage3_reference_refresh_report.md"
    )
    r.outputs["stage3_condition_reference_current_csv"] = str(current_ref_csv)

def run_reliability_layer(r, cfg: dict) -> None:
    """
    Pipeline reliability + final ranking layer.

    Key design:
      final_top_routes_current_csv is updated after each enrichment/rerank step.
      Downstream steps should preserve newly attached support/QC/reference columns.
    """
    refresh_stage3_reference_library_if_enabled(r, cfg)
    print("[SKIP hotfix] stage3_gap_recovery benchmark aux disabled")
    print("[SKIP hotfix] stage3_gap_closure benchmark aux disabled")
    build_v36_target_aware_stage2_candidates_if_enabled(r, cfg)

    print("[SKIP hotfix] V37 stage3 input preflight disabled")

    print("[SKIP hotfix] V37 stage3 export interface plan disabled")
   
    print("[SKIP hotfix] V38 stage3 feature source audit disabled")
   
    print("[SKIP hotfix] V39 stage3 input feature extension construction disabled")

    audit_stage2_retrieval_candidates_if_enabled(r, cfg)

    run_precursor_qc_if_enabled(r, cfg)

    attach_route_confidence_if_enabled(r, cfg)

    attach_stage3_condition_reference_support_if_enabled(r, cfg)

    attach_metadata_aware_stage3_reference_support_if_enabled(r, cfg)

    add_condition_distribution_confidence_if_enabled(r, cfg)

    audit_condition_diversity_if_enabled(r, cfg)

    postprocess_confidence_with_precursor_qc_if_enabled(r, cfg)

    export_final_report_if_enabled(r, cfg)

    build_joint_route_features_if_enabled(r, cfg)

    apply_v3_joint_route_rerank_if_enabled(r, cfg)

    export_final_report_v3_if_enabled(r, cfg)

    build_v3_learned_ranker_dataset_if_enabled(r, cfg)

    apply_v3_learned_ranker_if_enabled(r, cfg)

    add_v43_route_template_features_if_enabled(r, cfg)

    apply_v43_template_ranker_if_enabled(r, cfg)

    apply_v43_safe_strict_gate_if_enabled(r, cfg)

    finalize_recommended_routes_if_enabled(r, cfg)

    audit_final_recommended_routes_if_enabled(r, cfg)
