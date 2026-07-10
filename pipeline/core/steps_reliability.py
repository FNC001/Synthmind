#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
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


def run_stage3_gap_recovery_if_enabled(r, cfg: dict) -> None:
    if _get_bool(cfg, "stage3_reference_refresh.gap_recovery.enabled", False):
        print("[SKIP removed] stage3 gap recovery is not part of the public pipeline.")
    else:
        print("[SKIP disabled] stage3 gap recovery")


def run_stage3_gap_closure_if_enabled(r, cfg: dict) -> None:
    if _get_bool(cfg, "stage3_reference_refresh.gap_closure.enabled", False):
        print("[SKIP removed] stage3 gap closure is not part of the public pipeline.")
    else:
        print("[SKIP disabled] stage3 gap closure")


def _project_root(r, cfg: dict) -> Path:
    return Path(cfg.get("project_root", getattr(r, "project_root", "")))


def _project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


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

    qc_script = ROOT / "postprocess" / "qc_route_precursors.py"
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

    confidence_script = ROOT / "postprocess" / "attach_route_confidence.py"

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
    script = ROOT / "postprocess" / "attach_stage3_condition_reference_support.py"

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
    Attach optional metadata-aware Stage3 reference support.

    This step should run after real Stage3 condition-reference support and before
    condition_distribution_confidence, so all downstream tables can preserve:
      metadata_aware_stage3_reference_support_score
      metadata_aware_stage3_reference_level
      metadata_aware_stage3_reference_warning_level
      metadata_aware_stage3_reference_recommendation_status
      metadata_aware_stage3_mp_id
      metadata_aware_stage3_mp_formula

    This is disabled by default in the public config and only runs when a
    metadata alignment CSV is supplied.
    """
    if not _get_bool(cfg, "reliability.attach_metadata_aware_stage3_reference_support.enabled", False):
        print("[SKIP disabled] attach_metadata_aware_stage3_reference_support")
        return

    if "route_out_dir" not in r.outputs:
        print("[SKIP] attach metadata-aware stage3 reference support; missing route_out_dir.")
        return

    project_root = _project_root(r, cfg)
    route_out_dir = Path(r.outputs["route_out_dir"])

    script = ROOT / "postprocess" / "attach_metadata_aware_stage3_reference_support.py"

    input_csv = _get_current_csv(r, route_out_dir)

    metadata_alignment_value = _get_str(
        cfg,
        "reliability.attach_metadata_aware_stage3_reference_support.metadata_alignment_csv",
        "",
    )
    if not metadata_alignment_value:
        print("[SKIP] attach metadata-aware stage3 reference support; metadata_alignment_csv not configured")
        return

    metadata_alignment_csv = Path(metadata_alignment_value)
    if not metadata_alignment_csv.is_absolute():
        metadata_alignment_csv = project_root / metadata_alignment_csv

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

    script = ROOT / "postprocess" / "add_condition_distribution_confidence.py"

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

    script = ROOT / "postprocess" / "audit_condition_diversity.py"
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

    post_script = ROOT / "postprocess" / "postprocess_confidence_with_precursor_qc.py"

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

    report_script = ROOT / "postprocess" / "export_final_report_v21.py"
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

    script = ROOT / "postprocess" / "build_joint_route_features.py"

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

    script = ROOT / "postprocess" / "apply_v3_joint_route_rerank.py"

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
    script = ROOT / "postprocess" / "export_final_report_v3.py"
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
    script = ROOT / "postprocess" / "build_v3_learned_ranker_dataset.py"

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
    script = ROOT / "postprocess" / "apply_v3_learned_ranker.py"

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
    print("[SKIP removed] target-aware benchmark candidate construction is not part of the public pipeline.")


def run_v37_stage3_input_preflight_if_enabled(r, cfg: dict) -> None:
    print("[SKIP removed] stage3 input preflight benchmark is not part of the public pipeline.")


def run_v37_stage3_export_interface_plan_if_enabled(r, cfg: dict) -> None:
    print("[SKIP removed] stage3 export interface benchmark is not part of the public pipeline.")


def run_v38_stage3_feature_source_audit_if_enabled(r, cfg: dict) -> None:
    print("[SKIP removed] stage3 feature source benchmark is not part of the public pipeline.")


def run_v39_stage3_input_feature_extension_construction_if_enabled(r, cfg: dict) -> None:
    print("[SKIP removed] stage3 input extension benchmark is not part of the public pipeline.")


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

    script = ROOT / "postprocess" / "audit_stage2_retrieval_candidates.py"

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

    v43_cfg = cfg.get("stage35_v43", {}) or {}
    if not _bool_enabled(v43_cfg.get("enabled"), True):
        print("[SKIP disabled] stage35_v43")
        return

    if "route_out_dir" not in r.outputs:
        print("[SKIP] add v43 template features; missing route_out_dir in outputs.")
        return

    project_root = _project_root(r, cfg)
    route_out_dir = Path(r.outputs["route_out_dir"])

    script_dir = _project_path(
        project_root,
        v43_cfg.get("script_dir", "pipeline/ranking/v43_template_aware"),
    )
    script = script_dir / "01_add_route_template_features.py"

    input_csv = _first_existing([
        r.outputs.get("final_top_routes_v3_learned_reranked_csv", ""),
        str(
            route_out_dir
            / v43_cfg.get("input_csv_name", "final_top_routes_v3_learned_reranked.csv")
        ),
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

    output_csv = route_out_dir / v43_cfg.get(
        "feature_csv_name", "synthesis_routes_stage35_v43_template_features.csv"
    )
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
        "--top_n",
        str(_get_int(cfg, "stage35_v43.top_n", _get_int(cfg, "final.top_n", 30))),
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

    v43_cfg = cfg.get("stage35_v43", {}) or {}
    if not _bool_enabled(v43_cfg.get("enabled"), True):
        print("[SKIP disabled] stage35_v43")
        return

    if "route_out_dir" not in r.outputs:
        print("[SKIP] apply v43 template ranker; missing route_out_dir in outputs.")
        return

    project_root = _project_root(r, cfg)
    route_out_dir = Path(r.outputs["route_out_dir"])

    script_dir = _project_path(
        project_root,
        v43_cfg.get("script_dir", "pipeline/ranking/v43_template_aware"),
    )
    script = script_dir / "04_apply_v43_template_pairwise_ranker_chemonly.py"

    input_csv = _first_existing([
        r.outputs.get("stage35_v43_template_features_csv", ""),
        str(
            route_out_dir
            / v43_cfg.get("feature_csv_name", "synthesis_routes_stage35_v43_template_features.csv")
        ),
        str(route_out_dir / "synthesis_routes_stage35_v43_template_features.csv"),
        r.outputs.get("final_top_routes_current_csv", ""),
    ])

    if input_csv is None:
        input_csv = route_out_dir / v43_cfg.get(
            "feature_csv_name", "synthesis_routes_stage35_v43_template_features.csv"
        )

    model_path = _project_path(
        project_root,
        v43_cfg.get(
            "model_path",
            cfg.get(
                "v43_template_ranker_model_path",
                "runs/stage35/route_ranker_v43_template_aware/stage35_v43_template_pairwise_chemonly_extratrees.joblib",
            ),
        ),
    )

    feature_cols_json = _project_path(
        project_root,
        v43_cfg.get(
            "feature_cols_json",
            cfg.get(
                "v43_template_ranker_feature_cols_json",
                "runs/stage35/route_ranker_v43_template_aware/stage35_v43_template_pairwise_chemonly_feature_cols.json",
            ),
        ),
    )

    output_csv = route_out_dir / v43_cfg.get(
        "output_csv_name", "final_top_routes_v43_template_chemonly_reranked.csv"
    )
    output_md = route_out_dir / v43_cfg.get(
        "output_md_name", "final_top_routes_v43_template_chemonly_reranked.md"
    )
    summary_json = route_out_dir / v43_cfg.get(
        "summary_json_name", "final_top_routes_v43_template_chemonly_reranked_summary.json"
    )

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
        "--top_n",
        str(_get_int(cfg, "stage35_v43.top_n", _get_int(cfg, "final.top_n", 30))),
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
    script = ROOT / "postprocess" / "apply_v43_safe_strict_gate.py"

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

    script = ROOT / "postprocess" / "finalize_recommended_routes.py"

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

    script = ROOT / "postprocess" / "audit_final_recommended_routes.py"

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

    Disabled by default in the public release. If a project supplies its own
    refresh script, configure it explicitly before enabling this step.
    """
    if not _get_bool(cfg, "stage3_reference_refresh.enabled", False):
        print("[SKIP disabled] refresh_stage3_reference_library")
        return

    project_root = _project_root(r, cfg)

    script = (
        project_root
        / "pipeline/postprocess/"
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
