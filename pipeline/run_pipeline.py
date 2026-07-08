#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent


POSTPROCESS_STEP_NAMES = {
    "add_condition_distribution_confidence",
    "build_joint_route_features",
    "apply_v3_joint_route_rerank",
    "export_final_report_v3",
    "build_v3_learned_ranker_dataset",
    "apply_v3_learned_ranker",
    "add_v43_route_template_features",
    "apply_v43_template_ranker",
    "apply_v43_safe_strict_gate",
}


def build_step_funcs():
    from core import steps_common
    from core import steps_final
    from core import steps_reliability
    from core import steps_stage2
    from core import steps_stage3
    from core import steps_stage35

    return [
        # ---------- Common / structure input ----------
        ("make_infer_split", steps_common.make_infer_split),
        ("build_structdesc", steps_common.build_structdesc),
        ("build_chgnet_embedding", steps_common.build_chgnet_embedding),
        ("finalize_graph_embedding", steps_common.finalize_graph_embedding),

        # ---------- Stage2: structure -> precursor candidates ----------
        ("build_stage2_features", steps_stage2.build_stage2_features),
        ("build_stage2_npz", steps_stage2.build_stage2_npz),
        ("sample_stage2_gflownet", steps_stage2.sample_stage2_gflownet),
        ("constrain_stage2_by_composition", steps_stage2.constrain_stage2_by_composition),
        ("summarize_stage2", steps_stage2.summarize_stage2),
        ("add_composition_fallback", steps_stage2.add_composition_fallback),
        ("retrieve_stage2_candidates", steps_stage2.retrieve_stage2_candidates),
        ("predict_stage2_baseline", steps_stage2.predict_stage2_baseline),
        ("merge_stage2_sources", steps_stage2.merge_stage2_sources),
        ("rerank_stage2_by_elements", steps_stage2.rerank_stage2_by_elements),
        ("fix_stage2_global_rank", steps_stage2.fix_stage2_global_rank),

        # ---------- Stage3: precursor-conditioned condition prediction ----------
        ("build_stage3_features", steps_stage3.build_stage3_features),
        ("build_stage3_conditioned_table", steps_stage3.build_stage3_conditioned_table),
        ("run_stage3_flow", steps_stage3.run_stage3_flow),
        ("run_stage3_lgbm", steps_stage3.run_stage3_lgbm),
        ("compare_stage3_models", steps_stage3.compare_stage3_models),

        # ---------- Route formatting and Stage35 ranking ----------
        ("summarize_routes", steps_stage35.summarize_routes),
        ("filter_display_routes", steps_stage35.filter_display_routes),
        ("stage35_rule_rerank", steps_stage35.stage35_rule_rerank),
        ("stage35_learned_rerank", steps_stage35.stage35_learned_rerank),
        ("stage35_v21_rerank", steps_stage35.stage35_v21_rerank),
        ("best_route_per_precursor", steps_stage35.best_route_per_precursor),

        # ---------- Final route table ----------
        ("export_final_top_routes", steps_final.export_final_top_routes),

        # ---------- Optional precursor-only report ----------
        ("export_precursor_only_report", steps_stage2.export_precursor_only_report),
    ], steps_common, steps_reliability


def cfg_get(cfg, key: str, default=None):
    """
    Safely get config value from dict-like or object-like config.
    """
    try:
        return cfg.get(key, default)
    except Exception:
        return default


def print_final_outputs(r: PipelineRunner) -> None:
    print()
    print("============================================================")
    print("[DONE] pipeline_v3 finished")
    print("Outputs:")

    for k, v in r.outputs.items():
        print(f"  {k}: {v}")

    print()
    print("[FINAL OUTPUT]")
    if "final_recommended_routes_md" in r.outputs:
        print(f"  final recommended routes md: {r.outputs['final_recommended_routes_md']}")
    if "final_recommended_routes_csv" in r.outputs:
        print(f"  final recommended routes csv: {r.outputs['final_recommended_routes_csv']}")
    if "final_recommended_routes_source" in r.outputs:
        print(f"  final recommended routes source: {r.outputs['final_recommended_routes_source']}")

    if "best_per_precursor_md" in r.outputs:
        print(f"  best route per precursor md: {r.outputs['best_per_precursor_md']}")
    if "best_per_precursor_csv" in r.outputs:
        print(f"  best route per precursor csv: {r.outputs['best_per_precursor_csv']}")

    if "final_top_routes_md" in r.outputs:
        print(f"  final top routes md: {r.outputs['final_top_routes_md']}")
    if "final_top_routes_csv" in r.outputs:
        print(f"  final top routes csv: {r.outputs['final_top_routes_csv']}")

    if "final_top_routes_with_precursor_qc_md" in r.outputs:
        print(
            "  final top routes with precursor QC md: "
            f"{r.outputs['final_top_routes_with_precursor_qc_md']}"
        )
    if "final_top_routes_with_precursor_qc_csv" in r.outputs:
        print(
            "  final top routes with precursor QC csv: "
            f"{r.outputs['final_top_routes_with_precursor_qc_csv']}"
        )

    if "final_top_routes_with_confidence_md" in r.outputs:
        print(
            "  final top routes with confidence md: "
            f"{r.outputs['final_top_routes_with_confidence_md']}"
        )
    if "final_top_routes_with_confidence_csv" in r.outputs:
        print(
            "  final top routes with confidence csv: "
            f"{r.outputs['final_top_routes_with_confidence_csv']}"
        )

    if "precursor_only_md" in r.outputs:
        print(f"  precursor-only md: {r.outputs['precursor_only_md']}")
    if "precursor_only_csv" in r.outputs:
        print(f"  precursor-only csv: {r.outputs['precursor_only_csv']}")

    if "final_top_routes_with_joint_features_md" in r.outputs:
        print(f"  final top routes with joint features md: {r.outputs['final_top_routes_with_joint_features_md']}")
    if "final_top_routes_with_joint_features_csv" in r.outputs:
        print(f"  final top routes with joint features csv: {r.outputs['final_top_routes_with_joint_features_csv']}")

    if "final_top_routes_v3_joint_reranked_md" in r.outputs:
        print(f"  final top routes v3 joint reranked md: {r.outputs['final_top_routes_v3_joint_reranked_md']}")
    if "final_top_routes_v3_joint_reranked_csv" in r.outputs:
        print(f"  final top routes v3 joint reranked csv: {r.outputs['final_top_routes_v3_joint_reranked_csv']}")

    if "final_top_routes_v3_learned_reranked_md" in r.outputs:
        print(f"  final top routes v3 learned reranked md: {r.outputs['final_top_routes_v3_learned_reranked_md']}")
    if "final_top_routes_v3_learned_reranked_csv" in r.outputs:
        print(f"  final top routes v3 learned reranked csv: {r.outputs['final_top_routes_v3_learned_reranked_csv']}")

    if "final_top_routes_v43_template_chemonly_reranked_md" in r.outputs:
        print(
            "  final top routes v43 template chem-only reranked md: "
            f"{r.outputs['final_top_routes_v43_template_chemonly_reranked_md']}"
        )
    if "final_top_routes_v43_template_chemonly_reranked_csv" in r.outputs:
        print(
            "  final top routes v43 template chem-only reranked csv: "
            f"{r.outputs['final_top_routes_v43_template_chemonly_reranked_csv']}"
        )
def safe_copy(src: Path, dst: Path) -> bool:
    """
    Copy src to dst if src exists. Return True if copied.
    """
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def select_final_recommended_routes(r: PipelineRunner) -> None:
    """
    Backward-compatible final output selector.

    Priority:
      1. v4.3 template-aware chem-only reranker
      2. v3 learned reranker
      3. v3 joint reranker
      4. confidence route table
      5. original final_top_routes

    This function does not delete or overwrite historical outputs.
    It only creates a stable final user-facing alias:
      final_recommended_routes.csv/md
    """
    route_out_dir = Path(r.outputs.get("route_out_dir", r.out_dir))
    final_csv = route_out_dir / "final_recommended_routes.csv"
    final_md = route_out_dir / "final_recommended_routes.md"
    if final_csv.exists():

        r.outputs["final_recommended_routes_csv"] = str(final_csv)
        if final_md.exists():
            r.outputs["final_recommended_routes_md"] = str(final_md)
        r.outputs["final_recommended_routes_source"] = r.outputs.get(
            "final_recommended_routes_source",
            "existing_final_recommended_routes"
        )
        print(f"[KEEP] existing final recommended routes: {final_csv}")
        print(f"[INFO] final_recommended_routes_source = {r.outputs['final_recommended_routes_source']}")
        return

    candidates = [
        (
            "stage35_v43_template_chemonly",
            r.outputs.get("final_top_routes_v43_template_chemonly_reranked_csv"),
            r.outputs.get("final_top_routes_v43_template_chemonly_reranked_md"),
        ),
        (
            "v3_learned",
            r.outputs.get("final_top_routes_v3_learned_reranked_csv"),
            r.outputs.get("final_top_routes_v3_learned_reranked_md"),
        ),
        (
            "v3_joint",
            r.outputs.get("final_top_routes_v3_joint_reranked_csv"),
            r.outputs.get("final_top_routes_v3_joint_reranked_md"),
        ),
        (
            "confidence",
            r.outputs.get("final_top_routes_with_confidence_csv"),
            r.outputs.get("final_top_routes_with_confidence_md"),
        ),
        (
            "stage35_v21_or_basic_final",
            r.outputs.get("final_top_routes_csv"),
            r.outputs.get("final_top_routes_md"),
        ),
    ]

    selected_source = None

    for source, csv_path, md_path in candidates:
        if not csv_path:
            continue

        csv_path = Path(csv_path)
        if not csv_path.exists():
            continue

        safe_copy(csv_path, final_csv)

        if md_path and Path(md_path).exists():
            safe_copy(Path(md_path), final_md)
        else:
            final_md.write_text(
                f"# Final Recommended Routes\n\n"
                f"Source: `{source}`\n\n"
                f"CSV: `{final_csv}`\n",
                encoding="utf-8",
            )

        selected_source = source
        break

    if selected_source is None:
        print("[WARN] no available route table found for final_recommended_routes.")
        return

    r.outputs["final_recommended_routes_csv"] = str(final_csv)
    r.outputs["final_recommended_routes_md"] = str(final_md)
    r.outputs["final_recommended_routes_source"] = selected_source

    print(f"[SAVE] {final_csv}")
    print(f"[SAVE] {final_md}")
    print(f"[INFO] final_recommended_routes_source = {selected_source}")
    if "final_report_md" in r.outputs:
        print(f"  final report md: {r.outputs['final_report_md']}")

    print("============================================================")


def recursive_replace_infer_name(obj, old_name, new_name):
    """
    Recursively replace infer case name in config strings.
    This is robust when config contains pre-expanded paths such as
    poscar_dir/work_dir/out_dir inside nested dictionaries.
    """
    if isinstance(obj, dict):
        return {
            k: recursive_replace_infer_name(v, old_name, new_name)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [recursive_replace_infer_name(v, old_name, new_name) for v in obj]
    if isinstance(obj, str):
        return obj.replace(old_name, new_name)
    return obj


def apply_infer_name_override(cfg, infer_name):
    if not infer_name:
        return cfg

    old_name = cfg.get("infer_name", "demo_poscar_test")
    cfg = recursive_replace_infer_name(cfg, old_name, infer_name)
    cfg["infer_name"] = infer_name
    return cfg


def apply_project_root_override(cfg, project_root):
    if not project_root:
        return cfg

    old_root = cfg.get("project_root")
    if old_root:
        cfg = recursive_replace_infer_name(cfg, str(old_root), str(project_root))
    cfg["project_root"] = str(project_root)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Synthmind structure-to-synthesis route inference pipeline."
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--skip_preflight", action="store_true")
    ap.add_argument("--only_step", default=None, help="Run only one named step.")
    ap.add_argument("--start_from", default=None, help="Skip steps before this named step.")

    ap.add_argument("--infer_name", default=None, help="Override infer_name in config.")
    ap.add_argument("--project_root", default=None, help="Override project_root in config.")
    args = ap.parse_args()

    from core.config import load_config
    from core.runner import PipelineRunner

    step_funcs, steps_common, steps_reliability = build_step_funcs()

    cfg = load_config(
        args.config,
        overrides={
            "infer_name": getattr(args, "infer_name", None),
            "project_root": getattr(args, "project_root", None),
        },
    )
    
    r = PipelineRunner(cfg)

    print("============================================================")
    print("Synthmind structure-to-synthesis pipeline")
    print("pipeline_name =", cfg_get(cfg, "pipeline_name"))
    print("infer_name    =", cfg_get(cfg, "infer_name"))
    print("project_root  =", cfg_get(cfg, "project_root"))
    print("work_dir      =", r.work_dir)
    print("out_dir       =", r.out_dir)
    print("device        =", r.device)
    print("config        =", cfg_get(cfg, "_config_path"))
    print("============================================================")

    r.ensure_dirs()
    r.restore_existing_outputs()

    if not args.skip_preflight:
        steps_common.preflight(r)

    valid_step_names = [name for name, _ in step_funcs]
    known_config_step_names = set(valid_step_names) | POSTPROCESS_STEP_NAMES

    # Warn about unknown step names in config
    config_steps = cfg.get("steps", {})
    for name in config_steps:
        if name not in known_config_step_names:
            print(f"[WARN] config steps.{name} is not a recognized step name")

    if args.only_step is not None and args.only_step not in valid_step_names:
        raise ValueError(
            f"Unknown --only_step: {args.only_step}. "
            f"Available steps: {', '.join(valid_step_names)}"
        )

    if args.start_from is not None and args.start_from not in valid_step_names:
        raise ValueError(
            f"Unknown --start_from: {args.start_from}. "
            f"Available steps: {', '.join(valid_step_names)}"
        )

    started = args.start_from is None

    for name, func in step_funcs:
        if args.only_step is not None and name != args.only_step:
            print(f"[SKIP only_step] {name}")
            continue

        if args.start_from is not None and name == args.start_from:
            started = True

        if not started:
            print(f"[SKIP before start_from] {name}")
            continue

        if r.step_enabled(name):
            r.begin_step(name)
            func(r)
            r.end_step()
        else:
            print(f"[SKIP disabled] {name}")

    run_route_postprocess = (
        args.only_step is None
        and cfg_get(cfg, "pipeline_name") != "precursor_only"
    )

    if run_route_postprocess:
        # ---------- pipeline_v3 reliability layer ----------
        r.begin_step("reliability_layer")
        steps_reliability.run_reliability_layer(r, cfg)
        r.end_step()

        # Stable backward-compatible final user-facing route table.
        r.begin_step("select_final_recommended_routes")
        select_final_recommended_routes(r)
        r.end_step()
    else:
        print("[SKIP] route reliability/final recommendation postprocess")

    r.save_manifest()
    print_final_outputs(r)



if __name__ == "__main__":
    main()
