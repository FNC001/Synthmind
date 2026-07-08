#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path


def summarize_routes(r):
    r.log("===== STEP 17: summarize readable synthesis routes =====")
    out_dir = r.out_dir / "routes_flow_fallback_retrieval_baseline_element_reranked"
    r.run([
        "python",
        r.route_scripts_dir / "pipeline/core/12_summarize_structure_to_routes.py",
        "--stage3_flat_csv", r.outputs["flow_flat_csv"],
        "--output_dir", out_dir,
        "--top_n", 100,
    ])
    csv = out_dir / "synthesis_routes_readable.csv"
    md = out_dir / "synthesis_routes_readable.md"
    r.require_file(csv)
    r.require_file(md)
    r.outputs["route_csv"] = str(csv)
    r.outputs["route_md"] = str(md)
    r.outputs["route_out_dir"] = str(out_dir)


def filter_display_routes(r):
    r.log("===== STEP 18: filter display routes =====")
    d = r.cfg["display"]
    out_dir = Path(r.outputs["route_out_dir"])
    csv = out_dir / "synthesis_routes_display_filtered.csv"
    md = out_dir / "synthesis_routes_display_filtered.md"
    r.run([
        "python",
        r.route_scripts_dir / "pipeline/core/14_filter_synthesis_routes_for_display.py",
        "--input_csv", r.outputs["route_csv"],
        "--output_csv", csv,
        "--output_md", md,
        "--min_temperature_c", d["min_temperature_c"],
        "--max_temperature_c", d["max_temperature_c"],
        "--min_time_h", d["min_time_h"],
        "--max_time_h", d["max_time_h"],
        "--top_n", d["top_n"],
        "--prefer_top_component_mean",
    ])
    r.require_file(csv)
    r.require_file(md)
    r.outputs["display_csv"] = str(csv)
    r.outputs["display_md"] = str(md)


def stage35_rule_rerank(r):
    r.log("===== STEP 19: Stage35 rule-based final route rerank =====")
    s = r.cfg["stage35"]
    script = Path(s["rule_script"])
    if not script.exists():
        r.log(f"[WARN] missing Stage35 rule script: {script}")
        r.record_degradation("stage35_rule_rerank", f"missing script: {script}")
        return

    out_dir = Path(r.outputs["route_out_dir"])
    csv = out_dir / "synthesis_routes_stage35_rule_reranked.csv"
    md = out_dir / "synthesis_routes_stage35_rule_reranked.md"
    r.run([
        "python", script,
        "--input_csv", r.outputs["display_csv"],
        "--output_csv", csv,
        "--output_md", md,
        "--top_n", s["top_n"],
    ])
    r.outputs["stage35_rule_csv"] = str(csv)
    r.outputs["stage35_rule_md"] = str(md)


def stage35_learned_rerank(r):
    r.log("===== STEP 20: Stage35 learned final route rerank =====")
    s = r.cfg["stage35"]
    script = Path(s["learned_script"])
    model = Path(s["learned_model"])
    if not script.exists() or not model.exists():
        r.log("[WARN] missing Stage35 learned script or model.")
        r.record_degradation("stage35_learned_rerank", "missing script or model")
        return

    out_dir = Path(r.outputs["route_out_dir"])
    csv = out_dir / "synthesis_routes_stage35_learned_reranked.csv"
    md = out_dir / "synthesis_routes_stage35_learned_reranked.md"
    r.run([
        "python", script,
        "--input_csv", r.outputs["display_csv"],
        "--model_path", model,
        "--output_csv", csv,
        "--output_md", md,
        "--top_n", s["top_n"],
    ])
    r.outputs["stage35_learned_csv"] = str(csv)
    r.outputs["stage35_learned_md"] = str(md)


def stage35_v21_rerank(r):
    r.log("===== STEP 21: Stage35 v2.1 hybrid final route rerank =====")
    s = r.cfg["stage35"]
    script = Path(s["v21_script"])
    model = Path(s["v21_model"])
    feature_cols = Path(s["v21_feature_cols"])

    if not script.exists() or not model.exists() or not feature_cols.exists():
        r.log("[WARN] missing Stage35 v21 script/model/feature cols.")
        r.record_degradation("stage35_v21_rerank", "missing script, model, or feature_cols")
        return

    out_dir = Path(r.outputs["route_out_dir"])
    csv = out_dir / "synthesis_routes_stage35_v21_hybrid_reranked.csv"
    md = out_dir / "synthesis_routes_stage35_v21_hybrid_reranked.md"
    summary = out_dir / "synthesis_routes_stage35_v21_hybrid_reranked_summary.json"

    r.run([
        "python", script,
        "--input_csv", r.outputs["display_csv"],
        "--model_path", model,
        "--feature_cols_json", feature_cols,
        "--output_csv", csv,
        "--output_md", md,
        "--summary_json", summary,
        "--top_n", s["top_n"],
    ])
    r.outputs["stage35_v21_csv"] = str(csv)
    r.outputs["stage35_v21_md"] = str(md)


def best_route_per_precursor(r):
    r.log("===== STEP 22: best route per precursor =====")
    s = r.cfg["stage35"]
    script = Path(s["best_per_precursor_script"])
    if not script.exists():
        r.log(f"[WARN] missing best-per-precursor script: {script}")
        r.record_degradation("best_route_per_precursor", f"missing script: {script}")
        return

    out_dir = Path(r.outputs["route_out_dir"])
    csv = out_dir / "synthesis_routes_stage35_v21_best_per_precursor.csv"
    md = out_dir / "synthesis_routes_stage35_v21_best_per_precursor.md"
    summary = out_dir / "synthesis_routes_stage35_v21_best_per_precursor_summary.json"

    best_input = r.outputs.get("display_csv")
    if "stage35_v21_csv" in r.outputs:
        best_input = r.outputs["stage35_v21_csv"]
    elif "stage35_learned_csv" in r.outputs:
        best_input = r.outputs["stage35_learned_csv"]
    elif "stage35_rule_csv" in r.outputs:
        best_input = r.outputs["stage35_rule_csv"]

    d = r.cfg["display"]
    r.run([
        "python", script,
        "--input_csv", best_input,
        "--output_csv", csv,
        "--output_md", md,
        "--summary_json", summary,
        "--top_n", s["top_n"],
        "--prefer_top_component_mean",
        "--prefer_full_element_coverage",
        "--min_temperature_c", d["min_temperature_c"],
        "--max_temperature_c", d["max_temperature_c"],
        "--min_time_h", d["min_time_h"],
        "--max_time_h", d["max_time_h"],
    ])
    r.outputs["best_per_precursor_csv"] = str(csv)
    r.outputs["best_per_precursor_md"] = str(md)
