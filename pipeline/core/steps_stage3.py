#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path


def build_stage3_features(r):
    r.log("===== STEP 14: build Stage3 hybrid feature table =====")
    out_dir = r.work_dir / "stage3_hybrid"
    r.run([
        "python",
        r.project_root / "pipeline/core/05_build_hybrid_features_infer.py",
        "--task", "stage3",
        "--output_dir", out_dir,
        "--infer_descriptor_csv", r.outputs["infer_structdesc_csv"],
        "--infer_embedding_csv", r.outputs["final_graph_embed_csv"],
        "--embedding_prefix", "graph_emb",
        "--replicate_to_train_val",
    ])
    p = out_dir / "stage3_train_hybrid.csv"
    r.require_file(p)
    r.outputs["stage3_hybrid_csv"] = str(p)


def build_stage3_conditioned_table(r):
    r.log("===== STEP 15: build Stage3 precursor-conditioned feature table =====")
    out = r.work_dir / "stage3_conditioned_x_fallback_retrieval_baseline_element_reranked.csv"
    r.run([
        "python",
        r.project_root / "pipeline/core/05c_build_stage3_conditioned_feature_table_infer.py",
        "--infer_hybrid_csv", r.outputs["stage3_hybrid_csv"],
        "--stage2_candidates_csv", r.outputs["stage2_final_csv"],
        "--schema_json", r.cfg["stage3"]["schema_json"],
        "--output_csv", out,
        "--max_stage2_candidates", r.cfg["element_rerank"]["top_n"],
    ])
    r.require_file(out)
    r.outputs["stage3_conditioned_x"] = str(out)


def run_stage3_flow(r):
    r.log("===== STEP 16: run Stage3 Mixture Flow conditioned inference =====")
    s = r.cfg["stage3"]
    out_dir = r.out_dir / "stage3_condition_predictions_flow_fallback_retrieval_baseline_element_reranked"
    seed = s.get("seed", None)
    r.run([
        "python",
        r.route_scripts_dir / "pipeline/core/13_run_stage3_infer_mixture_flow_conditioned.py",
        "--conditioned_x_csv", r.outputs["stage3_conditioned_x"],
        "--schema_json", s["schema_json"],
        "--flow_ckpt", s["flow_ckpt"],
        "--flow_script", s["flow_script"],
        "--output_dir", out_dir,
        "--top_k_conditions", s["top_k_conditions"],
        "--n_flow_samples", s["n_flow_samples"],
        "--device", r.device,
    ] + (["--seed", str(seed)] if seed is not None else []))
    p = out_dir / "test_candidates_flat.csv"
    r.require_file(p)
    r.outputs["flow_flat_csv"] = str(p)
    r.outputs["flow_flat_csv_flow_only"] = str(p)


def run_stage3_lgbm(r):
    """STEP 16b: run Stage3 LightGBM quantile ensemble inference.

    Drop-in replacement for run_stage3_flow with better top-1 accuracy
    (MAE 148°C vs 180°C) and atmosphere/time_bucket predictions.
    """
    r.log("===== STEP 16b: run Stage3 LightGBM quantile ensemble inference =====")
    s = r.cfg["stage3"]
    lgbm_cfg = r.cfg.get("stage3_lgbm", {})
    out_dir = r.out_dir / "stage3_condition_predictions_lgbm"

    cmd = [
        "python",
        r.route_scripts_dir / "pipeline/core/13b_run_stage3_infer_lgbm_quantile.py",
        "--conditioned_x_csv", r.outputs["stage3_conditioned_x"],
        "--schema_json", s["schema_json"],
        "--output_dir", out_dir,
        "--top_k_conditions", s.get("top_k_conditions", 5),
    ]

    if lgbm_cfg.get("temp_model_dir"):
        cmd += ["--temp_model_dir", lgbm_cfg["temp_model_dir"]]
    if lgbm_cfg.get("time_model_dir"):
        cmd += ["--time_model_dir", lgbm_cfg["time_model_dir"]]
    if lgbm_cfg.get("atm_model"):
        cmd += ["--atm_model", lgbm_cfg["atm_model"]]
    if lgbm_cfg.get("time_bucket_model"):
        cmd += ["--time_bucket_model", lgbm_cfg["time_bucket_model"]]

    seed = s.get("seed", None)
    if seed is not None:
        cmd += ["--seed", str(seed)]

    r.run(cmd)
    p = out_dir / "test_candidates_flat.csv"
    r.require_file(p)
    r.outputs["lgbm_flat_csv"] = str(p)
    # If flow wasn't run, use lgbm as the primary output
    if "flow_flat_csv" not in r.outputs:
        r.outputs["flow_flat_csv"] = str(p)


def compare_stage3_models(r):
    """STEP 16c: compare Flow vs LightGBM Stage3 predictions side-by-side.

    Only runs when both run_stage3_flow and run_stage3_lgbm produced outputs.
    Generates a comparison CSV and markdown report.
    """
    import pandas as pd

    flow_csv = r.outputs.get("flow_flat_csv_flow_only")
    lgbm_csv = r.outputs.get("lgbm_flat_csv")

    if not flow_csv or not lgbm_csv:
        r.log("[SKIP] compare_stage3_models: need both flow and lgbm outputs")
        return

    if not Path(flow_csv).exists() or not Path(lgbm_csv).exists():
        r.log("[SKIP] compare_stage3_models: output files missing")
        return

    r.log("===== STEP 16c: compare Stage3 Flow vs LightGBM =====")

    df_flow = pd.read_csv(flow_csv)
    df_lgbm = pd.read_csv(lgbm_csv)

    out_dir = r.out_dir / "stage3_model_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Identify common columns for comparison
    temp_col = None
    for c in ["predicted_temperature", "temperature", "temp_mean", "condition_0"]:
        if c in df_flow.columns and c in df_lgbm.columns:
            temp_col = c
            break

    precursor_col = None
    for c in ["precursor_set", "precursor_set_str", "candidate_id"]:
        if c in df_flow.columns:
            precursor_col = c
            break

    lines = ["# Stage3 Model Comparison: Flow vs LightGBM", ""]
    lines.append(f"- Flow predictions: {len(df_flow)} rows")
    lines.append(f"- LightGBM predictions: {len(df_lgbm)} rows")
    lines.append("")

    if temp_col:
        flow_temps = df_flow[temp_col].dropna()
        lgbm_temps = df_lgbm[temp_col].dropna()
        lines.append(f"## Temperature ({temp_col})")
        lines.append(f"- Flow: mean={flow_temps.mean():.1f}, std={flow_temps.std():.1f}, "
                     f"min={flow_temps.min():.1f}, max={flow_temps.max():.1f}")
        lines.append(f"- LightGBM: mean={lgbm_temps.mean():.1f}, std={lgbm_temps.std():.1f}, "
                     f"min={lgbm_temps.min():.1f}, max={lgbm_temps.max():.1f}")
        lines.append("")

    # Per-precursor comparison if possible
    if precursor_col and temp_col and precursor_col in df_lgbm.columns:
        flow_top1 = df_flow.groupby(precursor_col)[temp_col].first().rename("flow_temp")
        lgbm_top1 = df_lgbm.groupby(precursor_col)[temp_col].first().rename("lgbm_temp")
        merged = pd.concat([flow_top1, lgbm_top1], axis=1).dropna()

        if len(merged) > 0:
            merged["diff"] = (merged["flow_temp"] - merged["lgbm_temp"]).abs()
            lines.append("## Per-precursor top-1 temperature comparison")
            lines.append(f"- Matched precursors: {len(merged)}")
            lines.append(f"- Mean absolute difference: {merged['diff'].mean():.1f}°C")
            lines.append(f"- Median absolute difference: {merged['diff'].median():.1f}°C")
            lines.append(f"- Max absolute difference: {merged['diff'].max():.1f}°C")
            lines.append("")

            merged_out = out_dir / "stage3_flow_vs_lgbm_per_precursor.csv"
            merged.reset_index().to_csv(merged_out, index=False)
            r.outputs["stage3_comparison_per_precursor_csv"] = str(merged_out)

    # LightGBM-specific columns
    lgbm_extra_cols = [c for c in df_lgbm.columns if c not in df_flow.columns]
    if lgbm_extra_cols:
        lines.append("## LightGBM-only columns")
        for c in lgbm_extra_cols[:10]:
            lines.append(f"- {c}")
        lines.append("")

    report_md = out_dir / "stage3_model_comparison.md"
    report_md.write_text("\n".join(lines), encoding="utf-8")
    r.log(f"[SAVE] {report_md}")

    r.outputs["stage3_model_comparison_md"] = str(report_md)

    # Use lgbm as primary (better top-1 accuracy per docstring)
    primary = r.cfg.get("stage3_comparison", {}).get("primary_model", "lgbm")
    if primary == "flow":
        r.outputs["flow_flat_csv"] = flow_csv
    else:
        r.outputs["flow_flat_csv"] = lgbm_csv
    r.log(f"[INFO] primary Stage3 model for downstream: {primary}")
