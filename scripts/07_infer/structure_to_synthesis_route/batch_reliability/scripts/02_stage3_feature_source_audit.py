#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from _common import load_config, get_paths, read_status_table, read_compact_watchlist, write_table_and_md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = get_paths(cfg)
    project_root = paths["project_root"]
    df = read_status_table(cfg)
    watch = read_compact_watchlist(cfg)

    # Stage3 recovery targets.
    # Prefer compact watchlist if available, but only keep cases that truly need Stage3-side recovery.
    source = watch if len(watch) > 0 else df

    targets = source[
        source["final_case_status"].astype(str).isin([
            "needs_stage3_recovery",
            "needs_condition_reexport",
            "needs_route_finalization_recovery",
        ])
    ].copy()

    rows = []
    for _, r in targets.iterrows():
        case_id = r["case_id"]
        infer_root = project_root / "data/interim/infer" / case_id
        out_root = project_root / "outputs/inference" / case_id

        stage3_hybrid = infer_root / "stage3_hybrid" / "stage3_train_hybrid.csv"
        conditioned_x = infer_root / "stage3_conditioned_x_fallback_retrieval_baseline_element_reranked.csv"
        flow_flat = out_root / "stage3_condition_predictions_flow_fallback_retrieval_baseline_element_reranked" / "test_candidates_flat.csv"
        final_routes = out_root / "routes_flow_fallback_retrieval_baseline_element_reranked" / "final_recommended_routes.csv"

        rows.append({
            "case_id": case_id,
            "final_case_status": r.get("final_case_status", ""),
            "problem_type": r.get("problem_type", ""),
            "infer_root": str(infer_root),
            "out_root": str(out_root),
            "has_stage3_hybrid_csv": stage3_hybrid.exists(),
            "stage3_hybrid_csv": str(stage3_hybrid),
            "has_stage3_conditioned_x": conditioned_x.exists(),
            "stage3_conditioned_x": str(conditioned_x),
            "has_flow_flat_csv": flow_flat.exists(),
            "flow_flat_csv": str(flow_flat),
            "has_final_routes": final_routes.exists(),
            "final_routes_csv": str(final_routes),
        })

    cols = [
        "case_id", "final_case_status", "problem_type",
        "infer_root", "out_root",
        "has_stage3_hybrid_csv", "stage3_hybrid_csv",
        "has_stage3_conditioned_x", "stage3_conditioned_x",
        "has_flow_flat_csv", "flow_flat_csv",
        "has_final_routes", "final_routes_csv",
    ]

    audit = pd.DataFrame(rows, columns=cols)

    out_dir = paths["reliability_root"] / "stage3_feature_source_audit"
    display_cols = [
        "case_id", "final_case_status", "problem_type",
        "has_stage3_hybrid_csv", "has_stage3_conditioned_x",
        "has_flow_flat_csv", "has_final_routes",
        "stage3_hybrid_csv", "stage3_conditioned_x", "flow_flat_csv", "final_routes_csv"
    ]

    write_table_and_md(
        audit,
        out_dir / "stage3_feature_source_audit.csv",
        out_dir / "stage3_feature_source_audit.md",
        "Stage3 Feature Source Audit",
        display_cols,
    )

    print("[SAVE]", out_dir / "stage3_feature_source_audit.csv")


if __name__ == "__main__":
    main()
