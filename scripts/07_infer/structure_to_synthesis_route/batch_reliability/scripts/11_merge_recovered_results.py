#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from _common import load_config, get_paths, read_status_table, write_table_and_md


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = get_paths(cfg)
    project_root = paths["project_root"]
    adaptive_root = paths["adaptive_root"]

    status_df = read_status_table(cfg)

    normalized_csv = (
        paths["reliability_root"]
        / "normalize_recovered_candidates"
        / "normalized_recovered_candidates.csv"
    )
    normalized = safe_read_csv(normalized_csv)

    rows = []

    for _, r in status_df.iterrows():
        case_id = str(r["case_id"])
        out_root = project_root / "outputs/inference" / case_id
        route_dir = out_root / "routes_flow_fallback_retrieval_baseline_element_reranked"

        final_csv = route_dir / "final_recommended_routes.csv"
        final_md = route_dir / "final_recommended_routes.md"
        flow_csv = (
            out_root
            / "stage3_condition_predictions_flow_fallback_retrieval_baseline_element_reranked"
            / "test_candidates_flat.csv"
        )

        n_final_routes = 0
        has_final = final_csv.exists()
        if has_final:
            try:
                n_final_routes = len(pd.read_csv(final_csv))
            except Exception:
                n_final_routes = -1

        n_normalized = 0
        if len(normalized) > 0 and "case_id" in normalized.columns:
            n_normalized = int((normalized["case_id"].astype(str) == case_id).sum())

        original_status = str(r.get("final_case_status", ""))
        refined_status = str(r.get("refined_case_status", ""))

        if has_final and n_final_routes > 0:
            closure_status = "closed_has_final_routes"
        elif n_normalized > 0:
            closure_status = "partial_has_normalized_candidates"
        elif original_status in ["review_required_only", "pass"]:
            closure_status = "no_recovery_needed"
        else:
            closure_status = "still_open"

        rows.append({
            "case_id": case_id,
            "original_final_case_status": original_status,
            "original_refined_case_status": refined_status,
            "has_final_routes_after_recovery": has_final,
            "n_final_routes_after_recovery": n_final_routes,
            "has_flow_flat_after_recovery": flow_csv.exists(),
            "n_normalized_recovered_candidates": n_normalized,
            "closure_status": closure_status,
            "final_routes_csv": str(final_csv),
            "final_routes_md": str(final_md),
            "flow_flat_csv": str(flow_csv),
            "input_poscar": r.get("input_poscar", ""),
        })

    merged = pd.DataFrame(rows)

    out_dir = paths["reliability_root"] / "merge_recovered_results"

    write_table_and_md(
        merged,
        out_dir / "merged_recovered_results.csv",
        out_dir / "merged_recovered_results.md",
        "Merged Recovered Results",
        [
            "case_id",
            "original_final_case_status",
            "original_refined_case_status",
            "closure_status",
            "has_final_routes_after_recovery",
            "n_final_routes_after_recovery",
            "has_flow_flat_after_recovery",
            "n_normalized_recovered_candidates",
            "final_routes_csv",
        ],
    )

    if len(merged) > 0:
        summary = (
            merged["closure_status"]
            .value_counts(dropna=False)
            .reset_index()
        )
        summary.columns = ["closure_status", "n_cases"]
    else:
        summary = pd.DataFrame(columns=["closure_status", "n_cases"])

    write_table_and_md(
        summary,
        out_dir / "recovery_closure_status.csv",
        out_dir / "recovery_closure_status.md",
        "Recovery Closure Status Summary",
    )

    print("[SAVE]", out_dir / "merged_recovered_results.csv")
    print("[SAVE]", out_dir / "recovery_closure_status.md")


if __name__ == "__main__":
    main()
