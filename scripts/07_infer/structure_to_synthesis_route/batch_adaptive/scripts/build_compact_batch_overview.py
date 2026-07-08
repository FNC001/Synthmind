#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def write_md_table(df: pd.DataFrame, path: Path, title: str, note: str = "") -> None:
    lines = [f"# {title}", ""]
    if note:
        lines.append(note)
        lines.append("")
    lines.append(f"- Rows: `{len(df)}`")
    lines.append("")
    if len(df) == 0:
        lines.append("No records.")
    else:
        lines.append(df.to_markdown(index=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def value_count_df(df: pd.DataFrame, col: str, name: str) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame(columns=[col, name])
    out = df[col].value_counts(dropna=False).reset_index()
    out.columns = [col, name]
    return out


def pick_existing(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default="/Users/wyc/SynPred")
    ap.add_argument("--batch_name", default="batch_500")
    ap.add_argument("--top_n_routes", type=int, default=100)
    ap.add_argument("--include_low_confidence", action="store_true")
    args = ap.parse_args()

    project_root = Path(args.project_root)
    batch_dir = project_root / "outputs" / "batch_adaptive" / args.batch_name

    master_csv = batch_dir / "master_status_standard.csv"
    routes_csv = batch_dir / "batch_recommended_routes_topn_review_refined.csv"

    if not routes_csv.exists():
        fallback = batch_dir / "batch_recommended_routes_topn.csv"
        if fallback.exists():
            routes_csv = fallback

    master = read_csv_required(master_csv)
    routes = read_csv_required(routes_csv)

    out_dir = batch_dir / "batch_compact_overview"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. case-level status summary
    case_status = value_count_df(master, "final_case_status", "n_cases")
    refined_status = value_count_df(master, "refined_case_status", "n_cases")

    case_status.to_csv(out_dir / "case_status_summary.csv", index=False)
    refined_status.to_csv(out_dir / "refined_case_status_summary.csv", index=False)

    write_md_table(
        case_status,
        out_dir / "case_status_summary.md",
        "Case-level Final Status Summary",
    )
    write_md_table(
        refined_status,
        out_dir / "refined_case_status_summary.md",
        "Case-level Refined Status Summary",
    )

    # 2. route-level status summary
    route_col = "route_refined_status" if "route_refined_status" in routes.columns else "review_severity"
    route_status = value_count_df(routes, route_col, "n_routes")
    route_status.to_csv(out_dir / "route_status_summary.csv", index=False)
    write_md_table(
        route_status,
        out_dir / "route_status_summary.md",
        "Route-level Status Summary",
    )

    # 3. watchlist
    watch_mask = pd.Series(False, index=master.index)

    if "final_case_status" in master.columns:
        watch_mask |= master["final_case_status"].astype(str).isin([
            "pipeline_failed",
            "needs_stage2_recovery",
            "needs_stage3_recovery",
            "needs_condition_reexport",
            "needs_route_finalization_recovery",
            "needs_manual_or_rule_recovery",
        ])

    if "refined_case_status" in master.columns:
        watch_mask |= master["refined_case_status"].astype(str).isin([
            "low_confidence_review",
            "needs_manual_check",
            "needs_manual_check_refined",
        ])

    watch = master[watch_mask].copy()

    watch_cols = pick_existing(master, [
        "case_id",
        "final_case_status",
        "refined_case_status",
        "refined_case_reason",
        "condition_support_score",
        "top1_condition_support_score",
        "top1_final_score",
        "top1_score",
        "top1_precursor_set",
        "problem_type",
        "recommended_action",
        "input_poscar",
        "pipeline_log",
    ])

    watch_out = watch[watch_cols].copy() if watch_cols else watch
    watch_out.to_csv(out_dir / "compact_watchlist.csv", index=False)
    write_md_table(
        watch_out,
        out_dir / "compact_watchlist.md",
        "Compact Watchlist",
        "Cases requiring attention: failed, low-confidence, condition re-export, or manual/rule review.",
    )

    # 4. compact top routes
    if route_col in routes.columns:
        if args.include_low_confidence:
            selected_routes = routes.copy()
        else:
            selected_routes = routes[
                routes[route_col].astype(str).isin([
                    "route_high_confidence",
                    "route_medium_confidence",
                    "high_confidence_review",
                    "medium_confidence_review",
                ])
            ].copy()
    else:
        selected_routes = routes.copy()

    score_col = None
    for c in [
        "final_recommendation_score",
        "stage35_v43_safe_strict_score",
        "top1_score",
        "score",
    ]:
        if c in selected_routes.columns:
            score_col = c
            break

    if score_col:
        selected_routes[score_col] = pd.to_numeric(selected_routes[score_col], errors="coerce")
        selected_routes = selected_routes.sort_values(score_col, ascending=False)

    selected_routes = selected_routes.head(args.top_n_routes).copy()

    route_cols = pick_existing(selected_routes, [
        "case_id",
        "case_status",
        "case_refined_status",
        "refined_case_status",
        "route_refined_status",
        "route_refined_reason",
        "review_severity",
        "precursor_set",
        "final_recommendation_score",
        "stage35_v43_safe_strict_score",
        "real_stage3_condition_reference_support_score",
        "condition_distribution_support_score",
        "final_recommendation_status",
        "input_poscar",
    ])

    compact_routes = selected_routes[route_cols].copy() if route_cols else selected_routes
    compact_routes.to_csv(out_dir / "compact_top_routes.csv", index=False)
    write_md_table(
        compact_routes,
        out_dir / "compact_top_routes.md",
        f"Compact Top Routes Top {args.top_n_routes}",
        "By default, this table keeps high- and medium-confidence routes only.",
    )

    # 5. overall report
    report = []
    report.append("# Compact Batch Overview Report")
    report.append("")
    report.append(f"- Batch name: `{args.batch_name}`")
    report.append(f"- Batch directory: `{batch_dir}`")
    report.append(f"- Number of cases: `{len(master)}`")
    report.append(f"- Number of exported routes: `{len(routes)}`")
    report.append(f"- Number of watchlist cases: `{len(watch_out)}`")
    report.append(f"- Number of compact top routes: `{len(compact_routes)}`")
    report.append("")

    report.append("## Case-level final status")
    report.append("")
    report.append(case_status.to_markdown(index=False) if len(case_status) else "No records.")
    report.append("")

    report.append("## Case-level refined status")
    report.append("")
    report.append(refined_status.to_markdown(index=False) if len(refined_status) else "No records.")
    report.append("")

    report.append("## Route-level status")
    report.append("")
    report.append(route_status.to_markdown(index=False) if len(route_status) else "No records.")
    report.append("")

    report.append("## Main files")
    report.append("")
    for label, path in [
        ("case_status_summary", out_dir / "case_status_summary.csv"),
        ("refined_case_status_summary", out_dir / "refined_case_status_summary.csv"),
        ("route_status_summary", out_dir / "route_status_summary.csv"),
        ("compact_watchlist", out_dir / "compact_watchlist.csv"),
        ("compact_top_routes", out_dir / "compact_top_routes.csv"),
    ]:
        report.append(f"- {label}: `{path}`")

    (out_dir / "compact_overview_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print("[SAVE]", out_dir / "case_status_summary.csv")
    print("[SAVE]", out_dir / "refined_case_status_summary.csv")
    print("[SAVE]", out_dir / "route_status_summary.csv")
    print("[SAVE]", out_dir / "compact_watchlist.csv")
    print("[SAVE]", out_dir / "compact_watchlist.md")
    print("[SAVE]", out_dir / "compact_top_routes.csv")
    print("[SAVE]", out_dir / "compact_top_routes.md")
    print("[SAVE]", out_dir / "compact_overview_report.md")


if __name__ == "__main__":
    main()
