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


def read_csv_optional(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def pick_existing(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def safe_markdown(df: pd.DataFrame, index: bool = False) -> str:
    """Convert DataFrame to markdown safely by replacing pd.NA/NaN with empty strings."""
    if df is None or len(df) == 0:
        return "No records."
    tmp = df.copy()
    tmp = tmp.astype(object).where(pd.notna(tmp), "")
    return tmp.to_markdown(index=index)


def contains_series(s: pd.Series, query: str) -> pd.Series:
    return s.astype(str).str.contains(query, case=False, na=False, regex=False)


def filter_by_query(df: pd.DataFrame, query: str) -> pd.DataFrame:
    if query is None or str(query).strip() == "":
        return df.copy()

    q = str(query).strip()
    mask = pd.Series(False, index=df.index)

    for c in [
        "case_id",
        "target_formula",
        "target_elements",
        "input_poscar",
        "precursor_set",
    ]:
        if c in df.columns:
            mask |= contains_series(df[c], q)

    return df[mask].copy()


def write_single_report(
    out_md: Path,
    query: str,
    evidence: pd.DataFrame,
    master: pd.DataFrame,
    all_routes: pd.DataFrame,
    manual_routes: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# Structure Synthesis Information Query")
    lines.append("")
    lines.append(f"- Query: `{query}`")
    lines.append(f"- Matched evidence rows: `{len(evidence)}`")
    lines.append("")

    if len(evidence) == 0:
        lines.append("No matched structure-route evidence was found.")
        out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    case_ids = sorted(evidence["case_id"].dropna().astype(str).unique().tolist())

    for case_id in case_ids:
        sub = evidence[evidence["case_id"].astype(str) == case_id].copy()

        lines.append(f"## {case_id}")
        lines.append("")

        first = sub.iloc[0]

        for label, col in [
            ("Input POSCAR", "input_poscar"),
            ("Target formula", "target_formula"),
            ("Target elements", "target_elements"),
            ("Target natoms", "target_natoms"),
            ("Case final status", "final_case_status"),
            ("Case refined status", "refined_case_status"),
            ("Problem type", "problem_type"),
            ("Recommended action", "recommended_action"),
        ]:
            if col in sub.columns:
                val = first.get(col, "")
                lines.append(f"- {label}: `{val}`")

        lines.append("")

        display_cols = pick_existing(sub, [
            "route_rank_within_case",
            "precursor_set",
            "route_refined_status",
            "route_refined_reason",
            "final_recommendation_score",
            "stage35_v43_safe_strict_score",
            "real_stage3_condition_reference_support_score",
            "condition_distribution_support_score",
            "final_recommendation_status",
            "needs_manual_review",
            "evidence_summary",
        ])

        lines.append("### Recommended synthesis routes")
        lines.append("")
        lines.append(safe_markdown(sub[display_cols], index=False))
        lines.append("")

        if "route_generation_method" in sub.columns:
            method = str(first.get("route_generation_method", ""))
            if method and method != "nan":
                lines.append("### How the route was obtained")
                lines.append("")
                lines.append(method)
                lines.append("")

        if "score_interpretation" in sub.columns:
            score_note = str(first.get("score_interpretation", ""))
            if score_note and score_note != "nan":
                lines.append("### Score interpretation")
                lines.append("")
                lines.append(score_note)
                lines.append("")

        # Add all top routes if final evidence table only includes high/medium subset.
        if len(all_routes) > 0 and "case_id" in all_routes.columns:
            rsub = all_routes[all_routes["case_id"].astype(str) == case_id].copy()
            if len(rsub) > 0:
                lines.append("### All exported top routes for this structure")
                lines.append("")
                cols = pick_existing(rsub, [
                    "precursor_set",
                    "route_refined_status",
                    "route_refined_reason",
                    "review_severity",
                    "final_recommendation_score",
                    "stage35_v43_safe_strict_score",
                    "real_stage3_condition_reference_support_score",
                    "condition_distribution_support_score",
                    "final_recommendation_status",
                ])
                lines.append(safe_markdown(rsub[cols].head(20), index=False))
                lines.append("")

        if len(manual_routes) > 0 and "case_id" in manual_routes.columns:
            msub = manual_routes[manual_routes["case_id"].astype(str) == case_id].copy()
            if len(msub) > 0:
                lines.append("### Manual-review-related routes")
                lines.append("")
                cols = pick_existing(msub, [
                    "precursor_set",
                    "route_refined_status",
                    "route_refined_reason",
                    "review_severity",
                    "final_recommendation_score",
                    "stage35_v43_safe_strict_score",
                    "real_stage3_condition_reference_support_score",
                    "condition_distribution_support_score",
                ])
                lines.append(safe_markdown(msub[cols], index=False))
                lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default="/Users/wyc/SynPred")
    ap.add_argument("--batch_name", default="batch_500")
    ap.add_argument("--query", required=True, help="case_id, target_formula, element string, POSCAR path keyword, or precursor keyword")
    ap.add_argument("--top_n", type=int, default=20)
    ap.add_argument("--output_dir", default="")
    args = ap.parse_args()

    project_root = Path(args.project_root)
    batch_dir = project_root / "outputs" / "batch_adaptive" / args.batch_name

    evidence_csv = batch_dir / "final_delivery" / "final_structure_route_evidence_table.csv"
    master_csv = batch_dir / "master_status_standard.csv"
    all_routes_csv = batch_dir / "batch_recommended_routes_topn_review_refined.csv"
    manual_routes_csv = batch_dir / "manual_review" / "manual_review_routes.csv"

    evidence = read_csv_required(evidence_csv)
    master = read_csv_optional(master_csv)
    all_routes = read_csv_optional(all_routes_csv)
    manual_routes = read_csv_optional(manual_routes_csv)

    matched = filter_by_query(evidence, args.query)

    # If no hit in final high/medium evidence table, try all routes.
    if len(matched) == 0 and len(all_routes) > 0:
        all_matched = filter_by_query(all_routes, args.query)
        if len(all_matched) > 0:
            # Attach case-level info if possible.
            if len(master) > 0 and "case_id" in master.columns and "case_id" in all_matched.columns:
                case_cols = pick_existing(master, [
                    "case_id",
                    "input_poscar",
                    "final_case_status",
                    "refined_case_status",
                    "refined_case_reason",
                    "problem_type",
                    "recommended_action",
                ])
                all_matched = all_matched.merge(master[case_cols], on="case_id", how="left", suffixes=("", "_case"))

            matched = all_matched.copy()

    # If still no route-level hit, fall back to case-level master table.
    # This is important for unsupported targets, e.g. O-only or H/O-only POSCARs.
    if len(matched) == 0 and len(master) > 0:
        case_matched = filter_by_query(master, args.query)
        if len(case_matched) > 0:
            case_matched = case_matched.copy()
            case_matched["route_rank_within_case"] = pd.NA
            case_matched["precursor_set"] = ""
            case_matched["route_refined_status"] = "no_route_generated"
            case_matched["route_refined_reason"] = case_matched.get(
                "refined_case_reason",
                "case_has_no_route_level_output",
            )
            case_matched["final_recommendation_score"] = pd.NA
            case_matched["stage35_v43_safe_strict_score"] = pd.NA
            case_matched["real_stage3_condition_reference_support_score"] = pd.NA
            case_matched["condition_distribution_support_score"] = pd.NA
            case_matched["needs_manual_review"] = case_matched.get(
                "final_case_status",
                "",
            ).astype(str).isin(["pipeline_failed", "needs_manual_or_rule_recovery"])

            case_matched["route_generation_method"] = (
                "No route was generated for this structure. "
                "The case-level status should be inspected to determine whether the target is unsupported, failed, or skipped."
            )
            case_matched["score_interpretation"] = (
                "No route-level score is available because no valid route-level recommendation was generated."
            )
            case_matched["evidence_summary"] = (
                "case_status="
                + case_matched.get("final_case_status", "").astype(str)
                + "; problem_type="
                + case_matched.get("problem_type", "").astype(str)
                + "; action="
                + case_matched.get("recommended_action", "").astype(str)
            )
            matched = case_matched.copy()

    # Sort.
    score_col = None
    for c in [
        "route_rank_within_case",
        "final_recommendation_score",
        "stage35_v43_safe_strict_score",
    ]:
        if c in matched.columns:
            score_col = c
            break

    if score_col == "route_rank_within_case":
        matched = matched.sort_values(["case_id", score_col], ascending=[True, True])
    elif score_col:
        matched[score_col] = pd.to_numeric(matched[score_col], errors="coerce")
        matched = matched.sort_values(["case_id", score_col], ascending=[True, False])

    matched = matched.head(args.top_n).copy()

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = batch_dir / "structure_query"

    out_dir.mkdir(parents=True, exist_ok=True)

    safe_q = "".join(ch if ch.isalnum() or ch in ["_", "-", "."] else "_" for ch in str(args.query))[:120]
    out_csv = out_dir / f"structure_synthesis_info__{safe_q}.csv"
    out_md = out_dir / f"structure_synthesis_info__{safe_q}.md"

    matched.to_csv(out_csv, index=False)

    write_single_report(
        out_md=out_md,
        query=args.query,
        evidence=matched,
        master=master,
        all_routes=all_routes,
        manual_routes=manual_routes,
    )

    print("[SAVE]", out_csv)
    print("[SAVE]", out_md)
    print(f"[INFO] n_matched_rows = {len(matched)}")

    if len(matched) > 0:
        cols = pick_existing(matched, [
            "case_id",
            "target_formula",
            "target_elements",
            "route_rank_within_case",
            "precursor_set",
            "route_refined_status",
            "final_recommendation_score",
            "stage35_v43_safe_strict_score",
            "real_stage3_condition_reference_support_score",
            "condition_distribution_support_score",
            "needs_manual_review",
        ])
        print()
        print(matched[cols].to_string(index=False))


if __name__ == "__main__":
    main()
