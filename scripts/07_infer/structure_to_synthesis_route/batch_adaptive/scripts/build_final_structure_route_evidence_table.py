#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
from pathlib import Path
import pandas as pd


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def pick_existing(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def parse_poscar_formula(poscar_path: str | Path) -> dict:
    """Parse target formula/elements/natoms from a VASP5 POSCAR.

    Expected VASP5 format:
      line 6: element symbols
      line 7: element counts
    """
    out = {
        "target_formula": "",
        "target_elements": "",
        "target_natoms": None,
    }

    if poscar_path is None or str(poscar_path).strip() in ["", "nan", "None"]:
        return out

    path = Path(str(poscar_path))
    if not path.exists():
        return out

    try:
        lines = path.read_text(errors="ignore").splitlines()
    except Exception:
        return out

    if len(lines) < 7:
        return out

    elems = lines[5].strip().split()
    counts_raw = lines[6].strip().split()

    if not elems or not all(re.fullmatch(r"[A-Z][a-z]?", x) for x in elems):
        return out

    try:
        counts = [int(float(x)) for x in counts_raw[:len(elems)]]
    except Exception:
        return out

    if len(counts) != len(elems):
        return out

    items = [(el, n) for el, n in zip(elems, counts) if n > 0]
    if not items:
        return out

    out["target_formula"] = "".join(f"{el}{n}" for el, n in items)
    out["target_elements"] = ";".join(el for el, _ in items)
    out["target_natoms"] = int(sum(n for _, n in items))
    return out


def add_target_formula_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "input_poscar" not in df.columns:
        df["target_formula"] = ""
        df["target_elements"] = ""
        df["target_natoms"] = None
        return df

    parsed = df["input_poscar"].apply(parse_poscar_formula)
    parsed_df = pd.DataFrame(parsed.tolist(), index=df.index)

    df["target_formula"] = parsed_df["target_formula"]
    df["target_elements"] = parsed_df["target_elements"]
    df["target_natoms"] = parsed_df["target_natoms"]

    return df


def infer_route_source(row: pd.Series) -> str:
    vals = []
    for c in [
        "decode_method",
        "route_source",
        "stage2_source",
        "source",
        "final_recommended_routes_source",
    ]:
        if c in row.index and pd.notna(row.get(c)) and str(row.get(c)).strip():
            vals.append(f"{c}={row.get(c)}")

    if vals:
        return "; ".join(vals)

    return "stage2_gflownet_plus_fallback_retrieval_baseline_element_rerank"


def make_evidence_summary(row: pd.Series) -> str:
    parts = []

    if "route_refined_status" in row.index and pd.notna(row.get("route_refined_status")):
        parts.append(f"route_status={row.get('route_refined_status')}")

    if "final_recommendation_score" in row.index and pd.notna(row.get("final_recommendation_score")):
        parts.append(f"final_score={float(row.get('final_recommendation_score')):.4f}")

    if "stage35_v43_safe_strict_score" in row.index and pd.notna(row.get("stage35_v43_safe_strict_score")):
        parts.append(f"v43_safe_strict={float(row.get('stage35_v43_safe_strict_score')):.4f}")

    if "real_stage3_condition_reference_support_score" in row.index and pd.notna(row.get("real_stage3_condition_reference_support_score")):
        parts.append(f"condition_ref_support={float(row.get('real_stage3_condition_reference_support_score')):.4f}")

    if "condition_distribution_support_score" in row.index and pd.notna(row.get("condition_distribution_support_score")):
        parts.append(f"condition_dist_support={float(row.get('condition_distribution_support_score')):.4f}")

    if "route_refined_reason" in row.index and pd.notna(row.get("route_refined_reason")):
        reason = str(row.get("route_refined_reason"))
        if reason and reason != "nan":
            parts.append(f"reason={reason}")

    return "; ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default="/Users/wyc/SynPred")
    ap.add_argument("--batch_name", default="batch_500")
    ap.add_argument("--top_n_per_case", type=int, default=3)
    ap.add_argument("--use_high_medium_only", action="store_true")
    args = ap.parse_args()

    project_root = Path(args.project_root)
    batch_dir = project_root / "outputs" / "batch_adaptive" / args.batch_name

    master_csv = batch_dir / "master_status_standard.csv"
    routes_all_csv = batch_dir / "batch_recommended_routes_topn_review_refined.csv"
    routes_hm_csv = batch_dir / "route_splits" / "route_recommended_high_medium.csv"

    master = read_csv_if_exists(master_csv)

    if args.use_high_medium_only and routes_hm_csv.exists():
        routes = pd.read_csv(routes_hm_csv)
        route_input_source = str(routes_hm_csv)
    else:
        routes = read_csv_if_exists(routes_all_csv)
        route_input_source = str(routes_all_csv)

    if len(master) == 0:
        raise FileNotFoundError(f"Missing or empty master status: {master_csv}")
    if len(routes) == 0:
        raise FileNotFoundError(f"Missing or empty routes table: {route_input_source}")

    case_cols = pick_existing(master, [
        "case_id",
        "input_poscar",
        "final_case_status",
        "refined_case_status",
        "refined_case_reason",
        "problem_type",
        "recommended_action",
        "top1_precursor_set",
        "top1_final_score",
        "condition_support_score",
        "top1_condition_support_score",
    ])

    case_info = master[case_cols].copy()
    merged = routes.merge(case_info, on="case_id", how="left", suffixes=("", "_case"))

    if "input_poscar" not in merged.columns and "input_poscar_case" in merged.columns:
        merged["input_poscar"] = merged["input_poscar_case"]

    score_col = None
    for c in [
        "final_recommendation_score",
        "stage35_v43_safe_strict_score",
        "real_stage3_condition_reference_support_score",
        "condition_distribution_support_score",
    ]:
        if c in merged.columns:
            score_col = c
            merged[c] = pd.to_numeric(merged[c], errors="coerce")
            break

    if score_col:
        merged = merged.sort_values(["case_id", score_col], ascending=[True, False])
    else:
        merged = merged.sort_values(["case_id"])

    merged["route_rank_within_case"] = merged.groupby("case_id").cumcount() + 1
    merged = merged[merged["route_rank_within_case"] <= args.top_n_per_case].copy()

    merged = add_target_formula_columns(merged)

    merged["route_source"] = merged.apply(infer_route_source, axis=1)

    merged["route_generation_method"] = (
        "Stage2 GFlowNet precursor candidate generation; "
        "composition-constrained decoding; "
        "fallback/retrieval/baseline candidate merging; "
        "element-aware reranking; "
        "Stage3 condition prediction; "
        "Stage35/V43 safe-strict route reranking; "
        "final recommendation finalizer."
    )

    merged["score_interpretation"] = (
        "Scores are internal ranking and consistency scores, not experimental validation. "
        "Higher final_recommendation_score indicates stronger combined route ranking, "
        "while Stage3 support scores indicate better agreement with learned condition-reference distributions."
    )

    merged["evidence_summary"] = merged.apply(make_evidence_summary, axis=1)

    if "route_refined_status" in merged.columns:
        merged["needs_manual_review"] = merged["route_refined_status"].astype(str).isin([
            "route_needs_manual_check",
            "needs_manual_check",
        ])
    else:
        merged["needs_manual_review"] = False

    output_cols = pick_existing(merged, [
        "case_id",
        "input_poscar",
        "target_formula",
        "target_elements",
        "target_natoms",
        "route_rank_within_case",
        "precursor_set",
        "route_refined_status",
        "route_refined_reason",
        "final_recommendation_score",
        "stage35_v43_safe_strict_score",
        "real_stage3_condition_reference_support_score",
        "condition_distribution_support_score",
        "final_recommendation_status",
        "final_case_status",
        "refined_case_status",
        "refined_case_reason",
        "problem_type",
        "recommended_action",
        "route_source",
        "route_generation_method",
        "score_interpretation",
        "evidence_summary",
        "needs_manual_review",
    ])

    out = merged[output_cols].copy()

    out_dir = batch_dir / "final_delivery"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "final_structure_route_evidence_table.csv"
    out_md = out_dir / "final_structure_route_evidence_table.md"
    summary_txt = out_dir / "final_structure_route_evidence_summary.txt"

    out.to_csv(out_csv, index=False)

    lines = []
    lines.append("# Final Structure–Route Evidence Table")
    lines.append("")
    lines.append(f"- Batch name: `{args.batch_name}`")
    lines.append(f"- Route input source: `{route_input_source}`")
    lines.append(f"- Top N per case: `{args.top_n_per_case}`")
    lines.append(f"- Use high/medium routes only: `{args.use_high_medium_only}`")
    lines.append(f"- Number of exported rows: `{len(out)}`")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Each row links one input structure to one predicted synthesis route. "
        "The target formula/elements are parsed directly from the original input POSCAR. "
        "The route is generated by Stage2 precursor-set generation and then reranked using "
        "composition constraints, fallback/retrieval/baseline candidates, Stage3 condition support, "
        "and Stage35/V43 safe-strict route scoring. Scores are internal ranking scores and should not be interpreted as experimental validation."
    )
    lines.append("")
    lines.append("## Table")
    lines.append("")

    if len(out) == 0:
        lines.append("No records.")
    else:
        display_cols = pick_existing(out, [
            "case_id",
            "target_formula",
            "target_elements",
            "target_natoms",
            "route_rank_within_case",
            "precursor_set",
            "route_refined_status",
            "final_recommendation_score",
            "stage35_v43_safe_strict_score",
            "real_stage3_condition_reference_support_score",
            "condition_distribution_support_score",
            "needs_manual_review",
        ])
        lines.append(out[display_cols].to_markdown(index=False))

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"batch_name={args.batch_name}\n")
        f.write(f"route_input_source={route_input_source}\n")
        f.write(f"top_n_per_case={args.top_n_per_case}\n")
        f.write(f"use_high_medium_only={args.use_high_medium_only}\n")
        f.write(f"n_rows={len(out)}\n")
        f.write(f"n_missing_target_formula={(out['target_formula'].astype(str).str.len() == 0).sum() if 'target_formula' in out.columns else 'NA'}\n")
        if "route_refined_status" in out.columns:
            f.write("\nroute_refined_status_counts:\n")
            f.write(str(out["route_refined_status"].value_counts(dropna=False)))
            f.write("\n")
        if "needs_manual_review" in out.columns:
            f.write("\nneeds_manual_review_counts:\n")
            f.write(str(out["needs_manual_review"].value_counts(dropna=False)))
            f.write("\n")

    print("[SAVE]", out_csv)
    print("[SAVE]", out_md)
    print("[SAVE]", summary_txt)
    print(f"[INFO] n_rows = {len(out)}")
    if "target_formula" in out.columns:
        n_missing = (out["target_formula"].astype(str).str.len() == 0).sum()
        print(f"[INFO] n_missing_target_formula = {n_missing}")


if __name__ == "__main__":
    main()
