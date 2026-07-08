#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd


DEFAULT_ROOT = Path("/Users/wyc/SynPred/outputs/inference")
DEFAULT_ROUTE_DIR = "routes_flow_fallback_retrieval_baseline_element_reranked"


def find_first_existing_csv(route_path: Path, candidate_csvs: list[str]) -> Path | None:
    """
    Prefer the newest v4.3 route-template-aware output, while keeping
    backward compatibility with older pipeline outputs.

    Important:
    - Do NOT blindly prefer final_recommended_routes.csv, because some older
      final_recommended_routes.csv files may still point to v3_learned outputs.
    - Prefer explicit v43 output when it exists.
    """
    for fname in candidate_csvs:
        p = route_path / fname
        if p.exists():
            return p
    return None


def sort_by_best_available_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure the first row is really the top-ranked route.

    Preferred rank:
      stage35_v43_template_chemonly_rank

    Backward-compatible fallback ranks:
      v3_learned_rerank_rank / v3_learned_rank / v3_joint_rerank_rank /
      v3_joint_rank / final_route_rank / stage35_v21_rank / best_route_rank
    """
    rank_priority = [
        "stage35_v43_template_chemonly_rank",
        "v3_learned_rerank_rank",
        "v3_learned_rank",
        "v3_joint_rerank_rank",
        "v3_joint_rank",
        "final_route_rank",
        "stage35_v21_rank",
        "best_route_rank",
    ]

    out = df.copy()

    for rc in rank_priority:
        if rc in out.columns:
            out[rc] = pd.to_numeric(out[rc], errors="coerce")
            out = out.sort_values(rc, ascending=True, na_position="last")
            return out

    return out


def get_value(row: dict, *names: str, default=""):
    for name in names:
        val = row.get(name, default)
        if pd.notna(val):
            return val
    return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=10)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--route_dir", default=DEFAULT_ROUTE_DIR)
    ap.add_argument(
        "--out_name",
        default=None,
        help="Output csv name. Default: _benchmark_top1_summary_START_END.csv",
    )
    args = ap.parse_args()

    root = Path(args.root)
    route_dir = args.route_dir

    # 核心改动：
    # 先读显式 v43 结果，再读 final_recommended_routes.csv。
    # 这样既不会覆盖旧文件，又能避免旧 final_recommended_routes.csv 误导汇总。
    candidate_csvs = [
        "final_top_routes_v43_template_chemonly_reranked.csv",
        "final_recommended_routes.csv",
        "final_top_routes_v3_learned_reranked.csv",
        "final_top_routes_v3_joint_reranked.csv",
        "final_top_routes_with_confidence.csv",
        "final_top_routes_with_precursor_qc.csv",
        "final_top_routes.csv",
    ]

    rows = []

    for i in range(args.start, args.end + 1):
        infer_name = f"benchmark_{i:03d}"
        route_path = root / infer_name / route_dir
        csv_path = find_first_existing_csv(route_path, candidate_csvs)

        if csv_path is None:
            rows.append({
                "infer_name": infer_name,
                "status": "missing",
                "csv_path": str(route_path / candidate_csvs[0]),
                "route_source_file": "",
            })
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            rows.append({
                "infer_name": infer_name,
                "status": f"read_error: {e}",
                "csv_path": str(csv_path),
                "route_source_file": csv_path.name,
            })
            continue

        if df.empty:
            rows.append({
                "infer_name": infer_name,
                "status": "empty",
                "csv_path": str(csv_path),
                "route_source_file": csv_path.name,
            })
            continue

        df = sort_by_best_available_rank(df)
        top = df.iloc[0].to_dict()

        rows.append({
            "infer_name": infer_name,
            "status": "ok",

            # identity
            "sample_id": get_value(top, "sample_id"),
            "material_id": get_value(top, "material_id"),
            "formula": get_value(top, "formula", "formula_x", "formula_y", "material_id", "sample_id"),

            # route
            "precursor_set": get_value(top, "precursor_set"),
            "temperature_c": get_value(top, "temperature_c"),
            "time_h": get_value(top, "time_h"),

            # legacy / shared scores
            "stage3_score": get_value(top, "stage3_score", "condition_score"),
            "stage35_v21_score": get_value(top, "stage35_v21_score"),
            "v3_joint_feature_score": get_value(top, "v3_joint_feature_score"),
            "v3_learned_ranker_score": get_value(top, "v3_learned_ranker_score"),

            # v4.3 template-aware ranker
            "stage35_v43_template_chemonly_rank": get_value(top, "stage35_v43_template_chemonly_rank"),
            "stage35_v43_template_chemonly_score": get_value(top, "stage35_v43_template_chemonly_score"),
            "stage35_v43_template_chemonly_mean_prob": get_value(top, "stage35_v43_template_chemonly_mean_prob"),
            "stage35_v43_template_chemonly_win_rate": get_value(top, "stage35_v43_template_chemonly_win_rate"),
            "stage35_v43_template_chemonly_wins": get_value(top, "stage35_v43_template_chemonly_wins"),
            "stage35_v43_template_chemonly_losses": get_value(top, "stage35_v43_template_chemonly_losses"),

            # v4.3 route-template features
            "route_template_primary": get_value(top, "route_template_primary"),
            "route_template_secondary": get_value(top, "route_template_secondary"),
            "route_template_type_signature": get_value(top, "route_template_type_signature"),
            "route_template_confidence": get_value(top, "route_template_confidence"),
            "route_template_matches_target_anion": get_value(top, "route_template_matches_target_anion"),
            "route_template_is_common_solid_state": get_value(top, "route_template_is_common_solid_state"),
            "route_template_is_overly_elemental": get_value(top, "route_template_is_overly_elemental"),
            "route_template_elemental_ratio": get_value(top, "route_template_elemental_ratio"),

            # useful route-template flags
            "route_has_oxide_template": get_value(top, "route_has_oxide_template"),
            "route_has_nitrate_template": get_value(top, "route_has_nitrate_template"),
            "route_has_carbonate_template": get_value(top, "route_has_carbonate_template"),
            "route_has_phosphate_template": get_value(top, "route_has_phosphate_template"),
            "route_has_sulfate_template": get_value(top, "route_has_sulfate_template"),
            "route_has_selenide_template": get_value(top, "route_has_selenide_template"),
            "route_has_selenite_selenate_template": get_value(top, "route_has_selenite_selenate_template"),
            "route_has_sulfide_template": get_value(top, "route_has_sulfide_template"),
            "route_has_halide_template": get_value(top, "route_has_halide_template"),
            "route_has_elemental_template": get_value(top, "route_has_elemental_template"),
            "route_has_organic_template": get_value(top, "route_has_organic_template"),
            "route_has_hydrate_template": get_value(top, "route_has_hydrate_template"),
            "route_has_ammonium_template": get_value(top, "route_has_ammonium_template"),

            # reliability / QC
            "confidence_level": get_value(top, "confidence_level", "route_confidence_level"),
            "recommendation_status": get_value(top, "recommendation_status", "route_recommendation_status"),
            "warning_level": get_value(top, "warning_level", "route_warning_level"),
            "precursor_qc_level": get_value(top, "precursor_qc_level"),
            "precursor_qc_status": get_value(top, "precursor_qc_status"),
            "element_coverage": get_value(top, "element_coverage", "element_coverage_recomputed"),
            "element_hit": get_value(top, "element_hit", "element_hit_recomputed"),
            "element_missing": get_value(top, "element_missing", "element_missing_recomputed"),

            # provenance
            "route_source_file": csv_path.name,
            "csv_path": str(csv_path),
        })

    out = pd.DataFrame(rows)

    if args.out_name is None:
        out_name = f"_benchmark_top1_summary_{args.start:03d}_{args.end:03d}.csv"
    else:
        out_name = args.out_name

    out_path = root / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    pd.set_option("display.max_columns", 240)
    pd.set_option("display.width", 260)
    print(out.to_string(index=False))
    print()
    print(f"[SAVE] {out_path}")


if __name__ == "__main__":
    main()
