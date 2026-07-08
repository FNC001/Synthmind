#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd


def export_final_top_routes(r):
    """
    Export final route table preserving the strongest available final reranking order.

    Priority:
    1. stage35_v21_csv
    2. stage35_learned_csv
    3. stage35_rule_csv
    4. display_csv
    """

    r.log("===== FINAL: export final top routes =====")

    if "stage35_v21_csv" in r.outputs:
        src = Path(r.outputs["stage35_v21_csv"])
        rank_col = "stage35_v21_rank"
        score_col = "stage35_v21_score"
        source = "stage35_v21"
    elif "stage35_learned_csv" in r.outputs:
        src = Path(r.outputs["stage35_learned_csv"])
        rank_col = "stage35_learned_rank"
        score_col = "stage35_learned_score"
        source = "stage35_learned"
    elif "stage35_rule_csv" in r.outputs:
        src = Path(r.outputs["stage35_rule_csv"])
        rank_col = "stage35_rule_rank"
        score_col = "stage35_rule_score"
        source = "stage35_rule"
    else:
        src = Path(r.outputs["display_csv"])
        rank_col = "rank"
        score_col = "stage3_score"
        source = "display_filtered"

    if not src.exists():
        raise FileNotFoundError(f"Missing final route source: {src}")

    df = pd.read_csv(src)

    if rank_col in df.columns:
        if "sample_id" in df.columns and df["sample_id"].nunique() > 1:
            df = df.sort_values(["sample_id", rank_col], ascending=[True, True]).reset_index(drop=True)
        else:
            df = df.sort_values(rank_col, ascending=True).reset_index(drop=True)
    elif score_col in df.columns:
        if "sample_id" in df.columns and df["sample_id"].nunique() > 1:
            df = df.sort_values(["sample_id", score_col], ascending=[True, False]).reset_index(drop=True)
        else:
            df = df.sort_values(score_col, ascending=False).reset_index(drop=True)

    if "sample_id" in df.columns and df["sample_id"].nunique() > 1:
        df["final_route_rank"] = df.groupby("sample_id").cumcount() + 1
    else:
        df["final_route_rank"] = range(1, len(df) + 1)
    df["final_route_source"] = source

    out_dir = Path(r.outputs.get("route_out_dir", r.out_dir / "final_routes"))
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "final_top_routes.csv"
    out_md = out_dir / "final_top_routes.md"
    summary_json = out_dir / "final_top_routes_summary.json"

    preferred_cols = [
        "final_route_rank",
        "final_route_source",
        rank_col,
        score_col,
        "stage35_v2_prob",
        "precursor_rank",
        "precursor_set",
        "temperature_c",
        "time_h",
        "condition_source",
        "stage3_score",
        "element_coverage",
        "element_hit",
        "element_missing",
        "missing_count",
        "extra_element_penalty",
    ]

    keep = [c for c in preferred_cols if c in df.columns]
    display = df[keep].copy() if keep else df.copy()

    df.to_csv(out_csv, index=False)
    display.head(50).to_markdown(out_md, index=False)

    summary = {
        "source": source,
        "input_csv": str(src),
        "output_csv": str(out_csv),
        "output_md": str(out_md),
        "n_rows": int(len(df)),
        "rank_col": rank_col,
        "score_col": score_col,
    }

    import json
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    r.outputs["final_top_routes_csv"] = str(out_csv)
    r.outputs["final_top_routes_md"] = str(out_md)
    r.outputs["final_top_routes_summary_json"] = str(summary_json)

    r.log(f"[SAVE] {out_csv}")
    r.log(f"[SAVE] {out_md}")
    r.log(f"[SAVE] {summary_json}")

    if len(display):
        print(display.head(10).to_string(index=False))
