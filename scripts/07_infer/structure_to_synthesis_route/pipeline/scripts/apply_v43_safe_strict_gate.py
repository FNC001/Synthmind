#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


ELEMENT_RE = re.compile(r"([A-Z][a-z]?)")

# H/O/N/C usually come from water, hydroxide, nitrate, carbonate, ammonium, organics, etc.
# They should not be treated as strict foreign functional elements here.
DEFAULT_ALLOWED_EXTRA = {"H", "O", "N", "C"}


def parse_elements(text: object) -> set[str]:
    if text is None:
        return set()
    if pd.isna(text):
        return set()
    s = str(text).strip()
    if not s:
        return set()
    return set(ELEMENT_RE.findall(s))


def get_first(row: pd.Series, names: list[str], default=""):
    for name in names:
        if name in row.index:
            v = row[name]
            if pd.notna(v):
                return v
    return default


def infer_target_elements(row: pd.Series) -> set[str]:
    """
    Prefer explicit target columns when available; otherwise infer from formula/material_id/sample_id.
    """
    for col in [
        "target_core_elements",
        "target_elements_qc",
        "element_hit",
        "element_hit_recomputed",
    ]:
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            raw = str(row[col]).replace(",", ";")
            parts = [x.strip() for x in raw.split(";") if x.strip()]
            if parts:
                return set(parts) - {"H", "O"}

    formula_text = str(
        get_first(row, ["formula", "formula_x", "formula_y", "material_id", "sample_id"], "")
    )
    return parse_elements(formula_text) - {"H", "O"}


def safe_float(x, default=0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def add_safe_strict_scores(
    df: pd.DataFrame,
    allowed_extra: set[str],
    score_col: str,
    penalty_per_extra: float,
    hard_review: bool,
) -> pd.DataFrame:
    out = df.copy()

    strict_extra_list = []
    n_strict_extra_list = []
    missing_core_list = []
    n_missing_core_list = []

    safe_score_list = []
    safe_bucket_list = []
    safe_reason_list = []

    for _, row in out.iterrows():
        target_core = infer_target_elements(row)
        precursor_set = get_first(row, ["precursor_set"], "")
        precursor_elements = parse_elements(precursor_set)

        missing_core = sorted(target_core - precursor_elements)
        extra_all = sorted(precursor_elements - target_core)
        strict_extra = sorted([e for e in extra_all if e not in allowed_extra])

        n_extra = len(strict_extra)
        n_missing = len(missing_core)

        base_score = safe_float(get_first(row, [score_col], 0.0), 0.0)

        penalty = penalty_per_extra * n_extra
        if n_missing > 0:
            penalty += 0.50 * n_missing

        safe_score = base_score - penalty

        reasons = []
        if n_extra > 0:
            reasons.append(f"strict_extra_elements={';'.join(strict_extra)}")
        if n_missing > 0:
            reasons.append(f"missing_core_elements={';'.join(missing_core)}")

        old_qc = str(get_first(row, ["qc_level_auto", "precursor_qc_level"], "")).strip()
        if old_qc and old_qc not in {"nan", "None", "pass"}:
            reasons.append(f"upstream_qc={old_qc}")

        if n_missing > 0:
            bucket = "review_required_missing_core"
        elif n_extra > 0:
            bucket = "review_required_strict_extra"
        else:
            bucket = "safe_candidate"

        strict_extra_list.append(";".join(strict_extra))
        n_strict_extra_list.append(n_extra)
        missing_core_list.append(";".join(missing_core))
        n_missing_core_list.append(n_missing)
        safe_score_list.append(safe_score)
        safe_bucket_list.append(bucket)
        safe_reason_list.append(" | ".join(reasons))

    out["v43_safe_strict_extra_elements"] = strict_extra_list
    out["v43_safe_n_strict_extra"] = n_strict_extra_list
    out["v43_safe_missing_core_elements"] = missing_core_list
    out["v43_safe_n_missing_core"] = n_missing_core_list
    out["stage35_v43_safe_strict_score"] = safe_score_list
    out["v43_safe_strict_bucket"] = safe_bucket_list
    out["v43_safe_strict_reason"] = safe_reason_list

    if hard_review:
        # Keep all routes, but rank safe candidates first.
        out["v43_safe_is_review_required"] = (
            (out["v43_safe_n_strict_extra"] > 0) | (out["v43_safe_n_missing_core"] > 0)
        ).astype(int)
    else:
        out["v43_safe_is_review_required"] = 0

    # Primary ordering:
    # 1. safe routes first
    # 2. higher safe strict score
    # 3. original v43 rank if available
    sort_cols = ["v43_safe_is_review_required", "stage35_v43_safe_strict_score"]
    ascending = [True, False]

    if "stage35_v43_template_chemonly_rank" in out.columns:
        out["stage35_v43_template_chemonly_rank"] = pd.to_numeric(
            out["stage35_v43_template_chemonly_rank"], errors="coerce"
        )
        sort_cols.append("stage35_v43_template_chemonly_rank")
        ascending.append(True)

    if "sample_id" in out.columns and out["sample_id"].nunique() > 1:
        sort_cols = ["sample_id"] + sort_cols
        ascending = [True] + ascending

    out = out.sort_values(sort_cols, ascending=ascending, na_position="last").reset_index(drop=True)
    if "sample_id" in out.columns and out["sample_id"].nunique() > 1:
        out["stage35_v43_safe_strict_rank"] = out.groupby("sample_id").cumcount() + 1
    else:
        out["stage35_v43_safe_strict_rank"] = range(1, len(out) + 1)

    # Put rank near the front when possible.
    front_cols = [
        "stage35_v43_safe_strict_rank",
        "stage35_v43_safe_strict_score",
        "v43_safe_strict_bucket",
        "v43_safe_strict_reason",
        "v43_safe_strict_extra_elements",
        "v43_safe_missing_core_elements",
    ]
    ordered = [c for c in front_cols if c in out.columns] + [
        c for c in out.columns if c not in front_cols
    ]
    return out[ordered]


def write_markdown(df: pd.DataFrame, output_md: Path, top_n: int) -> None:
    cols = [
        "stage35_v43_safe_strict_rank",
        "stage35_v43_safe_strict_score",
        "v43_safe_strict_bucket",
        "v43_safe_strict_reason",
        "stage35_v43_template_chemonly_score",
        "stage35_v43_template_chemonly_win_rate",
        "precursor_set",
        "temperature_c",
        "time_h",
        "route_template_primary",
        "route_template_type_signature",
    ]
    cols = [c for c in cols if c in df.columns]

    view = df[cols].head(top_n).copy()

    for c in [
        "stage35_v43_safe_strict_score",
        "stage35_v43_template_chemonly_score",
        "stage35_v43_template_chemonly_win_rate",
        "temperature_c",
        "time_h",
    ]:
        if c in view.columns:
            view[c] = pd.to_numeric(view[c], errors="coerce").round(4)

    lines = []
    lines.append("# V4.3 Safe-Strict Reranked Routes")
    lines.append("")
    lines.append(
        "This table applies a strict extra-element gate after the v4.3 template-aware ranker. "
        "Routes introducing target-external functional elements are demoted or marked for review."
    )
    lines.append("")
    lines.append(view.to_markdown(index=False))
    lines.append("")

    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--output_md", required=True)
    ap.add_argument("--summary_json", required=True)
    ap.add_argument("--top_n", type=int, default=30)
    ap.add_argument(
        "--score_col",
        default="stage35_v43_template_chemonly_score",
        help="Base score from v43 template-aware ranker.",
    )
    ap.add_argument("--penalty_per_extra", type=float, default=0.50)
    ap.add_argument(
        "--allowed_extra",
        default="H,O,N,C",
        help="Comma-separated elements allowed as non-strict extras.",
    )
    ap.add_argument(
        "--no_hard_review",
        action="store_true",
        help="Only score-penalize strict extras; do not force them behind safe routes.",
    )
    args = ap.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    summary_json = Path(args.summary_json)

    if not input_csv.exists():
        raise FileNotFoundError(input_csv)

    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError(f"Empty input_csv: {input_csv}")

    allowed_extra = {x.strip() for x in args.allowed_extra.split(",") if x.strip()}
    if not allowed_extra:
        allowed_extra = set(DEFAULT_ALLOWED_EXTRA)

    if args.score_col not in df.columns:
        # fallback to a reasonable score if v43 score is absent
        fallback_cols = [
            "v3_learned_ranker_score",
            "v3_joint_feature_score",
            "stage35_v21_score",
            "stage3_score",
        ]
        found = next((c for c in fallback_cols if c in df.columns), None)
        if found is None:
            raise ValueError(
                f"Score column {args.score_col!r} not found and no fallback score exists."
            )
        args.score_col = found

    out = add_safe_strict_scores(
        df=df,
        allowed_extra=allowed_extra,
        score_col=args.score_col,
        penalty_per_extra=args.penalty_per_extra,
        hard_review=not args.no_hard_review,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(output_csv, index=False)
    write_markdown(out, output_md, args.top_n)

    top = out.iloc[0].to_dict()

    summary = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "output_md": str(output_md),
        "rows_input": int(len(df)),
        "rows_output": int(len(out)),
        "score_col_used": args.score_col,
        "allowed_extra": sorted(allowed_extra),
        "penalty_per_extra": args.penalty_per_extra,
        "hard_review": not args.no_hard_review,
        "bucket_counts": out["v43_safe_strict_bucket"].value_counts(dropna=False).to_dict(),
        "n_routes_with_strict_extra": int((out["v43_safe_n_strict_extra"] > 0).sum()),
        "n_routes_with_missing_core": int((out["v43_safe_n_missing_core"] > 0).sum()),
        "top1_precursor_set": top.get("precursor_set", ""),
        "top1_safe_strict_score": top.get("stage35_v43_safe_strict_score", ""),
        "top1_safe_bucket": top.get("v43_safe_strict_bucket", ""),
        "top1_safe_reason": top.get("v43_safe_strict_reason", ""),
        "claim_boundary": "post_v43_rule_based_strict_extra_element_gate_not_experimental_validation",
    }

    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[SAVE] {output_csv}")
    print(f"[SAVE] {output_md}")
    print(f"[SAVE] {summary_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
