#!/usr/bin/env python3
"""Diagnose Stage35 validation/test distribution shift and ranking failure."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


NUMERIC_FEATURES = [
    "precursor_rank",
    "precursor_score",
    "precursor_score_norm",
    "element_coverage",
    "missing_element_count",
    "extra_element_count",
    "candidate_size",
    "contains_open_generated_precursor",
    "contains_repair_precursor",
    "temperature_c",
    "time_h",
    "temperature_point_score",
    "temperature_bin_score",
    "time_point_score",
    "time_bin_score",
    "atmosphere_probability",
    "solvent_probability",
    "condition_prior_score",
    "condition_calibrated_score_v3",
    "condition_rank_calibrated_v3",
    "condition_score",
    "condition_score_norm",
    "precursor_confidence",
    "condition_confidence",
    "precursor_condition_compatibility_score",
    "reaction_method_prior_score",
    "route_total_score_raw",
    "route_rank_raw",
]

CATEGORICAL_FEATURES = [
    "reaction_method",
    "candidate_source",
    "condition_source",
    "chemistry_check_status",
    "atmosphere",
    "solvent",
]

SCORE_CANDIDATES = [
    ("route_total_score_raw", False),
    ("total_score_raw", False),
    ("precursor_score_norm", False),
    ("precursor_score", False),
    ("condition_calibrated_score_v3", False),
    ("condition_score", False),
    ("route_rank_raw", True),
    ("precursor_rank", True),
    ("condition_rank", True),
]


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def binary_label(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.zeros(len(df), dtype=np.float32)
    return pd.to_numeric(df[col], errors="coerce").fillna(0).clip(0, 1).to_numpy(dtype=np.float32)


def rank_metrics(df: pd.DataFrame, score: np.ndarray) -> dict[str, float]:
    work = pd.DataFrame(
        {
            "sample_id": df["sample_id"].astype(str).to_numpy(),
            "score": np.nan_to_num(score.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0),
            "relaxed": binary_label(df, "relaxed_route_hit_if_eval"),
            "usable_relaxed": binary_label(df, "usable_relaxed_route_hit_if_eval"),
            "strict": binary_label(df, "strict_route_hit_if_eval"),
            "precursor_exact": binary_label(df, "precursor_exact_if_eval"),
            "condition_relaxed": binary_label(df, "relaxed_condition_hit_if_eval"),
            "known": binary_label(df, "atmosphere_known_mask") if "atmosphere_known_mask" in df.columns else np.ones(len(df)),
        }
    ).sort_values(["sample_id", "score"], ascending=[True, False])
    groups = work.groupby("sample_id", sort=False)
    n = max(len(groups), 1)
    out: dict[str, float] = {"num_samples": float(n), "num_candidates": float(len(work)), "candidates_per_sample": float(len(work) / n)}
    strict_ids = groups["known"].max()
    strict_ids = strict_ids[strict_ids > 0].index
    for k in (1, 10, 200, 500):
        top = groups.head(k)
        out[f"missing_top{k}_relaxed_route"] = float(top.groupby("sample_id")["relaxed"].max().mean())
        out[f"usable_top{k}_relaxed_route"] = float(top.groupby("sample_id")["usable_relaxed"].max().mean())
        out[f"top{k}_precursor_exact"] = float(top.groupby("sample_id")["precursor_exact"].max().mean())
        out[f"top{k}_condition_relaxed"] = float(top.groupby("sample_id")["condition_relaxed"].max().mean())
        strict_top = top[top["sample_id"].isin(strict_ids)]
        out[f"strict_top{k}_relaxed_route"] = float(strict_top.groupby("sample_id")["strict"].max().reindex(strict_ids, fill_value=0).mean())
    return out


def available_oracle_metrics(df: pd.DataFrame) -> dict[str, float]:
    groups = df.groupby(df["sample_id"].astype(str), sort=False)
    n = max(len(groups), 1)
    out = {
        "num_samples": float(n),
        "num_candidates": float(len(df)),
        "relaxed_route_available": float(groups["relaxed_route_hit_if_eval"].max().mean()) if "relaxed_route_hit_if_eval" in df.columns else 0.0,
        "usable_relaxed_route_available": float(groups["usable_relaxed_route_hit_if_eval"].max().mean()) if "usable_relaxed_route_hit_if_eval" in df.columns else 0.0,
        "strict_route_available": float(groups["strict_route_hit_if_eval"].max().mean()) if "strict_route_hit_if_eval" in df.columns else 0.0,
        "precursor_exact_available": float(groups["precursor_exact_if_eval"].max().mean()) if "precursor_exact_if_eval" in df.columns else 0.0,
        "condition_relaxed_available": float(groups["relaxed_condition_hit_if_eval"].max().mean()) if "relaxed_condition_hit_if_eval" in df.columns else 0.0,
        "condition_strict_available": float(groups["strict_condition_hit_if_eval"].max().mean()) if "strict_condition_hit_if_eval" in df.columns else 0.0,
    }
    if "atmosphere_known_mask" in df.columns:
        out["strict_comparable_fraction"] = float(groups["atmosphere_known_mask"].max().mean())
    return out


def score_from_column(df: pd.DataFrame, col: str, lower_is_better: bool) -> np.ndarray:
    values = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    return -values if lower_is_better else values


def metrics_for_scores(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, lower_is_better in SCORE_CANDIDATES:
        if col not in df.columns:
            continue
        row = {"score_name": ("neg_" + col) if lower_is_better else col}
        row.update(rank_metrics(df, score_from_column(df, col, lower_is_better)))
        rows.append(row)
    return pd.DataFrame(rows)


def per_method_score_metrics(df: pd.DataFrame, score_name: str, score: np.ndarray) -> pd.DataFrame:
    rows = []
    work = df.copy()
    work["_score_for_diag"] = score
    for method, group in work.groupby("reaction_method", dropna=False):
        if group["sample_id"].nunique() < 20:
            continue
        row = {"reaction_method": str(method)}
        row.update(rank_metrics(group, group["_score_for_diag"].to_numpy(dtype=np.float64)))
        row["score_name"] = score_name
        rows.append(row)
    return pd.DataFrame(rows)


def numeric_shift(val: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in NUMERIC_FEATURES:
        if col not in val.columns or col not in test.columns:
            continue
        v = pd.to_numeric(val[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        t = pd.to_numeric(test[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(v) == 0 or len(t) == 0:
            continue
        pooled = float(np.sqrt(0.5 * (v.var(ddof=0) + t.var(ddof=0))))
        mean_delta_std = float((t.mean() - v.mean()) / pooled) if pooled > 1e-12 else 0.0
        rows.append(
            {
                "feature": col,
                "val_mean": float(v.mean()),
                "test_mean": float(t.mean()),
                "mean_delta": float(t.mean() - v.mean()),
                "standardized_mean_delta": mean_delta_std,
                "val_p50": float(v.quantile(0.50)),
                "test_p50": float(t.quantile(0.50)),
                "val_p90": float(v.quantile(0.90)),
                "test_p90": float(t.quantile(0.90)),
                "val_missing_frac": float(pd.to_numeric(val[col], errors="coerce").isna().mean()),
                "test_missing_frac": float(pd.to_numeric(test[col], errors="coerce").isna().mean()),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.reindex(out["standardized_mean_delta"].abs().sort_values(ascending=False).index)
    return out


def categorical_shift(val: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in CATEGORICAL_FEATURES:
        if col not in val.columns or col not in test.columns:
            continue
        v = val[col].fillna("<MISSING>").astype(str).value_counts(normalize=True)
        t = test[col].fillna("<MISSING>").astype(str).value_counts(normalize=True)
        labels = sorted(set(v.index).union(t.index))
        tv = 0.5 * sum(abs(float(v.get(label, 0.0)) - float(t.get(label, 0.0))) for label in labels)
        for label in labels:
            rows.append(
                {
                    "feature": col,
                    "value": label,
                    "val_fraction": float(v.get(label, 0.0)),
                    "test_fraction": float(t.get(label, 0.0)),
                    "delta": float(t.get(label, 0.0) - v.get(label, 0.0)),
                    "total_variation": float(tv),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["total_variation", "feature", "delta"], ascending=[False, True, False])
    return out


def min_hit_rank(df: pd.DataFrame, rank_col: str, hit_col: str) -> pd.Series:
    if rank_col not in df.columns or hit_col not in df.columns:
        return pd.Series(dtype=float)
    hits = df[pd.to_numeric(df[hit_col], errors="coerce").fillna(0) > 0].copy()
    if hits.empty:
        return pd.Series(dtype=float)
    hits["_rank"] = pd.to_numeric(hits[rank_col], errors="coerce")
    return hits.groupby(hits["sample_id"].astype(str))["_rank"].min()


def hit_rank_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for hit_col in ["relaxed_route_hit_if_eval", "strict_route_hit_if_eval", "precursor_exact_if_eval", "relaxed_condition_hit_if_eval"]:
        ranks = min_hit_rank(df, "route_rank_raw", hit_col)
        if ranks.empty:
            continue
        rows.append(
            {
                "hit_type": hit_col,
                "available_samples": int(len(ranks)),
                "min_rank_p50": float(ranks.quantile(0.50)),
                "min_rank_p90": float(ranks.quantile(0.90)),
                "min_rank_p95": float(ranks.quantile(0.95)),
                "min_rank_lte_10": float((ranks <= 10).mean()),
                "min_rank_lte_200": float((ranks <= 200).mean()),
            }
        )
    return pd.DataFrame(rows)


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    return df.head(max_rows).to_markdown(index=False)


def write_report(out_dir: Path, payload: dict[str, Any], tables: dict[str, pd.DataFrame]) -> None:
    lines = [
        "# Stage35 Validation/Test Shift Diagnosis",
        "",
        "## Split Overview",
        "",
        "```json",
        json.dumps(payload["split_overview"], indent=2, sort_keys=True),
        "```",
        "",
        "## Oracle Availability",
        "",
        dataframe_to_markdown(tables["oracle_availability"]),
        "",
        "## Baseline Score Ranking Metrics",
        "",
        dataframe_to_markdown(tables["score_metrics"], 30),
        "",
        "## Top Numeric Shifts",
        "",
        dataframe_to_markdown(tables["numeric_shift"], 20),
        "",
        "## Top Categorical Shifts",
        "",
        dataframe_to_markdown(tables["categorical_shift"], 30),
        "",
        "## Per-Method Route Metrics",
        "",
        dataframe_to_markdown(tables["per_method_metrics"], 40),
        "",
        "## Hit Rank Summary",
        "",
        dataframe_to_markdown(tables["hit_rank_summary"], 20),
        "",
        "## Auto Conclusions",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["conclusions"])
    (out_dir / "STAGE35_VAL_TEST_SHIFT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--val_csv", default="outputs/evaluation/stage35_route_candidates_v3_final_20260612/val_route_candidates.csv")
    parser.add_argument("--test_csv", default="outputs/evaluation/stage35_route_candidates_v3_final_20260612/test_route_candidates.csv")
    parser.add_argument("--output_root", default="outputs/autorun/stage35_val_test_shift_20260614")
    parser.add_argument("--primary_score", default="precursor_score_norm")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_dir = Path(args.output_root)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    val_path = Path(args.val_csv)
    test_path = Path(args.test_csv)
    if not val_path.is_absolute():
        val_path = project_root / val_path
    if not test_path.is_absolute():
        test_path = project_root / test_path

    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)
    split_overview = {
        "val_csv": str(val_path),
        "test_csv": str(test_path),
        "val_rows": int(len(val)),
        "test_rows": int(len(test)),
        "val_samples": int(val["sample_id"].nunique()),
        "test_samples": int(test["sample_id"].nunique()),
        "val_candidates_per_sample": float(len(val) / max(val["sample_id"].nunique(), 1)),
        "test_candidates_per_sample": float(len(test) / max(test["sample_id"].nunique(), 1)),
    }

    oracle_df = pd.DataFrame(
        [
            {"split": "val", **available_oracle_metrics(val)},
            {"split": "test", **available_oracle_metrics(test)},
        ]
    )
    val_scores = metrics_for_scores(val)
    val_scores.insert(0, "split", "val")
    test_scores = metrics_for_scores(test)
    test_scores.insert(0, "split", "test")
    score_metrics = pd.concat([val_scores, test_scores], ignore_index=True)

    numeric = numeric_shift(val, test)
    cat = categorical_shift(val, test)

    if args.primary_score in val.columns:
        val_primary = score_from_column(val, args.primary_score, False)
        test_primary = score_from_column(test, args.primary_score, False)
        val_method = per_method_score_metrics(val, args.primary_score, val_primary)
        val_method.insert(0, "split", "val")
        test_method = per_method_score_metrics(test, args.primary_score, test_primary)
        test_method.insert(0, "split", "test")
        per_method = pd.concat([val_method, test_method], ignore_index=True)
    else:
        per_method = pd.DataFrame()

    hit_rank = pd.concat(
        [
            hit_rank_summary(val).assign(split="val"),
            hit_rank_summary(test).assign(split="test"),
        ],
        ignore_index=True,
    )

    conclusions = []
    val_oracle = oracle_df.loc[oracle_df["split"] == "val"].iloc[0]
    test_oracle = oracle_df.loc[oracle_df["split"] == "test"].iloc[0]
    for key in ["relaxed_route_available", "precursor_exact_available", "condition_relaxed_available", "strict_comparable_fraction"]:
        if key in oracle_df.columns:
            delta = float(test_oracle[key] - val_oracle[key])
            if abs(delta) >= 0.03:
                conclusions.append(f"{key} shifts by {delta:+.3f} from validation to test.")
    if not numeric.empty:
        top_num = numeric.iloc[0]
        conclusions.append(
            f"Largest numeric shift is {top_num['feature']} with standardized mean delta {top_num['standardized_mean_delta']:+.2f}."
        )
    if not cat.empty:
        top_cat = cat.iloc[0]
        conclusions.append(
            f"Largest categorical TV distance is {top_cat['feature']}={top_cat['value']} with TV={top_cat['total_variation']:.3f}."
        )
    if not per_method.empty:
        pivot = per_method.pivot_table(index="reaction_method", columns="split", values="missing_top1_relaxed_route", aggfunc="first")
        if "val" in pivot.columns and "test" in pivot.columns:
            pivot["test_minus_val"] = pivot["test"] - pivot["val"]
            worst = pivot["test_minus_val"].sort_values().head(3)
            conclusions.append("Worst per-method top1 shifts: " + ", ".join(f"{idx} {val:+.3f}" for idx, val in worst.items()))
    if not conclusions:
        conclusions.append("No large simple distribution shift detected; inspect learned score calibration and candidate text/source features next.")

    tables = {
        "oracle_availability": oracle_df,
        "score_metrics": score_metrics,
        "numeric_shift": numeric,
        "categorical_shift": cat,
        "per_method_metrics": per_method,
        "hit_rank_summary": hit_rank,
    }
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    payload = {"split_overview": split_overview, "conclusions": conclusions}
    atomic_write_json(out_dir / "stage35_val_test_shift_summary.json", payload)
    write_report(out_dir, payload, tables)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
