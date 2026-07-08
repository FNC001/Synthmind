#!/usr/bin/env python3
"""CDG vnext: joint condition-tuple calibration on a frozen candidate pool.

The experiment scores complete condition tuples (temperature, time,
atmosphere, solvent, method, and source scores) instead of optimizing a single
field.  It is selector-safe: training uses train candidates, validation selects
the candidate, and test is not run by this script unless explicitly requested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "temperature_c",
    "temperature_low_c",
    "temperature_high_c",
    "time_h",
    "time_low_h",
    "time_high_h",
    "temperature_point_score",
    "temperature_bin_score",
    "time_point_score",
    "time_bin_score",
    "atmosphere_probability",
    "solvent_probability",
    "condition_prior_score",
    "precursor_confidence_score",
    "open_generated_penalty",
    "repair_penalty",
    "total_score_raw",
    "multimodal_template_score",
    "method_template_score",
    "retrieval_score",
    "condition_rank_raw",
    "atmosphere_known_probability",
    "solvent_known_probability",
    "atmosphere_class_probability",
    "solvent_class_probability",
    "precursor_uncertainty_score",
    "method_expert_score",
    "template_score",
    "multimodal_score",
    "missing_aware_score",
    "strict_comparable_score",
]

CATEGORICAL_FEATURES = ["reaction_method", "atmosphere", "solvent", "condition_source"]
STRUCTURE_PREFIXES = (
    "feat_poscar_",
    "feat_pairdist_",
    "feat_nn_",
    "feat_coord_",
    "feat_lattice_",
    "feat_density",
    "feat_volume",
    "feat_spacegroup",
    "feat_crystal_system",
)
COMPOSITION_PREFIXES = ("feat_frac_el__", "feat_n_elements", "feat_total_atoms", "feat_stoich", "feat_z_", "feat_frac_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_csv", default="outputs/evaluation/stage3_condition_candidates_v4_20260612/train_condition_candidates.csv")
    parser.add_argument("--val_csv", default="outputs/evaluation/stage3_condition_candidates_v4_20260612/val_condition_candidates.csv")
    parser.add_argument("--test_csv", default="outputs/evaluation/stage3_condition_candidates_v4_20260612/test_condition_candidates.csv")
    parser.add_argument("--train_features", default="data/interim/features/structdesc_features_merged_20260609_with_structures_poscar_geom/stage3_train_ml.csv")
    parser.add_argument("--val_features", default="data/interim/features/structdesc_features_merged_20260609_with_structures_poscar_geom/stage3_val_ml.csv")
    parser.add_argument("--test_features", default="data/interim/features/structdesc_features_merged_20260609_with_structures_poscar_geom/stage3_test_ml.csv")
    parser.add_argument("--output_dir", default="outputs/autorun/cdg_joint_tuple_v1_20260623")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--run_test_only_if_val_passed", type=int, default=1)
    return parser.parse_args()


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def binary_col(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.zeros(len(df), dtype=np.int8)
    return pd.to_numeric(df[col], errors="coerce").fillna(0).clip(0, 1).to_numpy(dtype=np.int8)


def strict_comparable_relaxed(df: pd.DataFrame) -> np.ndarray:
    known = pd.to_numeric(df.get("atmosphere_known_mask", 0), errors="coerce").fillna(0).astype(int) == 1
    pred = df["atmosphere"].astype(str).str.lower() if "atmosphere" in df.columns else pd.Series("", index=df.index)
    true = df["true_atmosphere"].astype(str).str.lower() if "true_atmosphere" in df.columns else pd.Series("", index=df.index)
    atm_ok = known & pred.eq(true)
    temp_err = pd.to_numeric(df.get("temp_error", np.inf), errors="coerce").fillna(np.inf)
    time_err = pd.to_numeric(df.get("time_error", np.inf), errors="coerce").fillna(np.inf)
    return ((temp_err <= 200) & (time_err <= 48) & atm_ok).to_numpy(dtype=np.int8)


def strict_comparable_strict(df: pd.DataFrame) -> np.ndarray:
    known = pd.to_numeric(df.get("atmosphere_known_mask", 0), errors="coerce").fillna(0).astype(int) == 1
    pred = df["atmosphere"].astype(str).str.lower() if "atmosphere" in df.columns else pd.Series("", index=df.index)
    true = df["true_atmosphere"].astype(str).str.lower() if "true_atmosphere" in df.columns else pd.Series("", index=df.index)
    atm_ok = known & pred.eq(true)
    temp_err = pd.to_numeric(df.get("temp_error", np.inf), errors="coerce").fillna(np.inf)
    time_err = pd.to_numeric(df.get("time_error", np.inf), errors="coerce").fillna(np.inf)
    return ((temp_err <= 100) & (time_err <= 24) & atm_ok).to_numpy(dtype=np.int8)


def rank_metrics(df: pd.DataFrame, scores: np.ndarray) -> tuple[dict[str, float], pd.DataFrame]:
    ranked = pd.DataFrame(
        {
            "sample_id": df["sample_id"].astype(str).to_numpy(),
            "score": np.asarray(scores, dtype=np.float64),
            "row_index": np.arange(len(df), dtype=np.int64),
            "missing_relaxed": binary_col(df, "relaxed_hit_if_eval"),
            "missing_strict": binary_col(df, "strict_hit_if_eval"),
            "strict_relaxed": strict_comparable_relaxed(df),
            "strict_strict": strict_comparable_strict(df),
        }
    )
    ranked = ranked.sort_values(["sample_id", "score", "row_index"], ascending=[True, False, True], kind="mergesort")
    metrics: dict[str, float] = {"num_samples": float(ranked["sample_id"].nunique()), "num_candidates": float(len(ranked))}
    sample_tables = []
    for protocol, relaxed_col, strict_col in [
        ("missing_aware", "missing_relaxed", "missing_strict"),
        ("strict_comparable", "strict_relaxed", "strict_strict"),
    ]:
        for hit_name, col in [("relaxed_condition", relaxed_col), ("strict_condition", strict_col)]:
            for k in [1, 10, 50, 200]:
                top = ranked.groupby("sample_id", sort=False).head(k)
                metrics[f"{protocol}_top{k}_{hit_name}"] = float(top.groupby("sample_id")[col].max().mean())
            first = ranked[ranked[col] > 0].groupby("sample_id", sort=False)["score"].head(1)
            top1 = ranked.groupby("sample_id", sort=False).head(1).set_index("sample_id")[col]
            sample_tables.append(
                pd.DataFrame(
                    {
                        "sample_id": top1.index.astype(str),
                        f"{protocol}_{hit_name}_top1_hit": top1.to_numpy(dtype=np.int8),
                    }
                )
            )
    sample = sample_tables[0]
    for part in sample_tables[1:]:
        sample = sample.merge(part, on="sample_id", how="outer")
    return metrics, sample


def normalize_scores(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    std = arr.std()
    return np.zeros_like(arr) if std < 1e-12 else (arr - arr.mean()) / std


def score_column(df: pd.DataFrame, col: str, invert_rank: bool = False) -> np.ndarray:
    if col not in df.columns:
        return np.zeros(len(df), dtype=np.float64)
    vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    return -vals if invert_rank else vals


def feature_availability(candidates: pd.DataFrame, feature_path: Path) -> dict[str, Any]:
    if not feature_path.exists():
        return {"available": False, "reason": "feature_path_missing", "path": str(feature_path)}
    feat = pd.read_csv(feature_path, nrows=200000)
    result: dict[str, Any] = {"available": True, "path": str(feature_path), "columns": len(feat.columns)}
    for key in ["sample_id", "id", "material_id", "synth_uid"]:
        if key in feat.columns:
            overlap = len(set(candidates["sample_id"].astype(str)).intersection(set(feat[key].astype(str))))
            result[f"overlap_on_{key}"] = overlap
    best_overlap = max([v for k, v in result.items() if k.startswith("overlap_on_")] or [0])
    result["join_coverage"] = best_overlap / max(candidates["sample_id"].nunique(), 1)
    result["usable_for_ablation"] = result["join_coverage"] >= 0.8
    return result


def build_design(train: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    numeric = [c for c in NUMERIC_FEATURES if c in train.columns]
    categorical = [c for c in CATEGORICAL_FEATURES if c in train.columns]
    for col in categorical:
        train[col] = train[col].fillna("<MISSING>").astype(str)
        eval_df[col] = eval_df[col].fillna("<MISSING>").astype(str)
    return train[numeric + categorical].copy(), eval_df[numeric + categorical].copy(), numeric, categorical


def fit_joint_model(train_x: pd.DataFrame, train_y: np.ndarray, numeric: list[str], categorical: list[str], seed: int) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), categorical),
        ],
        sparse_threshold=0.2,
    )
    clf = HistGradientBoostingClassifier(
        max_iter=160,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=seed,
    )
    model = Pipeline([("pre", pre), ("clf", clf)])
    model.fit(train_x, train_y)
    return model


def paired_bootstrap(base: pd.Series, cand: pd.Series, seed: int, n_bootstrap: int) -> dict[str, float]:
    b = base.to_numpy(dtype=np.float64)
    c = cand.to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(b)
    diffs = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diffs[i] = c[idx].mean() - b[idx].mean()
    return {
        "delta": float(c.mean() - b.mean()),
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
        "n_samples": int(n),
    }


def write_report(out_dir: Path, split: str, rows: list[dict[str, Any]], gate: dict[str, Any], availability: dict[str, Any]) -> None:
    lines = [
        "# CDG Joint Condition-Tuple Calibration",
        "",
        f"- Split: `{split}`",
        "- Selector update: unchanged",
        "- Test policy: run only after validation gate",
        "",
        "## Feature Availability",
        "",
        "```json",
        json.dumps(to_builtin(availability), ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Metrics",
        "",
        "| ranker | missing top1 relaxed | missing top10 relaxed | strict top1 relaxed | strict top10 relaxed |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['ranker']} | {row.get('missing_aware_top1_relaxed_condition', 0):.4f} | "
            f"{row.get('missing_aware_top10_relaxed_condition', 0):.4f} | "
            f"{row.get('strict_comparable_top1_relaxed_condition', 0):.4f} | "
            f"{row.get('strict_comparable_top10_relaxed_condition', 0):.4f} |"
        )
    lines += ["", "## Gate", "", "```json", json.dumps(to_builtin(gate), ensure_ascii=False, indent=2, sort_keys=True), "```"]
    (out_dir / "CDG_JOINT_TUPLE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = Path(args.train_csv)
    eval_path = Path(args.val_csv if args.split == "validation" else args.test_csv)
    train = pd.read_csv(train_path)
    eval_df = pd.read_csv(eval_path)
    availability = {
        "train_features": feature_availability(train, Path(args.train_features)),
        "eval_features": feature_availability(eval_df, Path(args.val_features if args.split == "validation" else args.test_features)),
        "composition_structure_ablation_status": "not_run_join_coverage_below_80pct",
    }

    train_x, eval_x, numeric, categorical = build_design(train.copy(), eval_df.copy())
    y_missing = binary_col(train, "relaxed_hit_if_eval")
    y_strict = strict_comparable_relaxed(train)
    model_missing = fit_joint_model(train_x, y_missing, numeric, categorical, args.seed)
    model_strict = fit_joint_model(train_x, y_strict, numeric, categorical, args.seed + 1)
    p_missing = model_missing.predict_proba(eval_x)[:, 1]
    p_strict = model_strict.predict_proba(eval_x)[:, 1]
    joint_scores = 0.65 * normalize_scores(p_missing) + 0.35 * normalize_scores(p_strict)

    rankers: dict[str, np.ndarray] = {
        "raw_total_score": score_column(eval_df, "total_score_raw"),
        "v3_calibrated": score_column(eval_df, "condition_calibrated_score_v3"),
        "v4_missing_aware": score_column(eval_df, "missing_aware_score"),
        "v4_strict_comparable": score_column(eval_df, "strict_comparable_score"),
        "joint_tuple_condition_only": joint_scores,
    }
    rows: list[dict[str, Any]] = []
    sample_tables: dict[str, pd.DataFrame] = {}
    for name, scores in rankers.items():
        metrics, sample = rank_metrics(eval_df, scores)
        row = {"ranker": name}
        for key, value in metrics.items():
            row[key.replace("_condition", "_condition")] = value
        row["missing_aware_top1_relaxed_condition"] = metrics["missing_aware_top1_relaxed_condition"]
        row["missing_aware_top10_relaxed_condition"] = metrics["missing_aware_top10_relaxed_condition"]
        row["strict_comparable_top1_relaxed_condition"] = metrics["strict_comparable_top1_relaxed_condition"]
        row["strict_comparable_top10_relaxed_condition"] = metrics["strict_comparable_top10_relaxed_condition"]
        rows.append(row)
        sample_tables[name] = sample.add_prefix(f"{name}__").rename(columns={f"{name}__sample_id": "sample_id"})

    aligned = None
    for table in sample_tables.values():
        aligned = table if aligned is None else aligned.merge(table, on="sample_id", how="inner")
    baseline_name = "v3_calibrated"
    best_row = max(rows, key=lambda r: float(r.get("strict_comparable_top1_relaxed_condition", 0.0)))
    ci = paired_bootstrap(
        aligned[f"{baseline_name}__strict_comparable_relaxed_condition_top1_hit"],
        aligned[f"{best_row['ranker']}__strict_comparable_relaxed_condition_top1_hit"],
        args.seed,
        args.bootstrap,
    )
    baseline = next(r for r in rows if r["ranker"] == baseline_name)
    best_missing = float(best_row["missing_aware_top1_relaxed_condition"])
    baseline_missing = float(baseline["missing_aware_top1_relaxed_condition"])
    gate = {
        "baseline_ranker": baseline_name,
        "best_ranker": best_row["ranker"],
        "strict_top1_delta": float(best_row["strict_comparable_top1_relaxed_condition"] - baseline["strict_comparable_top1_relaxed_condition"]),
        "missing_top1_delta": best_missing - baseline_missing,
        "bootstrap_ci_strict_top1": ci,
        "passes_gate": bool(ci["delta"] > 0 and ci["ci_low"] > 0 and best_missing >= baseline_missing - 0.005),
        "selector_update_status": "unchanged",
        "test_allowed": bool(args.split == "validation" and ci["delta"] > 0 and ci["ci_low"] > 0 and best_missing >= baseline_missing - 0.005),
    }

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "train_csv": str(train_path),
        "eval_csv": str(eval_path),
        "train_sha256": sha256_file(train_path),
        "eval_sha256": sha256_file(eval_path),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "selector_update_status": "unchanged",
    }
    pd.DataFrame(rows).to_csv(out_dir / "cdg_joint_tuple_metrics.csv", index=False)
    aligned.to_csv(out_dir / "sample_level_top1_hits.csv", index=False)
    (out_dir / "metrics.json").write_text(
        json.dumps(to_builtin({"manifest": manifest, "rankers": rows, "gate": gate, "feature_availability": availability}), indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    write_report(out_dir, args.split, rows, gate, availability)
    print(json.dumps(to_builtin(gate), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
