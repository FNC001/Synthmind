#!/usr/bin/env python3
"""End-to-end same-pool route ranking with GRV and CDG joint-tuple scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "10_autorun"
if str(PROJECT_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SCRIPT_DIR))

from run_autodl_gpu_training_queue import normalize_scores  # noqa: E402
from synthmind.research.run_cdg_joint_tuple_calibration import (  # noqa: E402
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    binary_col,
    fit_joint_model,
    strict_comparable_relaxed,
)
from synthmind.research.run_grv_shared_pool_comparison import (  # noqa: E402
    RankerSpec,
    checkpoint_scores,
    paired_bootstrap,
    score_metrics,
    v3_final_scores,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--train_condition_csv", default="outputs/evaluation/stage3_condition_calibration_v3_final_20260612/train_condition_candidates_calibrated.csv")
    parser.add_argument("--route_csv", default="outputs/evaluation/stage35_route_candidates_v3_final_20260612/val_route_candidates.csv")
    parser.add_argument("--output_dir", default="outputs/autorun/route_grv_cdg_blend_20260623")
    parser.add_argument(
        "--checkpoint_runs",
        default=(
            "grv_v14=outputs/autorun/gpu_24h_training_20260618_v14_ooftrain_strict_replicate,"
            "grv_fast_v3_protocol=outputs/autorun/grv_shared_pool_fast_20260623_v3_protocol"
        ),
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline_ranker", default="grv_v14")
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


def parse_checkpoint_specs(text: str) -> list[RankerSpec]:
    specs: list[RankerSpec] = []
    for item in [part.strip() for part in text.split(",") if part.strip()]:
        name, run_dir = item.split("=", 1) if "=" in item else (Path(item).name, item)
        specs.append(RankerSpec(name=name.strip(), kind="checkpoint", run_dir=Path(run_dir.strip())))
    return specs


def build_cdg_design(train: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    numeric = [c for c in NUMERIC_FEATURES if c in train.columns and c in eval_df.columns]
    categorical = [c for c in CATEGORICAL_FEATURES if c in train.columns and c in eval_df.columns]
    train_work = train.copy()
    eval_work = eval_df.copy()
    for col in categorical:
        train_work[col] = train_work[col].fillna("<MISSING>").astype(str)
        eval_work[col] = eval_work[col].fillna("<MISSING>").astype(str)
    return train_work[numeric + categorical], eval_work[numeric + categorical], numeric, categorical


def cdg_joint_scores(train_condition: pd.DataFrame, route_df: pd.DataFrame, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    train_x, eval_x, numeric, categorical = build_cdg_design(train_condition, route_df)
    y_missing = binary_col(train_condition, "relaxed_hit_if_eval")
    y_strict = strict_comparable_relaxed(train_condition)
    model_missing = fit_joint_model(train_x, y_missing, numeric, categorical, seed)
    model_strict = fit_joint_model(train_x, y_strict, numeric, categorical, seed + 1)
    p_missing = model_missing.predict_proba(eval_x)[:, 1]
    p_strict = model_strict.predict_proba(eval_x)[:, 1]
    scores = 0.65 * normalize_scores(p_missing) + 0.35 * normalize_scores(p_strict)
    meta = {
        "numeric_features": numeric,
        "categorical_features": categorical,
        "target_missing_positive_rate": float(np.mean(y_missing)),
        "target_strict_positive_rate": float(np.mean(y_strict)),
    }
    return scores, meta


def aligned_sample_tables(sample_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    aligned = None
    for table in sample_tables.values():
        aligned = table if aligned is None else aligned.merge(table, on="sample_id", how="inner")
    if aligned is None:
        raise RuntimeError("no sample tables")
    return aligned


def write_report(out_dir: Path, split: str, rows: list[dict[str, Any]], gate: dict[str, Any]) -> None:
    lines = [
        "# Route GRV + CDG Same-Pool Blend",
        "",
        f"- Split: `{split}`",
        "- Candidate pool: frozen; no candidate expansion",
        "- Selector update: unchanged",
        "",
        "| ranker | missing top1 | missing top10 | missing top200 | strict top1 | usable top1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['ranker']} | {row.get('missing_aware_top1_relaxed_route', 0):.4f} | "
            f"{row.get('missing_aware_top10_relaxed_route', 0):.4f} | "
            f"{row.get('missing_aware_top200_relaxed_route', 0):.4f} | "
            f"{row.get('strict_comparable_top1_relaxed_route', 0):.4f} | "
            f"{row.get('missing_aware_top1_usable_relaxed_route', 0):.4f} |"
        )
    lines += ["", "## Gate", "", "```json", json.dumps(to_builtin(gate), ensure_ascii=False, indent=2, sort_keys=True), "```"]
    (out_dir / "ROUTE_GRV_CDG_BLEND_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    route_csv = Path(args.route_csv)
    train_condition_csv = Path(args.train_condition_csv)
    if not route_csv.is_absolute():
        route_csv = project_root / route_csv
    if not train_condition_csv.is_absolute():
        train_condition_csv = project_root / train_condition_csv
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    route_df = pd.read_csv(route_csv)
    train_condition = pd.read_csv(train_condition_csv)
    cdg_scores, cdg_meta = cdg_joint_scores(train_condition, route_df, args.seed)

    score_map: dict[str, np.ndarray] = {
        "legacy_v3_final_formula": v3_final_scores(route_df),
        "cdg_joint_condition_only": cdg_scores,
    }
    metadata: dict[str, Any] = {
        "legacy_v3_final_formula": {"kind": "formula"},
        "cdg_joint_condition_only": {"kind": "cdg_joint", **cdg_meta},
    }
    for spec in parse_checkpoint_specs(args.checkpoint_runs):
        scores, meta = checkpoint_scores(project_root, route_df, spec)
        score_map[spec.name] = scores
        metadata[spec.name] = meta

    rows: list[dict[str, Any]] = []
    sample_tables: dict[str, pd.DataFrame] = {}
    for name, scores in list(score_map.items()):
        metrics, sample = score_metrics(route_df, scores)
        rows.append({"ranker": name, **metrics})
        sample_tables[name] = sample.add_prefix(f"{name}__").rename(columns={f"{name}__sample_id": "sample_id"})

    blend_alphas = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]
    for base_name in [name for name in score_map if name.startswith("grv_")]:
        for alpha in blend_alphas:
            name = f"{base_name}_plus_cdg_a{alpha:.2f}"
            scores = alpha * normalize_scores(score_map[base_name]) + (1.0 - alpha) * normalize_scores(cdg_scores)
            metrics, sample = score_metrics(route_df, scores)
            rows.append({"ranker": name, **metrics, "blend_base": base_name, "alpha_grv": alpha})
            sample_tables[name] = sample.add_prefix(f"{name}__").rename(columns={f"{name}__sample_id": "sample_id"})

    pd.DataFrame(rows).to_csv(out_dir / "same_pool_route_grv_cdg_metrics.csv", index=False)
    aligned = aligned_sample_tables(sample_tables)
    aligned.to_csv(out_dir / "sample_level_top1_hits.csv", index=False)

    baseline_name = args.baseline_ranker
    if baseline_name not in sample_tables:
        baseline_name = "legacy_v3_final_formula"
    baseline_row = next(row for row in rows if row["ranker"] == baseline_name)
    best_row = max(rows, key=lambda row: float(row.get("missing_aware_top1_relaxed_route", 0.0)))
    ci = paired_bootstrap(
        aligned[f"{baseline_name}__missing_aware_relaxed_top1_hit"].to_numpy(),
        aligned[f"{best_row['ranker']}__missing_aware_relaxed_top1_hit"].to_numpy(),
        seed=args.seed,
        n_bootstrap=args.bootstrap,
    )
    strict_delta = float(best_row.get("strict_comparable_top1_relaxed_route", 0.0)) - float(
        baseline_row.get("strict_comparable_top1_relaxed_route", 0.0)
    )
    gate = {
        "baseline_ranker": baseline_name,
        "best_ranker": best_row["ranker"],
        "missing_aware_top1_delta": float(best_row["missing_aware_top1_relaxed_route"] - baseline_row["missing_aware_top1_relaxed_route"]),
        "strict_comparable_top1_delta": strict_delta,
        "bootstrap_ci_missing_top1": ci,
        "passes_gate": bool(ci["delta"] >= 0.01 and ci["ci_low"] > 0.0 and strict_delta >= -0.005),
        "test_allowed": bool(args.split == "validation" and ci["delta"] >= 0.01 and ci["ci_low"] > 0.0 and strict_delta >= -0.005),
        "selector_update_status": "unchanged",
    }
    manifest = {
        "split": args.split,
        "route_csv": str(route_csv),
        "route_sha256": sha256_file(route_csv),
        "train_condition_csv": str(train_condition_csv),
        "train_condition_sha256": sha256_file(train_condition_csv),
        "candidate_pool_status": "frozen",
        "selector_update_status": "unchanged",
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(to_builtin({"manifest": manifest, "rankers": rows, "metadata": metadata, "gate": gate}), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(out_dir, args.split, rows, gate)
    print(json.dumps(to_builtin(gate), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
