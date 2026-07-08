#!/usr/bin/env python3
"""Method-aware robust calibration for SynPred Stage35 checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from run_autodl_gpu_training_queue import (
    RouteRanker,
    atomic_write_json,
    blend_scores,
    build_matrix,
    passes_stage35_gate,
    robust_search_score_blend,
    route_metrics,
    score_in_batches,
    selection_score,
    stable_val_split_masks,
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_scores(
    project_root: Path,
    run_output_root: Path,
    checkpoint: Path,
    csv_path: Path,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    schema = load_json(run_output_root / "feature_schema.json")
    stats = load_json(run_output_root / "standardization_stats.json")
    df = pd.read_csv(csv_path)
    x_np = build_matrix(df, schema)
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)
    x_np = ((x_np - mean) / std).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device)
    variant = ckpt["variant"]
    model = RouteRanker(x_np.shape[1], int(variant["hidden"]), float(variant["dropout"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    scores = score_in_batches(model, torch.tensor(x_np, device=device), max(int(variant["batch_size"]) * 2, 65536))
    return df, scores, ckpt


def apply_method_blends(df: pd.DataFrame, scores: np.ndarray, blends: dict[str, dict[str, Any]], global_blend: dict[str, Any]) -> np.ndarray:
    final = np.zeros(len(df), dtype=np.float64)
    methods = df["reaction_method"].fillna("<MISSING>").astype(str).to_numpy()
    for method in sorted(set(methods)):
        mask = methods == method
        blend = blends.get(method, global_blend)
        final[mask] = blend_scores(df.loc[mask].copy(), scores[mask], blend)
    return final


def fit_method_blends(
    val_df: pd.DataFrame,
    val_scores: np.ndarray,
    tune_mask: np.ndarray,
    holdout_mask: np.ndarray,
    min_tune_samples: int,
    min_holdout_samples: int,
) -> dict[str, Any]:
    global_result = robust_search_score_blend(val_df, val_scores, tune_mask, holdout_mask)
    global_blend = global_result["blend"]
    method_results: dict[str, Any] = {}
    selected_blends: dict[str, dict[str, Any]] = {}

    methods = val_df["reaction_method"].fillna("<MISSING>").astype(str)
    for method in sorted(methods.unique()):
        method_mask = methods.to_numpy() == method
        method_tune = method_mask & tune_mask
        method_holdout = method_mask & holdout_mask
        tune_samples = int(val_df.loc[method_tune, "sample_id"].nunique())
        holdout_samples = int(val_df.loc[method_holdout, "sample_id"].nunique())
        if tune_samples < min_tune_samples or holdout_samples < min_holdout_samples:
            method_results[method] = {
                "selected": False,
                "reason": "too_few_samples",
                "tune_samples": tune_samples,
                "holdout_samples": holdout_samples,
                "blend": global_blend,
            }
            continue
        local_df = val_df.loc[method_mask].copy()
        local_scores = val_scores[method_mask]
        local_tune = tune_mask[method_mask]
        local_holdout = holdout_mask[method_mask]
        local_result = robust_search_score_blend(local_df, local_scores, local_tune, local_holdout)
        global_on_local_scores = blend_scores(local_df, local_scores, global_blend)
        global_holdout_metrics = route_metrics(local_df.loc[local_holdout].copy(), global_on_local_scores[local_holdout])
        local_score = float(selection_score(local_result["metrics"]))
        global_score = float(selection_score(global_holdout_metrics))
        top1_not_worse = (
            local_result["metrics"].get("missing_top1_relaxed_route", 0.0)
            >= global_holdout_metrics.get("missing_top1_relaxed_route", 0.0)
            and local_result["metrics"].get("strict_top1_relaxed_route", 0.0)
            >= global_holdout_metrics.get("strict_top1_relaxed_route", 0.0)
        )
        top10_not_much_worse = (
            local_result["metrics"].get("missing_top10_relaxed_route", 0.0)
            >= global_holdout_metrics.get("missing_top10_relaxed_route", 0.0) - 0.005
            and local_result["metrics"].get("strict_top10_relaxed_route", 0.0)
            >= global_holdout_metrics.get("strict_top10_relaxed_route", 0.0) - 0.005
        )
        selected = local_score >= global_score + 0.02 and top1_not_worse and top10_not_much_worse
        blend = local_result["blend"] if selected else global_blend
        if selected:
            selected_blends[method] = blend
        method_results[method] = {
            "selected": selected,
            "tune_samples": tune_samples,
            "holdout_samples": holdout_samples,
            "local_result": local_result,
            "global_holdout_metrics": global_holdout_metrics,
            "local_selection_score": local_score,
            "global_selection_score": global_score,
            "top1_not_worse": top1_not_worse,
            "top10_not_much_worse": top10_not_much_worse,
            "blend": blend,
        }

    final_scores = apply_method_blends(val_df, val_scores, selected_blends, global_blend)
    holdout_metrics = route_metrics(val_df.loc[holdout_mask].copy(), final_scores[holdout_mask])
    full_metrics = route_metrics(val_df, final_scores)
    tune_metrics = route_metrics(val_df.loc[tune_mask].copy(), final_scores[tune_mask])
    return {
        "global_result": global_result,
        "global_blend": global_blend,
        "selected_method_blends": selected_blends,
        "method_results": method_results,
        "tune_metrics": tune_metrics,
        "holdout_metrics": holdout_metrics,
        "full_val_metrics": full_metrics,
        "passes_gate": passes_stage35_gate(holdout_metrics) and passes_stage35_gate(full_metrics),
        "passes_gate_holdout": passes_stage35_gate(holdout_metrics),
        "passes_gate_full": passes_stage35_gate(full_metrics),
    }


def write_method_table(path: Path, method_results: dict[str, Any]) -> None:
    rows = []
    for method, result in method_results.items():
        row = {
            "reaction_method": method,
            "selected": bool(result.get("selected", False)),
            "tune_samples": result.get("tune_samples", 0),
            "holdout_samples": result.get("holdout_samples", 0),
            "base_score": result.get("blend", {}).get("base_score"),
            "alpha_model": result.get("blend", {}).get("alpha_model"),
        }
        row["local_selection_score"] = result.get("local_selection_score")
        row["global_selection_score"] = result.get("global_selection_score")
        row["top1_not_worse"] = result.get("top1_not_worse")
        row["top10_not_much_worse"] = result.get("top10_not_much_worse")
        local = result.get("local_result", {})
        for k, v in local.get("metrics", {}).items():
            row[f"local_holdout_{k}"] = v
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run_output_root", required=True)
    parser.add_argument("--val_csv", default="outputs/evaluation/stage35_route_candidates_v3_final_20260612/val_route_candidates.csv")
    parser.add_argument("--test_csv", default="outputs/evaluation/stage35_route_candidates_v3_final_20260612/test_route_candidates.csv")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--seed", type=int, default=7200)
    parser.add_argument("--holdout_fraction", type=float, default=0.5)
    parser.add_argument("--min_tune_samples", type=int, default=40)
    parser.add_argument("--min_holdout_samples", type=int, default=40)
    parser.add_argument("--run_test_only_if_gate_pass", type=int, default=1)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = project_root / checkpoint
    run_output_root = Path(args.run_output_root)
    if not run_output_root.is_absolute():
        run_output_root = project_root / run_output_root
    val_csv = Path(args.val_csv)
    if not val_csv.is_absolute():
        val_csv = project_root / val_csv
    test_csv = Path(args.test_csv)
    if not test_csv.is_absolute():
        test_csv = project_root / test_csv
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    val_df, val_scores, ckpt = load_scores(project_root, run_output_root, checkpoint, val_csv)
    tune_mask, holdout_mask = stable_val_split_masks(val_df, args.holdout_fraction, args.seed)
    calibration = fit_method_blends(val_df, val_scores, tune_mask, holdout_mask, args.min_tune_samples, args.min_holdout_samples)
    payload = {
        "checkpoint": str(checkpoint),
        "val_csv": str(val_csv),
        "seed": args.seed,
        "holdout_fraction": args.holdout_fraction,
        "variant": ckpt.get("variant", {}),
        "calibration": calibration,
        "test": None,
    }
    write_method_table(output_root / "method_blend_table.csv", calibration["method_results"])

    if args.run_test_only_if_gate_pass and calibration["passes_gate"]:
        test_df, test_scores, _ = load_scores(project_root, run_output_root, checkpoint, test_csv)
        final_test_scores = apply_method_blends(test_df, test_scores, calibration["selected_method_blends"], calibration["global_blend"])
        payload["test"] = {
            "test_csv": str(test_csv),
            "metrics": route_metrics(test_df, final_test_scores),
            "pure_model_metrics": route_metrics(test_df, test_scores),
        }

    atomic_write_json(output_root / "method_aware_calibration_result.json", payload)
    lines = [
        "# Stage35 Method-Aware Calibration",
        "",
        f"- Checkpoint: `{checkpoint}`",
        f"- Passes robust gate: `{calibration['passes_gate']}`",
        f"- Selected method blends: `{len(calibration['selected_method_blends'])}`",
        "",
        "## Holdout Metrics",
        "",
        "```json",
        json.dumps(calibration["holdout_metrics"], indent=2, sort_keys=True),
        "```",
        "",
        "## Full Validation Metrics",
        "",
        "```json",
        json.dumps(calibration["full_val_metrics"], indent=2, sort_keys=True),
        "```",
    ]
    if payload["test"]:
        lines += ["", "## Test Metrics", "", "```json", json.dumps(payload["test"], indent=2, sort_keys=True), "```"]
    (output_root / "METHOD_AWARE_CALIBRATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"passes_gate": calibration["passes_gate"], "test": payload["test"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
