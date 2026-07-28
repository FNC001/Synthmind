#!/usr/bin/env python3
"""Audit a Stage35 GPU checkpoint with tune/holdout validation splitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_autodl_gpu_training_queue import (
    RouteRanker,
    atomic_write_json,
    build_matrix,
    passes_stage35_gate,
    robust_search_score_blend,
    route_metrics,
    score_in_batches,
    stable_val_split_masks,
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run_output_root", required=True)
    parser.add_argument("--val_csv", default="outputs/evaluation/stage35_route_candidates_v3_final_20260612/val_route_candidates.csv")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--seed", type=int, default=9001)
    parser.add_argument("--holdout_fraction", type=float, default=0.5)
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
    output_json = Path(args.output_json)
    if not output_json.is_absolute():
        output_json = project_root / output_json

    schema = load_json(run_output_root / "feature_schema.json")
    stats = load_json(run_output_root / "standardization_stats.json")
    val_df = pd.read_csv(val_csv)
    x_np = build_matrix(val_df, schema)
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)
    x_np = ((x_np - mean) / std).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device)
    variant = ckpt["variant"]
    model = RouteRanker(x_np.shape[1], int(variant["hidden"]), float(variant["dropout"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    scores = score_in_batches(model, torch.tensor(x_np, device=device), max(int(variant["batch_size"]) * 2, 65536))
    tune_mask, holdout_mask = stable_val_split_masks(val_df, args.holdout_fraction, args.seed)
    audit = robust_search_score_blend(val_df, scores, tune_mask, holdout_mask)
    payload = {
        "checkpoint": str(checkpoint),
        "val_csv": str(val_csv),
        "seed": args.seed,
        "holdout_fraction": args.holdout_fraction,
        "variant": variant,
        "checkpoint_full_val_metrics": ckpt.get("metrics", {}),
        "pure_model_full_val_metrics": route_metrics(val_df, scores),
        "audit": audit,
        "passes_gate_for_test": bool(audit["passes_gate"]),
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
    }
    atomic_write_json(output_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
