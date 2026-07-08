#!/usr/bin/env python3
"""Evaluate a SynPred Stage35 GPU reranker checkpoint on a held-out split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_autodl_gpu_training_queue import (
    RouteRanker,
    atomic_write_json,
    blend_scores,
    build_matrix,
    route_metrics,
    score_in_batches,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run_output_root", required=True)
    parser.add_argument("--split_csv", default="outputs/evaluation/stage35_route_candidates_v3_final_20260612/test_route_candidates.csv")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_scores_csv", default="")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / checkpoint_path
    run_output_root = Path(args.run_output_root)
    if not run_output_root.is_absolute():
        run_output_root = project_root / run_output_root
    split_path = Path(args.split_csv)
    if not split_path.is_absolute():
        split_path = project_root / split_path
    output_json = Path(args.output_json)
    if not output_json.is_absolute():
        output_json = project_root / output_json

    schema_path = run_output_root / "feature_schema.json"
    stats_path = run_output_root / "standardization_stats.json"
    if not schema_path.exists():
        raise FileNotFoundError(schema_path)
    if not stats_path.exists():
        raise FileNotFoundError(stats_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    df = pd.read_csv(split_path)
    x_np = build_matrix(df, schema)
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)
    x_np = ((x_np - mean) / std).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    variant = ckpt["variant"]
    model = RouteRanker(x_np.shape[1], int(variant["hidden"]), float(variant["dropout"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    x = torch.tensor(x_np, device=device)
    model_scores = score_in_batches(model, x, max(int(variant["batch_size"]) * 2, 65536))
    final_scores = blend_scores(df, model_scores, ckpt.get("blend", {}))

    payload = {
        "checkpoint": str(checkpoint_path),
        "split_csv": str(split_path),
        "variant": variant,
        "blend": ckpt.get("blend", {}),
        "checkpoint_val_metrics": ckpt.get("metrics", {}),
        "pure_model_metrics": route_metrics(df, model_scores),
        "blended_metrics": route_metrics(df, final_scores),
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
    }
    atomic_write_json(output_json, payload)

    if args.output_scores_csv:
        output_scores = Path(args.output_scores_csv)
        if not output_scores.is_absolute():
            output_scores = project_root / output_scores
        output_scores.parent.mkdir(parents=True, exist_ok=True)
        score_df = pd.DataFrame(
            {
                "sample_id": df["sample_id"].astype(str),
                "route_candidate_id": df.get("route_candidate_id", pd.Series([""] * len(df))).astype(str),
                "model_score": model_scores,
                "blended_score": final_scores,
            }
        )
        score_df.to_csv(output_scores, index=False)
        payload["scores_csv"] = str(output_scores)
        atomic_write_json(output_json, payload)

    print(json.dumps(payload["blended_metrics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
