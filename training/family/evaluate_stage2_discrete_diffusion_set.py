#!/usr/bin/env python3
"""Evaluate and tune sampling for a trained Stage2 discrete set diffusion model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.family.train_stage2_discrete_diffusion_set import (  # noqa: E402
    DiscreteSetDiffusion,
    exact_metrics,
    generate_candidates,
    seed_everything,
    write_candidates,
)


def float_grid(value: str) -> List[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def int_grid(value: str) -> List[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid-search reverse sampling without retraining the diffusion model."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--sampling_steps", default="8,16")
    parser.add_argument("--guidance_scales", default="0.0,0.5,1.0,1.5")
    parser.add_argument("--temperature_starts", default="1.25")
    parser.add_argument("--temperature_ends", default="0.35,0.6,0.9")
    parser.add_argument("--length_temperatures", default="0.75,1.0")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--trajectory_batch", type=int, default=16)
    parser.add_argument("--selection_metric", default="exact_hit@100")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    pack = np.load(input_dir / f"{args.split}.npz", allow_pickle=True)
    x = np.asarray(pack["x"], dtype=np.float32)
    targets = [
        tuple(np.flatnonzero(row > 0.5).tolist())
        for row in np.asarray(pack["y_multi_hot"], dtype=np.float32)
    ]
    checkpoint = torch.load(
        Path(args.checkpoint).resolve(), map_location="cpu", weights_only=False
    )
    config = checkpoint["config"]
    label_chemistry = torch.as_tensor(checkpoint["label_chemistry"], dtype=torch.float32)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = DiscreteSetDiffusion(
        x_dim=int(checkpoint["x_dim"]),
        n_labels=int(checkpoint["n_labels"]),
        max_set_len=int(checkpoint["max_set_len"]),
        hidden=int(config["hidden"]),
        query_blocks=int(config["query_blocks"]),
        denoise_blocks=int(config["denoise_blocks"]),
        dropout=float(config["dropout"]),
        label_chemistry=label_chemistry,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    base_probabilities = torch.as_tensor(
        checkpoint["base_probabilities"], dtype=torch.float32, device=device
    )
    grid: List[Dict[str, Any]] = []
    best_value = -1.0
    best_rows = None
    best_scores = None
    best_config = None
    trial_index = 0
    for sampling_steps in int_grid(args.sampling_steps):
        for guidance_scale in float_grid(args.guidance_scales):
            for temperature_start in float_grid(args.temperature_starts):
                for temperature_end in float_grid(args.temperature_ends):
                    for length_temperature in float_grid(args.length_temperatures):
                        seed_everything(int(args.seed) + trial_index)
                        rows, scores = generate_candidates(
                            model,
                            x,
                            base_probabilities,
                            device,
                            int(args.eval_batch_size),
                            samples=int(args.samples),
                            sampling_steps=int(sampling_steps),
                            guidance_scale=float(guidance_scale),
                            temperature_start=float(temperature_start),
                            temperature_end=float(temperature_end),
                            length_temperature=float(length_temperature),
                            trajectory_batch=int(args.trajectory_batch),
                            diffusion_steps=int(config["diffusion_steps"]),
                        )
                        metrics = exact_metrics(targets, rows)
                        trial = {
                            "trial": trial_index,
                            "sampling_steps": int(sampling_steps),
                            "guidance_scale": float(guidance_scale),
                            "temperature_start": float(temperature_start),
                            "temperature_end": float(temperature_end),
                            "length_temperature": float(length_temperature),
                            "mean_unique_candidates": float(
                                np.mean([len(row) for row in rows])
                            ),
                            **metrics,
                        }
                        grid.append(trial)
                        print(json.dumps(trial), flush=True)
                        current = float(metrics[str(args.selection_metric)])
                        tie_break = float(metrics.get("exact_hit@10", 0.0))
                        best_tie = (
                            float(best_config.get("exact_hit@10", -1.0))
                            if best_config is not None
                            else -1.0
                        )
                        if current > best_value or (
                            current == best_value and tie_break > best_tie
                        ):
                            best_value = current
                            best_rows = rows
                            best_scores = scores
                            best_config = trial
                        trial_index += 1
    if best_rows is None or best_scores is None or best_config is None:
        raise RuntimeError("empty sampling grid")
    output_candidates = Path(args.output_candidates_jsonl).resolve()
    write_candidates(output_candidates, best_rows, best_scores)
    report = {
        "protocol": f"{args.split}_frozen_discrete_diffusion_sampling_grid",
        "config": vars(args),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_best_epoch": int(checkpoint["best_epoch"]),
        "selection_metric": str(args.selection_metric),
        "best": best_config,
        "grid": grid,
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
