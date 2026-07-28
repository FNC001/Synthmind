#!/usr/bin/env python3
"""Generate calibrated validation samples from a conditional Stage3 WGAN."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from training.family.train_stage3_conditional_wgan import (
    ConditionalGenerator,
    generate_samples,
)
from training.family.train_stage3_hybrid_cvae import load_pack


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrated validation sampling for Stage3 WGAN.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--latent_scale", type=float, default=1.0)
    parser.add_argument("--categorical_temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_path = Path(args.output_npz).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = ConditionalGenerator(
        int(checkpoint["structure_dim"]),
        int(checkpoint["precursor_dim"]),
        int(config["hidden"]),
        int(config["precursor_hidden"]),
        int(config["latent"]),
        int(config["generator_blocks"]),
        float(config["dropout"]),
        int(checkpoint["atmosphere_classes"]),
        int(checkpoint["method_classes"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    val = load_pack(input_dir / "val.npz")
    features = torch.from_numpy(
        np.hstack([
            np.asarray(val["x"], dtype=np.float32),
            np.asarray(val["y_set"], dtype=np.float32),
        ]).astype(np.float32)
    )
    continuous, discrete = generate_samples(
        model,
        features,
        int(args.samples),
        int(args.batch_size),
        checkpoint["target_stats"],
        device,
        int(args.seed),
        float(args.latent_scale),
        float(args.categorical_temperature),
    )
    np.savez_compressed(
        output_path,
        continuous_samples=continuous.astype(np.float32),
        discrete_samples=discrete.astype(np.int16),
        sample_id=np.asarray(val["sample_id"]).astype(str),
    )
    report = {
        "protocol": "validation_only_conditional_wgan_calibrated_sampling",
        "checkpoint": str(checkpoint_path),
        "output_npz": str(output_path),
        "rows": int(len(features)),
        "samples_per_row": int(args.samples),
        "latent_scale": float(args.latent_scale),
        "categorical_temperature": float(args.categorical_temperature),
        "seed": int(args.seed),
        "device": str(device),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
