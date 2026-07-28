#!/usr/bin/env python3
"""Train a validation-selected conditional WGAN-GP for synthesis conditions.

The adversarial critic models the joint temperature/log-time distribution on
rows where both values are observed.  Noisy categorical heads use all rows
with observed labels.  The frozen test split is never loaded.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from training.family.train_stage3_hybrid_cvae import (
    ResidualBlock,
    inverse_continuous,
    load_pack,
    transform_continuous,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ConditionEncoder(nn.Module):
    def __init__(
        self,
        structure_dim: int,
        precursor_dim: int,
        hidden: int,
        precursor_hidden: int,
        blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.structure_dim = int(structure_dim)
        structure_hidden = max(256, hidden // 2)
        self.structure = nn.Sequential(
            nn.Linear(structure_dim, structure_hidden),
            nn.LayerNorm(structure_hidden),
            nn.GELU(),
            nn.Linear(structure_hidden, structure_hidden),
            nn.GELU(),
        )
        self.precursor = nn.Sequential(
            nn.Linear(precursor_dim, precursor_hidden),
            nn.LayerNorm(precursor_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(structure_hidden + precursor_hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden, dropout) for _ in range(int(blocks))]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        structure = self.structure(features[:, : self.structure_dim])
        precursor = self.precursor(features[:, self.structure_dim :])
        return self.blocks(self.fusion(torch.cat([structure, precursor], dim=-1)))


class ConditionalGenerator(nn.Module):
    def __init__(
        self,
        structure_dim: int,
        precursor_dim: int,
        hidden: int,
        precursor_hidden: int,
        latent: int,
        blocks: int,
        dropout: float,
        atmosphere_classes: int,
        method_classes: int,
    ) -> None:
        super().__init__()
        self.latent = int(latent)
        self.condition = ConditionEncoder(
            structure_dim,
            precursor_dim,
            hidden,
            precursor_hidden,
            blocks,
            dropout,
        )
        self.input = nn.Sequential(
            nn.Linear(hidden + latent, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden, dropout) for _ in range(int(blocks))]
        )
        self.continuous = nn.Linear(hidden, 2)
        self.atmosphere = nn.Linear(hidden, atmosphere_classes)
        self.method = nn.Linear(hidden, method_classes)

    def forward(self, features: torch.Tensor, noise: torch.Tensor) -> Dict[str, torch.Tensor]:
        context = self.condition(features)
        hidden = self.blocks(self.input(torch.cat([context, noise], dim=-1)))
        return {
            "continuous": self.continuous(hidden),
            "atmosphere": self.atmosphere(hidden),
            "method": self.method(hidden),
        }


class ConditionalCritic(nn.Module):
    def __init__(
        self,
        structure_dim: int,
        precursor_dim: int,
        hidden: int,
        precursor_hidden: int,
        blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.condition = ConditionEncoder(
            structure_dim,
            precursor_dim,
            hidden,
            precursor_hidden,
            blocks,
            dropout,
        )
        self.value = nn.Sequential(
            nn.Linear(2, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        condition = self.condition(features)
        encoded = self.value(values)
        combined = torch.cat(
            [condition, encoded, condition * encoded, torch.abs(condition - encoded)], dim=-1
        )
        return self.scorer(combined).squeeze(-1)


def gradient_penalty(
    critic: ConditionalCritic,
    features: torch.Tensor,
    real: torch.Tensor,
    fake: torch.Tensor,
) -> torch.Tensor:
    alpha = torch.rand((len(real), 1), device=real.device)
    mixed = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
    score = critic(features, mixed)
    gradient = torch.autograd.grad(
        outputs=score.sum(),
        inputs=mixed,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return (gradient.norm(2, dim=1) - 1.0).square().mean()


def categorical_loss(
    output: Dict[str, torch.Tensor],
    discrete: torch.Tensor,
    discrete_mask: torch.Tensor,
) -> torch.Tensor:
    losses = []
    for index, name in enumerate(("atmosphere", "method")):
        valid = discrete_mask[:, index] > 0.5
        if bool(valid.any()):
            losses.append(F.cross_entropy(output[name][valid], discrete[valid, index]))
    return torch.stack(losses).mean() if losses else output["continuous"].sum() * 0.0


@torch.no_grad()
def validation_score(
    generator: ConditionalGenerator,
    loader: DataLoader,
    device: torch.device,
    samples: int,
    categorical_weight: float,
    seed: int,
) -> Dict[str, float]:
    generator.eval()
    rng = torch.Generator(device=device).manual_seed(int(seed))
    continuous_total = 0.0
    continuous_count = 0.0
    categorical_total = 0.0
    categorical_count = 0
    for features, continuous, continuous_mask, discrete, discrete_mask in loader:
        features = features.to(device, non_blocking=True)
        continuous = continuous.to(device, non_blocking=True)
        continuous_mask = continuous_mask.to(device, non_blocking=True)
        discrete = discrete.to(device, non_blocking=True)
        discrete_mask = discrete_mask.to(device, non_blocking=True)
        row_count = len(features)
        expanded = features[:, None, :].expand(-1, int(samples), -1).reshape(
            row_count * int(samples), -1
        )
        noise = torch.randn(
            (row_count * int(samples), generator.latent),
            device=device,
            generator=rng,
        )
        output = generator(expanded, noise)
        generated = output["continuous"].reshape(row_count, int(samples), 2)
        observed_error = torch.abs(generated - continuous[:, None, :]).mean(dim=1)
        pairwise = torch.abs(generated[:, :, None, :] - generated[:, None, :, :]).mean(
            dim=(1, 2)
        )
        crps = observed_error - 0.5 * pairwise
        continuous_total += float((crps * continuous_mask).sum())
        continuous_count += float(continuous_mask.sum())
        for index, name in enumerate(("atmosphere", "method")):
            valid = discrete_mask[:, index] > 0.5
            if bool(valid.any()):
                classes = output[name].shape[-1]
                probabilities = F.softmax(
                    output[name].reshape(row_count, int(samples), classes), dim=-1
                ).mean(dim=1)
                categorical_total += float(
                    F.nll_loss(
                        probabilities[valid].clamp_min(1e-8).log(),
                        discrete[valid, index],
                        reduction="sum",
                    )
                )
                categorical_count += int(valid.sum())
    continuous_crps = continuous_total / max(continuous_count, 1.0)
    categorical_nll = categorical_total / max(categorical_count, 1)
    return {
        "score": continuous_crps + float(categorical_weight) * categorical_nll,
        "continuous_crps": continuous_crps,
        "categorical_nll": categorical_nll,
    }


@torch.no_grad()
def generate_samples(
    generator: ConditionalGenerator,
    features: torch.Tensor,
    samples: int,
    batch_size: int,
    stats: dict[str, dict[str, float]],
    device: torch.device,
    seed: int,
    latent_scale: float,
    categorical_temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    generator.eval()
    rng = torch.Generator(device=device).manual_seed(int(seed))
    continuous_rows = []
    discrete_rows = []
    temperature = max(float(categorical_temperature), 1e-4)
    for start in range(0, len(features), int(batch_size)):
        batch = features[start : start + int(batch_size)].to(device)
        row_count = len(batch)
        expanded = batch[:, None, :].expand(-1, int(samples), -1).reshape(
            row_count * int(samples), -1
        )
        noise = torch.randn(
            (row_count * int(samples), generator.latent),
            device=device,
            generator=rng,
        ) * float(latent_scale)
        output = generator(expanded, noise)
        normalized = output["continuous"].reshape(row_count, int(samples), 2)
        atmosphere = torch.multinomial(
            F.softmax(output["atmosphere"] / temperature, dim=-1),
            1,
            generator=rng,
        ).reshape(row_count, int(samples))
        method = torch.multinomial(
            F.softmax(output["method"] / temperature, dim=-1),
            1,
            generator=rng,
        ).reshape(row_count, int(samples))
        continuous_rows.append(
            inverse_continuous(normalized.float().cpu().numpy(), stats)
        )
        discrete_rows.append(torch.stack([atmosphere, method], dim=-1).cpu().numpy())
    return np.concatenate(continuous_rows), np.concatenate(discrete_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a validation-only Stage3 conditional WGAN-GP.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--precursor_hidden", type=int, default=512)
    parser.add_argument("--latent", type=int, default=128)
    parser.add_argument("--generator_blocks", type=int, default=4)
    parser.add_argument("--critic_blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--generator_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=2e-4)
    parser.add_argument("--critic_steps", type=int, default=5)
    parser.add_argument("--gradient_penalty", type=float, default=10.0)
    parser.add_argument("--categorical_weight", type=float, default=1.0)
    parser.add_argument("--reconstruction_weight", type=float, default=0.05)
    parser.add_argument("--validation_samples", type=int, default=16)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--latent_scale", type=float, default=1.0)
    parser.add_argument("--categorical_temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    seed_everything(int(args.seed))
    started = time.time()
    input_dir = Path(args.input_dir).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    schema = json.loads((input_dir / "schema.json").read_text(encoding="utf-8"))
    train = load_pack(input_dir / "train.npz")
    val = load_pack(input_dir / "val.npz")
    train_continuous, val_continuous, stats = transform_continuous(train, val)
    structure_dim = int(np.asarray(train["x"]).shape[1])
    precursor_dim = int(np.asarray(train["y_set"]).shape[1])
    atmosphere_classes = len(schema["discrete_schema"]["atmosphere_coarse"]["vocab"])
    method_classes = len(schema["discrete_schema"]["reaction_method"]["vocab"])

    def make_dataset(pack: dict[str, np.ndarray], continuous: np.ndarray) -> TensorDataset:
        features = np.hstack([
            np.asarray(pack["x"], dtype=np.float32),
            np.asarray(pack["y_set"], dtype=np.float32),
        ]).astype(np.float32)
        return TensorDataset(
            torch.from_numpy(features),
            torch.from_numpy(continuous.astype(np.float32)),
            torch.from_numpy(np.asarray(pack["y_cond_continuous_mask"], dtype=np.float32)),
            torch.from_numpy(np.asarray(pack["y_cond_discrete"], dtype=np.int64)),
            torch.from_numpy(np.asarray(pack["y_cond_discrete_mask"], dtype=np.float32)),
        )

    train_dataset = make_dataset(train, train_continuous)
    val_dataset = make_dataset(val, val_continuous)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    generator = ConditionalGenerator(
        structure_dim,
        precursor_dim,
        int(args.hidden),
        int(args.precursor_hidden),
        int(args.latent),
        int(args.generator_blocks),
        float(args.dropout),
        atmosphere_classes,
        method_classes,
    ).to(device)
    critic = ConditionalCritic(
        structure_dim,
        precursor_dim,
        int(args.hidden),
        int(args.precursor_hidden),
        int(args.critic_blocks),
        float(args.dropout),
    ).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(128, min(int(args.batch_size), 512)),
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
    )
    generator_optimizer = torch.optim.AdamW(
        generator.parameters(), lr=float(args.generator_lr), betas=(0.0, 0.9), weight_decay=1e-5
    )
    critic_optimizer = torch.optim.AdamW(
        critic.parameters(), lr=float(args.critic_lr), betas=(0.0, 0.9), weight_decay=1e-5
    )
    best_score = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0
    training_log = []
    for epoch in range(1, int(args.epochs) + 1):
        generator.train()
        critic.train()
        critic_losses = []
        generator_losses = []
        for features, continuous, continuous_mask, discrete, discrete_mask in train_loader:
            features = features.to(device, non_blocking=True)
            continuous = continuous.to(device, non_blocking=True)
            continuous_mask = continuous_mask.to(device, non_blocking=True)
            discrete = discrete.to(device, non_blocking=True)
            discrete_mask = discrete_mask.to(device, non_blocking=True)
            full = continuous_mask.min(dim=1).values > 0.5
            if int(full.sum()) < 2:
                continue
            full_features = features[full]
            real = continuous[full]
            for _ in range(int(args.critic_steps)):
                critic_optimizer.zero_grad(set_to_none=True)
                noise = torch.randn((len(full_features), int(args.latent)), device=device)
                with torch.no_grad():
                    fake = generator(full_features, noise)["continuous"]
                real_score = critic(full_features, real)
                fake_score = critic(full_features, fake)
                penalty = gradient_penalty(critic, full_features, real, fake)
                critic_loss = fake_score.mean() - real_score.mean() + float(
                    args.gradient_penalty
                ) * penalty
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
                critic_optimizer.step()
                critic_losses.append(float(critic_loss.detach()))

            generator_optimizer.zero_grad(set_to_none=True)
            noise = torch.randn((len(features), int(args.latent)), device=device)
            output = generator(features, noise)
            adversarial = -critic(features[full], output["continuous"][full]).mean()
            supervised = categorical_loss(output, discrete, discrete_mask)
            reconstruction = F.smooth_l1_loss(output["continuous"][full], real)
            generator_loss = (
                adversarial
                + float(args.categorical_weight) * supervised
                + float(args.reconstruction_weight) * reconstruction
            )
            generator_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 5.0)
            generator_optimizer.step()
            generator_losses.append(float(generator_loss.detach()))

        validation = validation_score(
            generator,
            val_loader,
            device,
            int(args.validation_samples),
            float(args.categorical_weight),
            int(args.seed) + epoch * 1009,
        )
        row = {
            "epoch": epoch,
            "critic_loss": float(np.mean(critic_losses)),
            "generator_loss": float(np.mean(generator_losses)),
            **{f"val_{key}": value for key, value in validation.items()},
        }
        training_log.append(row)
        print(json.dumps(row), flush=True)
        if validation["score"] < best_score - 1e-4:
            best_score = float(validation["score"])
            best_epoch = epoch
            best_state = copy.deepcopy(generator.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= int(args.patience):
            break
    if best_state is None:
        raise RuntimeError("training produced no generator checkpoint")
    generator.load_state_dict(best_state)
    val_features = val_dataset.tensors[0]
    continuous_samples, discrete_samples = generate_samples(
        generator,
        val_features,
        int(args.samples),
        max(1, min(64, int(args.batch_size))),
        stats,
        device,
        int(args.seed) + 100000,
        float(args.latent_scale),
        float(args.categorical_temperature),
    )
    sample_path = run_dir / "val_samples.npz"
    np.savez_compressed(
        sample_path,
        continuous_samples=continuous_samples.astype(np.float32),
        discrete_samples=discrete_samples.astype(np.int16),
        sample_id=np.asarray(val["sample_id"]).astype(str),
    )
    checkpoint = {
        "state_dict": {name: value.cpu() for name, value in best_state.items()},
        "config": vars(args),
        "structure_dim": structure_dim,
        "precursor_dim": precursor_dim,
        "atmosphere_classes": atmosphere_classes,
        "method_classes": method_classes,
        "target_stats": stats,
        "best_epoch": best_epoch,
        "schema_version": schema["schema_version"],
    }
    torch.save(checkpoint, run_dir / "best_generator.pt")
    summary = {
        "model": "stage3_conditional_wgan_gp",
        "protocol": "validation_only_formula_disjoint_model_selection",
        "config": vars(args),
        "device": str(device),
        "parameter_count": {
            "generator": int(sum(value.numel() for value in generator.parameters())),
            "critic": int(sum(value.numel() for value in critic.parameters())),
        },
        "rows": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "train_full_continuous": int(
                (np.asarray(train["y_cond_continuous_mask"]) > 0.5).all(axis=1).sum()
            ),
        },
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "elapsed_seconds": time.time() - started,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "training_log": training_log,
        "artifacts": {
            "checkpoint": str(run_dir / "best_generator.pt"),
            "val_samples": str(sample_path),
        },
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
