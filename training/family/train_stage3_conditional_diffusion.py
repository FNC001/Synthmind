#!/usr/bin/env python3
"""Train a validation-only conditional diffusion model for synthesis conditions.

Continuous temperature/time targets use a DDPM noise objective. Categorical
atmosphere/method heads are sampled from the denoised state. The frozen test
split is never loaded.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

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


def cosine_schedule(timesteps: int, offset: float = 0.008) -> torch.Tensor:
    steps = torch.arange(timesteps + 1, dtype=torch.float64)
    values = torch.cos(((steps / timesteps + offset) / (1.0 + offset)) * math.pi / 2).square()
    alpha_bar = values / values[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1e-5, 0.999).float()


def sinusoidal_embedding(timestep: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    frequency = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=timestep.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = timestep.float()[:, None] * frequency[None, :]
    embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
    if embedding.shape[-1] < dimension:
        embedding = F.pad(embedding, (0, dimension - embedding.shape[-1]))
    return embedding


class ConditionalDiffusion(nn.Module):
    def __init__(
        self,
        structure_dim: int,
        precursor_dim: int,
        hidden: int,
        precursor_hidden: int,
        blocks: int,
        dropout: float,
        time_dim: int,
        atmosphere_classes: int,
        method_classes: int,
    ) -> None:
        super().__init__()
        self.structure_dim = int(structure_dim)
        self.time_dim = int(time_dim)
        structure_hidden = max(256, hidden // 2)
        self.structure_encoder = nn.Sequential(
            nn.Linear(structure_dim, structure_hidden),
            nn.LayerNorm(structure_hidden),
            nn.GELU(),
            nn.Linear(structure_hidden, structure_hidden),
            nn.GELU(),
        )
        self.precursor_encoder = nn.Sequential(
            nn.Linear(precursor_dim, precursor_hidden),
            nn.LayerNorm(precursor_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.context_in = nn.Sequential(
            nn.Linear(structure_hidden + precursor_hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.context_blocks = nn.Sequential(
            *[ResidualBlock(hidden, dropout) for _ in range(max(2, blocks // 2))]
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(time_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.noisy_encoder = nn.Sequential(nn.Linear(2, hidden), nn.GELU())
        self.denoise_in = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        self.denoise_blocks = nn.Sequential(
            *[ResidualBlock(hidden, dropout) for _ in range(blocks)]
        )
        self.noise_head = nn.Linear(hidden, 2)
        self.atmosphere_head = nn.Linear(hidden, atmosphere_classes)
        self.method_head = nn.Linear(hidden, method_classes)

    def encode_condition(self, features: torch.Tensor) -> torch.Tensor:
        structure = self.structure_encoder(features[:, : self.structure_dim])
        precursor = self.precursor_encoder(features[:, self.structure_dim :])
        return self.context_blocks(self.context_in(torch.cat([structure, precursor], dim=-1)))

    def denoise(
        self, condition: torch.Tensor, noisy: torch.Tensor, timestep: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        time = self.time_encoder(sinusoidal_embedding(timestep, self.time_dim))
        noisy_hidden = self.noisy_encoder(noisy)
        hidden = self.denoise_blocks(
            self.denoise_in(torch.cat([condition, time, noisy_hidden], dim=-1))
        )
        return {
            "noise": self.noise_head(hidden),
            "atmosphere": self.atmosphere_head(hidden),
            "method": self.method_head(hidden),
        }


def make_dataset(pack: dict[str, np.ndarray], continuous: np.ndarray) -> TensorDataset:
    features = np.hstack(
        [np.asarray(pack["x"], dtype=np.float32), np.asarray(pack["y_set"], dtype=np.float32)]
    ).astype(np.float32)
    return TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(continuous.astype(np.float32)),
        torch.from_numpy(np.asarray(pack["y_cond_continuous_mask"], dtype=np.float32)),
        torch.from_numpy(np.asarray(pack["y_cond_discrete"], dtype=np.int64)),
        torch.from_numpy(np.asarray(pack["y_cond_discrete_mask"], dtype=np.float32)),
    )


def diffusion_loss(
    model: ConditionalDiffusion,
    batch: list[torch.Tensor],
    alpha_bar: torch.Tensor,
    timesteps: int,
    categorical_weight: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    features, continuous, continuous_mask, discrete, discrete_mask = batch
    row_count = len(features)
    timestep = torch.randint(
        0, timesteps, (row_count,), device=features.device, generator=generator
    )
    noise = torch.randn(continuous.shape, device=continuous.device, generator=generator)
    selected = alpha_bar[timestep].unsqueeze(-1)
    clean = continuous * continuous_mask
    noisy = selected.sqrt() * clean + (1.0 - selected).sqrt() * noise
    condition = model.encode_condition(features)
    output = model.denoise(condition, noisy, timestep)
    continuous_loss = (
        (output["noise"] - noise).square() * continuous_mask
    ).sum() / continuous_mask.sum().clamp_min(1.0)
    categorical_losses = []
    for index, name in enumerate(("atmosphere", "method")):
        valid = discrete_mask[:, index] > 0.5
        if valid.any():
            categorical_losses.append(F.cross_entropy(output[name][valid], discrete[valid, index]))
    categorical_loss = (
        torch.stack(categorical_losses).mean()
        if categorical_losses
        else continuous_loss.new_zeros(())
    )
    total = continuous_loss + float(categorical_weight) * categorical_loss
    return total, {
        "continuous": float(continuous_loss.detach()),
        "categorical": float(categorical_loss.detach()),
    }


@torch.no_grad()
def validate(
    model: ConditionalDiffusion,
    loader: DataLoader,
    alpha_bar: torch.Tensor,
    timesteps: int,
    categorical_weight: float,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    totals = {"loss": 0.0, "continuous": 0.0, "categorical": 0.0}
    batches = 0
    for raw_batch in loader:
        batch = [value.to(device, non_blocking=True) for value in raw_batch]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss, parts = diffusion_loss(
                model, batch, alpha_bar, timesteps, categorical_weight, generator
            )
        totals["loss"] += float(loss)
        totals["continuous"] += parts["continuous"]
        totals["categorical"] += parts["categorical"]
        batches += 1
    return {key: value / max(1, batches) for key, value in totals.items()}


@torch.no_grad()
def generate_samples(
    model: ConditionalDiffusion,
    features: torch.Tensor,
    samples: int,
    row_batch_size: int,
    alpha_bar: torch.Tensor,
    sampling_steps: int,
    ddim_eta: float,
    categorical_temperature: float,
    stats: dict[str, dict[str, float]],
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    total_steps = len(alpha_bar)
    schedule = torch.linspace(total_steps - 1, 0, sampling_steps, device=device).round().long()
    schedule = torch.unique_consecutive(schedule)
    continuous_rows: list[np.ndarray] = []
    discrete_rows: list[np.ndarray] = []
    for start in range(0, len(features), row_batch_size):
        batch = features[start : start + row_batch_size].to(device)
        rows = len(batch)
        condition = model.encode_condition(batch)
        condition = condition[:, None, :].expand(-1, samples, -1).reshape(rows * samples, -1)
        current = torch.randn((rows * samples, 2), generator=generator, device=device)
        last_output: dict[str, torch.Tensor] | None = None
        for index, timestep_value in enumerate(schedule):
            timestep = torch.full(
                (rows * samples,), int(timestep_value), device=device, dtype=torch.long
            )
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                output = model.denoise(condition, current, timestep)
            predicted_noise = output["noise"].float()
            alpha_t = alpha_bar[int(timestep_value)].float()
            predicted_clean = (
                current - (1.0 - alpha_t).sqrt() * predicted_noise
            ) / alpha_t.sqrt().clamp_min(1e-5)
            predicted_clean = predicted_clean.clamp(-5.0, 5.0)
            if index + 1 == len(schedule):
                current = predicted_clean
            else:
                next_timestep = int(schedule[index + 1])
                alpha_next = alpha_bar[next_timestep].float()
                sigma = float(ddim_eta) * torch.sqrt(
                    ((1.0 - alpha_next) / (1.0 - alpha_t).clamp_min(1e-8))
                    * (1.0 - alpha_t / alpha_next).clamp_min(0.0)
                )
                direction = torch.sqrt((1.0 - alpha_next - sigma.square()).clamp_min(0.0))
                random_noise = torch.randn(
                    current.shape, generator=generator, device=device
                )
                current = (
                    alpha_next.sqrt() * predicted_clean
                    + direction * predicted_noise
                    + sigma * random_noise
                )
            last_output = output
        final_timestep = torch.zeros(rows * samples, device=device, dtype=torch.long)
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            last_output = model.denoise(condition, current, final_timestep)
        atmosphere = torch.multinomial(
            F.softmax(
                last_output["atmosphere"].float()
                / max(float(categorical_temperature), 1e-4),
                dim=-1,
            ),
            1,
            generator=generator,
        ).reshape(rows, samples)
        method = torch.multinomial(
            F.softmax(
                last_output["method"].float() / max(float(categorical_temperature), 1e-4),
                dim=-1,
            ),
            1,
            generator=generator,
        ).reshape(rows, samples)
        normalized = current.reshape(rows, samples, 2).cpu().numpy()
        continuous_rows.append(inverse_continuous(normalized, stats))
        discrete_rows.append(torch.stack([atmosphere, method], dim=-1).cpu().numpy())
    return np.concatenate(continuous_rows), np.concatenate(discrete_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--precursor_hidden", type=int, default=512)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--time_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--sampling_steps", type=int, default=50)
    parser.add_argument("--ddim_eta", type=float, default=0.8)
    parser.add_argument("--categorical_temperature", type=float, default=1.0)
    parser.add_argument("--categorical_weight", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    seed_everything(args.seed)
    started = time.time()
    input_dir = Path(args.input_dir).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    schema = json.loads((input_dir / "schema.json").read_text(encoding="utf-8"))
    train = load_pack(input_dir / "train.npz")
    val = load_pack(input_dir / "val.npz")
    train_continuous, val_continuous, stats = transform_continuous(train, val)
    train_dataset = make_dataset(train, train_continuous)
    val_dataset = make_dataset(val, val_continuous)
    structure_dim = int(np.asarray(train["x"]).shape[1])
    precursor_dim = int(np.asarray(train["y_set"]).shape[1])
    atmosphere_classes = len(schema["discrete_schema"]["atmosphere_coarse"]["vocab"])
    method_classes = len(schema["discrete_schema"]["reaction_method"]["vocab"])
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = ConditionalDiffusion(
        structure_dim,
        precursor_dim,
        args.hidden,
        args.precursor_hidden,
        args.blocks,
        args.dropout,
        args.time_dim,
        atmosphere_classes,
        method_classes,
    ).to(device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(args.batch_size, 512),
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
    )
    betas = cosine_schedule(args.timesteps).to(device)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0
    training_log: list[dict[str, float | int]] = []
    peak_cuda_memory = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = []
        train_cont = []
        train_cat = []
        for raw_batch in train_loader:
            batch = [value.to(device, non_blocking=True) for value in raw_batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                loss, parts = diffusion_loss(
                    model, batch, alpha_bar, args.timesteps, args.categorical_weight
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_total.append(float(loss.detach()))
            train_cont.append(parts["continuous"])
            train_cat.append(parts["categorical"])
        scheduler.step()
        validation = validate(
            model,
            val_loader,
            alpha_bar,
            args.timesteps,
            args.categorical_weight,
            device,
            args.seed + 100_000,
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_total)),
            "train_continuous": float(np.mean(train_cont)),
            "train_categorical": float(np.mean(train_cat)),
            **{f"val_{key}": value for key, value in validation.items()},
        }
        training_log.append(row)
        print(json.dumps(row), flush=True)
        if validation["loss"] < best_loss - 1e-4:
            best_loss = validation["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if device.type == "cuda":
            peak_cuda_memory = max(peak_cuda_memory, torch.cuda.max_memory_allocated() / 2**20)
        if bad_epochs >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    continuous_samples, discrete_samples = generate_samples(
        model,
        val_dataset.tensors[0],
        args.samples,
        max(1, min(32, args.batch_size)),
        alpha_bar,
        args.sampling_steps,
        args.ddim_eta,
        args.categorical_temperature,
        stats,
        device,
        args.seed + 1000,
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
    torch.save(checkpoint, run_dir / "best_model.pt")
    summary = {
        "model": "stage3_conditional_diffusion",
        "protocol": "validation_only_formula_disjoint_model_selection",
        "config": vars(args),
        "device": str(device),
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "rows": {"train": len(train_dataset), "val": len(val_dataset)},
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "elapsed_seconds": time.time() - started,
        "peak_cuda_memory_mb": peak_cuda_memory,
        "training_log": training_log,
        "artifacts": {
            "checkpoint": str(run_dir / "best_model.pt"),
            "val_samples": str(sample_path),
        },
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
