#!/usr/bin/env python3
"""Train a conditional VAE for mixed continuous/categorical synthesis conditions.

Model selection and sample export are validation-only.  The frozen test split is
never loaded, so this script is safe to use during architecture selection.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pack(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as pack:
        return {key: np.asarray(pack[key]) for key in pack.files}


def transform_continuous(
    train: dict[str, np.ndarray], val: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, float]]]:
    transformed: dict[str, np.ndarray] = {}
    for name, pack in (("train", train), ("val", val)):
        values = np.asarray(pack["y_cond_continuous_raw"], dtype=np.float32).copy()
        values[:, 1] = np.log1p(np.clip(values[:, 1], 0.0, None))
        transformed[name] = values
    mask = np.asarray(train["y_cond_continuous_mask"], dtype=np.float32) > 0.5
    stats: dict[str, dict[str, float]] = {}
    for index, name in enumerate(("temperature_c", "log1p_time_h")):
        observed = transformed["train"][mask[:, index], index]
        mean = float(observed.mean())
        std = max(float(observed.std()), 1e-6)
        stats[name] = {"mean": mean, "std": std}
        for split in transformed:
            transformed[split][:, index] = (transformed[split][:, index] - mean) / std
    return transformed["train"], transformed["val"], stats


def inverse_continuous(values: np.ndarray, stats: dict[str, dict[str, float]]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).copy()
    result[..., 0] = (
        result[..., 0] * stats["temperature_c"]["std"]
        + stats["temperature_c"]["mean"]
    )
    result[..., 1] = (
        result[..., 1] * stats["log1p_time_h"]["std"]
        + stats["log1p_time_h"]["mean"]
    )
    result[..., 1] = np.expm1(result[..., 1])
    result[..., 0] = np.clip(result[..., 0], 0.0, 2500.0)
    result[..., 1] = np.clip(result[..., 1], 0.0, 10000.0)
    return result


class ResidualBlock(nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden * 2)
        self.fc2 = nn.Linear(hidden * 2, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.fc2(self.dropout(F.gelu(self.fc1(self.norm(values)))))
        return values + self.dropout(hidden)


class HybridCVAE(nn.Module):
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
        self.structure_dim = int(structure_dim)
        self.atmosphere_classes = int(atmosphere_classes)
        self.method_classes = int(method_classes)
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
        self.condition_fusion = nn.Sequential(
            nn.Linear(structure_hidden + precursor_hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.condition_blocks = nn.Sequential(
            *[ResidualBlock(hidden, dropout) for _ in range(blocks)]
        )
        target_dim = 2 + 2 + atmosphere_classes + method_classes + 2
        self.posterior = nn.Sequential(
            nn.Linear(hidden + target_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            ResidualBlock(hidden, dropout),
            nn.Linear(hidden, latent * 2),
        )
        self.decoder_in = nn.Sequential(
            nn.Linear(hidden + latent, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        self.decoder_blocks = nn.Sequential(
            *[ResidualBlock(hidden, dropout) for _ in range(blocks)]
        )
        self.continuous_head = nn.Linear(hidden, 4)
        self.atmosphere_head = nn.Linear(hidden, atmosphere_classes)
        self.method_head = nn.Linear(hidden, method_classes)

    def encode_condition(self, features: torch.Tensor) -> torch.Tensor:
        structure = self.structure_encoder(features[:, : self.structure_dim])
        precursor = self.precursor_encoder(features[:, self.structure_dim :])
        return self.condition_blocks(
            self.condition_fusion(torch.cat([structure, precursor], dim=-1))
        )

    def target_context(
        self,
        continuous: torch.Tensor,
        continuous_mask: torch.Tensor,
        discrete: torch.Tensor,
        discrete_mask: torch.Tensor,
    ) -> torch.Tensor:
        atmosphere = F.one_hot(
            discrete[:, 0].clamp(0, self.atmosphere_classes - 1), self.atmosphere_classes
        ).to(continuous.dtype)
        method = F.one_hot(
            discrete[:, 1].clamp(0, self.method_classes - 1), self.method_classes
        ).to(continuous.dtype)
        return torch.cat(
            [
                continuous * continuous_mask,
                continuous_mask,
                atmosphere * discrete_mask[:, :1],
                method * discrete_mask[:, 1:2],
                discrete_mask,
            ],
            dim=-1,
        )

    def decode(self, condition: torch.Tensor, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.decoder_blocks(self.decoder_in(torch.cat([condition, latent], dim=-1)))
        continuous = self.continuous_head(hidden)
        return {
            "continuous_mean": continuous[:, :2],
            "continuous_log_scale": continuous[:, 2:].clamp(-4.0, 2.0),
            "atmosphere": self.atmosphere_head(hidden),
            "method": self.method_head(hidden),
        }

    def forward(
        self,
        features: torch.Tensor,
        continuous: torch.Tensor,
        continuous_mask: torch.Tensor,
        discrete: torch.Tensor,
        discrete_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        condition = self.encode_condition(features)
        posterior = self.posterior(
            torch.cat(
                [
                    condition,
                    self.target_context(continuous, continuous_mask, discrete, discrete_mask),
                ],
                dim=-1,
            )
        )
        mean, log_variance = posterior.chunk(2, dim=-1)
        log_variance = log_variance.clamp(-8.0, 5.0)
        latent = mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)
        output = self.decode(condition, latent)
        output["latent_mean"] = mean
        output["latent_log_variance"] = log_variance
        return output


def loss_components(
    output: dict[str, torch.Tensor],
    continuous: torch.Tensor,
    continuous_mask: torch.Tensor,
    discrete: torch.Tensor,
    discrete_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    scale = torch.exp(output["continuous_log_scale"])
    gaussian = 0.5 * ((continuous - output["continuous_mean"]) / scale).square()
    gaussian = gaussian + output["continuous_log_scale"]
    continuous_loss = (gaussian * continuous_mask).sum() / continuous_mask.sum().clamp_min(1.0)
    discrete_losses = []
    for index, name in enumerate(("atmosphere", "method")):
        valid = discrete_mask[:, index] > 0.5
        if valid.any():
            discrete_losses.append(F.cross_entropy(output[name][valid], discrete[valid, index]))
    discrete_loss = torch.stack(discrete_losses).mean()
    mean = output["latent_mean"]
    log_variance = output["latent_log_variance"]
    kl = -0.5 * (1.0 + log_variance - mean.square() - log_variance.exp()).sum(dim=-1).mean()
    return {"continuous": continuous_loss, "discrete": discrete_loss, "kl": kl}


@torch.no_grad()
def validation_loss(
    model: HybridCVAE, loader: DataLoader, device: torch.device, beta: float
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "continuous": 0.0, "discrete": 0.0, "kl": 0.0}
    batches = 0
    for batch in loader:
        values = [item.to(device, non_blocking=True) for item in batch]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(*values)
            parts = loss_components(output, *values[1:])
            loss = parts["continuous"] + parts["discrete"] + beta * parts["kl"]
        totals["loss"] += float(loss)
        for name in parts:
            totals[name] += float(parts[name])
        batches += 1
    return {name: value / max(1, batches) for name, value in totals.items()}


@torch.no_grad()
def generate_samples(
    model: HybridCVAE,
    features: torch.Tensor,
    samples: int,
    batch_size: int,
    latent_dim: int,
    stats: dict[str, dict[str, float]],
    device: torch.device,
    seed: int,
    latent_scale: float = 1.0,
    continuous_noise_scale: float = 1.0,
    categorical_temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    continuous_rows: list[np.ndarray] = []
    discrete_rows: list[np.ndarray] = []
    for start in range(0, len(features), batch_size):
        batch = features[start : start + batch_size].to(device)
        condition = model.encode_condition(batch)
        row_count = len(batch)
        condition = condition[:, None, :].expand(-1, samples, -1).reshape(row_count * samples, -1)
        latent = torch.randn(
            (row_count * samples, latent_dim), generator=generator, device=device
        ) * float(latent_scale)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model.decode(condition, latent)
        scale = torch.exp(output["continuous_log_scale"].float()) * float(continuous_noise_scale)
        normalized = output["continuous_mean"].float() + scale * torch.randn(
            scale.shape, generator=generator, device=device
        )
        normalized = normalized.reshape(row_count, samples, 2).cpu().numpy()
        atmosphere = torch.multinomial(
            F.softmax(
                output["atmosphere"].float() / max(float(categorical_temperature), 1e-4),
                dim=-1,
            ),
            1,
            generator=generator,
        ).reshape(row_count, samples)
        method = torch.multinomial(
            F.softmax(
                output["method"].float() / max(float(categorical_temperature), 1e-4),
                dim=-1,
            ),
            1,
            generator=generator,
        ).reshape(row_count, samples)
        continuous_rows.append(inverse_continuous(normalized, stats))
        discrete_rows.append(torch.stack([atmosphere, method], dim=-1).cpu().numpy())
    return np.concatenate(continuous_rows), np.concatenate(discrete_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a validation-only Stage3 hybrid conditional VAE.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--precursor_hidden", type=int, default=512)
    parser.add_argument("--latent", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--kl_beta", type=float, default=0.02)
    parser.add_argument("--kl_warmup_epochs", type=int, default=30)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    seed_everything(args.seed)
    started = time.time()
    input_dir = Path(args.input_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    schema = json.loads((input_dir / "schema.json").read_text(encoding="utf-8"))
    train = load_pack(input_dir / "train.npz")
    val = load_pack(input_dir / "val.npz")
    train_cont, val_cont, stats = transform_continuous(train, val)
    structure_dim = int(np.asarray(train["x"]).shape[1])
    precursor_dim = int(np.asarray(train["y_set"]).shape[1])
    atmosphere_classes = len(schema["discrete_schema"]["atmosphere_coarse"]["vocab"])
    method_classes = len(schema["discrete_schema"]["reaction_method"]["vocab"])

    def dataset(pack: dict[str, np.ndarray], continuous: np.ndarray) -> TensorDataset:
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

    train_dataset = dataset(train, train_cont)
    val_dataset = dataset(val, val_cont)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = HybridCVAE(
        structure_dim,
        precursor_dim,
        args.hidden,
        args.precursor_hidden,
        args.latent,
        args.blocks,
        args.dropout,
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0
    training_log = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        beta = args.kl_beta * min(1.0, epoch / max(1, args.kl_warmup_epochs))
        train_losses = []
        for batch in train_loader:
            values = [item.to(device, non_blocking=True) for item in batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(*values)
                parts = loss_components(output, *values[1:])
                loss = parts["continuous"] + parts["discrete"] + beta * parts["kl"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        scheduler.step()
        # The training beta is annealed, but checkpoint scores must remain
        # comparable across epochs.  Always select with the final target beta.
        validation = validation_loss(model, val_loader, device, float(args.kl_beta))
        row = {
            "epoch": epoch,
            "train_beta": beta,
            "validation_beta": float(args.kl_beta),
            "train_loss": float(np.mean(train_losses)),
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
        if bad_epochs >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    val_features = val_dataset.tensors[0]
    continuous_samples, discrete_samples = generate_samples(
        model,
        val_features,
        args.samples,
        max(1, min(64, args.batch_size)),
        args.latent,
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
        "model": "stage3_hybrid_conditional_vae",
        "protocol": "validation_only_formula_disjoint_model_selection",
        "config": vars(args),
        "device": str(device),
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "rows": {"train": len(train_dataset), "val": len(val_dataset)},
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "elapsed_seconds": time.time() - started,
        "training_log": training_log,
        "artifacts": {"checkpoint": str(run_dir / "best_model.pt"), "val_samples": str(sample_path)},
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
