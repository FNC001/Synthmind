#!/usr/bin/env python3
"""Train a conditional normalizing flow for mixed synthesis conditions.

The flow models the joint temperature/log-time distribution when both fields
are observed. An auxiliary diagonal Gaussian retains partially observed rows,
and separate categorical heads model atmosphere and synthesis method. Model
selection is validation-only; the frozen test split is never opened.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict

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


class ConditionalAffineCoupling(nn.Module):
    def __init__(
        self,
        context_dim: int,
        hidden: int,
        mask: tuple[float, float],
        dropout: float,
    ) -> None:
        super().__init__()
        self.register_buffer("mask", torch.tensor(mask, dtype=torch.float32))
        self.network = nn.Sequential(
            nn.Linear(context_dim + 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )

    def parameters_for(self, values: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masked = values * self.mask
        scale, shift = self.network(torch.cat([masked, context], dim=-1)).chunk(2, dim=-1)
        active = 1.0 - self.mask
        scale = 2.0 * torch.tanh(scale / 2.0) * active
        shift = shift * active
        return scale, shift

    def to_base(self, values: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale, shift = self.parameters_for(values, context)
        transformed = values * self.mask + (1.0 - self.mask) * (values - shift) * torch.exp(-scale)
        return transformed, -scale.sum(dim=-1)

    def from_base(self, values: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        scale, shift = self.parameters_for(values, context)
        return values * self.mask + (1.0 - self.mask) * (values * torch.exp(scale) + shift)


class ConditionalSynthesisFlow(nn.Module):
    def __init__(
        self,
        structure_dim: int,
        precursor_dim: int,
        hidden: int,
        precursor_hidden: int,
        context_blocks: int,
        flow_layers: int,
        coupling_hidden: int,
        dropout: float,
        atmosphere_classes: int,
        method_classes: int,
    ) -> None:
        super().__init__()
        self.structure_dim = int(structure_dim)
        self.context_dim = int(hidden)
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
        self.context_fusion = nn.Sequential(
            nn.Linear(structure_hidden + precursor_hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.context_blocks = nn.Sequential(
            *[ResidualBlock(hidden, dropout) for _ in range(context_blocks)]
        )
        self.flow = nn.ModuleList([
            ConditionalAffineCoupling(
                hidden,
                coupling_hidden,
                (1.0, 0.0) if layer % 2 == 0 else (0.0, 1.0),
                dropout,
            )
            for layer in range(flow_layers)
        ])
        self.partial_gaussian_head = nn.Linear(hidden, 4)
        self.atmosphere_head = nn.Linear(hidden, atmosphere_classes)
        self.method_head = nn.Linear(hidden, method_classes)

    def encode_context(self, features: torch.Tensor) -> torch.Tensor:
        structure = self.structure_encoder(features[:, : self.structure_dim])
        precursor = self.precursor_encoder(features[:, self.structure_dim :])
        return self.context_blocks(
            self.context_fusion(torch.cat([structure, precursor], dim=-1))
        )

    def to_base(self, values: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(len(values), dtype=values.dtype, device=values.device)
        current = values
        for layer in self.flow:
            current, contribution = layer.to_base(current, context)
            log_det = log_det + contribution
        return current, log_det

    def sample_continuous(
        self,
        context: torch.Tensor,
        generator: torch.Generator,
        base_scale: float,
    ) -> torch.Tensor:
        current = torch.randn(
            (len(context), 2), device=context.device, generator=generator
        ) * float(base_scale)
        for layer in reversed(self.flow):
            current = layer.from_base(current, context)
        return current


def loss_components(
    model: ConditionalSynthesisFlow,
    features: torch.Tensor,
    continuous: torch.Tensor,
    continuous_mask: torch.Tensor,
    discrete: torch.Tensor,
    discrete_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    context = model.encode_context(features)
    full = continuous_mask.min(dim=1).values > 0.5
    if bool(full.any()):
        base, log_det = model.to_base(continuous[full], context[full])
        base_nll = 0.5 * (base.square() + math.log(2.0 * math.pi)).sum(dim=-1)
        flow_nll = (base_nll - log_det).mean()
    else:
        flow_nll = context.sum() * 0.0

    gaussian = model.partial_gaussian_head(context)
    mean, log_scale = gaussian[:, :2], gaussian[:, 2:].clamp(-4.0, 2.0)
    diagonal_nll = 0.5 * ((continuous - mean) * torch.exp(-log_scale)).square() + log_scale
    partial_nll = (diagonal_nll * continuous_mask).sum() / continuous_mask.sum().clamp_min(1.0)

    categorical_losses = []
    for index, logits in enumerate((model.atmosphere_head(context), model.method_head(context))):
        valid = discrete_mask[:, index] > 0.5
        if bool(valid.any()):
            categorical_losses.append(F.cross_entropy(logits[valid], discrete[valid, index]))
    discrete_loss = (
        torch.stack(categorical_losses).mean() if categorical_losses else context.sum() * 0.0
    )
    return {
        "flow": flow_nll,
        "partial": partial_nll,
        "discrete": discrete_loss,
        "full_rows": full.to(torch.float32).sum(),
    }


@torch.no_grad()
def validation_loss(
    model: ConditionalSynthesisFlow,
    loader: DataLoader,
    device: torch.device,
    partial_weight: float,
) -> Dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "flow": 0.0, "partial": 0.0, "discrete": 0.0}
    batches = 0
    for batch in loader:
        values = [item.to(device, non_blocking=True) for item in batch]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            parts = loss_components(model, *values)
            loss = parts["flow"] + float(partial_weight) * parts["partial"] + parts["discrete"]
        totals["loss"] += float(loss)
        for key in ("flow", "partial", "discrete"):
            totals[key] += float(parts[key])
        batches += 1
    return {key: value / max(1, batches) for key, value in totals.items()}


@torch.no_grad()
def generate_samples(
    model: ConditionalSynthesisFlow,
    features: torch.Tensor,
    samples: int,
    batch_size: int,
    stats: dict[str, dict[str, float]],
    device: torch.device,
    seed: int,
    base_scale: float,
    categorical_temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(int(seed))
    continuous_rows = []
    discrete_rows = []
    temperature = max(float(categorical_temperature), 1e-4)
    for start in range(0, len(features), int(batch_size)):
        batch = features[start : start + int(batch_size)].to(device)
        context = model.encode_context(batch)
        row_count = len(context)
        expanded = context[:, None, :].expand(-1, int(samples), -1).reshape(
            row_count * int(samples), -1
        )
        normalized = model.sample_continuous(expanded, generator, float(base_scale))
        atmosphere = torch.multinomial(
            F.softmax(model.atmosphere_head(expanded).float() / temperature, dim=-1),
            1,
            generator=generator,
        ).reshape(row_count, int(samples))
        method = torch.multinomial(
            F.softmax(model.method_head(expanded).float() / temperature, dim=-1),
            1,
            generator=generator,
        ).reshape(row_count, int(samples))
        continuous_rows.append(
            inverse_continuous(
                normalized.reshape(row_count, int(samples), 2).float().cpu().numpy(), stats
            )
        )
        discrete_rows.append(torch.stack([atmosphere, method], dim=-1).cpu().numpy())
    return np.concatenate(continuous_rows), np.concatenate(discrete_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a validation-only conditional Stage3 flow.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--precursor_hidden", type=int, default=512)
    parser.add_argument("--context_blocks", type=int, default=4)
    parser.add_argument("--flow_layers", type=int, default=12)
    parser.add_argument("--coupling_hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--partial_weight", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--base_scale", type=float, default=1.0)
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
    model = ConditionalSynthesisFlow(
        structure_dim,
        precursor_dim,
        int(args.hidden),
        int(args.precursor_hidden),
        int(args.context_blocks),
        int(args.flow_layers),
        int(args.coupling_hidden),
        float(args.dropout),
        atmosphere_classes,
        method_classes,
    ).to(device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(512, int(args.batch_size)),
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.epochs))
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0
    training_log = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            values = [item.to(device, non_blocking=True) for item in batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                parts = loss_components(model, *values)
                loss = parts["flow"] + float(args.partial_weight) * parts["partial"] + parts["discrete"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        scheduler.step()
        validation = validation_loss(model, val_loader, device, float(args.partial_weight))
        row = {
            "epoch": int(epoch),
            "train_loss": float(np.mean(train_losses)),
            **{f"val_{key}": float(value) for key, value in validation.items()},
        }
        training_log.append(row)
        print(json.dumps(row), flush=True)
        if validation["loss"] < best_loss - 1e-4:
            best_loss = float(validation["loss"])
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= int(args.patience):
            break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    continuous_samples, discrete_samples = generate_samples(
        model,
        val_dataset.tensors[0],
        int(args.samples),
        max(1, min(64, int(args.batch_size))),
        stats,
        device,
        int(args.seed) + 1000,
        float(args.base_scale),
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
    torch.save(checkpoint, run_dir / "best_model.pt")
    report: Dict[str, Any] = {
        "model": "stage3_conditional_affine_normalizing_flow",
        "protocol": "validation_only_formula_disjoint_model_selection",
        "config": vars(args),
        "device": str(device),
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "rows": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "train_both_continuous": int(
                np.sum(np.asarray(train["y_cond_continuous_mask"]).min(axis=1) > 0.5)
            ),
        },
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "elapsed_seconds": time.time() - started,
        "training_log": training_log,
        "artifacts": {
            "checkpoint": str(run_dir / "best_model.pt"),
            "val_samples": str(sample_path),
        },
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
