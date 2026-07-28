#!/usr/bin/env python3
"""Conditional discrete denoising diffusion for variable-length precursor sets.

The forward process removes true set members and injects frequency-smoothed
decoys.  The reverse model predicts the clean set and is sampled repeatedly to
produce a diverse candidate slate.  This is intentionally a set model: label
order never enters either training or decoding.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.family.train_stage2_listwise_ranker import (  # noqa: E402
    ResidualBlock,
    precursor_formula_features,
)


SetKey = Tuple[int, ...]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sinusoidal_embedding(timesteps: torch.Tensor, width: int) -> torch.Tensor:
    half = width // 2
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = timesteps.float()[:, None] * frequencies[None, :]
    values = torch.cat([angles.sin(), angles.cos()], dim=-1)
    if values.shape[1] < width:
        values = F.pad(values, (0, width - values.shape[1]))
    return values


class DenoisingBlock(nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.film = nn.Linear(hidden, hidden * 2)
        self.fc1 = nn.Linear(hidden, hidden * 4)
        self.fc2 = nn.Linear(hidden * 4, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(conditioning).chunk(2, dim=-1)
        hidden = self.norm(value) * (1.0 + 0.1 * scale) + shift
        hidden = self.fc2(F.silu(self.fc1(hidden)))
        return value + self.dropout(hidden)


class DiscreteSetDiffusion(nn.Module):
    def __init__(
        self,
        x_dim: int,
        n_labels: int,
        max_set_len: int,
        hidden: int,
        query_blocks: int,
        denoise_blocks: int,
        dropout: float,
        label_chemistry: torch.Tensor,
    ) -> None:
        super().__init__()
        self.x_dim = int(x_dim)
        self.n_labels = int(n_labels)
        self.max_set_len = int(max_set_len)
        self.hidden = int(hidden)
        self.query_input = nn.Sequential(
            nn.Linear(x_dim, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
        )
        self.query_blocks = nn.Sequential(
            *[ResidualBlock(hidden, dropout) for _ in range(query_blocks)]
        )
        self.query_norm = nn.LayerNorm(hidden)
        self.null_query = nn.Parameter(torch.zeros(hidden))
        self.label_embedding = nn.Embedding(n_labels, hidden)
        self.register_buffer("label_chemistry", label_chemistry.float(), persistent=True)
        self.label_chemistry_encoder = nn.Sequential(
            nn.Linear(label_chemistry.shape[1], hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.set_context = nn.Sequential(
            nn.Linear(hidden + 2, hidden), nn.LayerNorm(hidden), nn.SiLU()
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.SiLU(), nn.Linear(hidden * 2, hidden)
        )
        self.conditioning = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden), nn.SiLU()
        )
        self.state_input = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden), nn.SiLU()
        )
        self.denoise_blocks = nn.ModuleList(
            [DenoisingBlock(hidden, dropout) for _ in range(denoise_blocks)]
        )
        self.output_norm = nn.LayerNorm(hidden)
        self.label_bias = nn.Parameter(torch.zeros(n_labels))
        self.untied_head = nn.Linear(hidden, n_labels, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(12.0)))
        self.cardinality_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, max_set_len)
        )

    def label_representations(self) -> torch.Tensor:
        return self.label_embedding.weight + self.label_chemistry_encoder(self.label_chemistry)

    def encode_query(self, x: torch.Tensor) -> torch.Tensor:
        return self.query_norm(self.query_blocks(self.query_input(x)))

    def cardinality_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.cardinality_head(self.encode_query(x))

    def forward(
        self,
        x: torch.Tensor,
        noisy_set: torch.Tensor,
        timesteps: torch.Tensor,
        condition_keep: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.encode_query(x)
        cardinality_logits = self.cardinality_head(query).float()
        if condition_keep is not None:
            keep = condition_keep.to(query.dtype)[:, None]
            query = keep * query + (1.0 - keep) * self.null_query[None, :]
        labels = self.label_representations()
        counts = noisy_set.sum(dim=1, keepdim=True)
        pooled = noisy_set @ labels / counts.clamp_min(1.0)
        context = self.set_context(
            torch.cat(
                [
                    pooled,
                    counts / max(self.max_set_len, 1),
                    (counts > 0).to(pooled.dtype),
                ],
                dim=1,
            )
        )
        time = self.time_encoder(sinusoidal_embedding(timesteps, self.hidden))
        conditioning = self.conditioning(torch.cat([query, time], dim=1))
        state = self.state_input(torch.cat([query, context], dim=1))
        for block in self.denoise_blocks:
            state = block(state, conditioning)
        state = self.output_norm(state)
        normalized_state = F.normalize(state.float(), dim=1)
        normalized_labels = F.normalize(labels.float(), dim=1)
        scale = self.logit_scale.exp().clamp(1.0, 100.0)
        logits = scale * (normalized_state @ normalized_labels.t())
        logits = logits + self.untied_head(state).float() + self.label_bias.float()
        return logits, cardinality_logits


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    ema_values = dict(ema.named_parameters())
    for name, parameter in model.named_parameters():
        ema_values[name].mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)
    ema_buffers = dict(ema.named_buffers())
    for name, buffer in model.named_buffers():
        ema_buffers[name].copy_(buffer)


def corrupt_sets(
    clean: torch.Tensor,
    timesteps: torch.Tensor,
    diffusion_steps: int,
    base_probabilities: torch.Tensor,
    noise_multiplier: float,
) -> torch.Tensor:
    phase = timesteps.float() / float(diffusion_steps)
    keep_probability = torch.cos(phase * math.pi / 2.0).square()[:, None]
    retained = clean * (torch.rand_like(clean) < keep_probability).to(clean.dtype)
    expected_decoys = (
        (1.0 - keep_probability)
        * clean.sum(dim=1, keepdim=True).clamp_min(1.0)
        * float(noise_multiplier)
    )
    decoy_probability = (expected_decoys * base_probabilities[None, :]).clamp(max=0.35)
    decoys = (torch.rand_like(clean) < decoy_probability).to(clean.dtype)
    return torch.maximum(retained, decoys * (1.0 - clean))


def denoising_loss(
    logits: torch.Tensor,
    cardinality_logits: torch.Tensor,
    targets: torch.Tensor,
    lengths_zero_based: torch.Tensor,
    negative_weight: float,
    distribution_weight: float,
    cardinality_weight: float,
    hard_negatives: int,
    positive_weights: torch.Tensor,
) -> tuple[torch.Tensor, Dict[str, float]]:
    weighted_targets = targets * positive_weights[None, :]
    positives = weighted_targets.sum(dim=1).clamp_min(1.0)
    positive_loss = (F.softplus(-logits) * weighted_targets).sum(dim=1) / positives
    negative_values = F.softplus(logits).masked_fill(targets > 0.5, -torch.inf)
    negative_count = min(int(hard_negatives), logits.shape[1] - 1)
    hard_negative_loss = torch.topk(negative_values, k=negative_count, dim=1).values.mean(dim=1)
    reconstruction = (positive_loss + float(negative_weight) * hard_negative_loss).mean()
    target_distribution = weighted_targets / positives[:, None]
    distribution = -(target_distribution * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    cardinality = F.cross_entropy(cardinality_logits, lengths_zero_based)
    total = (
        reconstruction
        + float(distribution_weight) * distribution
        + float(cardinality_weight) * cardinality
    )
    return total, {
        "reconstruction": float(reconstruction.detach().cpu()),
        "positive": float(positive_loss.mean().detach().cpu()),
        "hard_negative": float(hard_negative_loss.mean().detach().cpu()),
        "distribution": float(distribution.detach().cpu()),
        "cardinality": float(cardinality.detach().cpu()),
    }


def _gumbel_like(values: torch.Tensor) -> torch.Tensor:
    uniform = torch.rand_like(values).clamp_(1e-6, 1.0 - 1e-6)
    return -torch.log(-torch.log(uniform))


@torch.no_grad()
def sample_batch(
    model: DiscreteSetDiffusion,
    x: torch.Tensor,
    base_probabilities: torch.Tensor,
    samples: int,
    sampling_steps: int,
    guidance_scale: float,
    temperature_start: float,
    temperature_end: float,
    length_temperature: float,
    trajectory_batch: int,
    diffusion_steps: int,
) -> tuple[List[List[SetKey]], List[List[float]]]:
    model.eval()
    device = x.device
    all_candidates: List[Dict[SetKey, List[float]]] = [dict() for _ in range(len(x))]
    cardinality_logits = model.cardinality_logits(x).float()
    length_probs = F.softmax(cardinality_logits / float(length_temperature), dim=1)
    for sample_start in range(0, int(samples), int(trajectory_batch)):
        current_samples = min(int(trajectory_batch), int(samples) - sample_start)
        expanded_x = x[:, None, :].expand(-1, current_samples, -1).reshape(-1, x.shape[1])
        expanded_length_probs = length_probs[:, None, :].expand(-1, current_samples, -1).reshape(
            -1, model.max_set_len
        )
        lengths = torch.multinomial(expanded_length_probs, 1).squeeze(1) + 1
        initial_scores = torch.log(base_probabilities.clamp_min(1e-12))[None, :].expand(
            len(expanded_x), -1
        )
        initial_scores = initial_scores + _gumbel_like(initial_scores)
        initial_labels = torch.topk(initial_scores, k=model.max_set_len, dim=1).indices
        noisy = torch.zeros(
            (len(expanded_x), model.n_labels), dtype=expanded_x.dtype, device=device
        )
        ranks = torch.arange(model.max_set_len, device=device)[None, :]
        active = ranks < lengths[:, None]
        rows = torch.arange(len(expanded_x), device=device)[:, None].expand_as(initial_labels)[active]
        noisy[rows, initial_labels[active]] = 1.0
        final_logits = None
        for reverse_index in range(int(sampling_steps), 0, -1):
            model_timestep = max(
                1,
                int(
                    round(
                        reverse_index
                        * float(diffusion_steps)
                        / max(int(sampling_steps), 1)
                    )
                ),
            )
            timesteps = torch.full(
                (len(expanded_x),), model_timestep, dtype=torch.long, device=device
            )
            conditional, _ = model(
                expanded_x,
                noisy,
                timesteps,
                torch.ones(len(expanded_x), device=device),
            )
            if float(guidance_scale) != 0.0:
                unconditional, _ = model(
                    expanded_x,
                    noisy,
                    timesteps,
                    torch.zeros(len(expanded_x), device=device),
                )
                logits = conditional + float(guidance_scale) * (conditional - unconditional)
            else:
                logits = conditional
            progress = 1.0 - (reverse_index - 1) / max(int(sampling_steps) - 1, 1)
            temperature = float(temperature_start) * (1.0 - progress) + float(
                temperature_end
            ) * progress
            perturbed = logits / max(temperature, 1e-4) + _gumbel_like(logits)
            labels = torch.topk(perturbed, k=model.max_set_len, dim=1).indices
            noisy.zero_()
            rows = torch.arange(len(expanded_x), device=device)[:, None].expand_as(labels)[active]
            noisy[rows, labels[active]] = 1.0
            final_logits = logits
        if final_logits is None:
            raise RuntimeError("sampling_steps must be positive")
        log_label_probs = F.log_softmax(final_logits, dim=1)
        length_log_probs = F.log_softmax(
            torch.log(expanded_length_probs.clamp_min(1e-12)), dim=1
        )
        noisy_np = noisy.cpu().numpy()
        member_scores = (log_label_probs * noisy).sum(dim=1) / lengths.float()
        scores = member_scores + length_log_probs.gather(1, (lengths - 1)[:, None]).squeeze(1)
        scores_np = scores.float().cpu().numpy()
        for flat_index, (mask, score) in enumerate(zip(noisy_np, scores_np)):
            row_index = flat_index // current_samples
            candidate = tuple(np.flatnonzero(mask > 0.5).astype(int).tolist())
            if candidate:
                all_candidates[row_index].setdefault(candidate, []).append(float(score))
    rows: List[List[SetKey]] = []
    score_rows: List[List[float]] = []
    for candidate_scores in all_candidates:
        aggregated = [
            (float(np.logaddexp.reduce(values)), candidate)
            for candidate, values in candidate_scores.items()
        ]
        aggregated.sort(key=lambda item: (-item[0], item[1]))
        score_rows.append([score for score, _ in aggregated])
        rows.append([candidate for _, candidate in aggregated])
    return rows, score_rows


@torch.no_grad()
def generate_candidates(
    model: DiscreteSetDiffusion,
    x: np.ndarray,
    base_probabilities: torch.Tensor,
    device: torch.device,
    batch_size: int,
    **sampling: Any,
) -> tuple[List[List[SetKey]], List[List[float]]]:
    rows: List[List[SetKey]] = []
    score_rows: List[List[float]] = []
    for start in range(0, len(x), int(batch_size)):
        batch = torch.from_numpy(x[start : start + int(batch_size)]).to(
            device, non_blocking=True
        )
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            batch_rows, batch_scores = sample_batch(
                model, batch, base_probabilities, **sampling
            )
        rows.extend(batch_rows)
        score_rows.extend(batch_scores)
    return rows, score_rows


def exact_metrics(
    targets: Sequence[SetKey], candidates: Sequence[Sequence[SetKey]]
) -> Dict[str, float]:
    return {
        f"exact_hit@{k}": float(
            np.mean([target in set(row[:k]) for target, row in zip(targets, candidates)])
        )
        for k in (1, 3, 5, 10, 20, 50, 100, 200)
    }


def write_candidates(
    path: Path, rows: Sequence[Sequence[SetKey]], scores: Sequence[Sequence[float]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row_index, (values, row_scores) in enumerate(zip(rows, scores)):
            handle.write(
                json.dumps(
                    {
                        "row_index": row_index,
                        "candidate_label_ids": [list(value) for value in values],
                        "scores": list(row_scores),
                    }
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a chemistry-aware conditional discrete set diffusion model."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--query_blocks", type=int, default=4)
    parser.add_argument("--denoise_blocks", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--noise_multiplier", type=float, default=2.0)
    parser.add_argument("--condition_dropout", type=float, default=0.10)
    parser.add_argument("--negative_weight", type=float, default=0.35)
    parser.add_argument("--distribution_weight", type=float, default=0.35)
    parser.add_argument("--cardinality_weight", type=float, default=0.60)
    parser.add_argument("--hard_negatives", type=int, default=64)
    parser.add_argument("--family_balance_power", type=float, default=0.0)
    parser.add_argument("--group_balance_power", type=float, default=0.0)
    parser.add_argument("--rare_label_power", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--grad_clip", type=float, default=2.0)
    parser.add_argument("--eval_samples", type=int, default=48)
    parser.add_argument("--final_samples", type=int, default=256)
    parser.add_argument("--sampling_steps", type=int, default=16)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--temperature_start", type=float, default=1.25)
    parser.add_argument("--temperature_end", type=float, default=0.22)
    parser.add_argument("--length_temperature", type=float, default=0.75)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--trajectory_batch", type=int, default=32)
    parser.add_argument("--selection_metric", default="exact_hit@100")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    seed_everything(int(args.seed))
    input_dir = Path(args.input_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    train_pack = np.load(input_dir / "train.npz", allow_pickle=True)
    val_pack = np.load(input_dir / "val.npz", allow_pickle=True)
    x_train = np.asarray(train_pack["x"], dtype=np.float32)
    y_train = np.asarray(train_pack["y_multi_hot"], dtype=np.float32)
    x_val = np.asarray(val_pack["x"], dtype=np.float32)
    y_val = np.asarray(val_pack["y_multi_hot"], dtype=np.float32)
    lengths_train = np.asarray(train_pack["set_len"], dtype=np.int64)
    max_set_len = int(max(lengths_train.max(), np.asarray(val_pack["set_len"]).max()))
    targets_val = [tuple(np.flatnonzero(row > 0.5).tolist()) for row in y_val]
    precursor_names = json.loads(
        (input_dir / "precursor_names.json").read_text(encoding="utf-8")
    )
    label_chemistry = torch.from_numpy(precursor_formula_features(precursor_names))
    raw_frequency = y_train.sum(axis=0).astype(np.float64) + 0.25
    frequency = np.power(raw_frequency, 0.65)
    frequency /= frequency.sum()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    base_probabilities = torch.from_numpy(frequency.astype(np.float32)).to(device)
    positive_weight_values = np.power(
        raw_frequency.max() / raw_frequency,
        float(args.rare_label_power),
    )
    positive_weight_values /= np.mean(positive_weight_values)
    positive_weights = torch.from_numpy(positive_weight_values.astype(np.float32)).to(device)
    model = DiscreteSetDiffusion(
        x_dim=x_train.shape[1],
        n_labels=y_train.shape[1],
        max_set_len=max_set_len,
        hidden=int(args.hidden),
        query_blocks=int(args.query_blocks),
        denoise_blocks=int(args.denoise_blocks),
        dropout=float(args.dropout),
        label_chemistry=label_chemistry,
    ).to(device)
    ema_model = copy.deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )
    total_steps = int(args.epochs) * math.ceil(len(x_train) / int(args.batch_size))
    warmup_steps = max(100, int(0.04 * total_steps))

    def learning_rate(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate)
    train_dataset = TensorDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(y_train),
            torch.from_numpy(lengths_train - 1),
        )
    sampler = None
    if float(args.family_balance_power) > 0 or float(args.group_balance_power) > 0:
        meta = pd.read_csv(
            input_dir / "train_meta.csv",
            usecols=["family_signature_primary", "family_group_key"],
            low_memory=False,
        ).fillna("UNK")
        family_values = meta["family_signature_primary"].astype(str)
        group_values = meta["family_group_key"].astype(str)
        family_counts = family_values.value_counts().to_dict()
        group_counts = group_values.value_counts().to_dict()
        row_weights = np.asarray(
            [
                family_counts[family] ** (-float(args.family_balance_power))
                * group_counts[group] ** (-float(args.group_balance_power))
                for family, group in zip(family_values, group_values)
            ],
            dtype=np.float64,
        )
        row_weights /= row_weights.mean()
        sampler = WeightedRandomSampler(
            torch.from_numpy(row_weights),
            num_samples=len(train_dataset),
            replacement=True,
        )
    loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
    )
    logs: List[Dict[str, Any]] = []
    best_metric = -1.0
    best_epoch = 0
    best_state = None
    bad_evaluations = 0
    global_step = 0
    sampling = {
        "sampling_steps": int(args.sampling_steps),
        "guidance_scale": float(args.guidance_scale),
        "temperature_start": float(args.temperature_start),
        "temperature_end": float(args.temperature_end),
        "length_temperature": float(args.length_temperature),
        "trajectory_batch": int(args.trajectory_batch),
        "diffusion_steps": int(args.diffusion_steps),
    }
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        epoch_losses: List[float] = []
        components: Dict[str, List[float]] = {
            "reconstruction": [],
            "positive": [],
            "hard_negative": [],
            "distribution": [],
            "cardinality": [],
        }
        for batch_x, batch_y, batch_lengths in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_lengths = batch_lengths.to(device, non_blocking=True)
            timesteps = torch.randint(
                1,
                int(args.diffusion_steps) + 1,
                (len(batch_x),),
                device=device,
            )
            noisy = corrupt_sets(
                batch_y,
                timesteps,
                int(args.diffusion_steps),
                base_probabilities,
                float(args.noise_multiplier),
            )
            condition_keep = (
                torch.rand(len(batch_x), device=device) >= float(args.condition_dropout)
            ).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                logits, cardinality_logits = model(
                    batch_x, noisy, timesteps, condition_keep
                )
                loss, current = denoising_loss(
                    logits,
                    cardinality_logits,
                    batch_y,
                    batch_lengths,
                    float(args.negative_weight),
                    float(args.distribution_weight),
                    float(args.cardinality_weight),
                    int(args.hard_negatives),
                    positive_weights,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            scheduler.step()
            # Early warm-up avoids an almost-random EMA during the first
            # validation checks; the decay asymptotically reaches ema_decay.
            effective_ema_decay = min(
                float(args.ema_decay),
                (1.0 + float(global_step)) / (10.0 + float(global_step)),
            )
            update_ema(ema_model, model, effective_ema_decay)
            global_step += 1
            epoch_losses.append(float(loss.detach().cpu()))
            for key, value in current.items():
                components[key].append(value)
        row: Dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": float(np.mean(epoch_losses)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{
                f"train_{key}": float(np.mean(values))
                for key, values in components.items()
            },
        }
        should_evaluate = epoch == 1 or epoch % int(args.eval_every) == 0
        if should_evaluate:
            val_rows, _ = generate_candidates(
                ema_model,
                x_val,
                base_probabilities,
                device,
                int(args.eval_batch_size),
                samples=int(args.eval_samples),
                **sampling,
            )
            metrics = exact_metrics(targets_val, val_rows)
            row.update(metrics)
            current_metric = float(metrics[str(args.selection_metric)])
            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in ema_model.state_dict().items()
                }
                bad_evaluations = 0
            else:
                bad_evaluations += 1
        logs.append(row)
        print(json.dumps(row), flush=True)
        if should_evaluate and bad_evaluations >= int(args.patience):
            break
    if best_state is None:
        raise RuntimeError("no diffusion checkpoint selected")
    last_state = {
        key: value.detach().cpu().clone()
        for key, value in ema_model.state_dict().items()
    }
    torch.save(
        {
            "state_dict": last_state,
            "config": vars(args),
            "x_dim": int(x_train.shape[1]),
            "n_labels": int(y_train.shape[1]),
            "max_set_len": max_set_len,
            "label_chemistry": label_chemistry,
            "base_probabilities": frequency.astype(np.float32),
            "best_epoch": int(logs[-1]["epoch"]),
            "best_selection_metric": float(
                logs[-1].get(str(args.selection_metric), float("nan"))
            ),
        },
        run_dir / "last_discrete_diffusion.pt",
    )
    ema_model.load_state_dict(best_state)
    final_rows, final_scores = generate_candidates(
        ema_model,
        x_val,
        base_probabilities,
        device,
        int(args.eval_batch_size),
        samples=int(args.final_samples),
        **sampling,
    )
    validation = exact_metrics(targets_val, final_rows)
    write_candidates(run_dir / "val_candidates.jsonl", final_rows, final_scores)
    checkpoint = {
        "state_dict": best_state,
        "config": vars(args),
        "x_dim": int(x_train.shape[1]),
        "n_labels": int(y_train.shape[1]),
        "max_set_len": max_set_len,
        "label_chemistry": label_chemistry,
        "base_probabilities": frequency.astype(np.float32),
        "best_epoch": best_epoch,
        "best_selection_metric": best_metric,
    }
    torch.save(checkpoint, run_dir / "best_discrete_diffusion.pt")
    report = {
        "model": "chemistry_aware_conditional_discrete_set_diffusion",
        "protocol": "train_only_fit_formula_disjoint_validation_sampling",
        "config": vars(args),
        "data": {
            "train_rows": int(len(x_train)),
            "val_rows": int(len(x_val)),
            "x_dim": int(x_train.shape[1]),
            "n_labels": int(y_train.shape[1]),
            "max_set_len": max_set_len,
        },
        "best_epoch": best_epoch,
        "best_selection_metric": best_metric,
        "final_validation": validation,
        "mean_unique_candidates": float(np.mean([len(row) for row in final_rows])),
        "training_log": logs,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
