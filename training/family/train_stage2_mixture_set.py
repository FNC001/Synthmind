#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


SetKey = Tuple[int, ...]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ResidualBlock(nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden * 2)
        self.fc2 = nn.Linear(hidden * 2, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.dropout(self.fc2(F.gelu(self.fc1(self.norm(value)))))


class MixtureSetNet(nn.Module):
    def __init__(
        self,
        x_dim: int,
        n_labels: int,
        max_set_len: int,
        experts: int,
        hidden: int,
        blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.n_labels = int(n_labels)
        self.max_set_len = int(max_set_len)
        self.experts = int(experts)
        self.input = nn.Sequential(
            nn.Linear(x_dim, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden, dropout) for _ in range(blocks)])
        self.norm = nn.LayerNorm(hidden)
        self.expert_embedding = nn.Parameter(torch.randn(experts, hidden) * 0.02)
        self.expert_film = nn.Linear(hidden, hidden * 2)
        self.label_head = nn.Linear(hidden, n_labels)
        self.length_head = nn.Linear(hidden, max_set_len)
        self.gate_head = nn.Linear(hidden, experts)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shared = self.norm(self.blocks(self.input(x)))
        scale, shift = self.expert_film(self.expert_embedding).chunk(2, dim=-1)
        hidden = shared[:, None, :] * (1.0 + 0.1 * scale[None, :, :]) + shift[None, :, :]
        hidden = F.gelu(hidden)
        return self.label_head(hidden), self.length_head(hidden), self.gate_head(shared)


def mixture_loss(
    label_logits: torch.Tensor,
    length_logits: torch.Tensor,
    gate_logits: torch.Tensor,
    targets: torch.Tensor,
    lengths: torch.Tensor,
    negative_weight: float,
    length_weight: float,
    balance_weight: float,
    temperature: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    expanded = targets[:, None, :]
    positive_count = expanded.sum(dim=-1).clamp_min(1.0)
    negative_count = (1.0 - expanded).sum(dim=-1).clamp_min(1.0)
    positive_loss = (F.softplus(-label_logits) * expanded).sum(dim=-1) / positive_count
    negative_loss = (F.softplus(label_logits) * (1.0 - expanded)).sum(dim=-1) / negative_count
    label_loss = positive_loss + float(negative_weight) * negative_loss
    expanded_lengths = lengths[:, None].expand(-1, label_logits.shape[1])
    length_loss = F.cross_entropy(
        length_logits.reshape(-1, length_logits.shape[-1]),
        expanded_lengths.reshape(-1),
        reduction="none",
    ).reshape(label_logits.shape[:2])
    route_loss = label_loss + float(length_weight) * length_loss
    log_gate = F.log_softmax(gate_logits, dim=1)
    log_joint = log_gate - route_loss / float(temperature)
    nll = -torch.logsumexp(log_joint, dim=1).mean() * float(temperature)
    responsibilities = F.softmax(log_joint, dim=1).mean(dim=0)
    uniform = torch.full_like(responsibilities, 1.0 / len(responsibilities))
    balance = torch.sum(responsibilities * torch.log((responsibilities + 1e-8) / uniform))
    loss = nll + float(balance_weight) * balance
    return loss, {
        "nll": float(nll.detach().cpu()),
        "balance": float(balance.detach().cpu()),
        "min_route_loss": float(route_loss.min(dim=1).values.mean().detach().cpu()),
    }


def decode(
    label_logits: np.ndarray,
    length_logits: np.ndarray,
    gate_logits: np.ndarray,
) -> List[List[SetKey]]:
    rows: List[List[SetKey]] = []
    for labels, lengths, gates in zip(label_logits, length_logits, gate_logits):
        expert_order = np.argsort(-gates)
        candidates: List[SetKey] = []
        seen = set()
        for expert in expert_order:
            length = int(np.argmax(lengths[expert])) + 1
            candidate = tuple(sorted(np.argsort(-labels[expert])[:length].astype(int).tolist()))
            if candidate and candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)
        # Expert collisions are uncommon but should not silently reduce K.
        if len(candidates) < len(expert_order):
            for expert in expert_order:
                base_length = int(np.argmax(lengths[expert])) + 1
                label_order = np.argsort(-labels[expert])
                for offset in range(1, 4):
                    length = min(len(label_order), max(1, base_length + (offset if offset % 2 else -offset // 2)))
                    candidate = tuple(sorted(label_order[:length].astype(int).tolist()))
                    if candidate and candidate not in seen:
                        seen.add(candidate)
                        candidates.append(candidate)
                    if len(candidates) >= len(expert_order):
                        break
                if len(candidates) >= len(expert_order):
                    break
        rows.append(candidates)
    return rows


@torch.no_grad()
def predict(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    label_values, length_values, gate_values = [], [], []
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)
    for (batch_x,) in loader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            labels, lengths, gates = model(batch_x.to(device, non_blocking=True))
        label_values.append(labels.float().cpu().numpy())
        length_values.append(lengths.float().cpu().numpy())
        gate_values.append(gates.float().cpu().numpy())
    return np.concatenate(label_values), np.concatenate(length_values), np.concatenate(gate_values)


def exact_metrics(targets: Sequence[SetKey], rows: Sequence[Sequence[SetKey]]) -> Dict[str, float]:
    return {
        f"exact_hit@{k}": float(np.mean([target in set(row[:k]) for target, row in zip(targets, rows)]))
        for k in (1, 3, 5, 10)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a latent mixture that directly emits a Top-K slate of precursor sets.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--experts", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--negative_weight", type=float, default=2.0)
    parser.add_argument("--length_weight", type=float, default=0.5)
    parser.add_argument("--balance_weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed_everything(args.seed)
    input_dir = Path(args.input_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    packs = {
        split: {key: value for key, value in np.load(input_dir / f"{split}.npz", allow_pickle=True).items()}
        for split in ("train", "val")
    }
    x_train = np.asarray(packs["train"]["x"], dtype=np.float32)
    y_train = np.asarray(packs["train"]["y_multi_hot"], dtype=np.float32)
    x_val = np.asarray(packs["val"]["x"], dtype=np.float32)
    y_val = np.asarray(packs["val"]["y_multi_hot"], dtype=np.float32)
    lengths_train = np.asarray(packs["train"]["set_len"], dtype=np.int64) - 1
    targets_val = [tuple(np.flatnonzero(row > 0.5).tolist()) for row in y_val]
    max_set_len = int(max(packs["train"]["set_len"].max(), packs["val"]["set_len"].max()))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(lengths_train)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = MixtureSetNet(
        x_train.shape[1], y_train.shape[1], max_set_len,
        args.experts, args.hidden, args.blocks, args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    best_metric = -1.0
    best_epoch = 0
    best_state = None
    bad_epochs = 0
    logs: List[Dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        components: Dict[str, List[float]] = {"nll": [], "balance": [], "min_route_loss": []}
        for batch_x, batch_y, batch_lengths in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_lengths = batch_lengths.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                labels, lengths, gates = model(batch_x)
                loss, current = mixture_loss(
                    labels, lengths, gates, batch_y, batch_lengths,
                    args.negative_weight, args.length_weight,
                    args.balance_weight, args.temperature,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            for key, value in current.items():
                components[key].append(value)
        scheduler.step()
        val_labels, val_lengths, val_gates = predict(model, x_val, device, args.batch_size * 2)
        val_rows = decode(val_labels, val_lengths, val_gates)
        val_metrics = exact_metrics(targets_val, val_rows)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **{f"train_{key}": float(np.mean(values)) for key, values in components.items()},
            **val_metrics,
        }
        logs.append(row)
        print(json.dumps(row), flush=True)
        current_metric = float(val_metrics["exact_hit@10"])
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("no mixture checkpoint selected")
    model.load_state_dict(best_state)
    val_labels, val_lengths, val_gates = predict(model, x_val, device, args.batch_size * 2)
    val_rows = decode(val_labels, val_lengths, val_gates)
    validation = exact_metrics(targets_val, val_rows)
    checkpoint = {
        "state_dict": {key: value.cpu() for key, value in best_state.items()},
        "config": vars(args),
        "x_dim": x_train.shape[1],
        "n_labels": y_train.shape[1],
        "max_set_len": max_set_len,
        "best_epoch": best_epoch,
        "best_val_exact_hit_at_10": best_metric,
    }
    torch.save(checkpoint, run_dir / "best_mixture_set.pt")
    with (run_dir / "val_candidates.jsonl").open("w", encoding="utf-8") as handle:
        for row, candidates in enumerate(val_rows):
            handle.write(json.dumps({"row_index": row, "candidate_label_ids": [list(value) for value in candidates]}) + "\n")
    report = {
        "model": "stage2_latent_mixture_topk_set",
        "config": vars(args),
        "data": {"train_rows": len(x_train), "val_rows": len(x_val), "x_dim": x_train.shape[1], "n_labels": y_train.shape[1]},
        "best_epoch": best_epoch,
        "validation": validation,
        "training_log": logs,
    }
    (run_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
