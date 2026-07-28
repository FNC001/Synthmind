#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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


class FactorizedSetGenerator(nn.Module):
    def __init__(self, x_dim: int, n_labels: int, max_set_len: int, hidden: int, blocks: int, dropout: float) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(x_dim, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden, dropout) for _ in range(blocks)])
        self.label_head = nn.Linear(hidden, n_labels)
        self.length_head = nn.Linear(hidden, max_set_len)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.blocks(self.input(x))
        return self.label_head(hidden), self.length_head(hidden)


def asymmetric_loss(logits: torch.Tensor, targets: torch.Tensor, gamma_neg: float, gamma_pos: float) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    positive = targets * torch.log(probabilities.clamp_min(1e-8))
    negative = (1.0 - targets) * torch.log((1.0 - probabilities).clamp_min(1e-8))
    if gamma_pos > 0:
        positive = positive * (1.0 - probabilities).pow(gamma_pos)
    if gamma_neg > 0:
        negative = negative * probabilities.pow(gamma_neg)
    return -(positive + negative).sum(dim=1).mean()


def decode_candidates(
    label_logits: np.ndarray,
    length_logits: np.ndarray,
    top_labels: int,
    candidate_limit: int,
    length_weight: float,
    max_enumerated_length: int,
) -> tuple[List[List[SetKey]], List[List[float]]]:
    all_candidates: List[List[SetKey]] = []
    all_scores: List[List[float]] = []
    length_log_probs = length_logits - np.logaddexp.reduce(length_logits, axis=1, keepdims=True)
    for row_logits, row_length_scores in zip(label_logits, length_log_probs):
        label_ids = np.argsort(-row_logits)[:top_labels]
        scored: List[tuple[float, SetKey]] = []
        max_length = min(len(row_length_scores), max_enumerated_length, top_labels)
        for length in range(1, max_length + 1):
            # Independent Bernoulli log likelihood differs between equal-length sets
            # only by the sum of selected label logits.
            length_score = float(length_weight * row_length_scores[length - 1])
            for local_indices in itertools.combinations(range(top_labels), length):
                ids = tuple(sorted(int(label_ids[index]) for index in local_indices))
                score = length_score + float(sum(row_logits[value] for value in ids))
                scored.append((score, ids))
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = scored[:candidate_limit]
        all_candidates.append([item[1] for item in selected])
        all_scores.append([item[0] for item in selected])
    return all_candidates, all_scores


def exact_metrics(targets: Sequence[SetKey], candidates: Sequence[Sequence[SetKey]]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for k in (1, 3, 5, 10, 20, 50, 100, 500):
        metrics[f"exact_hit@{k}"] = float(np.mean([target in set(row[:k]) for target, row in zip(targets, candidates)]))
    return metrics


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels, lengths = [], []
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)
    for (batch_x,) in loader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            label_logits, length_logits = model(batch_x.to(device, non_blocking=True))
        labels.append(label_logits.float().cpu().numpy())
        lengths.append(length_logits.float().cpu().numpy())
    return np.vstack(labels), np.vstack(lengths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a factorized precursor-member/cardinality model and decode exact sets.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--length_loss_weight", type=float, default=1.0)
    parser.add_argument("--gamma_neg", type=float, default=2.0)
    parser.add_argument("--gamma_pos", type=float, default=0.0)
    parser.add_argument("--top_labels", type=int, default=16)
    parser.add_argument("--candidate_limit", type=int, default=500)
    parser.add_argument("--max_enumerated_length", type=int, default=4)
    parser.add_argument("--length_score_weights", default="0.5,1,2,4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed_everything(args.seed)
    input_dir = Path(args.input_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    packs = {split: np.load(input_dir / f"{split}.npz", allow_pickle=True) for split in ("train", "val")}
    x_train = np.asarray(packs["train"]["x"], dtype=np.float32)
    y_train = np.asarray(packs["train"]["y_multi_hot"], dtype=np.float32)
    x_val = np.asarray(packs["val"]["x"], dtype=np.float32)
    y_val = np.asarray(packs["val"]["y_multi_hot"], dtype=np.float32)
    lengths_train = np.asarray(packs["train"]["set_len"], dtype=np.int64) - 1
    targets_val = [tuple(np.flatnonzero(row > 0.5).tolist()) for row in y_val]
    max_set_len = int(max(packs["train"]["set_len"].max(), packs["val"]["set_len"].max()))
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(lengths_train)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = FactorizedSetGenerator(x_train.shape[1], y_train.shape[1], max_set_len, args.hidden, args.blocks, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_score = -math.inf
    best_epoch = 0
    best_state = None
    bad_epochs = 0
    logs: List[Dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y, batch_lengths in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_lengths = batch_lengths.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                label_logits, length_logits = model(batch_x)
                label_loss = asymmetric_loss(label_logits, batch_y, args.gamma_neg, args.gamma_pos)
                length_loss = F.cross_entropy(length_logits, batch_lengths)
                loss = label_loss + args.length_loss_weight * length_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_label_logits, val_length_logits = predict(model, x_val, device, args.batch_size)
        label_top16 = np.argsort(-val_label_logits, axis=1)[:, : min(16, y_val.shape[1])]
        label_recall = float(np.mean([all(int(value) in set(row) for value in target) for target, row in zip(targets_val, label_top16)]))
        length_accuracy = float(np.mean(np.argmax(val_length_logits, axis=1) == (packs["val"]["set_len"] - 1)))
        score = label_recall + 0.05 * length_accuracy
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_all_labels_in_top16": label_recall, "val_length_accuracy": length_accuracy, "selection_score": score}
        logs.append(row)
        print(json.dumps(row), flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    model.load_state_dict(best_state)
    val_label_logits, val_length_logits = predict(model, x_val, device, args.batch_size)
    trials = []
    best_trial = None
    best_candidates: List[List[SetKey]] = []
    best_scores: List[List[float]] = []
    for length_weight in [float(value) for value in args.length_score_weights.split(",") if value.strip()]:
        candidates, scores = decode_candidates(val_label_logits, val_length_logits, args.top_labels, args.candidate_limit, length_weight, args.max_enumerated_length)
        trial = {"length_score_weight": length_weight, **exact_metrics(targets_val, candidates)}
        trials.append(trial)
        if best_trial is None or (trial["exact_hit@10"], trial["exact_hit@500"]) > (best_trial["exact_hit@10"], best_trial["exact_hit@500"]):
            best_trial, best_candidates, best_scores = trial, candidates, scores
    torch.save({"state_dict": {key: value.cpu() for key, value in best_state.items()}, "config": vars(args), "x_dim": x_train.shape[1], "n_labels": y_train.shape[1], "max_set_len": max_set_len, "best_epoch": best_epoch}, run_dir / "best_factorized_generator.pt")
    summary = {"config": vars(args), "best_epoch": best_epoch, "selection_score": best_score, "validation": best_trial, "trials": trials, "training_log": logs}
    (run_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "val_candidates.jsonl").open("w", encoding="utf-8") as handle:
        for index, (candidates, scores) in enumerate(zip(best_candidates, best_scores)):
            handle.write(json.dumps({"row_index": index, "candidate_label_ids": [list(value) for value in candidates], "scores": scores}) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
