#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.family.train_stage2_listwise_ranker import (
    ResidualBlock,
    append_query_formula_features,
    precursor_formula_features,
)


SetKey = Tuple[int, ...]


def parse_family_filter(value: str) -> List[str]:
    return sorted({item.strip() for item in str(value).split(",") if item.strip()})


def family_row_indices(meta_path: Path, families: Sequence[str]) -> np.ndarray:
    if not families:
        return np.asarray([], dtype=np.int64)
    values = pd.read_csv(meta_path, usecols=["family_signature_primary"])[
        "family_signature_primary"
    ].fillna("UNK").astype(str).to_numpy()
    return np.flatnonzero(np.isin(values, np.asarray(families, dtype=object)))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AutoregressiveSetGenerator(nn.Module):
    """Permutation-tolerant set decoder with chemistry-tied label scores."""

    def __init__(
        self,
        x_dim: int,
        n_labels: int,
        max_set_len: int,
        hidden: int,
        blocks: int,
        dropout: float,
        label_chemistry: torch.Tensor,
        query_chemistry_dim: int = 147,
    ) -> None:
        super().__init__()
        self.n_labels = int(n_labels)
        self.stop_id = int(n_labels)
        self.max_set_len = int(max_set_len)
        self.hidden = int(hidden)
        self.query_chemistry_dim = int(query_chemistry_dim)
        if label_chemistry.shape[0] != self.n_labels:
            raise ValueError("label chemistry row count must match n_labels")
        self.register_buffer("label_chemistry", label_chemistry.float(), persistent=False)
        self.target_encoder = nn.Sequential(
            nn.Linear(x_dim, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
        )
        self.target_blocks = nn.Sequential(*[ResidualBlock(hidden, dropout) for _ in range(blocks)])
        self.query_chemistry_encoder = nn.Sequential(
            nn.Linear(query_chemistry_dim, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        self.label_embedding = nn.Embedding(n_labels, hidden)
        self.label_chemistry_encoder = nn.Sequential(
            nn.Linear(label_chemistry.shape[1], hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        self.step_embedding = nn.Embedding(max_set_len + 1, hidden)
        self.state_encoder = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.stop_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.length_head = nn.Linear(hidden, max_set_len)
        self.label_bias = nn.Parameter(torch.zeros(n_labels))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def encode_target(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.target_blocks(self.target_encoder(x))
        return encoded + self.query_chemistry_encoder(x[:, -self.query_chemistry_dim :])

    def label_representations(self) -> torch.Tensor:
        return self.label_embedding.weight + self.label_chemistry_encoder(self.label_chemistry)

    def forward_state(
        self,
        target: torch.Tensor,
        selected_mask: torch.Tensor,
        step_ids: torch.Tensor,
        label_representations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        labels = self.label_representations() if label_representations is None else label_representations
        counts = selected_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        selected_context = selected_mask @ labels / counts
        state = self.state_encoder(
            torch.cat([target, selected_context, self.step_embedding(step_ids)], dim=-1)
        )
        scale = self.logit_scale.exp().clamp(max=100.0) / math.sqrt(self.hidden)
        label_logits = scale * (state @ labels.t()) + self.label_bias
        return torch.cat([label_logits, self.stop_head(state)], dim=-1)

    def auxiliary_logits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target = self.encode_target(x)
        labels = self.label_representations()
        scale = self.logit_scale.exp().clamp(max=100.0) / math.sqrt(self.hidden)
        return scale * (target @ labels.t()) + self.label_bias, self.length_head(target)


def asymmetric_multilabel_loss(logits: torch.Tensor, targets: torch.Tensor, gamma_neg: float = 2.0) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    positive = targets * torch.log(probabilities.clamp_min(1e-8))
    negative = (1.0 - targets) * torch.log((1.0 - probabilities).clamp_min(1e-8))
    negative = negative * probabilities.pow(gamma_neg)
    return -(positive + negative).sum(dim=1).mean()


def order_invariant_teacher_loss(
    model: AutoregressiveSetGenerator,
    x: torch.Tensor,
    y: torch.Tensor,
    lengths: torch.Tensor,
    label_smoothing: float,
    remaining_mass_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = model.encode_target(x)
    labels = model.label_representations()
    batch_size = len(x)
    selected = torch.zeros((batch_size, model.n_labels), dtype=x.dtype, device=x.device)
    # A fresh target-only random ordering each batch prevents label-ID order
    # from becoming part of the learned set probability.
    priorities = torch.rand_like(y).masked_fill(y < 0.5, 2.0)
    order = priorities.argsort(dim=1)
    losses: List[torch.Tensor] = []
    mass_losses: List[torch.Tensor] = []
    for step in range(model.max_set_len + 1):
        step_ids = torch.full((batch_size,), min(step, model.max_set_len), dtype=torch.long, device=x.device)
        logits = model.forward_state(target, selected, step_ids, labels)
        logits[:, : model.n_labels] = logits[:, : model.n_labels].masked_fill(selected > 0.5, -1e4)
        active = step < lengths
        actions = torch.full((batch_size,), model.stop_id, dtype=torch.long, device=x.device)
        if active.any():
            actions[active] = order[active, step]
            logits[active, model.stop_id] = -1e4
        losses.append(F.cross_entropy(logits, actions, label_smoothing=float(label_smoothing), reduction="none"))
        if active.any() and remaining_mass_weight > 0:
            log_probs = F.log_softmax(logits[active], dim=-1)
            remaining = (y[active] > 0.5) & (selected[active] < 0.5)
            remaining_log_probs = log_probs[:, : model.n_labels].masked_fill(~remaining, -torch.inf)
            mass_losses.append(-torch.logsumexp(remaining_log_probs, dim=1).mean())
        if active.any():
            rows = torch.nonzero(active, as_tuple=False).squeeze(1)
            # Matmul backward keeps the previous selected mask for label-vector
            # gradients, so each teacher-forcing transition must create a new
            # state rather than mutate that saved tensor in place.
            next_selected = selected.clone()
            next_selected[rows, actions[rows]] = 1.0
            selected = next_selected
    sequence_loss = torch.stack(losses, dim=1).sum(dim=1).mean()
    remaining_loss = torch.stack(mass_losses).mean() if mass_losses else sequence_loss.new_zeros(())
    return sequence_loss + float(remaining_mass_weight) * remaining_loss, sequence_loss, remaining_loss


@torch.no_grad()
def beam_decode_batch(
    model: AutoregressiveSetGenerator,
    x: torch.Tensor,
    beam_width: int,
    branch_factor: int,
) -> tuple[List[List[SetKey]], List[List[float]]]:
    device = x.device
    batch_size = len(x)
    width = int(beam_width)
    target = model.encode_target(x)
    label_representations = model.label_representations()
    selected = torch.zeros((batch_size, width, model.n_labels), device=device)
    stopped = torch.zeros((batch_size, width), dtype=torch.bool, device=device)
    scores = torch.full((batch_size, width), -torch.inf, device=device)
    scores[:, 0] = 0.0
    for step in range(model.max_set_len + 1):
        flat_selected = selected.reshape(batch_size * width, model.n_labels)
        flat_target = target[:, None, :].expand(-1, width, -1).reshape(batch_size * width, -1)
        step_ids = torch.full((batch_size * width,), min(step, model.max_set_len), dtype=torch.long, device=device)
        logits = model.forward_state(flat_target, flat_selected, step_ids, label_representations)
        logits[:, : model.n_labels] = logits[:, : model.n_labels].masked_fill(flat_selected > 0.5, -1e4)
        if step == 0:
            logits[:, model.stop_id] = -1e4
        if step == model.max_set_len:
            logits[:, : model.n_labels] = -1e4
        flat_stopped = stopped.reshape(-1)
        if flat_stopped.any():
            logits[flat_stopped] = -1e4
            logits[flat_stopped, model.stop_id] = 0.0
        log_probs = F.log_softmax(logits, dim=-1).reshape(batch_size, width, -1)
        branches = min(int(branch_factor), log_probs.shape[-1])
        branch_scores, branch_actions = torch.topk(log_probs, k=branches, dim=-1)
        joint = scores[:, :, None] + branch_scores
        top_scores, top_flat = torch.topk(joint.reshape(batch_size, -1), k=width, dim=-1)
        parent = torch.div(top_flat, branches, rounding_mode="floor")
        actions = torch.gather(branch_actions.reshape(batch_size, -1), 1, top_flat)
        selected = torch.gather(selected, 1, parent[:, :, None].expand(-1, -1, model.n_labels))
        stopped = torch.gather(stopped, 1, parent)
        active = (~stopped) & (actions != model.stop_id)
        if active.any():
            rows, beams = torch.nonzero(active, as_tuple=True)
            selected[rows, beams, actions[rows, beams]] = 1.0
        stopped = stopped | (actions == model.stop_id)
        scores = top_scores
        if stopped.all():
            break
    selected_np = selected.cpu().numpy()
    scores_np = scores.float().cpu().numpy()
    all_candidates: List[List[SetKey]] = []
    all_scores: List[List[float]] = []
    for row_index in range(batch_size):
        paths: Dict[SetKey, List[float]] = {}
        for beam_index in range(width):
            key = tuple(np.flatnonzero(selected_np[row_index, beam_index] > 0.5).tolist())
            if key:
                paths.setdefault(key, []).append(float(scores_np[row_index, beam_index]))
        values = [(float(np.logaddexp.reduce(path_scores)), key) for key, path_scores in paths.items()]
        values.sort(key=lambda item: (-item[0], item[1]))
        all_scores.append([score for score, _ in values])
        all_candidates.append([key for _, key in values])
    return all_candidates, all_scores


@torch.no_grad()
def evaluate_beam(
    model: AutoregressiveSetGenerator,
    x: np.ndarray,
    targets: Sequence[SetKey],
    device: torch.device,
    beam_width: int,
    branch_factor: int,
    batch_size: int,
) -> Dict[str, float]:
    model.eval()
    rows: List[List[SetKey]] = []
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)
    for (batch_x,) in loader:
        candidates, _ = beam_decode_batch(model, batch_x.to(device), beam_width, branch_factor)
        rows.extend(candidates)
    return {
        **{
            f"exact_hit@{k}": float(np.mean([target in set(row[:k]) for target, row in zip(targets, rows)]))
            for k in (1, 3, 5, 10, 20, 50)
        },
        "mean_unique_candidates": float(np.mean([len(row) for row in rows])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a chemistry-tied autoregressive precursor-set generator.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.02)
    parser.add_argument("--remaining_mass_weight", type=float, default=0.25)
    parser.add_argument("--membership_weight", type=float, default=0.05)
    parser.add_argument("--length_weight", type=float, default=0.2)
    parser.add_argument("--eval_beam_width", type=int, default=50)
    parser.add_argument("--eval_branch_factor", type=int, default=32)
    parser.add_argument("--eval_every", type=int, default=2)
    parser.add_argument(
        "--selection_metric",
        choices=("exact_hit@10", "last"),
        default="exact_hit@10",
        help="Use validation Top-10 early stopping, or save the last fixed epoch for honest OOF training.",
    )
    parser.add_argument("--train_families", default="")
    parser.add_argument("--val_families", default="")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed_everything(args.seed)
    input_dir = Path(args.input_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    packs = {split: np.load(input_dir / f"{split}.npz", allow_pickle=True) for split in ("train", "val")}
    train_x = append_query_formula_features(np.asarray(packs["train"]["x"], dtype=np.float32), input_dir / "train_meta.csv")
    val_x = append_query_formula_features(np.asarray(packs["val"]["x"], dtype=np.float32), input_dir / "val_meta.csv")
    train_y = np.asarray(packs["train"]["y_multi_hot"], dtype=np.float32)
    val_y = np.asarray(packs["val"]["y_multi_hot"], dtype=np.float32)
    train_lengths = np.asarray(packs["train"]["set_len"], dtype=np.int64)
    train_families = parse_family_filter(args.train_families)
    val_families = parse_family_filter(args.val_families)
    if train_families:
        train_indices = family_row_indices(input_dir / "train_meta.csv", train_families)
        train_x = train_x[train_indices]
        train_y = train_y[train_indices]
        train_lengths = train_lengths[train_indices]
    if val_families:
        val_indices = family_row_indices(input_dir / "val_meta.csv", val_families)
        val_x = val_x[val_indices]
        val_y = val_y[val_indices]
    if not len(train_x) or not len(val_x):
        raise RuntimeError("family restriction removed every training or validation row")
    val_targets = [tuple(np.flatnonzero(row > 0.5).tolist()) for row in val_y]
    precursor_names = json.loads((input_dir / "precursor_names.json").read_text(encoding="utf-8"))
    label_chemistry = torch.from_numpy(precursor_formula_features([str(value) for value in precursor_names]))
    max_set_len = int(max(train_lengths.max(), np.asarray(packs["val"]["set_len"]).max()))
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y), torch.from_numpy(train_lengths)),
        batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True,
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = AutoregressiveSetGenerator(
        train_x.shape[1], train_y.shape[1], max_set_len, args.hidden, args.blocks, args.dropout, label_chemistry
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_metric = -1.0
    best_epoch = 0
    best_state = None
    bad_evaluations = 0
    logs: List[Dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        sequence_losses = []
        mass_losses = []
        for batch_x, batch_y, batch_lengths in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_lengths = batch_lengths.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                set_loss, sequence_loss, mass_loss = order_invariant_teacher_loss(
                    model, batch_x, batch_y, batch_lengths, args.label_smoothing, args.remaining_mass_weight
                )
                membership_logits, length_logits = model.auxiliary_logits(batch_x)
                membership_loss = asymmetric_multilabel_loss(membership_logits, batch_y)
                length_loss = F.cross_entropy(length_logits, batch_lengths - 1)
                loss = set_loss + args.membership_weight * membership_loss + args.length_weight * length_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            sequence_losses.append(float(sequence_loss.detach().cpu()))
            mass_losses.append(float(mass_loss.detach().cpu()))
        row: Dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)),
            "sequence_loss": float(np.mean(sequence_losses)),
            "remaining_mass_loss": float(np.mean(mass_losses)),
        }
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            validation = evaluate_beam(
                model, val_x, val_targets, device, args.eval_beam_width,
                args.eval_branch_factor, max(1, args.batch_size // 16),
            )
            row.update({f"val_{key}": value for key, value in validation.items()})
            current = float(validation["exact_hit@10"])
            if args.selection_metric == "last":
                best_metric = current
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                bad_evaluations = 0
            elif current > best_metric:
                best_metric = current
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                bad_evaluations = 0
            else:
                bad_evaluations += 1
        logs.append(row)
        print(json.dumps(row), flush=True)
        if args.selection_metric != "last" and bad_evaluations >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("no autoregressive checkpoint selected")
    model.load_state_dict(best_state)
    final_metrics = evaluate_beam(
        model, val_x, val_targets, device, max(100, args.eval_beam_width),
        max(args.eval_branch_factor, 64), max(1, args.batch_size // 16),
    )
    checkpoint = {
        "state_dict": {key: value.cpu() for key, value in best_state.items()},
        "config": vars(args),
        "x_dim": train_x.shape[1],
        "n_labels": train_y.shape[1],
        "max_set_len": max_set_len,
        "best_epoch": best_epoch,
        "best_val_exact_hit_at_10": best_metric,
    }
    torch.save(checkpoint, run_dir / "best_autoregressive_set.pt")
    summary = {
        "protocol": "stage2_formula_disjoint_train_to_val_exact_precursor_set",
        "checkpoint_selection": (
            "fixed_last_epoch_no_query_label_selection"
            if args.selection_metric == "last"
            else "validation_exact_hit_at_10"
        ),
        "config": vars(args),
        "data": {
            "n_train": len(train_x), "n_val": len(val_x), "n_labels": train_y.shape[1],
            "train_families": train_families, "val_families": val_families,
        },
        "best_epoch": best_epoch,
        "selection_val_exact_hit_at_10": best_metric,
        "validation": final_metrics,
        "training_log": logs,
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
