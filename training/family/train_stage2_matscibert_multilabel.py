#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer


def sequence_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def normalized(values: torch.Tensor) -> torch.Tensor:
    return values / values.norm(dim=-1, keepdim=True).clamp_min(1e-8)


class FormulaPrecursorModel(nn.Module):
    def __init__(
        self,
        model_path: Path,
        label_initialization: np.ndarray,
        label_bias: np.ndarray,
        train_last_layers: int,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path, local_files_only=True)
        hidden = int(self.encoder.config.hidden_size)
        if label_initialization.shape[1] != hidden:
            raise ValueError("MatSciBERT cache hidden size does not match encoder")
        self.label_embedding = nn.Parameter(torch.from_numpy(label_initialization.astype(np.float32)))
        self.label_bias = nn.Parameter(torch.from_numpy(label_bias.astype(np.float32)))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0), dtype=torch.float32))
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        layers = list(self.encoder.encoder.layer)
        if not 0 <= int(train_last_layers) <= len(layers):
            raise ValueError(f"train_last_layers must be between 0 and {len(layers)}")
        for layer in layers[-int(train_last_layers) :] if int(train_last_layers) else []:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        for parameter in self.encoder.pooler.parameters() if self.encoder.pooler is not None else []:
            parameter.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        content = attention_mask.bool().clone()
        content[:, 0] = False
        final_indices = attention_mask.sum(dim=1).sub(1).clamp_min(0)
        content[torch.arange(len(content), device=content.device), final_indices] = False
        empty = ~content.any(dim=1)
        if bool(empty.any()):
            content[empty] = attention_mask[empty].bool()
        weights = content.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        scale = self.logit_scale.exp().clamp(1.0, 100.0)
        return scale * (normalized(pooled) @ normalized(self.label_embedding).T) + self.label_bias


def multilabel_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    positive_weight: torch.Tensor,
    row_weight: torch.Tensor,
    negative_focal_gamma: float,
) -> torch.Tensor:
    positive = targets * torch.nn.functional.softplus(-logits) * positive_weight
    negative_probability = torch.sigmoid(logits)
    negative = (1.0 - targets) * torch.nn.functional.softplus(logits)
    if float(negative_focal_gamma) > 0:
        negative = negative * negative_probability.pow(float(negative_focal_gamma))
    per_row = (positive + negative).mean(dim=1)
    return (per_row * row_weight).sum() / row_weight.sum().clamp_min(1e-8)


def group_balance_weights(groups: Sequence[str], power: float) -> np.ndarray:
    counts = Counter(map(str, groups))
    values = np.asarray([counts[str(value)] ** (-float(power)) for value in groups], dtype=np.float32)
    return values / max(float(values.mean()), 1e-8)


def tokenize_formulas(tokenizer, formulas: Sequence[str], max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = tokenizer(
        [f"Target material formula: {formula}." for formula in formulas],
        padding="max_length",
        truncation=True,
        max_length=int(max_length),
        return_tensors="pt",
    )
    return tokens["input_ids"], tokens["attention_mask"]


def prediction_metrics(logits: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    output: dict[str, float] = {}
    positive_count = np.maximum(targets.sum(axis=1), 1.0)
    order = np.argsort(-logits, axis=1, kind="stable")[:, :50]
    for k in (1, 3, 5, 10, 20, 50):
        selected = np.take_along_axis(targets, order[:, :k], axis=1).sum(axis=1)
        output[f"label_recall@{k}"] = float(np.mean(selected / positive_count))
        output[f"all_gold_labels_in_top@{k}"] = float(np.mean(selected >= positive_count))
    return output


def infer_logits(
    model: FormulaPrecursorModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(TensorDataset(input_ids, attention_mask), batch_size=int(batch_size))
    batches = []
    model.eval()
    with torch.inference_mode():
        for ids, mask in loader:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(ids.to(device), mask.to(device))
            batches.append(logits.float().cpu().numpy())
    return np.concatenate(batches, axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune MatSciBERT on train-only Stage2 multi-label precursor supervision."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--embedding_cache", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--score_cache", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--inference_batch_size", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--encoder_lr", type=float, default=2e-5)
    parser.add_argument("--head_lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--train_last_layers", type=int, default=4)
    parser.add_argument("--group_balance_power", type=float, default=0.5)
    parser.add_argument("--positive_weight_cap", type=float, default=50.0)
    parser.add_argument("--negative_focal_gamma", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    input_dir = Path(args.input_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    score_cache = Path(args.score_cache).resolve()
    score_cache.parent.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model).resolve()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    splits = ("train", "val", "test")
    packs = {split: np.load(input_dir / f"{split}.npz", allow_pickle=True) for split in splits}
    targets = {split: np.asarray(packs[split]["y_multi_hot"], dtype=np.float32) for split in splits}
    meta = {split: pd.read_csv(input_dir / f"{split}_meta.csv", low_memory=False) for split in splits}
    formulas = {split: meta[split]["formula"].fillna("").astype(str).tolist() for split in splits}
    precursor_names = json.loads((input_dir / "precursor_names.json").read_text(encoding="utf-8"))
    with np.load(Path(args.embedding_cache).resolve(), allow_pickle=False) as cache:
        if str(cache["precursor_names_sha256"].item()) != sequence_sha256(precursor_names):
            raise ValueError("embedding cache precursor vocabulary mismatch")
        label_initialization = np.asarray(cache["precursor_role_mean"], dtype=np.float32)

    train_y = targets["train"]
    label_positive = train_y.sum(axis=0)
    positive_weight = np.sqrt((len(train_y) - label_positive + 1.0) / (label_positive + 1.0))
    positive_weight = np.clip(positive_weight, 1.0, float(args.positive_weight_cap)).astype(np.float32)
    prior = (label_positive + 1.0) / (len(train_y) + 2.0)
    label_bias = np.clip(np.log(prior / (1.0 - prior)), -8.0, -1.0).astype(np.float32)
    row_weights = group_balance_weights(
        meta["train"]["family_group_key"].fillna("UNK").astype(str).tolist(),
        float(args.group_balance_power),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokens = {
        split: tokenize_formulas(tokenizer, formulas[split], int(args.max_length))
        for split in splits
    }
    train_dataset = TensorDataset(
        tokens["train"][0],
        tokens["train"][1],
        torch.from_numpy(train_y),
        torch.from_numpy(row_weights),
    )
    generator = torch.Generator().manual_seed(int(args.seed))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        generator=generator,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    model = FormulaPrecursorModel(
        model_path,
        label_initialization,
        label_bias,
        int(args.train_last_layers),
    ).to(device)
    encoder_parameters = [
        parameter for name, parameter in model.named_parameters()
        if name.startswith("encoder.") and parameter.requires_grad
    ]
    head_parameters = [
        parameter for name, parameter in model.named_parameters()
        if not name.startswith("encoder.") and parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": float(args.encoder_lr)},
            {"params": head_parameters, "lr": float(args.head_lr)},
        ],
        weight_decay=float(args.weight_decay),
    )
    positive_weight_tensor = torch.from_numpy(positive_weight).to(device)
    history = []
    best_score = -1.0
    best_epoch = 0
    best_state = None
    stale = 0
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for input_ids, attention_mask, batch_targets, batch_weights in train_loader:
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            batch_weights = batch_weights.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(input_ids, attention_mask)
                loss = multilabel_loss(
                    logits.float(), batch_targets, positive_weight_tensor,
                    batch_weights, float(args.negative_focal_gamma),
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(input_ids)
            total_rows += len(input_ids)
        val_logits = infer_logits(
            model, *tokens["val"], int(args.inference_batch_size), device
        )
        val_metrics = prediction_metrics(val_logits, targets["val"])
        score = val_metrics["label_recall@10"]
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_rows, 1),
            **val_metrics,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if score > best_score + 1e-7:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(args.patience):
                break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    torch.save(
        {"state_dict": best_state, "config": vars(args), "best_epoch": best_epoch},
        run_dir / "best_model.pt",
    )

    logits = {
        split: infer_logits(model, *tokens[split], int(args.inference_batch_size), device)
        for split in splits
    }
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray("stage2_matscibert_multilabel_scores_v1"),
        "precursor_names_sha256": np.asarray(sequence_sha256(precursor_names)),
    }
    for split in splits:
        arrays[f"{split}_formula_sha256"] = np.asarray(sequence_sha256(formulas[split]))
        arrays[f"{split}_logits"] = logits[split].astype(np.float16)
    np.savez_compressed(score_cache, **arrays)
    report = {
        "protocol": "train_only_matscibert_multilabel_formula_to_precursor",
        "config": vars(args),
        "best_epoch": best_epoch,
        "best_validation": prediction_metrics(logits["val"], targets["val"]),
        "history": history,
        "score_cache": str(score_cache),
        "test_policy": "Test labels were not evaluated or used for checkpoint selection.",
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
