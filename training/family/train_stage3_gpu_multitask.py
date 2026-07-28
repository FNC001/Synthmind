#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset


SPLITS = ("train", "val", "test")
QUANTILES = (0.1, 0.5, 0.9)
HEAVY_TRIALS = (
    {"name": "wide_512", "hidden": 512, "precursor_hidden": 256, "blocks": 3, "dropout": 0.15, "lr": 8e-4, "batch_size": 512},
    {"name": "wide_768", "hidden": 768, "precursor_hidden": 384, "blocks": 4, "dropout": 0.12, "lr": 6e-4, "batch_size": 512},
    {"name": "wide_1024", "hidden": 1024, "precursor_hidden": 512, "blocks": 4, "dropout": 0.18, "lr": 4e-4, "batch_size": 768},
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(to_builtin(value), ensure_ascii=False, indent=2), encoding="utf-8")


class ResidualBlock(nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden * 2)
        self.fc2 = nn.Linear(hidden * 2, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.norm(value)
        value = self.fc2(self.dropout(F.gelu(self.fc1(value))))
        return residual + self.dropout(value)


class Stage3MultiTaskNet(nn.Module):
    def __init__(
        self,
        structure_dim: int,
        precursor_dim: int,
        hidden: int,
        precursor_hidden: int,
        blocks: int,
        dropout: float,
        atmosphere_classes: int,
        method_classes: int,
    ) -> None:
        super().__init__()
        structure_hidden = max(128, hidden // 2)
        self.structure_dim = int(structure_dim)
        self.structure_encoder = nn.Sequential(
            nn.Linear(structure_dim, structure_hidden),
            nn.LayerNorm(structure_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(structure_hidden, structure_hidden),
            nn.GELU(),
        )
        self.precursor_encoder = nn.Sequential(
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
        self.blocks = nn.Sequential(*[ResidualBlock(hidden, dropout) for _ in range(blocks)])
        self.final_norm = nn.LayerNorm(hidden)
        self.quantile_head = nn.Linear(hidden, 2 * len(QUANTILES))
        self.atmosphere_head = nn.Linear(hidden, atmosphere_classes)
        self.method_head = nn.Linear(hidden, method_classes)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        structure = self.structure_encoder(features[:, : self.structure_dim])
        precursor = self.precursor_encoder(features[:, self.structure_dim :])
        hidden = self.final_norm(self.blocks(self.fusion(torch.cat([structure, precursor], dim=1))))
        return {
            "quantiles": self.quantile_head(hidden).reshape(-1, 2, len(QUANTILES)),
            "atmosphere": self.atmosphere_head(hidden),
            "method": self.method_head(hidden),
        }


def pinball_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses = []
    for target_index in range(2):
        valid = mask[:, target_index] > 0.5
        if not torch.any(valid):
            continue
        error = target[valid, target_index, None] - prediction[valid, target_index]
        q = torch.tensor(QUANTILES, device=prediction.device, dtype=prediction.dtype)[None, :]
        losses.append(torch.maximum(q * error, (q - 1.0) * error).mean())
    return torch.stack(losses).mean()


def class_weights(labels: np.ndarray, mask: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(labels[mask], minlength=n_classes).astype(np.float64)
    weights = np.zeros(n_classes, dtype=np.float32)
    present = counts > 0
    weights[present] = 1.0 / np.sqrt(counts[present])
    weights[present] /= weights[present].mean()
    return torch.from_numpy(weights)


def transform_continuous(packs: Mapping[str, Mapping[str, np.ndarray]]) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    train_raw = np.asarray(packs["train"]["y_cond_continuous_raw"], dtype=np.float32)
    train_mask = np.asarray(packs["train"]["y_cond_continuous_mask"], dtype=np.float32) > 0.5
    stats: Dict[str, Any] = {}
    transformed: Dict[str, np.ndarray] = {}
    for split in SPLITS:
        raw = np.asarray(packs[split]["y_cond_continuous_raw"], dtype=np.float32).copy()
        raw[:, 1] = np.log1p(np.clip(raw[:, 1], 0.0, None))
        transformed[split] = raw
    train_for_stats = train_raw.copy()
    train_for_stats[:, 1] = np.log1p(np.clip(train_for_stats[:, 1], 0.0, None))
    for index, name in enumerate(("temperature_c", "log1p_time_h")):
        values = train_for_stats[train_mask[:, index], index]
        mean = float(values.mean())
        std = float(values.std()) if float(values.std()) > 1e-8 else 1.0
        stats[name] = {"mean": mean, "std": std}
        for split in SPLITS:
            transformed[split][:, index] = (transformed[split][:, index] - mean) / std
    return transformed, stats


def inverse_continuous(normalized: np.ndarray, stats: Mapping[str, Any]) -> np.ndarray:
    output = np.asarray(normalized, dtype=np.float32).copy()
    output[:, 0] = output[:, 0] * float(stats["temperature_c"]["std"]) + float(stats["temperature_c"]["mean"])
    output[:, 1] = output[:, 1] * float(stats["log1p_time_h"]["std"]) + float(stats["log1p_time_h"]["mean"])
    output[:, 1] = np.expm1(output[:, 1])
    return output


def make_dataset(x: np.ndarray, pack: Mapping[str, np.ndarray], y_cont: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(y_cont.astype(np.float32)),
        torch.from_numpy(np.asarray(pack["y_cond_continuous_mask"], dtype=np.float32)),
        torch.from_numpy(np.asarray(pack["y_cond_discrete"], dtype=np.int64)),
        torch.from_numpy(np.asarray(pack["y_cond_discrete_mask"], dtype=np.float32)),
    )


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    collected: Dict[str, list[np.ndarray]] = {"quantiles": [], "atmosphere": [], "method": []}
    for batch in loader:
        features = batch[0].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(features)
        for name in collected:
            collected[name].append(output[name].float().cpu().numpy())
    result = {name: np.concatenate(values, axis=0) for name, values in collected.items()}
    # Quantile heads are trained independently; sorting is the standard
    # non-parametric monotonicity correction for interval reporting.
    result["quantiles"] = np.sort(result["quantiles"], axis=2)
    return result


def validation_score(
    prediction: Mapping[str, np.ndarray],
    pack: Mapping[str, np.ndarray],
) -> tuple[float, Dict[str, float]]:
    true_cont = np.asarray(pack["y_cont_transformed"], dtype=np.float32)
    cont_mask = np.asarray(pack["y_cond_continuous_mask"], dtype=np.float32) > 0.5
    median = prediction["quantiles"][:, :, 1]
    temp_mae = float(np.abs(true_cont[cont_mask[:, 0], 0] - median[cont_mask[:, 0], 0]).mean())
    time_mae = float(np.abs(true_cont[cont_mask[:, 1], 1] - median[cont_mask[:, 1], 1]).mean())
    true_disc = np.asarray(pack["y_cond_discrete"], dtype=np.int64)
    disc_mask = np.asarray(pack["y_cond_discrete_mask"], dtype=np.float32) > 0.5
    atmosphere_pred = prediction["atmosphere"].argmax(axis=1)
    method_pred = prediction["method"].argmax(axis=1)
    atmosphere_f1 = float(f1_score(true_disc[disc_mask[:, 0], 0], atmosphere_pred[disc_mask[:, 0]], average="macro", zero_division=0))
    method_f1 = float(f1_score(true_disc[disc_mask[:, 1], 1], method_pred[disc_mask[:, 1]], average="macro", zero_division=0))
    components = {"temp_norm_mae": temp_mae, "time_log_norm_mae": time_mae, "atmosphere_macro_f1": atmosphere_f1, "method_macro_f1": method_f1}
    return temp_mae + time_mae + (1.0 - atmosphere_f1) + (1.0 - method_f1), components


def train_trial(
    config: Mapping[str, Any],
    x: Mapping[str, np.ndarray],
    packs: Mapping[str, Dict[str, np.ndarray]],
    datasets: Mapping[str, TensorDataset],
    stats: Mapping[str, Any],
    atmosphere_classes: int,
    method_classes: int,
    device: torch.device,
    epochs: int,
    patience: int,
    seed: int,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    seed_everything(seed)
    model = Stage3MultiTaskNet(
        structure_dim=int(packs["structure_dim"]),
        precursor_dim=int(packs["precursor_dim"]),
        hidden=int(config["hidden"]),
        precursor_hidden=int(config["precursor_hidden"]),
        blocks=int(config["blocks"]),
        dropout=float(config["dropout"]),
        atmosphere_classes=atmosphere_classes,
        method_classes=method_classes,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=1e-4)
    train_loader = DataLoader(datasets["train"], batch_size=int(config["batch_size"]), shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(datasets["val"], batch_size=1024, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    train_disc = np.asarray(packs["train"]["y_cond_discrete"], dtype=np.int64)
    train_disc_mask = np.asarray(packs["train"]["y_cond_discrete_mask"], dtype=np.float32) > 0.5
    atmosphere_weights = class_weights(train_disc[:, 0], train_disc_mask[:, 0], atmosphere_classes).to(device)
    method_weights = class_weights(train_disc[:, 1], train_disc_mask[:, 1], method_classes).to(device)
    best_score = float("inf")
    best_epoch = 0
    best_state: Dict[str, Any] | None = None
    log = []
    bad_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for features, y_cont, cont_mask, y_disc, disc_mask in train_loader:
            features = features.to(device, non_blocking=True)
            y_cont = y_cont.to(device, non_blocking=True)
            cont_mask = cont_mask.to(device, non_blocking=True)
            y_disc = y_disc.to(device, non_blocking=True)
            disc_mask = disc_mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(features)
                loss = pinball_loss(output["quantiles"], y_cont, cont_mask)
                atmosphere_valid = disc_mask[:, 0] > 0.5
                method_valid = disc_mask[:, 1] > 0.5
                if torch.any(atmosphere_valid):
                    loss = loss + 0.5 * F.cross_entropy(output["atmosphere"][atmosphere_valid], y_disc[atmosphere_valid, 0], weight=atmosphere_weights)
                if torch.any(method_valid):
                    loss = loss + 0.5 * F.cross_entropy(output["method"][method_valid], y_disc[method_valid, 1], weight=method_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            n_batches += 1
        scheduler.step()
        val_prediction = predict(model, val_loader, device)
        score, components = validation_score(val_prediction, packs["val"])
        row = {"epoch": epoch, "train_loss": total_loss / max(1, n_batches), "val_score": score, **components}
        log.append(row)
        print(json.dumps({"trial": config["name"], **row}), flush=True)
        if score < best_score - 1e-5:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    return {
        "config": dict(config),
        "best_val_score": best_score,
        "best_epoch": best_epoch,
        "log": log,
    }, {"state_dict": best_state, "stats": dict(stats), "config": dict(config)}


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else math.nan,
    }


def evaluate_test(pack: Mapping[str, np.ndarray], prediction: Mapping[str, np.ndarray], stats: Mapping[str, Any]) -> tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    median_raw = inverse_continuous(prediction["quantiles"][:, :, 1], stats)
    lower_raw = inverse_continuous(prediction["quantiles"][:, :, 0], stats)
    upper_raw = inverse_continuous(prediction["quantiles"][:, :, 2], stats)
    true_raw = np.asarray(pack["y_cond_continuous_raw"], dtype=np.float32)
    cont_mask = np.asarray(pack["y_cond_continuous_mask"], dtype=np.float32) > 0.5
    metrics: Dict[str, Any] = {"continuous": {}, "discrete": {}}
    for index, name in enumerate(("temperature_c", "time_h")):
        valid = cont_mask[:, index]
        coverage = np.mean((true_raw[valid, index] >= lower_raw[valid, index]) & (true_raw[valid, index] <= upper_raw[valid, index]))
        metrics["continuous"][name] = {**regression_metrics(true_raw[valid, index], median_raw[valid, index]), "interval_10_90_coverage": float(coverage), "n": int(valid.sum())}
    true_disc = np.asarray(pack["y_cond_discrete"], dtype=np.int64)
    disc_mask = np.asarray(pack["y_cond_discrete_mask"], dtype=np.float32) > 0.5
    discrete_pred = np.column_stack([prediction["atmosphere"].argmax(axis=1), prediction["method"].argmax(axis=1)])
    for index, name in enumerate(("atmosphere_coarse", "reaction_method")):
        valid = disc_mask[:, index]
        metrics["discrete"][name] = {
            "accuracy": float(accuracy_score(true_disc[valid, index], discrete_pred[valid, index])),
            "macro_f1": float(f1_score(true_disc[valid, index], discrete_pred[valid, index], average="macro", zero_division=0)),
            "n": int(valid.sum()),
        }
    return metrics, median_raw, lower_raw, upper_raw


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU hyperparameter search for Stage3 family-conditioned multi-task prediction.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--drop_family_features", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_trials", type=int, default=3)
    args = parser.parse_args()
    input_dir = Path(args.input_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    schema = json.loads((input_dir / "schema.json").read_text(encoding="utf-8"))
    packs: Dict[str, Dict[str, np.ndarray]] = {
        split: {key: value for key, value in np.load(input_dir / f"{split}.npz", allow_pickle=True).items()}
        for split in SPLITS
    }
    base_count = int(schema["base_feature_count"])
    family_count = int(schema["family_feature_count"])
    precursor_dim = len(schema["precursor_vocab"])
    structure_dim = base_count if args.drop_family_features else base_count + family_count
    x: Dict[str, np.ndarray] = {}
    for split in SPLITS:
        structure = np.asarray(packs[split]["x"], dtype=np.float32)[:, :structure_dim]
        x[split] = np.hstack([structure, np.asarray(packs[split]["y_set"], dtype=np.float32)]).astype(np.float32)
    transformed, stats = transform_continuous(packs)
    for split in SPLITS:
        packs[split]["y_cont_transformed"] = transformed[split]
    packs["structure_dim"] = structure_dim  # type: ignore[assignment]
    packs["precursor_dim"] = precursor_dim  # type: ignore[assignment]
    datasets = {split: make_dataset(x[split], packs[split], transformed[split]) for split in SPLITS}
    atmosphere_classes = int(np.max(packs["train"]["y_cond_discrete"][:, 0])) + 1
    method_classes = int(np.max(packs["train"]["y_cond_discrete"][:, 1])) + 1
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    trials = []
    checkpoints = []
    for index, config in enumerate(HEAVY_TRIALS[: max(1, int(args.max_trials))]):
        trial, checkpoint = train_trial(
            config, x, packs, datasets, stats, atmosphere_classes, method_classes,
            device, args.epochs, args.patience, args.seed + index * 101,
        )
        trials.append(trial)
        checkpoints.append(checkpoint)
        torch.cuda.empty_cache()
    best_index = int(np.argmin([trial["best_val_score"] for trial in trials]))
    best = checkpoints[best_index]
    config = best["config"]
    model = Stage3MultiTaskNet(
        structure_dim=structure_dim, precursor_dim=precursor_dim,
        hidden=int(config["hidden"]), precursor_hidden=int(config["precursor_hidden"]),
        blocks=int(config["blocks"]), dropout=float(config["dropout"]),
        atmosphere_classes=atmosphere_classes, method_classes=method_classes,
    ).to(device)
    model.load_state_dict(best["state_dict"])
    predictions = {
        split: predict(model, DataLoader(datasets[split], batch_size=1024, shuffle=False, num_workers=2, pin_memory=True), device)
        for split in ("val", "test")
    }
    test_metrics, median_raw, lower_raw, upper_raw = evaluate_test(packs["test"], predictions["test"], stats)
    test_meta = pd.read_csv(input_dir / "test_meta.csv", low_memory=False)
    output = test_meta[[column for column in ("sample_id", "formula", "family_signature_primary", "family_id_primary", "source_dataset", "reaction_method") if column in test_meta]].copy()
    for index, name in enumerate(("temperature_c", "time_h")):
        output[f"true_{name}"] = packs["test"]["y_cond_continuous_raw"][:, index]
        output[f"has_{name}"] = packs["test"]["y_cond_continuous_mask"][:, index]
        output[f"pred_{name}"] = median_raw[:, index]
        output[f"pred_{name}_q10"] = lower_raw[:, index]
        output[f"pred_{name}_q90"] = upper_raw[:, index]
    discrete_pred = np.column_stack([predictions["test"]["atmosphere"].argmax(axis=1), predictions["test"]["method"].argmax(axis=1)])
    for index, name in enumerate(("atmosphere_coarse", "reaction_method")):
        output[f"true_{name}_id"] = packs["test"]["y_cond_discrete"][:, index]
        output[f"has_{name}"] = packs["test"]["y_cond_discrete_mask"][:, index]
        output[f"pred_{name}_id"] = discrete_pred[:, index]
    output.to_csv(run_dir / "pred_test.csv", index=False)
    checkpoint = {
        "state_dict": {name: tensor.cpu() for name, tensor in best["state_dict"].items()},
        "config": config,
        "structure_dim": structure_dim,
        "precursor_dim": precursor_dim,
        "atmosphere_classes": atmosphere_classes,
        "method_classes": method_classes,
        "target_stats": stats,
        "schema_version": schema["schema_version"],
        "drop_family_features": bool(args.drop_family_features),
    }
    torch.save(checkpoint, run_dir / "best_model.pt")
    summary = {
        "model": "stage3_gpu_multitask_residual_mlp",
        "config": vars(args),
        "device": str(device),
        "data": {"rows": {split: int(len(x[split])) for split in SPLITS}, "structure_dim": structure_dim, "precursor_dim": precursor_dim},
        "trials": trials,
        "selected_trial": trials[best_index],
        "test_metrics": test_metrics,
        "artifacts": {"checkpoint": str(run_dir / "best_model.pt"), "pred_test": str(run_dir / "pred_test.csv")},
    }
    write_json(run_dir / "metrics.json", summary)
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
