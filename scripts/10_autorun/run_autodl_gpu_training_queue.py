#!/usr/bin/env python3
"""Run a 24h AutoDL GPU training queue for SynPred.

This queue is intentionally self contained: it only needs pandas, numpy and
PyTorch, which are already present on the current AutoDL image.  It trains
neural Stage35 route rerankers from existing route-candidate CSVs, validates
every few epochs, and keeps launching controlled variants until the time budget
expires.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


BASELINE_STAGE35 = {
    "missing_top1_relaxed_route": 0.2072,
    "missing_top10_relaxed_route": 0.3455,
    "missing_top200_relaxed_route": 0.4885,
    "strict_top1_relaxed_route": 0.1045,
    "strict_top10_relaxed_route": 0.1804,
    "strict_top200_relaxed_route": 0.2697,
}

LEAKY_TOKENS = (
    "true_",
    "_hit",
    "hit_if_eval",
    "exact",
    "jaccard",
    "f1",
    "_error",
    "correct",
    "known_mask",
)

ID_COLUMNS = {
    "sample_id",
    "sample_index",
    "route_candidate_id",
    "condition_candidate_id",
    "formula",
    "true_precursors",
    "pred_precursors",
    "candidate_set",
    "candidate_source_mix",
}

PREFERRED_NUMERIC_FEATURES = [
    "rank",
    "calibrated_score_v5",
    "calibrated_rank_v5",
    "total_score_v5",
    "precursor_rank",
    "precursor_score",
    "element_coverage",
    "missing_element_count",
    "extra_element_count",
    "candidate_size",
    "contains_open_generated_precursor",
    "contains_repair_precursor",
    "precursor_score_norm",
    "temperature_c",
    "temperature_low_c",
    "temperature_high_c",
    "time_h",
    "time_low_h",
    "time_high_h",
    "retrieval_score",
    "temperature_point_score",
    "temperature_bin_score",
    "time_point_score",
    "time_bin_score",
    "atmosphere_probability",
    "solvent_probability",
    "condition_prior_score",
    "precursor_confidence_score",
    "open_generated_penalty",
    "repair_penalty",
    "total_score_raw",
    "multimodal_template_score",
    "method_template_score",
    "condition_calibrated_score_v3",
    "condition_rank_calibrated_v3",
    "condition_rank",
    "condition_score",
    "condition_score_norm",
    "precursor_confidence",
    "condition_confidence",
    "precursor_condition_compatibility_score",
    "reaction_method_prior_score",
    "route_total_score_raw",
    "route_rank_raw",
    "route_calibrated_score_v3_final_formula",
]

CATEGORICAL_FEATURES = [
    "reaction_method",
    "candidate_source",
    "chemistry_check_status",
    "condition_source",
    "atmosphere",
    "solvent",
]


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def maybe_update_best_json(path: Path, payload: dict[str, Any]) -> None:
    current_score = float(payload.get("score", -1.0))
    previous_score = -1.0
    if path.exists():
        try:
            previous_score = float(json.loads(path.read_text(encoding="utf-8")).get("score", -1.0))
        except Exception:
            previous_score = -1.0
    if current_score > previous_score:
        atomic_write_json(path, payload)


def run_cmd(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=30)
        return out.strip()
    except Exception as exc:
        return f"COMMAND_FAILED {cmd!r}: {exc!r}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_leaky_column(name: str) -> bool:
    lower = name.lower()
    if lower in ID_COLUMNS:
        return True
    return any(token in lower for token in LEAKY_TOKENS)


def parse_feature_list(text: str) -> set[str]:
    return {part.strip() for part in text.split(",") if part.strip()}


def detect_feature_schema(
    train_df: pd.DataFrame,
    exclude_features: set[str] | None = None,
    exclude_categorical_features: set[str] | None = None,
) -> dict[str, Any]:
    exclude_features = exclude_features or set()
    exclude_categorical_features = exclude_categorical_features or set()
    numeric_features: list[str] = []
    for col in PREFERRED_NUMERIC_FEATURES:
        if col in train_df.columns and col not in exclude_features and not is_leaky_column(col):
            numeric_features.append(col)
    for col in train_df.columns:
        if col in numeric_features or col in exclude_features or is_leaky_column(col):
            continue
        if pd.api.types.is_numeric_dtype(train_df[col]):
            numeric_features.append(col)

    categorical_maps: dict[str, dict[str, int]] = {}
    for col in CATEGORICAL_FEATURES:
        if col not in train_df.columns or col in exclude_categorical_features:
            continue
        values = train_df[col].fillna("<MISSING>").astype(str)
        uniques = sorted(values.value_counts().head(128).index.tolist())
        categorical_maps[col] = {value: i for i, value in enumerate(uniques)}

    return {"numeric_features": numeric_features, "categorical_maps": categorical_maps}


def build_matrix(df: pd.DataFrame, schema: dict[str, Any]) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for col in schema["numeric_features"]:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            arrays.append(values.fillna(0.0).to_numpy(dtype=np.float32).reshape(-1, 1))
        else:
            arrays.append(np.zeros((len(df), 1), dtype=np.float32))

    categorical_maps = schema["categorical_maps"]
    ordered_categorical_cols = [col for col in CATEGORICAL_FEATURES if col in categorical_maps]
    ordered_categorical_cols.extend(sorted(col for col in categorical_maps if col not in set(ordered_categorical_cols)))
    for col in ordered_categorical_cols:
        mapping = categorical_maps[col]
        width = len(mapping) + 1
        mat = np.zeros((len(df), width), dtype=np.float32)
        if col in df.columns:
            codes = df[col].fillna("<MISSING>").astype(str).map(mapping).fillna(len(mapping)).astype(int)
            mat[np.arange(len(df)), codes.to_numpy()] = 1.0
        arrays.append(mat)

    if not arrays:
        raise RuntimeError("No usable non-leaky features were detected.")
    return np.concatenate(arrays, axis=1)


def standardize(train_x: np.ndarray, other: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray], dict[str, Any]]:
    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    train_z = (train_x - mean) / std
    other_z = [(x - mean) / std for x in other]
    stats = {"mean": mean.tolist(), "std": std.tolist()}
    return train_z.astype(np.float32), [x.astype(np.float32) for x in other_z], stats


def binary_label(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.zeros(len(df), dtype=np.float32)
    return pd.to_numeric(df[col], errors="coerce").fillna(0).clip(0, 1).to_numpy(dtype=np.float32)


def route_metrics(df: pd.DataFrame, scores: np.ndarray, prefix: str = "") -> dict[str, float]:
    eval_df = pd.DataFrame(
        {
            "sample_id": df["sample_id"].astype(str).to_numpy(),
            "score": scores.astype(np.float64),
            "relaxed": binary_label(df, "relaxed_route_hit_if_eval"),
            "usable_relaxed": binary_label(df, "usable_relaxed_route_hit_if_eval"),
            "strict": binary_label(df, "strict_route_hit_if_eval"),
            "known": binary_label(df, "atmosphere_known_mask") if "atmosphere_known_mask" in df.columns else np.ones(len(df)),
        }
    )
    eval_df = eval_df.sort_values(["sample_id", "score"], ascending=[True, False])
    grouped = eval_df.groupby("sample_id", sort=False)
    n_groups = max(len(grouped), 1)

    metrics: dict[str, float] = {}
    for k in (1, 10, 200):
        top = grouped.head(k)
        relaxed_hit = top.groupby("sample_id")["relaxed"].max().mean()
        usable_hit = top.groupby("sample_id")["usable_relaxed"].max().mean()
        strict_groups = eval_df.groupby("sample_id")["known"].max()
        strict_ids = strict_groups[strict_groups > 0].index
        strict_top = top[top["sample_id"].isin(strict_ids)]
        strict_hit = strict_top.groupby("sample_id")["strict"].max().reindex(strict_ids, fill_value=0).mean()
        metrics[f"{prefix}missing_top{k}_relaxed_route"] = float(relaxed_hit if not math.isnan(relaxed_hit) else 0.0)
        metrics[f"{prefix}usable_top{k}_relaxed_route"] = float(usable_hit if not math.isnan(usable_hit) else 0.0)
        metrics[f"{prefix}strict_top{k}_relaxed_route"] = float(strict_hit if not math.isnan(strict_hit) else 0.0)
    metrics[f"{prefix}num_samples"] = float(n_groups)
    metrics[f"{prefix}num_candidates"] = float(len(eval_df))
    return metrics


def passes_stage35_gate(metrics: dict[str, float]) -> bool:
    return (
        metrics.get("missing_top1_relaxed_route", 0.0) > BASELINE_STAGE35["missing_top1_relaxed_route"]
        and metrics.get("missing_top10_relaxed_route", 0.0) >= BASELINE_STAGE35["missing_top10_relaxed_route"]
        and metrics.get("strict_top1_relaxed_route", 0.0) > BASELINE_STAGE35["strict_top1_relaxed_route"]
        and metrics.get("strict_top10_relaxed_route", 0.0) >= BASELINE_STAGE35["strict_top10_relaxed_route"]
    )


def selection_score(metrics: dict[str, float]) -> float:
    """Gate-aware validation objective used only for model selection."""

    top1 = metrics.get("missing_top1_relaxed_route", 0.0) + metrics.get("strict_top1_relaxed_route", 0.0)
    top10 = metrics.get("missing_top10_relaxed_route", 0.0) + metrics.get("strict_top10_relaxed_route", 0.0)
    top200 = metrics.get("missing_top200_relaxed_route", 0.0) + metrics.get("strict_top200_relaxed_route", 0.0)
    usable = metrics.get("usable_top1_relaxed_route", 0.0)
    margin = (
        max(0.0, metrics.get("missing_top1_relaxed_route", 0.0) - BASELINE_STAGE35["missing_top1_relaxed_route"])
        + max(0.0, metrics.get("strict_top1_relaxed_route", 0.0) - BASELINE_STAGE35["strict_top1_relaxed_route"])
        + 0.5 * max(0.0, metrics.get("missing_top10_relaxed_route", 0.0) - BASELINE_STAGE35["missing_top10_relaxed_route"])
        + 0.5 * max(0.0, metrics.get("strict_top10_relaxed_route", 0.0) - BASELINE_STAGE35["strict_top10_relaxed_route"])
    )
    gate_bonus = 10.0 if passes_stage35_gate(metrics) else 0.0
    return float(gate_bonus + 8.0 * top1 + 2.0 * top10 + 0.4 * top200 + 0.5 * usable + 20.0 * margin)


def normalize_scores(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    std = float(arr.std())
    if std < 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - float(arr.mean())) / std


def quantile_minmax(df: pd.DataFrame, col: str) -> np.ndarray:
    if col in df.columns:
        vals = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        vals = pd.Series(0.0, index=df.index)
    lo = vals.quantile(0.01)
    hi = vals.quantile(0.99)
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        return ((vals - lo) / (hi - lo)).clip(0, 1).to_numpy(dtype=np.float64)
    return vals.to_numpy(dtype=np.float64)


def v3_final_formula_score(df: pd.DataFrame) -> np.ndarray:
    """Inference-safe reproduction of the v3 final calibration search_id=4."""

    route_raw = quantile_minmax(df, "route_total_score_raw")
    precursor_rank = pd.to_numeric(df.get("precursor_rank", 999), errors="coerce").clip(lower=1).fillna(999).to_numpy(dtype=np.float64)
    condition_rank = pd.to_numeric(df.get("condition_rank", 999), errors="coerce").clip(lower=1).fillna(999).to_numpy(dtype=np.float64)
    precursor_rank_score = 1.0 / precursor_rank
    condition_rank_score = 1.0 / condition_rank
    rank_product_score = precursor_rank_score * condition_rank_score
    open_flag = quantile_minmax(df, "contains_open_generated_precursor")
    repair_flag = quantile_minmax(df, "contains_repair_precursor")
    return (
        0.8 * route_raw
        + 0.5 * precursor_rank_score
        + 0.5 * condition_rank_score
        + 0.7 * rank_product_score
        - 0.2 * open_flag
        - 0.1 * repair_flag
    )


def base_score_candidates(df: pd.DataFrame) -> dict[str, np.ndarray]:
    candidates: dict[str, np.ndarray] = {}
    score_columns = [
        "route_total_score_raw",
        "total_score_raw",
        "condition_calibrated_score_v3",
        "condition_score",
        "precursor_score",
        "precursor_score_norm",
    ]
    rank_columns = ["route_rank_raw", "condition_rank_calibrated_v3", "condition_rank", "precursor_rank"]
    for col in score_columns:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            candidates[col] = normalize_scores(values)
    for col in rank_columns:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").fillna(1e9).to_numpy(dtype=np.float64)
            candidates[f"neg_{col}"] = normalize_scores(-values)
    if "route_total_score_raw" in df.columns and "precursor_rank" in df.columns and "condition_rank" in df.columns:
        candidates["route_calibrated_score_v3_final_formula"] = normalize_scores(v3_final_formula_score(df))
    return candidates


def blend_scores(df: pd.DataFrame, model_scores: np.ndarray, blend: dict[str, Any]) -> np.ndarray:
    alpha = float(blend.get("alpha_model", 1.0))
    base_name = str(blend.get("base_score", "__model_only__"))
    model_z = normalize_scores(model_scores)
    if base_name == "__model_only__":
        return model_z
    candidates = base_score_candidates(df)
    base = candidates.get(base_name)
    if base is None:
        return model_z
    return alpha * model_z + (1.0 - alpha) * base


def search_score_blend(df: pd.DataFrame, model_scores: np.ndarray, max_alpha_model: float = 1.0) -> dict[str, Any]:
    alpha_grid = [a for a in [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 1.0] if a <= max_alpha_model + 1e-12]
    if not alpha_grid:
        alpha_grid = [0.0]
    model_z = normalize_scores(model_scores)
    base_candidates = base_score_candidates(df)
    all_candidates = {"__model_only__": np.zeros_like(model_z), **base_candidates}

    best: dict[str, Any] | None = None
    for base_name, base_values in all_candidates.items():
        local_grid = [1.0] if base_name == "__model_only__" else alpha_grid
        for alpha in local_grid:
            final_scores = model_z if base_name == "__model_only__" else alpha * model_z + (1.0 - alpha) * base_values
            metrics = route_metrics(df, final_scores)
            score = selection_score(metrics)
            payload = {
                "score": score,
                "metrics": metrics,
                "blend": {
                    "base_score": base_name,
                    "alpha_model": float(alpha),
                    "score_formula": "alpha*z(model_score)+(1-alpha)*z(base_score)",
                },
            }
            if best is None or score > best["score"]:
                best = payload
    if best is None:
        metrics = route_metrics(df, model_z)
        best = {
            "score": selection_score(metrics),
            "metrics": metrics,
            "blend": {"base_score": "__model_only__", "alpha_model": 1.0, "score_formula": "z(model_score)"},
        }
    return best


def stable_val_split_masks(df: pd.DataFrame, holdout_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    sample_ids = df["sample_id"].astype(str)
    unique_ids = sorted(sample_ids.unique().tolist())
    holdout_ids = set()
    for sample_id in unique_ids:
        digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()
        bucket = int(digest[:12], 16) / float(16**12 - 1)
        if bucket < holdout_fraction:
            holdout_ids.add(sample_id)
    if not holdout_ids or len(holdout_ids) == len(unique_ids):
        midpoint = max(1, len(unique_ids) // 2)
        holdout_ids = set(unique_ids[:midpoint])
    holdout_mask = sample_ids.isin(holdout_ids).to_numpy(dtype=bool)
    tune_mask = ~holdout_mask
    return tune_mask, holdout_mask


def robust_search_score_blend(
    df: pd.DataFrame,
    model_scores: np.ndarray,
    tune_mask: np.ndarray,
    holdout_mask: np.ndarray,
    max_alpha_model: float = 1.0,
) -> dict[str, Any]:
    tune_df = df.loc[tune_mask].copy()
    holdout_df = df.loc[holdout_mask].copy()
    tune_scores = model_scores[tune_mask]
    holdout_scores = model_scores[holdout_mask]

    tune_result = search_score_blend(tune_df, tune_scores, max_alpha_model=max_alpha_model)
    blend = tune_result["blend"]
    holdout_final = blend_scores(holdout_df, holdout_scores, blend)
    full_final = blend_scores(df, model_scores, blend)
    holdout_metrics = route_metrics(holdout_df, holdout_final)
    full_metrics = route_metrics(df, full_final)
    return {
        "score": selection_score(holdout_metrics),
        "metrics": holdout_metrics,
        "tune_metrics": tune_result["metrics"],
        "full_val_metrics": full_metrics,
        "blend": blend,
        "passes_gate": passes_stage35_gate(holdout_metrics) and passes_stage35_gate(full_metrics),
        "passes_gate_holdout": passes_stage35_gate(holdout_metrics),
        "passes_gate_full": passes_stage35_gate(full_metrics),
    }


class RouteRanker(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden // 4),
            nn.SiLU(),
            nn.Linear(hidden // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class Variant:
    seed: int
    hidden: int
    dropout: float
    lr: float
    batch_size: int
    weight_decay: float
    pairwise_weight: float = 0.0
    pairwise_pairs_per_epoch: int = 0


def build_pair_groups(df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    relaxed = binary_label(df, "relaxed_route_hit_if_eval")
    strict = binary_label(df, "strict_route_hit_if_eval")
    usable = binary_label(df, "usable_relaxed_route_hit_if_eval")
    relevance = relaxed + 2.0 * strict + 0.5 * usable
    work = pd.DataFrame({"sample_id": df["sample_id"].astype(str).to_numpy(), "row": np.arange(len(df)), "rel": relevance})
    pair_groups: list[tuple[np.ndarray, np.ndarray]] = []
    for _, group in work.groupby("sample_id", sort=False):
        positives = group.loc[group["rel"] > 0, "row"].to_numpy(dtype=np.int64)
        negatives = group.loc[group["rel"] <= 0, "row"].to_numpy(dtype=np.int64)
        if len(positives) and len(negatives):
            pair_groups.append((positives, negatives))
    return pair_groups


def sample_pair_indices(
    pair_groups: list[tuple[np.ndarray, np.ndarray]],
    num_pairs: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    group_ids = rng.integers(0, len(pair_groups), size=num_pairs)
    pos = np.empty(num_pairs, dtype=np.int64)
    neg = np.empty(num_pairs, dtype=np.int64)
    for i, gid in enumerate(group_ids):
        positives, negatives = pair_groups[int(gid)]
        pos[i] = positives[int(rng.integers(0, len(positives)))]
        neg[i] = negatives[int(rng.integers(0, len(negatives)))]
    return pos, neg


def score_in_batches(model: nn.Module, x: torch.Tensor, batch_size: int) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, x.shape[0], batch_size):
            logits = model(x[start : start + batch_size])
            chunks.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(chunks)


def train_variant(
    variant: Variant,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_w: torch.Tensor,
    pair_groups: list[tuple[np.ndarray, np.ndarray]],
    val_x: torch.Tensor,
    val_df: pd.DataFrame,
    val_tune_mask: np.ndarray | None,
    val_holdout_mask: np.ndarray | None,
    robust_val_split: bool,
    out_dir: Path,
    log,
    end_time: float,
    heartbeat_minutes: float,
    eval_every_epochs: int,
    max_alpha_model: float,
) -> dict[str, Any]:
    random.seed(variant.seed)
    np.random.seed(variant.seed)
    torch.manual_seed(variant.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(variant.seed)

    model = RouteRanker(train_x.shape[1], variant.hidden, variant.dropout).to(train_x.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=variant.lr, weight_decay=variant.weight_decay)
    pos = float(train_y.sum().detach().cpu().item())
    neg = float(train_y.numel() - pos)
    pos_weight = torch.tensor(min(max(neg / max(pos, 1.0), 1.0), 50.0), device=train_x.device)
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    rng = np.random.default_rng(variant.seed + 17)

    best: dict[str, Any] = {"score": -1.0, "epoch": 0, "metrics": {}, "checkpoint": None}
    last_eval = 0
    last_heartbeat = time.time()
    epoch = 0

    while time.time() < end_time:
        epoch += 1
        model.train()
        order = torch.randperm(train_x.shape[0], device=train_x.device)
        losses = []
        pair_losses = []
        for start in range(0, train_x.shape[0], variant.batch_size):
            idx = order[start : start + variant.batch_size]
            logits = model(train_x[idx])
            loss_vec = criterion(logits, train_y[idx])
            loss = (loss_vec * train_w[idx]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            if time.time() >= end_time:
                break

        if variant.pairwise_weight > 0 and variant.pairwise_pairs_per_epoch > 0 and pair_groups and time.time() < end_time:
            remaining = int(variant.pairwise_pairs_per_epoch)
            while remaining > 0 and time.time() < end_time:
                num_pairs = min(remaining, max(4096, variant.batch_size // 4))
                pos_np, neg_np = sample_pair_indices(pair_groups, num_pairs, rng)
                pos_idx = torch.tensor(pos_np, dtype=torch.long, device=train_x.device)
                neg_idx = torch.tensor(neg_np, dtype=torch.long, device=train_x.device)
                diff = model(train_x[pos_idx]) - model(train_x[neg_idx])
                pair_loss = torch.nn.functional.softplus(-diff).mean()
                loss = variant.pairwise_weight * pair_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                pair_losses.append(float(pair_loss.detach().cpu().item()))
                remaining -= num_pairs

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        mean_pair_loss = float(np.mean(pair_losses)) if pair_losses else 0.0
        log(f"variant={asdict(variant)} epoch={epoch} loss={mean_loss:.6f} pair_loss={mean_pair_loss:.6f}")

        should_eval = epoch == 1 or epoch - last_eval >= eval_every_epochs or time.time() - last_heartbeat >= heartbeat_minutes * 60
        if should_eval:
            last_eval = epoch
            last_heartbeat = time.time()
            model_scores = score_in_batches(model, val_x, max(variant.batch_size * 2, 65536))
            pure_metrics = route_metrics(val_df, model_scores)
            if robust_val_split and val_tune_mask is not None and val_holdout_mask is not None:
                blend_result = robust_search_score_blend(
                    val_df,
                    model_scores,
                    val_tune_mask,
                    val_holdout_mask,
                    max_alpha_model=max_alpha_model,
                )
            else:
                blend_result = search_score_blend(val_df, model_scores, max_alpha_model=max_alpha_model)
                blend_result["full_val_metrics"] = blend_result["metrics"]
                blend_result["tune_metrics"] = blend_result["metrics"]
                blend_result["passes_gate"] = passes_stage35_gate(blend_result["metrics"])
                blend_result["passes_gate_holdout"] = blend_result["passes_gate"]
                blend_result["passes_gate_full"] = blend_result["passes_gate"]
            metrics = blend_result["metrics"]
            combined = blend_result["score"]
            log(
                "VAL "
                + json.dumps(
                    {
                        "variant": asdict(variant),
                        "epoch": epoch,
                        "loss": mean_loss,
                        "pair_loss": mean_pair_loss,
                        "metrics": metrics,
                        "tune_metrics": blend_result.get("tune_metrics"),
                        "full_val_metrics": blend_result.get("full_val_metrics"),
                        "pure_model_metrics": pure_metrics,
                        "blend": blend_result["blend"],
                        "passes_gate": blend_result.get("passes_gate", passes_stage35_gate(metrics)),
                        "passes_gate_holdout": blend_result.get("passes_gate_holdout", passes_stage35_gate(metrics)),
                        "passes_gate_full": blend_result.get("passes_gate_full", passes_stage35_gate(metrics)),
                    },
                    sort_keys=True,
                )
            )
            atomic_write_json(
                out_dir / "latest_progress.json",
                {
                    "variant": asdict(variant),
                    "epoch": epoch,
                    "metrics": metrics,
                    "tune_metrics": blend_result.get("tune_metrics"),
                    "full_val_metrics": blend_result.get("full_val_metrics"),
                    "pure_model_metrics": pure_metrics,
                    "blend": blend_result["blend"],
                    "passes_gate": blend_result.get("passes_gate", passes_stage35_gate(metrics)),
                },
            )
            if combined > best["score"]:
                ckpt = out_dir / "checkpoints" / f"stage35_gpu_ranker_seed{variant.seed}_epoch{epoch}.pt"
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "variant": asdict(variant),
                        "metrics": metrics,
                        "tune_metrics": blend_result.get("tune_metrics"),
                        "full_val_metrics": blend_result.get("full_val_metrics"),
                        "pure_model_metrics": pure_metrics,
                        "blend": blend_result["blend"],
                        "passes_gate": blend_result.get("passes_gate", passes_stage35_gate(metrics)),
                    },
                    ckpt,
                )
                best = {
                    "score": combined,
                    "epoch": epoch,
                    "metrics": metrics,
                    "tune_metrics": blend_result.get("tune_metrics"),
                    "full_val_metrics": blend_result.get("full_val_metrics"),
                    "pure_model_metrics": pure_metrics,
                    "blend": blend_result["blend"],
                    "passes_gate": blend_result.get("passes_gate", passes_stage35_gate(metrics)),
                    "checkpoint": str(ckpt),
                }
                variant_dir = out_dir / "variant_bests"
                variant_dir.mkdir(parents=True, exist_ok=True)
                atomic_write_json(variant_dir / f"seed{variant.seed}_best.json", best)
                maybe_update_best_json(out_dir / "online_best_val_model.json", best)

        if epoch >= 200 and time.time() < end_time:
            break

    return best


def make_variants(seed: int) -> list[Variant]:
    variants: list[Variant] = []
    hidden_options = [512, 1024, 1536, 2048]
    lr_options = [8e-4, 4e-4, 2e-4]
    dropout_options = [0.05, 0.10, 0.15]
    batch_options = [65536, 131072]
    pairwise_options = [(1.0, 65536), (0.5, 65536), (1.5, 131072), (0.0, 0)]
    i = 0
    for hidden in hidden_options:
        for lr in lr_options:
            for dropout in dropout_options:
                batch_size = batch_options[i % len(batch_options)]
                pairwise_weight, pairwise_pairs = pairwise_options[i % len(pairwise_options)]
                variants.append(
                    Variant(
                        seed=seed + i,
                        hidden=hidden,
                        dropout=dropout,
                        lr=lr,
                        batch_size=batch_size,
                        weight_decay=1e-4,
                        pairwise_weight=pairwise_weight,
                        pairwise_pairs_per_epoch=pairwise_pairs,
                    )
                )
                i += 1
    return variants


def build_manifest(out_dir: Path, project_root: Path) -> None:
    manifest = []
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            manifest.append(
                {
                    "path": relpath(path, project_root),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "description": "AutoDL GPU 24h training artifact",
                    "paper_primary_result": path.name in {"best_val_model.json", "test_metrics_if_gate_passed.json"},
                    "diagnostic_result": path.suffix in {".log", ".json", ".csv"},
                    "reproducible": True,
                }
            )
        except Exception:
            continue
    atomic_write_json(out_dir / "artifact_manifest.json", manifest)


def write_final_report(out_dir: Path, started_at: str, ended_at: str, best: dict[str, Any], test_metrics: dict[str, Any] | None) -> None:
    lines = [
        "# SynPred AutoDL GPU 24h Training Report",
        "",
        f"- Started: {started_at}",
        f"- Ended: {ended_at}",
        f"- CUDA available: {torch.cuda.is_available()}",
        f"- GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU-only fallback'}",
        "- Default inference selector updated: no",
        "",
        "## Best Validation Candidate",
        "",
        "```json",
        json.dumps(best, indent=2, sort_keys=True),
        "```",
        "",
        "## Test Evaluation",
        "",
    ]
    if test_metrics:
        lines += ["Validation passed Stage35 replacement gate, so test was evaluated.", "", "```json", json.dumps(test_metrics, indent=2, sort_keys=True), "```"]
    else:
        lines += ["No test run was promoted unless a validation candidate passed the strict Stage35 gate."]
    lines += [
        "",
        "## Notes",
        "",
        "- This run uses the existing v3 final route-candidate CSVs and trains neural route rerankers on AutoDL GPU.",
        "- Evaluation is validation-first; test is only used after the gate passes.",
        "- Passwords, keys, and connection credentials are not written by this script.",
    ]
    (out_dir / "GPU_24H_TRAINING_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--output_root", default="outputs/autorun/gpu_24h_training_20260614")
    parser.add_argument("--max_hours", type=float, default=24.0)
    parser.add_argument("--heartbeat_minutes", type=float, default=30.0)
    parser.add_argument("--eval_every_epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_test_only_if_val_pass", type=int, default=1)
    parser.add_argument("--robust_val_split", type=int, default=1)
    parser.add_argument("--val_holdout_fraction", type=float, default=0.5)
    parser.add_argument("--exclude_features", default="")
    parser.add_argument("--exclude_categorical_features", default="")
    parser.add_argument("--max_alpha_model", type=float, default=1.0)
    parser.add_argument(
        "--label_mode",
        choices=["route_mix", "usable", "strict", "relaxed"],
        default="route_mix",
        help="Training target recipe. route_mix is the historical default.",
    )
    parser.add_argument("--train_csv", default="outputs/evaluation/stage35_route_candidates_v3_final_20260612/train_route_candidates.csv")
    parser.add_argument("--val_csv", default="outputs/evaluation/stage35_route_candidates_v3_final_20260612/val_route_candidates.csv")
    parser.add_argument("--test_csv", default="outputs/evaluation/stage35_route_candidates_v3_final_20260612/test_route_candidates.csv")
    args = parser.parse_args()

    started_at = now()
    project_root = Path(args.project_root).resolve()
    out_dir = (project_root / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "gpu_training_queue.log"

    def log(message: str) -> None:
        text = f"[{now()}] {message}"
        print(text, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(text + "\n")

    log("starting AutoDL GPU training queue")
    log(f"python={sys.executable}")
    log(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
    log("nvidia-smi=" + run_cmd(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"]))
    log("disk=" + run_cmd(["df", "-h", str(project_root)]))

    train_path = project_root / args.train_csv
    val_path = project_root / args.val_csv
    test_path = project_root / args.test_csv
    for path in [train_path, val_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    log(f"loading train CSV {train_path}")
    train_df = pd.read_csv(train_path)
    log(f"loading val CSV {val_path}")
    val_df = pd.read_csv(val_path)
    val_tune_mask = val_holdout_mask = None
    if args.robust_val_split:
        val_tune_mask, val_holdout_mask = stable_val_split_masks(val_df, args.val_holdout_fraction, args.seed)
        log(
            "robust_val_split="
            + json.dumps(
                {
                    "enabled": True,
                    "tune_candidates": int(val_tune_mask.sum()),
                    "holdout_candidates": int(val_holdout_mask.sum()),
                    "tune_samples": int(val_df.loc[val_tune_mask, "sample_id"].nunique()),
                    "holdout_samples": int(val_df.loc[val_holdout_mask, "sample_id"].nunique()),
                    "holdout_fraction": args.val_holdout_fraction,
                    "seed": args.seed,
                },
                sort_keys=True,
            )
        )
    exclude_features = parse_feature_list(args.exclude_features)
    exclude_categorical_features = parse_feature_list(args.exclude_categorical_features)
    schema = detect_feature_schema(train_df, exclude_features, exclude_categorical_features)
    atomic_write_json(
        out_dir / "feature_exclusion_policy.json",
        {
            "exclude_features": sorted(exclude_features),
            "exclude_categorical_features": sorted(exclude_categorical_features),
            "reason": "User-requested/diagnostic shift-prone feature exclusion for robust Stage35 training.",
        },
    )
    atomic_write_json(out_dir / "feature_schema.json", schema)
    log(f"detected {len(schema['numeric_features'])} numeric features and {len(schema['categorical_maps'])} categorical fields")

    train_x_np = build_matrix(train_df, schema)
    val_x_np = build_matrix(val_df, schema)
    train_x_np, [val_x_np], stats = standardize(train_x_np, [val_x_np])
    atomic_write_json(out_dir / "standardization_stats.json", stats)

    train_relaxed = binary_label(train_df, "relaxed_route_hit_if_eval")
    train_strict = binary_label(train_df, "strict_route_hit_if_eval")
    train_usable = binary_label(train_df, "usable_relaxed_route_hit_if_eval")
    if args.label_mode == "usable":
        train_y_np = np.clip(train_usable + 0.35 * train_relaxed + 0.25 * train_strict, 0.0, 1.0).astype(np.float32)
        train_w_np = (1.0 + 2.0 * train_usable + 1.0 * train_relaxed + 1.0 * train_strict).astype(np.float32)
    elif args.label_mode == "strict":
        train_y_np = np.clip(train_strict + 0.4 * train_relaxed + 0.2 * train_usable, 0.0, 1.0).astype(np.float32)
        train_w_np = (1.0 + 3.0 * train_strict + 1.0 * train_relaxed + 0.5 * train_usable).astype(np.float32)
    elif args.label_mode == "relaxed":
        train_y_np = np.clip(train_relaxed + 0.35 * train_strict + 0.15 * train_usable, 0.0, 1.0).astype(np.float32)
        train_w_np = (1.0 + 2.0 * train_relaxed + 1.0 * train_strict + 0.5 * train_usable).astype(np.float32)
    else:
        train_y_np = np.clip(train_relaxed + 0.5 * train_strict + 0.25 * train_usable, 0.0, 1.0).astype(np.float32)
        train_w_np = (1.0 + 2.0 * train_strict + 1.0 * train_usable).astype(np.float32)
    log(
        "label_mode="
        + json.dumps(
            {
                "mode": args.label_mode,
                "positive_mean": float(train_y_np.mean()),
                "strict_rate": float(train_strict.mean()),
                "relaxed_rate": float(train_relaxed.mean()),
                "usable_rate": float(train_usable.mean()),
            },
            sort_keys=True,
        )
    )
    pair_groups = build_pair_groups(train_df)
    log(f"built {len(pair_groups)} sample-level pairwise hard-negative groups")

    baseline_payloads = []
    for base_name, base_values in base_score_candidates(val_df).items():
        metrics = route_metrics(val_df, base_values)
        baseline_payloads.append({"base_score": base_name, "score": selection_score(metrics), "metrics": metrics})
    baseline_payloads.sort(key=lambda x: x["score"], reverse=True)
    baseline_metrics = baseline_payloads[0]["metrics"] if baseline_payloads else {}
    atomic_write_json(out_dir / "baseline_val_score_candidates.json", baseline_payloads)
    log("baseline_val=" + json.dumps(baseline_payloads[:3], sort_keys=True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x = torch.tensor(train_x_np, device=device)
    train_y = torch.tensor(train_y_np, device=device)
    train_w = torch.tensor(train_w_np, device=device)
    val_x = torch.tensor(val_x_np, device=device)
    del train_x_np, val_x_np

    end_time = time.time() + args.max_hours * 3600.0
    all_bests: list[dict[str, Any]] = []
    best_global: dict[str, Any] = {"score": -1.0, "metrics": {}}

    variants = make_variants(args.seed)
    variant_index = 0
    while time.time() < end_time:
        variant = variants[variant_index % len(variants)]
        cycle = variant_index // len(variants)
        if cycle:
            variant = Variant(
                seed=variant.seed + cycle * 1000,
                hidden=variant.hidden,
                dropout=variant.dropout,
                lr=variant.lr * (0.85**cycle),
                batch_size=variant.batch_size,
                weight_decay=variant.weight_decay,
                pairwise_weight=variant.pairwise_weight,
                pairwise_pairs_per_epoch=variant.pairwise_pairs_per_epoch,
            )
        log(f"starting variant {variant_index + 1}: {asdict(variant)}")
        best = train_variant(
            variant,
            train_x,
            train_y,
            train_w,
            pair_groups,
            val_x,
            val_df,
            val_tune_mask,
            val_holdout_mask,
            bool(args.robust_val_split),
            out_dir,
            log,
            end_time,
            args.heartbeat_minutes,
            args.eval_every_epochs,
            float(args.max_alpha_model),
        )
        all_bests.append(best)
        atomic_write_json(out_dir / "variant_bests.json", all_bests)
        if best.get("score", -1.0) > best_global.get("score", -1.0):
            best_global = best
            atomic_write_json(out_dir / "best_val_model.json", best_global)
        variant_index += 1

    test_metrics = None
    best_metrics = best_global.get("metrics", {})
    best_full_metrics = best_global.get("full_val_metrics", best_metrics)
    if (
        args.run_test_only_if_val_pass
        and passes_stage35_gate(best_metrics)
        and passes_stage35_gate(best_full_metrics)
        and test_path.exists()
        and best_global.get("checkpoint")
    ):
        log("validation gate passed; loading test CSV")
        test_df = pd.read_csv(test_path)
        test_x_np = build_matrix(test_df, schema)
        mean = np.array(stats["mean"], dtype=np.float32)
        std = np.array(stats["std"], dtype=np.float32)
        test_x_np = ((test_x_np - mean) / std).astype(np.float32)
        test_x = torch.tensor(test_x_np, device=device)
        ckpt = torch.load(best_global["checkpoint"], map_location=device)
        variant_payload = ckpt["variant"]
        model = RouteRanker(test_x.shape[1], int(variant_payload["hidden"]), float(variant_payload["dropout"])).to(device)
        model.load_state_dict(ckpt["model_state"])
        model_scores = score_in_batches(model, test_x, max(int(variant_payload["batch_size"]) * 2, 65536))
        final_scores = blend_scores(test_df, model_scores, ckpt.get("blend", best_global.get("blend", {})))
        test_metrics = route_metrics(test_df, final_scores)
        test_payload = {
            "metrics": test_metrics,
            "blend": ckpt.get("blend", best_global.get("blend", {})),
            "pure_model_metrics": route_metrics(test_df, model_scores),
        }
        atomic_write_json(out_dir / "test_metrics_if_gate_passed.json", test_payload)
        log("test_metrics=" + json.dumps(test_payload, sort_keys=True))
    else:
        log("validation gate not passed or test disabled; skipping test")

    ended_at = now()
    atomic_write_json(
        out_dir / "run_summary.json",
        {
            "started_at": started_at,
            "ended_at": ended_at,
            "max_hours": args.max_hours,
            "best": best_global,
            "test_metrics": test_metrics,
            "baseline_stage35": BASELINE_STAGE35,
            "default_inference_selector_updated": False,
            "device": str(device),
        },
    )
    write_final_report(out_dir, started_at, ended_at, best_global, test_metrics)
    build_manifest(out_dir, project_root)
    log("finished AutoDL GPU training queue")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
