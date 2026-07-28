#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pymatgen.core import Composition, Element


SetKey = Tuple[int, ...]
ELEMENT_PATTERN = re.compile(r"[A-Z][a-z]?")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_candidates(paths: Sequence[str], n_rows: int) -> tuple[List[List[SetKey]], List[List[float]]]:
    merged: List[List[SetKey]] = [[] for _ in range(n_rows)]
    merged_scores: List[List[float]] = [[] for _ in range(n_rows)]
    seen = [set() for _ in range(n_rows)]
    for raw_path in paths:
        path = Path(raw_path).resolve()
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                index = int(row["row_index"])
                values = row["candidate_label_ids"]
                raw_scores = row.get("scores", [])
                for rank, value in enumerate(values):
                    key = tuple(sorted({int(item) for item in value}))
                    if not key or key in seen[index]:
                        continue
                    seen[index].add(key)
                    merged[index].append(key)
                    score = float(raw_scores[rank]) if rank < len(raw_scores) else -math.log1p(rank)
                    merged_scores[index].append(score)
    return merged, merged_scores


def append_source_features(x: np.ndarray, meta_path: Path, source_vocab: Sequence[str]) -> np.ndarray:
    meta = pd.read_csv(meta_path, usecols=["source_dataset"])
    values = meta["source_dataset"].fillna("").astype(str).to_numpy()
    matrix = np.zeros((len(values), len(source_vocab)), dtype=np.float32)
    lookup = {value: index for index, value in enumerate(source_vocab)}
    for row, value in enumerate(values):
        if value in lookup:
            matrix[row, lookup[value]] = 1.0
    return np.hstack([x, matrix]).astype(np.float32)


def precursor_formula_features(names: Sequence[str]) -> np.ndarray:
    """Element/group descriptors let the ranker generalize to rare precursor IDs."""
    output = np.zeros((len(names), 118 + 18 + 7 + 4), dtype=np.float32)
    for row, raw_name in enumerate(names):
        try:
            amounts = Composition(str(raw_name)).get_el_amt_dict()
        except Exception:
            amounts = {symbol: 1.0 for symbol in ELEMENT_PATTERN.findall(str(raw_name))}
        parsed = []
        for symbol, raw_amount in amounts.items():
            try:
                element = Element(str(symbol))
                amount = max(float(raw_amount), 0.0)
            except Exception:
                continue
            if amount > 0:
                parsed.append((element, amount))
        total = sum(amount for _, amount in parsed)
        if total <= 0:
            continue
        electronegativity = 0.0
        for element, amount in parsed:
            fraction = amount / total
            output[row, int(element.Z) - 1] += fraction
            if element.group is not None and 1 <= int(element.group) <= 18:
                output[row, 118 + int(element.group) - 1] += fraction
            if element.row is not None and 1 <= int(element.row) <= 7:
                output[row, 118 + 18 + int(element.row) - 1] += fraction
            electronegativity += fraction * float(element.X or 0.0)
        base = 118 + 18 + 7
        output[row, base] = min(len(parsed), 10) / 10.0
        output[row, base + 1] = sum(element.Z * amount / total for element, amount in parsed) / 100.0
        output[row, base + 2] = electronegativity / 4.0
        output[row, base + 3] = min(total, 20.0) / 20.0
    return output


def append_query_formula_features(x: np.ndarray, meta_path: Path) -> np.ndarray:
    formulas = pd.read_csv(meta_path, usecols=["formula"])["formula"].fillna("").astype(str).tolist()
    return np.hstack([x, precursor_formula_features(formulas)]).astype(np.float32)


def restrict_dataset_families(dataset: Dataset, meta_path: Path, raw_families: str) -> List[str]:
    families = sorted({value.strip() for value in str(raw_families).split(",") if value.strip()})
    if not families:
        return []
    family_values = pd.read_csv(meta_path, usecols=["family_signature_primary"])[
        "family_signature_primary"
    ].fillna("UNK").astype(str).to_numpy()
    allowed = {index for index, value in enumerate(family_values) if value in set(families)}
    dataset.row_indices = [index for index in dataset.row_indices if index in allowed]
    return families


def restrict_dataset_min_set_len(dataset: Dataset, set_lengths: np.ndarray, minimum: int) -> None:
    if int(minimum) <= 0:
        return
    dataset.row_indices = [index for index in dataset.row_indices if int(set_lengths[index]) >= int(minimum)]


class CandidateDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        candidates: Sequence[Sequence[SetKey]],
        candidate_scores: Sequence[Sequence[float]],
        n_candidates: int,
        pool_limit: int,
        max_set_len: int,
        pad_id: int,
        training: bool,
        seed: int,
    ) -> None:
        self.x = np.asarray(x, dtype=np.float32)
        self.targets = [tuple(np.flatnonzero(row > 0.5).tolist()) for row in y]
        self.candidates = candidates
        self.candidate_scores = candidate_scores
        self.n_candidates = int(n_candidates)
        self.pool_limit = max(int(pool_limit), self.n_candidates)
        self.max_set_len = int(max_set_len)
        self.pad_id = int(pad_id)
        self.training = bool(training)
        self.seed = int(seed)
        # A listwise loss is only well-defined when the generated pool contains
        # the exact positive.  Earlier runs injected missing ground truth sets
        # into the final slot, teaching the model an inference-time-impossible
        # shortcut.  Keep every row for evaluation, but train only on honest
        # positive-containing candidate pools.
        if self.training:
            self.row_indices = [
                index
                for index, target in enumerate(self.targets)
                if target in set(self.candidates[index][: self.pool_limit])
            ]
        else:
            self.row_indices = list(range(len(self.x)))

    def __len__(self) -> int:
        return len(self.row_indices)

    def _select(self, index: int) -> tuple[List[SetKey], List[float], List[int], int]:
        target = self.targets[index]
        source_scores = list(self.candidate_scores[index])
        if not self.training:
            selected = list(self.candidates[index][: self.n_candidates])
            ranks = list(range(len(selected)))
            return selected, source_scores[: self.n_candidates], ranks, selected.index(target) if target in selected else -1
        pool = list(self.candidates[index][: self.pool_limit])
        if target not in pool:
            raise RuntimeError("training row lacks an honest positive candidate")
        positive_rank = pool.index(target)
        # Training on every member of a deep candidate pool is unnecessarily
        # expensive.  Retain the highest-ranked hard negatives and, when the
        # positive lies deeper than the tensor budget, replace the final slot
        # with that *honestly generated* positive.  Its original source rank is
        # kept below, so the model never sees an artificially favorable prior.
        selected_indices = list(range(min(self.n_candidates, len(pool))))
        if positive_rank not in selected_indices:
            selected_indices[-1] = positive_rank
            selected_indices.sort()
        selected = [pool[value] for value in selected_indices]
        selected_scores = [source_scores[value] if value < len(source_scores) else -math.log1p(value) for value in selected_indices]
        return selected, selected_scores, selected_indices, selected_indices.index(positive_rank)

    def __getitem__(self, index: int):
        row_index = self.row_indices[index]
        selected, selected_scores, source_ranks, positive_index = self._select(row_index)
        labels = np.full((self.n_candidates, self.max_set_len), self.pad_id, dtype=np.int64)
        mask = np.zeros(self.n_candidates, dtype=np.float32)
        numeric = np.zeros((self.n_candidates, 2), dtype=np.float32)
        if selected_scores:
            score_values = np.asarray(selected_scores, dtype=np.float32)
            score_values = (score_values - score_values.mean()) / max(float(score_values.std()), 1e-6)
        else:
            score_values = np.zeros(0, dtype=np.float32)
        for candidate_index, candidate in enumerate(selected):
            label_values = candidate[: self.max_set_len]
            labels[candidate_index, : len(label_values)] = label_values
            mask[candidate_index] = 1.0
            numeric[candidate_index, 0] = float(score_values[candidate_index]) if candidate_index < len(score_values) else 0.0
            numeric[candidate_index, 1] = -math.log1p(source_ranks[candidate_index])
        return (
            torch.from_numpy(self.x[row_index]),
            torch.from_numpy(labels),
            torch.from_numpy(numeric),
            torch.from_numpy(mask),
            torch.tensor(positive_index, dtype=torch.long),
        )


class GoldContrastiveDataset(Dataset):
    """Gold-set contrastive pretraining without exposing a rank/slot shortcut."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        candidates: Sequence[Sequence[SetKey]],
        pool_limit: int,
        negative_count: int,
        max_set_len: int,
        pad_id: int,
        seed: int,
    ) -> None:
        self.x = np.asarray(x, dtype=np.float32)
        self.targets = [tuple(np.flatnonzero(row > 0.5).tolist()) for row in y]
        self.candidates = candidates
        self.pool_limit = int(pool_limit)
        self.negative_count = int(negative_count)
        self.max_set_len = int(max_set_len)
        self.pad_id = int(pad_id)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        target = self.targets[index]
        pool = [value for value in self.candidates[index][: self.pool_limit] if value != target]
        hard_count = min(len(pool), max(1, self.negative_count // 2))
        chosen = list(pool[:hard_count])
        remaining = pool[hard_count:]
        if remaining and len(chosen) < self.negative_count:
            rng = random.Random(self.seed + index * 1000003)
            take = min(self.negative_count - len(chosen), len(remaining))
            chosen.extend(rng.sample(remaining, take))
        values = [target, *chosen]
        rng = random.Random(self.seed * 17 + index * 7919)
        rng.shuffle(values)
        positive_index = values.index(target)
        labels = np.full((self.negative_count + 1, self.max_set_len), self.pad_id, dtype=np.int64)
        mask = np.zeros(self.negative_count + 1, dtype=np.float32)
        for candidate_index, candidate in enumerate(values):
            labels[candidate_index, : len(candidate)] = candidate[: self.max_set_len]
            mask[candidate_index] = 1.0
        return (
            torch.from_numpy(self.x[index]),
            torch.from_numpy(labels),
            torch.zeros((self.negative_count + 1, 2), dtype=torch.float32),
            torch.from_numpy(mask),
            torch.tensor(positive_index, dtype=torch.long),
        )


class ResidualBlock(nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden * 2)
        self.fc2 = nn.Linear(hidden * 2, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.dropout(self.fc2(F.gelu(self.fc1(self.norm(value)))))


class ListwiseSetRanker(nn.Module):
    def __init__(
        self,
        x_dim: int,
        n_labels: int,
        hidden: int,
        blocks: int,
        dropout: float,
        max_set_len: int,
        use_membership_energy: bool = False,
        membership_score_scale: float = 1.0,
        length_score_scale: float = 0.0,
        label_chemistry: torch.Tensor | None = None,
        query_chemistry_dim: int = 0,
        use_relative_chemistry_features: bool = False,
        candidate_transformer_layers: int = 0,
        candidate_transformer_heads: int = 8,
        joint_transformer_layers: int = 0,
        joint_transformer_heads: int = 8,
    ) -> None:
        super().__init__()
        self.pad_id = int(n_labels)
        self.n_labels = int(n_labels)
        self.use_membership_energy = bool(use_membership_energy)
        self.membership_score_scale = float(membership_score_scale)
        self.length_score_scale = float(length_score_scale)
        self.query_chemistry_dim = int(query_chemistry_dim)
        self.use_relative_chemistry_features = bool(use_relative_chemistry_features)
        self.candidate_transformer_layers = int(candidate_transformer_layers)
        self.joint_transformer_layers = int(joint_transformer_layers)
        if self.use_relative_chemistry_features and (label_chemistry is None or self.query_chemistry_dim <= 0):
            raise ValueError("relative chemistry requires label and query formula descriptors")
        self.target_encoder = nn.Sequential(
            nn.Linear(x_dim, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
        )
        self.target_blocks = nn.Sequential(*[ResidualBlock(hidden, dropout) for _ in range(blocks)])
        self.label_embedding = nn.Embedding(n_labels + 1, hidden, padding_idx=self.pad_id)
        if label_chemistry is not None:
            if label_chemistry.shape[0] != n_labels:
                raise ValueError("label chemistry row count must match n_labels")
            padded_chemistry = torch.cat(
                [label_chemistry.float(), torch.zeros((1, label_chemistry.shape[1]), dtype=torch.float32)], dim=0
            )
            self.register_buffer("label_chemistry", padded_chemistry, persistent=False)
            self.chemical_encoder = nn.Sequential(
                nn.Linear(label_chemistry.shape[1], hidden), nn.LayerNorm(hidden), nn.GELU()
            )
        else:
            self.label_chemistry = None
        if self.query_chemistry_dim > 0:
            self.query_chemical_encoder = nn.Sequential(
                nn.Linear(self.query_chemistry_dim, hidden), nn.LayerNorm(hidden), nn.GELU()
            )
        if self.use_relative_chemistry_features:
            # Four descriptor-wise relations plus eight normalized coverage/
            # distance summaries.  The group slices make Li/Na and Cl/Br
            # transferable even when the exact elements were sparse in train.
            relation_dim = self.query_chemistry_dim * 4 + 8
            self.relative_chemical_encoder = nn.Sequential(
                nn.Linear(relation_dim, hidden * 2),
                nn.LayerNorm(hidden * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden * 2, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
            )
        if self.candidate_transformer_layers > 0:
            if hidden % int(candidate_transformer_heads) != 0:
                raise ValueError("hidden must be divisible by candidate_transformer_heads")
            layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=int(candidate_transformer_heads),
                dim_feedforward=hidden * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.candidate_transformer = nn.TransformerEncoder(
                layer, num_layers=self.candidate_transformer_layers, norm=nn.LayerNorm(hidden)
            )
        if self.joint_transformer_layers > 0:
            if hidden % int(joint_transformer_heads) != 0:
                raise ValueError("hidden must be divisible by joint_transformer_heads")
            joint_layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=int(joint_transformer_heads),
                dim_feedforward=hidden * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.joint_transformer = nn.TransformerEncoder(
                joint_layer, num_layers=self.joint_transformer_layers, norm=nn.LayerNorm(hidden)
            )
        self.length_embedding = nn.Embedding(max_set_len + 1, hidden)
        self.candidate_encoder = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.numeric_encoder = nn.Sequential(
            nn.Linear(2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(
                hidden * (
                    4
                    + int(self.use_relative_chemistry_features)
                    + int(self.joint_transformer_layers > 0)
                ),
                hidden * 2,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        if self.use_membership_energy:
            self.membership_head = nn.Linear(hidden, n_labels)
            self.cardinality_head = nn.Linear(hidden, max_set_len)

    def encode_target(self, x: torch.Tensor) -> torch.Tensor:
        target = self.target_blocks(self.target_encoder(x))
        if self.query_chemistry_dim > 0:
            target = target + self.query_chemical_encoder(x[:, -self.query_chemistry_dim :])
        return target

    def auxiliary_logits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_membership_energy:
            raise RuntimeError("membership energy is disabled")
        target = self.encode_target(x)
        return self.membership_head(target), self.cardinality_head(target)

    def forward(self, x: torch.Tensor, candidate_labels: torch.Tensor, candidate_numeric: torch.Tensor) -> torch.Tensor:
        target = self.encode_target(x)
        valid = candidate_labels != self.pad_id
        lengths = valid.sum(dim=-1).clamp_min(1)
        embedded = self.label_embedding(candidate_labels)
        if self.label_chemistry is not None:
            embedded = embedded + self.chemical_encoder(self.label_chemistry[candidate_labels])
        if self.candidate_transformer_layers > 0:
            original_shape = embedded.shape
            flat_embedded = embedded.reshape(-1, original_shape[-2], original_shape[-1])
            padding_mask = (~valid).reshape(-1, original_shape[-2])
            # Fully padded candidate slots exist at the tail of short rows.
            # Keep one zero token visible to avoid all-masked attention NaNs;
            # the original validity mask still removes it during pooling.
            all_padding = padding_mask.all(dim=1)
            if all_padding.any():
                padding_mask = padding_mask.clone()
                padding_mask[all_padding, 0] = False
                flat_embedded = flat_embedded.clone()
                flat_embedded[all_padding, 0] = 0.0
            embedded = self.candidate_transformer(
                flat_embedded, src_key_padding_mask=padding_mask
            ).reshape(original_shape)
        joint_query_context = None
        if self.joint_transformer_layers > 0:
            batch_size, candidate_count, _, hidden = embedded.shape
            query_tokens = target[:, None, None, :].expand(-1, candidate_count, 1, -1)
            joint_tokens = torch.cat([query_tokens, embedded], dim=-2)
            query_padding = torch.zeros(
                (batch_size, candidate_count, 1), dtype=torch.bool, device=valid.device
            )
            joint_padding = torch.cat([query_padding, ~valid], dim=-1)
            joint_output = self.joint_transformer(
                joint_tokens.reshape(-1, joint_tokens.shape[-2], hidden),
                src_key_padding_mask=joint_padding.reshape(-1, joint_padding.shape[-1]),
            ).reshape(batch_size, candidate_count, joint_tokens.shape[-2], hidden)
            joint_query_context = joint_output[:, :, 0, :]
        candidate = (embedded * valid[..., None]).sum(dim=-2) / lengths[..., None]
        candidate = self.candidate_encoder(candidate + self.length_embedding(lengths))
        candidate = candidate + self.numeric_encoder(candidate_numeric)
        expanded_target = target[:, None, :].expand_as(candidate)
        feature_parts = [
            expanded_target, candidate, expanded_target * candidate, torch.abs(expanded_target - candidate)
        ]
        if joint_query_context is not None:
            feature_parts.append(joint_query_context)
        if self.use_relative_chemistry_features:
            query_chemistry = x[:, -self.query_chemistry_dim :]
            label_descriptors = self.label_chemistry[candidate_labels]
            valid_float = valid[..., None].to(label_descriptors.dtype)
            candidate_mean = (label_descriptors * valid_float).sum(dim=-2) / lengths[..., None]
            candidate_max = label_descriptors.masked_fill(~valid[..., None], 0.0).amax(dim=-2)
            expanded_query = query_chemistry[:, None, :].expand_as(candidate_mean)

            query_element = expanded_query[..., :118]
            candidate_element = candidate_max[..., :118]
            query_group = expanded_query[..., 118:136]
            candidate_group = candidate_max[..., 118:136]
            eps = 1e-6
            query_element_present = query_element > eps
            candidate_element_present = candidate_element > eps
            query_group_present = query_group > eps
            candidate_group_present = candidate_group > eps
            scalars = torch.stack(
                [
                    (query_element_present & candidate_element_present).sum(dim=-1)
                    / query_element_present.sum(dim=-1).clamp_min(1),
                    (query_element_present & candidate_element_present).sum(dim=-1)
                    / candidate_element_present.sum(dim=-1).clamp_min(1),
                    (query_group_present & candidate_group_present).sum(dim=-1)
                    / query_group_present.sum(dim=-1).clamp_min(1),
                    (query_group_present & candidate_group_present).sum(dim=-1)
                    / candidate_group_present.sum(dim=-1).clamp_min(1),
                    torch.minimum(query_element, candidate_element).sum(dim=-1),
                    torch.minimum(query_group, candidate_group).sum(dim=-1),
                    torch.abs(query_element - candidate_mean[..., :118]).mean(dim=-1),
                    torch.abs(query_group - candidate_mean[..., 118:136]).mean(dim=-1),
                ],
                dim=-1,
            ).to(candidate_mean.dtype)
            relations = torch.cat(
                [
                    candidate_mean,
                    candidate_max,
                    torch.abs(expanded_query - candidate_mean),
                    expanded_query * candidate_max,
                    scalars,
                ],
                dim=-1,
            )
            feature_parts.append(self.relative_chemical_encoder(relations))
        features = torch.cat(feature_parts, dim=-1)
        score = self.scorer(features).squeeze(-1)
        if self.use_membership_energy:
            membership_logits = self.membership_head(target)
            safe_labels = candidate_labels.clamp_max(self.n_labels - 1)
            expanded_logits = membership_logits[:, None, :].expand(-1, candidate_labels.shape[1], -1)
            selected_logits = torch.gather(expanded_logits, 2, safe_labels)
            membership_energy = (selected_logits * valid).sum(dim=-1)
            score = score + self.membership_score_scale * membership_energy
            if self.length_score_scale:
                length_log_probs = F.log_softmax(self.cardinality_head(target), dim=-1)
                length_indices = (lengths - 1).clamp_max(length_log_probs.shape[1] - 1)
                expanded_lengths = length_log_probs[:, None, :].expand(-1, candidate_labels.shape[1], -1)
                score = score + self.length_score_scale * torch.gather(
                    expanded_lengths, 2, length_indices[..., None]
                ).squeeze(-1)
        return score


def asymmetric_multilabel_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma_neg: float,
    gamma_pos: float,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    positive = targets * torch.log(probabilities.clamp_min(1e-8))
    negative = (1.0 - targets) * torch.log((1.0 - probabilities).clamp_min(1e-8))
    if gamma_pos > 0:
        positive = positive * (1.0 - probabilities).pow(gamma_pos)
    if gamma_neg > 0:
        negative = negative * probabilities.pow(gamma_neg)
    return -(positive + negative).sum(dim=1).mean()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    rank_bias_weight: float,
    residual_scale: float,
) -> Dict[str, float]:
    model.eval()
    hits = {k: 0 for k in (1, 3, 5, 10, 20, 50, 100)}
    count = 0
    for x, labels, candidate_numeric, candidate_mask, positive_index in loader:
        x = x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        candidate_numeric = candidate_numeric.to(device, non_blocking=True)
        candidate_mask = candidate_mask.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            residual = model(x, labels, candidate_numeric) * float(residual_scale)
            rank_prior = float(rank_bias_weight) * candidate_numeric[..., 1]
            scores = (rank_prior + residual).masked_fill(candidate_mask < 0.5, -torch.inf)
        order = torch.argsort(scores, dim=1, descending=True)
        positive_index = positive_index.to(device)
        found = positive_index >= 0
        positive_rank = torch.argmax(
            (order == positive_index[:, None]).to(torch.int64), dim=1
        ) + 1
        for k in hits:
            hits[k] += int((found & (positive_rank <= k)).sum().item())
        count += len(x)
    return {f"exact_hit@{k}": hits[k] / max(1, count) for k in hits}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a GPU listwise scorer on beam/substitution precursor sets.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--train_candidates", nargs="+", required=True)
    parser.add_argument("--val_candidates", nargs="+", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--train_candidate_limit", type=int, default=100)
    parser.add_argument(
        "--train_pool_limit", type=int, default=0,
        help="Search this many generated candidates for an honest training positive; 0 uses train_candidate_limit.",
    )
    parser.add_argument("--val_candidate_limit", type=int, default=600)
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--rank_bias_weight", type=float, default=1.0)
    parser.add_argument("--residual_scale", type=float, default=0.1)
    parser.add_argument("--objective", choices=("cross_entropy", "topk_boundary"), default="cross_entropy")
    parser.add_argument("--training_topk", type=int, default=10)
    parser.add_argument("--append_source_features", action="store_true")
    parser.add_argument("--membership_pretrain_epochs", type=int, default=0)
    parser.add_argument("--membership_score_scale", type=float, default=1.0)
    parser.add_argument("--length_score_scale", type=float, default=0.0)
    parser.add_argument("--membership_length_loss_weight", type=float, default=1.0)
    parser.add_argument("--membership_gamma_neg", type=float, default=2.0)
    parser.add_argument("--membership_gamma_pos", type=float, default=0.0)
    parser.add_argument("--use_formula_features", action="store_true")
    parser.add_argument("--append_query_formula_features", action="store_true")
    parser.add_argument("--aligned_query_formula_encoder", action="store_true")
    parser.add_argument("--use_relative_chemistry_features", action="store_true")
    parser.add_argument("--candidate_transformer_layers", type=int, default=0)
    parser.add_argument("--candidate_transformer_heads", type=int, default=8)
    parser.add_argument("--joint_transformer_layers", type=int, default=0)
    parser.add_argument("--joint_transformer_heads", type=int, default=8)
    parser.add_argument("--train_families", default="")
    parser.add_argument("--val_families", default="")
    parser.add_argument("--train_min_set_len", type=int, default=0)
    parser.add_argument("--val_min_set_len", type=int, default=0)
    parser.add_argument("--gold_contrastive_pretrain_epochs", type=int, default=0)
    parser.add_argument("--gold_negative_count", type=int, default=31)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.use_relative_chemistry_features and not (
        args.use_formula_features and args.append_query_formula_features
    ):
        parser.error("--use_relative_chemistry_features requires --use_formula_features and --append_query_formula_features")
    seed_everything(args.seed)
    input_dir = Path(args.input_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    packs = {split: np.load(input_dir / f"{split}.npz", allow_pickle=True) for split in ("train", "val")}
    train_x = np.asarray(packs["train"]["x"], dtype=np.float32)
    val_x = np.asarray(packs["val"]["x"], dtype=np.float32)
    source_vocab: List[str] = []
    if args.append_source_features:
        source_vocab = sorted(
            pd.read_csv(input_dir / "train_meta.csv", usecols=["source_dataset"])["source_dataset"]
            .fillna("").astype(str).unique().tolist()
        )
        train_x = append_source_features(train_x, input_dir / "train_meta.csv", source_vocab)
        val_x = append_source_features(val_x, input_dir / "val_meta.csv", source_vocab)
    if args.append_query_formula_features:
        train_x = append_query_formula_features(train_x, input_dir / "train_meta.csv")
        val_x = append_query_formula_features(val_x, input_dir / "val_meta.csv")
    train_y = np.asarray(packs["train"]["y_multi_hot"], dtype=np.float32)
    val_y = np.asarray(packs["val"]["y_multi_hot"], dtype=np.float32)
    n_labels = train_y.shape[1]
    label_chemistry = None
    if args.use_formula_features:
        precursor_names = json.loads((input_dir / "precursor_names.json").read_text(encoding="utf-8"))
        label_chemistry = torch.from_numpy(precursor_formula_features([str(value) for value in precursor_names]))
    max_set_len = int(max(packs["train"]["set_len"].max(), packs["val"]["set_len"].max()))
    train_candidates, train_candidate_scores = load_candidates(args.train_candidates, len(train_x))
    val_candidates, val_candidate_scores = load_candidates(args.val_candidates, len(val_x))
    train_pool_limit = int(args.train_pool_limit) if int(args.train_pool_limit) > 0 else int(args.train_candidate_limit)
    train_dataset = CandidateDataset(
        train_x, train_y, train_candidates, train_candidate_scores, args.train_candidate_limit,
        train_pool_limit, max_set_len, n_labels, True, args.seed
    )
    val_dataset = CandidateDataset(
        val_x, val_y, val_candidates, val_candidate_scores, args.val_candidate_limit,
        args.val_candidate_limit, max_set_len, n_labels, False, args.seed
    )
    train_families = restrict_dataset_families(train_dataset, input_dir / "train_meta.csv", args.train_families)
    val_families = restrict_dataset_families(val_dataset, input_dir / "val_meta.csv", args.val_families)
    restrict_dataset_min_set_len(train_dataset, np.asarray(packs["train"]["set_len"]), args.train_min_set_len)
    restrict_dataset_min_set_len(val_dataset, np.asarray(packs["val"]["set_len"]), args.val_min_set_len)
    if not train_dataset.row_indices:
        raise RuntimeError("family restriction removed every training row")
    if not val_dataset.row_indices:
        raise RuntimeError("family restriction removed every validation row")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=max(1, args.batch_size // 2), shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_membership_energy = int(args.membership_pretrain_epochs) > 0
    model = ListwiseSetRanker(
        train_x.shape[1], n_labels, args.hidden, args.blocks, args.dropout, max_set_len,
        use_membership_energy=use_membership_energy,
        membership_score_scale=args.membership_score_scale,
        length_score_scale=args.length_score_scale,
        label_chemistry=label_chemistry,
        query_chemistry_dim=(147 if (args.aligned_query_formula_encoder or args.use_relative_chemistry_features) else 0),
        use_relative_chemistry_features=args.use_relative_chemistry_features,
        candidate_transformer_layers=args.candidate_transformer_layers,
        candidate_transformer_heads=args.candidate_transformer_heads,
        joint_transformer_layers=args.joint_transformer_layers,
        joint_transformer_heads=args.joint_transformer_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    pretraining_log: List[Dict[str, float]] = []
    if use_membership_energy:
        auxiliary_loader = DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(train_x),
                torch.from_numpy(train_y),
                torch.from_numpy(np.asarray(packs["train"]["set_len"], dtype=np.int64) - 1),
            ),
            batch_size=max(64, args.batch_size * 8), shuffle=True, num_workers=2,
            pin_memory=True, persistent_workers=True,
        )
        for pretrain_epoch in range(1, int(args.membership_pretrain_epochs) + 1):
            model.train()
            pretrain_losses = []
            for batch_x, batch_y, batch_lengths in auxiliary_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                batch_lengths = batch_lengths.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    member_logits, length_logits = model.auxiliary_logits(batch_x)
                    member_loss = asymmetric_multilabel_loss(
                        member_logits, batch_y, args.membership_gamma_neg, args.membership_gamma_pos
                    )
                    length_loss = F.cross_entropy(length_logits, batch_lengths)
                    pretrain_loss = member_loss + float(args.membership_length_loss_weight) * length_loss
                pretrain_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                pretrain_losses.append(float(pretrain_loss.detach().cpu()))
            pretrain_row = {"pretrain_epoch": pretrain_epoch, "train_loss": float(np.mean(pretrain_losses))}
            pretraining_log.append(pretrain_row)
            print(json.dumps(pretrain_row), flush=True)
    gold_pretraining_log: List[Dict[str, float]] = []
    if int(args.gold_contrastive_pretrain_epochs) > 0:
        gold_dataset = GoldContrastiveDataset(
            train_x, train_y, train_candidates, train_pool_limit,
            args.gold_negative_count, max_set_len, n_labels, args.seed,
        )
        gold_loader = DataLoader(
            gold_dataset, batch_size=max(16, args.batch_size * 2), shuffle=True,
            num_workers=2, pin_memory=True, persistent_workers=True,
        )
        for pretrain_epoch in range(1, int(args.gold_contrastive_pretrain_epochs) + 1):
            model.train()
            gold_losses = []
            for batch_x, batch_labels, batch_numeric, batch_mask, positive_index in gold_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_labels = batch_labels.to(device, non_blocking=True)
                batch_numeric = batch_numeric.to(device, non_blocking=True)
                batch_mask = batch_mask.to(device, non_blocking=True)
                positive_index = positive_index.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    compatibility = model(batch_x, batch_labels, batch_numeric)
                    compatibility = compatibility.masked_fill(batch_mask < 0.5, -1e4)
                    gold_loss = F.cross_entropy(compatibility, positive_index)
                gold_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                gold_losses.append(float(gold_loss.detach().cpu()))
            gold_row = {"gold_pretrain_epoch": pretrain_epoch, "train_loss": float(np.mean(gold_losses))}
            gold_pretraining_log.append(gold_row)
            print(json.dumps(gold_row), flush=True)
    best_metric = -1.0
    best_epoch = 0
    best_state = None
    bad_epochs = 0
    logs = []
    if pretraining_log or gold_pretraining_log:
        initial_metrics = evaluate(model, val_loader, device, args.rank_bias_weight, args.residual_scale)
        initial_row = {"epoch": 0, "train_loss": None, **initial_metrics, "checkpoint_stage": "post_pretraining"}
        logs.append(initial_row)
        print(json.dumps(initial_row), flush=True)
        best_metric = float(initial_metrics["exact_hit@10"])
        best_state = copy.deepcopy(model.state_dict())
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, labels, candidate_numeric, candidate_mask, positive_index in train_loader:
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            candidate_numeric = candidate_numeric.to(device, non_blocking=True)
            candidate_mask = candidate_mask.to(device, non_blocking=True)
            positive_index = positive_index.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                residual = model(x, labels, candidate_numeric) * float(args.residual_scale)
                rank_prior = float(args.rank_bias_weight) * candidate_numeric[..., 1]
                scores = (rank_prior + residual).masked_fill(candidate_mask < 0.5, -1e4)
                if args.objective == "cross_entropy":
                    loss = F.cross_entropy(scores, positive_index)
                else:
                    positive_score = scores.gather(1, positive_index[:, None]).squeeze(1)
                    negative_scores = scores.clone()
                    negative_scores.scatter_(1, positive_index[:, None], -torch.inf)
                    available = max(1, negative_scores.shape[1] - 1)
                    boundary_k = min(max(1, int(args.training_topk)), available)
                    kth_negative = torch.topk(
                        negative_scores, k=boundary_k, dim=1, largest=True, sorted=True
                    ).values[:, -1]
                    loss = F.softplus(kth_negative - positive_score).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_metrics = evaluate(model, val_loader, device, args.rank_bias_weight, args.residual_scale)
        current = float(val_metrics["exact_hit@10"])
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **val_metrics}
        logs.append(row)
        print(json.dumps(row), flush=True)
        if current > best_metric:
            best_metric = current
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("no ranker checkpoint selected")
    model.load_state_dict(best_state)
    final_metrics = evaluate(model, val_loader, device, args.rank_bias_weight, args.residual_scale)
    checkpoint = {
        "state_dict": {key: value.cpu() for key, value in best_state.items()},
        "config": vars(args),
        "x_dim": train_x.shape[1],
        "n_labels": n_labels,
        "max_set_len": max_set_len,
        "source_vocab": source_vocab,
        "best_epoch": best_epoch,
        "best_val_exact_hit_at_10": best_metric,
    }
    torch.save(checkpoint, run_dir / "best_listwise_ranker.pt")
    summary = {
        "config": vars(args),
        "data": {
            "n_train": len(train_dataset),
            "n_val": len(val_dataset),
            "n_train_excluded_without_positive": int(len(train_x) - len(train_dataset)),
            "train_pool_limit": int(train_pool_limit),
            "train_families": train_families,
            "val_families": val_families,
            "train_min_set_len": int(args.train_min_set_len),
            "val_min_set_len": int(args.val_min_set_len),
            "mean_train_candidates": float(np.mean([len(row) for row in train_candidates])),
            "mean_val_candidates": float(np.mean([len(row) for row in val_candidates])),
        },
        "best_epoch": best_epoch,
        "validation": final_metrics,
        "membership_pretraining_log": pretraining_log,
        "gold_contrastive_pretraining_log": gold_pretraining_log,
        "training_log": logs,
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
