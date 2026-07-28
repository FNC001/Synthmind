#!/usr/bin/env python3
"""Train a chemistry-structured energy model for exact precursor variants.

The current Stage-2 system already places the correct *precursor-family*
template in the validation Top-10 more often than it places the exact formula
set.  This model therefore solves a deliberately narrower residual problem:
given a target material and several exact precursor sets with the same family
template, score the chemically most plausible exact variant.

Training negatives come from formula-group-disjoint OOF candidate sources.
The frozen test split is never read.  A candidate label may have zero training
frequency: its representation is derived from formula composition and a
label-free MatSciBERT embedding, rather than a learned label-ID embedding.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Composition, Element
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader, Dataset

from training.family.evaluate_stage2_candidate_fusion import load_source
from training.family.evaluate_stage2_oof_chemistry_rescore import (
    chemistry_features_for_candidate,
    family_length_modes,
    json_set,
    label_chemistry,
)
from training.family.evaluate_stage2_precursor_family_slate import family_key, precursor_family
from training.family.train_stage2_oof_candidate_stacker import (
    CandidatePriorBuilder,
    TemplatePriorBuilder,
    precursor_route_token,
)
from training.family.train_stage2_within_family_variant_ranker import exact_metrics


SetKey = Tuple[int, ...]
TOP_K = (1, 3, 5, 10, 20, 50, 100)
ROLE_NAMES = (
    "acetate",
    "alcohol",
    "carbonate",
    "elemental",
    "glucose",
    "halide",
    "hydrazine",
    "hydroxide",
    "nitrate",
    "organic",
    "oxide_or_other_oxygenate",
    "phosphate",
    "pvp",
    "sulfate",
    "thioamide",
    "thiourea",
    "urea",
    "other",
)
ELEMENT_PATTERN = re.compile(r"[A-Z][a-z]?")


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def normalized_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def targets_from_matrix(matrix: np.ndarray) -> List[SetKey]:
    return [tuple(np.flatnonzero(row > 0.5).astype(int).tolist()) for row in matrix]


def load_matsci_pca_views(
    path: Path,
    components: int,
    seed: int,
    evaluation_split: str = "val",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    with np.load(path, allow_pickle=False) as cache:
        label_raw = np.concatenate(
            [cache["precursor_common_mean"], cache["precursor_role_mean"]], axis=1
        ).astype(np.float32)
        train_raw = np.concatenate(
            [cache["train_query_common_mean"], cache["train_query_role_mean"]], axis=1
        ).astype(np.float32)
        evaluation_raw = np.concatenate(
            [
                cache[f"{str(evaluation_split)}_query_common_mean"],
                cache[f"{str(evaluation_split)}_query_role_mean"],
            ],
            axis=1,
        ).astype(np.float32)
        schema = str(cache["schema_version"].item())
    maximum = min(label_raw.shape[0] + train_raw.shape[0] - 1, label_raw.shape[1])
    if not 1 <= int(components) <= maximum:
        raise ValueError(f"matsci_components must be between 1 and {maximum}")
    pca = PCA(
        n_components=int(components),
        svd_solver="randomized",
        random_state=int(seed),
    )
    pca.fit(np.vstack([label_raw, train_raw]))
    label = normalized_rows(pca.transform(label_raw))
    train = normalized_rows(pca.transform(train_raw))
    evaluation = normalized_rows(pca.transform(evaluation_raw))
    metadata = {
        "schema": schema,
        "components": int(components),
        "explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
        "evaluation_split": str(evaluation_split),
    }
    return label, train, evaluation, metadata


def safe_composition(raw_name: str) -> tuple[Dict[str, float], bool]:
    raw = str(raw_name).strip()
    candidates = [raw, re.split(r"[·.](?=[0-9]*H2O)", raw, maxsplit=1)[0]]
    for candidate in candidates:
        try:
            values = Composition(candidate).remove_charges().get_el_amt_dict()
            if values:
                return {str(key): float(value) for key, value in values.items()}, True
        except Exception:
            continue
    elements = set(ELEMENT_PATTERN.findall(raw))
    return {symbol: 1.0 for symbol in elements}, False


def label_structured_features(names: Sequence[str]) -> tuple[np.ndarray, Dict[str, int]]:
    """Formula-derived, label-ID-free features for every precursor label."""
    feature_rows: List[np.ndarray] = []
    role_to_index = {role: index for index, role in enumerate(ROLE_NAMES)}
    parsed = 0
    for name in names:
        amounts, ok = safe_composition(str(name))
        parsed += int(ok)
        total = max(1e-8, float(sum(amounts.values())))
        element_fraction = np.zeros(118, dtype=np.float32)
        element_presence = np.zeros(118, dtype=np.float32)
        group_fraction = np.zeros(18, dtype=np.float32)
        metal_amount = 0.0
        atomic_numbers = []
        for symbol, amount in amounts.items():
            try:
                element = Element(symbol)
            except ValueError:
                continue
            index = int(element.Z) - 1
            if not 0 <= index < 118:
                continue
            fraction = float(amount) / total
            element_fraction[index] = fraction
            element_presence[index] = 1.0
            atomic_numbers.append(float(element.Z))
            if bool(element.is_metal):
                metal_amount += float(amount)
            if element.group is not None:
                group_fraction[int(element.group) - 1] += fraction
        role, token_groups, hydrated, token_element_count = precursor_route_token(str(name))
        role_one_hot = np.zeros(len(ROLE_NAMES), dtype=np.float32)
        role_one_hot[role_to_index.get(str(role), role_to_index["other"])] = 1.0
        token_group_presence = np.zeros(18, dtype=np.float32)
        for group in token_groups:
            if 1 <= int(group) <= 18:
                token_group_presence[int(group) - 1] = 1.0
        scalars = np.asarray(
            [
                min(len(amounts), 12) / 12.0,
                math.log1p(total) / math.log(65.0),
                float(metal_amount / total),
                float(np.mean(atomic_numbers) / 118.0) if atomic_numbers else 0.0,
                float(np.std(atomic_numbers) / 59.0) if atomic_numbers else 0.0,
                float(hydrated),
                min(int(token_element_count), 8) / 8.0,
                float(ok),
            ],
            dtype=np.float32,
        )
        feature_rows.append(
            np.concatenate(
                [
                    element_fraction,
                    element_presence,
                    group_fraction,
                    token_group_presence,
                    role_one_hot,
                    scalars,
                ]
            ).astype(np.float32)
        )
    return np.asarray(feature_rows, dtype=np.float32), {
        "labels": int(len(names)),
        "composition_parsed": int(parsed),
        "text_fallback": int(len(names) - parsed),
    }


def metadata_features(
    train_meta: pd.DataFrame,
    val_meta: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    columns = ("source_dataset", "synthesis_type", "family_routing_level", "quality_tier")
    train_rows: List[np.ndarray] = []
    val_rows: List[np.ndarray] = []
    names: List[str] = []
    for column in columns:
        train_values = train_meta[column].fillna("UNK").astype(str)
        categories = sorted(set(train_values.tolist()) | {"UNK"})
        mapping = {value: index for index, value in enumerate(categories)}
        train_block = np.zeros((len(train_meta), len(categories)), dtype=np.float32)
        val_block = np.zeros((len(val_meta), len(categories)), dtype=np.float32)
        for row_index, value in enumerate(train_values):
            train_block[row_index, mapping.get(str(value), mapping["UNK"])] = 1.0
        for row_index, value in enumerate(val_meta[column].fillna("UNK").astype(str)):
            val_block[row_index, mapping.get(str(value), mapping["UNK"])] = 1.0
        train_rows.append(train_block)
        val_rows.append(val_block)
        names.extend(f"meta_{column}={value}" for value in categories)
    train_quality = (
        train_meta["quality_weight"].fillna(0.5).to_numpy(dtype=np.float32).reshape(-1, 1)
    )
    val_quality = (
        val_meta["quality_weight"].fillna(0.5).to_numpy(dtype=np.float32).reshape(-1, 1)
    )
    names.append("meta_quality_weight")
    return (
        np.hstack([*train_rows, train_quality]).astype(np.float32),
        np.hstack([*val_rows, val_quality]).astype(np.float32),
        names,
    )


def standardize_from_train(
    train: np.ndarray, val: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(train, axis=0, dtype=np.float64).astype(np.float32)
    std = np.nanstd(train, axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return (
        np.nan_to_num((train - mean) / std).astype(np.float32),
        np.nan_to_num((val - mean) / std).astype(np.float32),
        mean,
        std,
    )


def merge_candidate_sources(
    sources: Sequence[Sequence[Sequence[SetKey]]],
    row_index: int,
    limit: int,
) -> List[SetKey]:
    statistics: Dict[SetKey, List[float]] = {}
    for source in sources:
        for rank, candidate in enumerate(source[int(row_index)], start=1):
            key = tuple(sorted(set(int(value) for value in candidate)))
            if not key:
                continue
            current = statistics.setdefault(key, [float(rank), 0.0, 0.0])
            current[0] = min(current[0], float(rank))
            current[1] += 1.0 / (60.0 + float(rank))
            current[2] += 1.0
    ordered = sorted(
        statistics,
        key=lambda key: (
            statistics[key][0],
            -statistics[key][2],
            -statistics[key][1],
            key,
        ),
    )
    return ordered[: int(limit)]


def build_pair_features(
    candidate: SetKey,
    row: pd.Series,
    family: str,
    expected_length: int,
    label_elements: Sequence[set[str]],
    label_groups: Sequence[set[int]],
    label_metals: Sequence[set[str]],
    train_seen: np.ndarray,
    prior_builder: CandidatePriorBuilder,
    template_builder: TemplatePriorBuilder,
) -> np.ndarray:
    target_cations = json_set(row["target_cation_elements"])
    target_anions = json_set(row["target_anion_elements"])
    base = chemistry_features_for_candidate(
        candidate,
        target_cations,
        target_anions,
        label_elements,
        label_groups,
        label_metals,
        train_seen,
        int(expected_length),
    )
    candidate_elements: set[str] = set()
    candidate_metals: set[str] = set()
    per_label_target_fraction = []
    for label in candidate:
        elements = label_elements[int(label)]
        metals = label_metals[int(label)]
        candidate_elements.update(elements)
        candidate_metals.update(metals)
        per_label_target_fraction.append(len(metals & target_cations) / max(1, len(metals)))
    target_union = target_cations | target_anions
    exact_intersection = len(candidate_elements & target_union)
    extra_elements = candidate_elements - target_union - {"H", "C", "N", "O", "S", "P"}
    extra = np.asarray(
        [
            len(candidate_metals & target_cations) / max(1, len(candidate_metals)),
            float(candidate_metals == target_cations),
            exact_intersection / max(1, len(target_union)),
            -len(extra_elements) / max(1, len(candidate_elements)),
            float(np.mean(per_label_target_fraction)) if per_label_target_fraction else 0.0,
            float(np.min(per_label_target_fraction)) if per_label_target_fraction else 0.0,
            float(np.max(per_label_target_fraction)) if per_label_target_fraction else 0.0,
            min(len(candidate), 8) / 8.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [
            base,
            extra,
            prior_builder.features(candidate, str(family)),
            template_builder.features(candidate, str(family), target_anions),
        ]
    ).astype(np.float32)


@dataclass
class PoolData:
    query_indices: np.ndarray
    candidate_ids: np.ndarray
    mask: np.ndarray
    pair_features: np.ndarray
    target_in_source_pool: int
    rows_with_same_family_negative: int
    rows_with_cross_family_negative: int


class CandidateRegistry:
    def __init__(self) -> None:
        self.keys: List[SetKey] = [tuple()]
        self.mapping: Dict[SetKey, int] = {tuple(): 0}

    def id_for(self, candidate: SetKey) -> int:
        key = tuple(sorted(set(int(value) for value in candidate)))
        if key not in self.mapping:
            self.mapping[key] = len(self.keys)
            self.keys.append(key)
        return int(self.mapping[key])


def build_training_pools(
    targets: Sequence[SetKey],
    meta: pd.DataFrame,
    source_rows: Sequence[Sequence[Sequence[SetKey]]],
    label_families: Sequence[str],
    pool_limit: int,
    source_union_limit: int,
    registry: CandidateRegistry,
    feature_builder,
    seed: int,
    cross_family_negatives: int = 0,
    include_indices: set[int] | None = None,
) -> PoolData:
    family_truth_pool: Dict[Tuple[str, ...], List[SetKey]] = defaultdict(list)
    for target in targets:
        key = family_key(target, label_families)
        if target not in family_truth_pool[key]:
            family_truth_pool[key].append(target)
    rng = random.Random(int(seed))
    query_indices: List[int] = []
    candidate_rows: List[List[int]] = []
    mask_rows: List[List[bool]] = []
    feature_rows: List[np.ndarray] = []
    target_in_source_pool = 0
    rows_with_same_family_negative = 0
    rows_with_cross_family_negative = 0
    for row_index, target in enumerate(targets):
        if include_indices is not None and int(row_index) not in include_indices:
            continue
        target_family = family_key(target, label_families)
        merged = merge_candidate_sources(source_rows, row_index, int(source_union_limit))
        target_in_source_pool += int(target in set(merged))
        same_family_negatives = [
            candidate
            for candidate in merged
            if candidate != target and family_key(candidate, label_families) == target_family
        ]
        cross_family_pool = [
            candidate
            for candidate in merged
            if candidate != target and family_key(candidate, label_families) != target_family
        ]
        rows_with_same_family_negative += int(bool(same_family_negatives))
        rows_with_cross_family_negative += int(bool(cross_family_pool))
        fallback = list(family_truth_pool.get(target_family, []))
        rng.shuffle(fallback)
        for candidate in fallback:
            if candidate != target and candidate not in same_family_negatives:
                same_family_negatives.append(candidate)
        negative_limit = max(1, int(pool_limit) - 1)
        cross_limit = min(max(0, int(cross_family_negatives)), negative_limit)
        same_limit = max(0, negative_limit - cross_limit)
        negatives = list(same_family_negatives[:same_limit])
        for candidate in cross_family_pool[:cross_limit]:
            if candidate not in negatives:
                negatives.append(candidate)
        if cross_limit > 0 and len(negatives) < negative_limit:
            for candidate in [
                *same_family_negatives[same_limit:],
                *cross_family_pool[cross_limit:],
            ]:
                if candidate not in negatives:
                    negatives.append(candidate)
                if len(negatives) >= negative_limit:
                    break
        if not negatives:
            continue
        pool = [target, *negatives]
        ids = [registry.id_for(candidate) for candidate in pool]
        features = np.asarray(
            [feature_builder(int(row_index), candidate) for candidate in pool],
            dtype=np.float32,
        )
        pad = int(pool_limit) - len(pool)
        if pad > 0:
            ids.extend([0] * pad)
            features = np.vstack(
                [features, np.zeros((pad, features.shape[1]), dtype=np.float32)]
            )
        query_indices.append(int(row_index))
        candidate_rows.append(ids[: int(pool_limit)])
        mask_rows.append([True] * len(pool) + [False] * max(0, pad))
        feature_rows.append(features[: int(pool_limit)])
    return PoolData(
        query_indices=np.asarray(query_indices, dtype=np.int64),
        candidate_ids=np.asarray(candidate_rows, dtype=np.int64),
        mask=np.asarray(mask_rows, dtype=bool),
        pair_features=np.asarray(feature_rows, dtype=np.float32),
        target_in_source_pool=int(target_in_source_pool),
        rows_with_same_family_negative=int(rows_with_same_family_negative),
        rows_with_cross_family_negative=int(rows_with_cross_family_negative),
    )


def candidate_label_tensor(
    registry: CandidateRegistry,
    max_labels: int,
) -> tuple[np.ndarray, np.ndarray]:
    label_ids = np.full((len(registry.keys), int(max_labels)), -1, dtype=np.int64)
    mask = np.zeros((len(registry.keys), int(max_labels)), dtype=bool)
    for candidate_id, candidate in enumerate(registry.keys):
        values = list(candidate)[: int(max_labels)]
        if values:
            label_ids[candidate_id, : len(values)] = values
            mask[candidate_id, : len(values)] = True
    return label_ids, mask


class QueryPoolDataset(Dataset):
    def __init__(self, pool: PoolData) -> None:
        self.pool = pool

    def __len__(self) -> int:
        return int(len(self.pool.query_indices))

    def __getitem__(self, index: int):
        return (
            self.pool.query_indices[int(index)],
            self.pool.candidate_ids[int(index)],
            self.pool.mask[int(index)],
            self.pool.pair_features[int(index)],
        )


class StructuredEnergyRanker(nn.Module):
    def __init__(
        self,
        query_dim: int,
        label_dim: int,
        pair_dim: int,
        hidden_dim: int,
        dropout: float,
        label_features: torch.Tensor,
        candidate_labels: torch.Tensor,
        candidate_label_mask: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("label_features", label_features)
        self.register_buffer("candidate_labels", candidate_labels)
        self.register_buffer("candidate_label_mask", candidate_label_mask)
        self.query_encoder = nn.Sequential(
            nn.Linear(int(query_dim), int(hidden_dim) * 2),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim) * 2),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
        )
        self.label_encoder = nn.Sequential(
            nn.Linear(int(label_dim), int(hidden_dim) * 2),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim) * 2),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(int(hidden_dim) * 3, int(hidden_dim) * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
        )
        self.pair_encoder = nn.Sequential(
            nn.Linear(int(pair_dim), int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
        )
        self.energy_head = nn.Sequential(
            nn.Linear(int(hidden_dim) * 5, int(hidden_dim) * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(
        self,
        query: torch.Tensor,
        candidate_ids: torch.Tensor,
        pair_features: torch.Tensor,
    ) -> torch.Tensor:
        batch, candidates = candidate_ids.shape
        query_encoded = self.query_encoder(query)
        flat_candidate_ids = candidate_ids.reshape(-1)
        label_ids = self.candidate_labels[flat_candidate_ids]
        label_mask = self.candidate_label_mask[flat_candidate_ids]
        safe_label_ids = label_ids.clamp(min=0)
        labels = self.label_encoder(self.label_features[safe_label_ids])
        mask_float = label_mask.unsqueeze(-1).to(labels.dtype)
        label_sum = (labels * mask_float).sum(dim=1)
        label_mean = label_sum / mask_float.sum(dim=1).clamp(min=1.0)
        label_max = labels.masked_fill(~label_mask.unsqueeze(-1), -1e4).max(dim=1).values
        label_max = torch.where(label_mask.any(dim=1, keepdim=True), label_max, torch.zeros_like(label_max))
        lengths = mask_float.sum(dim=1) / max(1, int(label_mask.shape[1]))
        length_channel = lengths.expand(-1, labels.shape[-1])
        candidate_encoded = self.candidate_encoder(
            torch.cat([label_mean, label_max, length_channel], dim=-1)
        ).reshape(batch, candidates, -1)
        query_expanded = query_encoded.unsqueeze(1).expand(-1, candidates, -1)
        pair_encoded = self.pair_encoder(pair_features)
        joined = torch.cat(
            [
                query_expanded,
                candidate_encoded,
                query_expanded * candidate_encoded,
                torch.abs(query_expanded - candidate_encoded),
                pair_encoded,
            ],
            dim=-1,
        )
        return self.energy_head(joined).squeeze(-1)


def family_slot_rerank(
    base: Sequence[SetKey],
    scores: np.ndarray,
    label_families: Sequence[str],
    protected_prefix: int,
    slate_size: int = 10,
    minimum_gain: float = 0.0,
) -> List[SetKey]:
    unique = list(dict.fromkeys(base))
    index_by_candidate = {candidate: index for index, candidate in enumerate(unique)}
    current_scores = np.asarray(scores[: len(unique)], dtype=np.float32)
    keys = [family_key(candidate, label_families) for candidate in unique]
    by_family: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        by_family[key].append(index)
    for indices in by_family.values():
        indices.sort(key=lambda index: (-float(current_scores[index]), index))

    return _family_slot_rerank_prepared(
        unique,
        current_scores,
        keys,
        by_family,
        index_by_candidate,
        protected_prefix,
        slate_size,
        minimum_gain,
        append_remaining=True,
    )


def _family_slot_rerank_prepared(
    unique: Sequence[SetKey],
    current_scores: np.ndarray,
    keys: Sequence[Tuple[str, ...]],
    by_family: Dict[Tuple[str, ...], List[int]],
    index_by_candidate: Dict[SetKey, int],
    protected_prefix: int,
    slate_size: int,
    minimum_gain: float,
    append_remaining: bool,
) -> List[SetKey]:
    """Rerank a slate using precomputed family keys and within-family order."""

    selected = list(unique[: int(protected_prefix)])
    selected_set = set(selected)
    for slot_index, key in enumerate(
        keys[len(selected) : int(slate_size)], start=len(selected)
    ):
        chosen = None
        indices = by_family.get(key, [])
        for index in indices:
            candidate = unique[index]
            if candidate not in selected_set:
                chosen = candidate
                break
        base_candidate = unique[int(slot_index)]
        if base_candidate not in selected_set and chosen is not None:
            chosen_index = index_by_candidate[chosen]
            base_index = int(slot_index)
            if float(current_scores[chosen_index]) < (
                float(current_scores[base_index]) + float(minimum_gain)
            ):
                chosen = base_candidate
        if chosen is None:
            chosen = next((candidate for candidate in unique if candidate not in selected_set), None)
        if chosen is not None:
            selected.append(chosen)
            selected_set.add(chosen)
    if not append_remaining:
        return selected
    return selected + [candidate for candidate in unique if candidate not in selected_set]


def _global_score_rerank_prepared(
    unique: Sequence[SetKey],
    current_scores: np.ndarray,
    protected_prefix: int,
    slate_size: int,
    candidate_window: int,
    append_remaining: bool,
) -> List[SetKey]:
    """Allow the energy model to reallocate slots across precursor families."""

    protected = min(int(protected_prefix), int(slate_size), len(unique))
    selected = list(unique[:protected])
    selected_set = set(selected)
    limit = min(len(unique), max(int(candidate_window), int(slate_size)))
    remaining_indices = list(range(protected, limit))
    remaining_indices.sort(key=lambda index: (-float(current_scores[index]), index))
    for index in remaining_indices:
        candidate = unique[index]
        if candidate in selected_set:
            continue
        selected.append(candidate)
        selected_set.add(candidate)
        if len(selected) >= int(slate_size):
            break
    if len(selected) < int(slate_size):
        for candidate in unique:
            if candidate not in selected_set:
                selected.append(candidate)
                selected_set.add(candidate)
            if len(selected) >= int(slate_size):
                break
    if not append_remaining:
        return selected
    return selected + [candidate for candidate in unique if candidate not in selected_set]


def _global_safe_swap_prepared(
    unique: Sequence[SetKey],
    current_scores: np.ndarray,
    protected_prefix: int,
    slate_size: int,
    candidate_window: int,
    minimum_gain: float,
    append_remaining: bool,
) -> List[SetKey]:
    """Replace only weak unprotected base slots with confident outsiders."""

    slate_size = min(int(slate_size), len(unique))
    protected = min(int(protected_prefix), slate_size)
    selected = list(unique[:slate_size])
    replaceable = list(range(protected, slate_size))
    replaceable.sort(key=lambda index: (float(current_scores[index]), -index))
    limit = min(len(unique), max(int(candidate_window), slate_size))
    outsiders = list(range(slate_size, limit))
    outsiders.sort(key=lambda index: (-float(current_scores[index]), index))
    for outsider_index in outsiders:
        if not replaceable:
            break
        base_index = replaceable[0]
        if float(current_scores[outsider_index]) < (
            float(current_scores[base_index]) + float(minimum_gain)
        ):
            break
        selected[base_index] = unique[outsider_index]
        replaceable.pop(0)
    selected_set = set(selected)
    if not append_remaining:
        return selected
    return selected + [candidate for candidate in unique if candidate not in selected_set]


def score_candidates(
    model: StructuredEnergyRanker,
    query_features: np.ndarray,
    candidate_ids: np.ndarray,
    pair_features: np.ndarray,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    model.eval()
    output = np.zeros(len(candidate_ids), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(candidate_ids), int(chunk_size)):
            end = min(len(candidate_ids), start + int(chunk_size))
            query = torch.from_numpy(query_features[start:end]).to(device)
            candidates = torch.from_numpy(candidate_ids[start:end, None]).to(device)
            pair = torch.from_numpy(pair_features[start:end, None]).to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                values = model(query, candidates, pair).squeeze(1)
            output[start:end] = values.float().cpu().numpy()
    return output


def build_validation_pairs(
    base_rows: Sequence[Sequence[SetKey]],
    meta: pd.DataFrame,
    registry: CandidateRegistry,
    feature_builder,
    include_indices: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    query_indices: List[int] = []
    candidate_ids: List[int] = []
    pair_features: List[np.ndarray] = []
    spans: List[Tuple[int, int]] = []
    offset = 0
    for row_index, candidates in enumerate(base_rows):
        values = (
            list(dict.fromkeys(candidates))
            if include_indices is None or int(row_index) in include_indices
            else []
        )
        for candidate in values:
            query_indices.append(int(row_index))
            candidate_ids.append(registry.id_for(candidate))
            pair_features.append(feature_builder(int(row_index), candidate))
        spans.append((offset, offset + len(values)))
        offset += len(values)
    return (
        np.asarray(query_indices, dtype=np.int64),
        np.asarray(candidate_ids, dtype=np.int64),
        np.asarray(pair_features, dtype=np.float32),
        spans,
    )


def evaluate_grid(
    targets: Sequence[SetKey],
    base_rows: Sequence[Sequence[SetKey]],
    raw_scores: np.ndarray,
    spans: Sequence[Tuple[int, int]],
    label_families: Sequence[str],
    alphas: Sequence[float],
    protected_prefixes: Sequence[int],
    minimum_gains: Sequence[float],
    candidate_windows: Sequence[int] = (20, 50, 100),
) -> tuple[Dict[str, object], List[List[SetKey]], List[Dict[str, object]]]:
    trials: List[Dict[str, object]] = []
    best_key = None
    best_trial: Dict[str, object] = {}
    best_parameters: Dict[str, object] | None = None
    base_top10 = np.asarray(
        [target in set(row[:10]) for target, row in zip(targets, base_rows)], dtype=bool
    )
    prepared_rows = []
    row_group_ids: List[int] = []
    row_group_lookup: Dict[tuple[object, ...], int] = {}
    for row, (start, end) in zip(base_rows, spans):
        unique = list(dict.fromkeys(row))
        keys = [family_key(candidate, label_families) for candidate in unique]
        prepared_rows.append(
            (
                unique,
                keys,
                {candidate: index for index, candidate in enumerate(unique)},
            )
        )
        signature = (
            tuple(unique),
            np.asarray(raw_scores[int(start) : int(end)], dtype=np.float32).tobytes(),
        )
        group_id = row_group_lookup.get(signature)
        if group_id is None:
            group_id = len(row_group_lookup)
            row_group_lookup[signature] = int(group_id)
        row_group_ids.append(int(group_id))
    for alpha in alphas:
        score_rows = []
        ordered_family_rows = []
        prepared_group_cache: Dict[
            int, tuple[np.ndarray, Dict[Tuple[str, ...], List[int]]]
        ] = {}
        for (unique, keys, _), (start, end), group_id in zip(
            prepared_rows, spans, row_group_ids
        ):
            cached_group = prepared_group_cache.get(group_id)
            if cached_group is None:
                scores = raw_scores[start:end].copy()
                prior = -np.log1p(np.arange(len(scores), dtype=np.float32))
                blended_scores = scores + float(alpha) * prior
                by_family: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
                for index, key in enumerate(keys):
                    by_family[key].append(index)
                for indices in by_family.values():
                    indices.sort(
                        key=lambda index: (-float(blended_scores[index]), index)
                    )
                cached_group = (blended_scores, by_family)
                prepared_group_cache[group_id] = cached_group
            score_rows.append(cached_group[0])
            ordered_family_rows.append(cached_group[1])
        for protected in protected_prefixes:
            for minimum_gain in minimum_gains:
                ranked_cache: Dict[int, List[SetKey]] = {}
                ranked: List[List[SetKey]] = []
                for (
                    (unique, keys, index_by_candidate),
                    scores,
                    by_family,
                    group_id,
                ) in zip(prepared_rows, score_rows, ordered_family_rows, row_group_ids):
                    if group_id not in ranked_cache:
                        ranked_cache[group_id] = _family_slot_rerank_prepared(
                        unique,
                        scores,
                        keys,
                        by_family,
                        index_by_candidate,
                        int(protected),
                        10,
                        float(minimum_gain),
                        append_remaining=False,
                    )
                    ranked.append(ranked_cache[group_id])
                metrics = {
                    f"exact_hit@{cutoff}": float(
                        np.mean(
                            [
                                target in row[:cutoff]
                                for target, row in zip(targets, ranked)
                            ]
                        )
                    )
                    for cutoff in (1, 3, 5, 10)
                }
                current_top10 = np.asarray(
                    [target in set(row[:10]) for target, row in zip(targets, ranked)],
                    dtype=bool,
                )
                trial: Dict[str, object] = {
                    "strategy": "family_slot",
                    "alpha": float(alpha),
                    "protected_prefix": int(protected),
                    "minimum_gain": float(minimum_gain),
                    "new_hits_over_base": int((current_top10 & ~base_top10).sum()),
                    "lost_hits_vs_base": int((base_top10 & ~current_top10).sum()),
                    **metrics,
                }
                trials.append(trial)
                key = (
                    float(metrics["exact_hit@10"]),
                    float(metrics["exact_hit@5"]),
                    float(metrics["exact_hit@1"]),
                    -int(trial["lost_hits_vs_base"]),
                    -int(protected),
                    -float(minimum_gain),
                    -float(alpha),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_trial = trial
                    best_parameters = {
                        "strategy": "family_slot",
                        "alpha": float(alpha),
                        "protected_prefix": int(protected),
                        "minimum_gain": float(minimum_gain),
                    }

        for protected in protected_prefixes:
            for candidate_window in candidate_windows:
                ranked_cache = {}
                ranked = []
                for (unique, _, _), scores, group_id in zip(
                    prepared_rows, score_rows, row_group_ids
                ):
                    if group_id not in ranked_cache:
                        ranked_cache[group_id] = _global_score_rerank_prepared(
                            unique,
                            scores,
                            int(protected),
                            10,
                            int(candidate_window),
                            append_remaining=False,
                        )
                    ranked.append(ranked_cache[group_id])
                metrics = {
                    f"exact_hit@{cutoff}": float(
                        np.mean(
                            [
                                target in row[:cutoff]
                                for target, row in zip(targets, ranked)
                            ]
                        )
                    )
                    for cutoff in (1, 3, 5, 10)
                }
                current_top10 = np.asarray(
                    [target in set(row[:10]) for target, row in zip(targets, ranked)],
                    dtype=bool,
                )
                trial = {
                    "strategy": "global_score",
                    "alpha": float(alpha),
                    "protected_prefix": int(protected),
                    "candidate_window": int(candidate_window),
                    "new_hits_over_base": int((current_top10 & ~base_top10).sum()),
                    "lost_hits_vs_base": int((base_top10 & ~current_top10).sum()),
                    **metrics,
                }
                trials.append(trial)
                key = (
                    float(metrics["exact_hit@10"]),
                    float(metrics["exact_hit@5"]),
                    float(metrics["exact_hit@1"]),
                    -int(trial["lost_hits_vs_base"]),
                    -int(protected),
                    0.0,
                    -float(alpha),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_trial = trial
                    best_parameters = {
                        "strategy": "global_score",
                        "alpha": float(alpha),
                        "protected_prefix": int(protected),
                        "candidate_window": int(candidate_window),
                    }

                for minimum_gain in minimum_gains:
                    safe_ranked_cache: Dict[int, List[SetKey]] = {}
                    safe_ranked: List[List[SetKey]] = []
                    for (unique, _, _), scores, group_id in zip(
                        prepared_rows, score_rows, row_group_ids
                    ):
                        if group_id not in safe_ranked_cache:
                            safe_ranked_cache[group_id] = _global_safe_swap_prepared(
                                unique,
                                scores,
                                int(protected),
                                10,
                                int(candidate_window),
                                float(minimum_gain),
                                append_remaining=False,
                            )
                        safe_ranked.append(safe_ranked_cache[group_id])
                    safe_metrics = {
                        f"exact_hit@{cutoff}": float(
                            np.mean(
                                [
                                    target in row[:cutoff]
                                    for target, row in zip(targets, safe_ranked)
                                ]
                            )
                        )
                        for cutoff in (1, 3, 5, 10)
                    }
                    safe_top10 = np.asarray(
                        [
                            target in set(row[:10])
                            for target, row in zip(targets, safe_ranked)
                        ],
                        dtype=bool,
                    )
                    safe_trial = {
                        "strategy": "global_safe_swap",
                        "alpha": float(alpha),
                        "protected_prefix": int(protected),
                        "candidate_window": int(candidate_window),
                        "minimum_gain": float(minimum_gain),
                        "new_hits_over_base": int((safe_top10 & ~base_top10).sum()),
                        "lost_hits_vs_base": int((base_top10 & ~safe_top10).sum()),
                        **safe_metrics,
                    }
                    trials.append(safe_trial)
                    safe_key = (
                        float(safe_metrics["exact_hit@10"]),
                        float(safe_metrics["exact_hit@5"]),
                        float(safe_metrics["exact_hit@1"]),
                        -int(safe_trial["lost_hits_vs_base"]),
                        -int(protected),
                        -float(minimum_gain),
                        -float(alpha),
                    )
                    if best_key is None or safe_key > best_key:
                        best_key = safe_key
                        best_trial = safe_trial
                        best_parameters = {
                            "strategy": "global_safe_swap",
                            "alpha": float(alpha),
                            "protected_prefix": int(protected),
                            "candidate_window": int(candidate_window),
                            "minimum_gain": float(minimum_gain),
                        }

    if best_parameters is None:
        raise RuntimeError("empty structured-energy evaluation grid")
    best_alpha = float(best_parameters["alpha"])
    best_protected = int(best_parameters["protected_prefix"])
    best_rows: List[List[SetKey]] = []
    for (unique, keys, index_by_candidate), (start, end) in zip(prepared_rows, spans):
        scores = raw_scores[start:end].copy()
        prior = -np.log1p(np.arange(len(scores), dtype=np.float32))
        scores += float(best_alpha) * prior
        if best_parameters["strategy"] == "global_score":
            best_rows.append(
                _global_score_rerank_prepared(
                    unique,
                    scores,
                    best_protected,
                    10,
                    int(best_parameters["candidate_window"]),
                    append_remaining=True,
                )
            )
        elif best_parameters["strategy"] == "global_safe_swap":
            best_rows.append(
                _global_safe_swap_prepared(
                    unique,
                    scores,
                    best_protected,
                    10,
                    int(best_parameters["candidate_window"]),
                    float(best_parameters["minimum_gain"]),
                    append_remaining=True,
                )
            )
        else:
            by_family: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
            for index, key in enumerate(keys):
                by_family[key].append(index)
            for indices in by_family.values():
                indices.sort(key=lambda index: (-float(scores[index]), index))
            best_rows.append(
                _family_slot_rerank_prepared(
                    unique,
                    scores,
                    keys,
                    by_family,
                    index_by_candidate,
                    best_protected,
                    10,
                    float(best_parameters["minimum_gain"]),
                    append_remaining=True,
                )
            )
    best_trial = {**best_trial, **exact_metrics(targets, best_rows)}
    return best_trial, best_rows, trials


def apply_fixed_trial(
    targets: Sequence[SetKey],
    base_rows: Sequence[Sequence[SetKey]],
    raw_scores: np.ndarray,
    spans: Sequence[Tuple[int, int]],
    label_families: Sequence[str],
    parameters: Dict[str, object],
) -> tuple[Dict[str, object], List[List[SetKey]]]:
    """Apply a validation-selected slate policy without searching evaluation labels."""

    strategy = str(parameters["strategy"])
    if strategy not in {"family_slot", "global_score", "global_safe_swap"}:
        raise ValueError(f"unsupported fixed slate strategy: {strategy!r}")
    alpha = float(parameters["alpha"])
    protected = int(parameters["protected_prefix"])
    rows: List[List[SetKey]] = []
    for base, (start, end) in zip(base_rows, spans):
        unique = list(dict.fromkeys(base))
        scores = np.asarray(raw_scores[int(start) : int(end)], dtype=np.float32).copy()
        if len(scores) != len(unique):
            raise ValueError("fixed slate score span does not match candidate row")
        scores += float(alpha) * -np.log1p(np.arange(len(scores), dtype=np.float32))
        if strategy == "global_score":
            ranked = _global_score_rerank_prepared(
                unique,
                scores,
                protected,
                10,
                int(parameters["candidate_window"]),
                append_remaining=True,
            )
        elif strategy == "global_safe_swap":
            ranked = _global_safe_swap_prepared(
                unique,
                scores,
                protected,
                10,
                int(parameters["candidate_window"]),
                float(parameters["minimum_gain"]),
                append_remaining=True,
            )
        else:
            keys = [family_key(candidate, label_families) for candidate in unique]
            by_family: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
            for index, key in enumerate(keys):
                by_family[key].append(index)
            for indices in by_family.values():
                indices.sort(key=lambda index: (-float(scores[index]), index))
            ranked = _family_slot_rerank_prepared(
                unique,
                scores,
                keys,
                by_family,
                {candidate: index for index, candidate in enumerate(unique)},
                protected,
                10,
                float(parameters["minimum_gain"]),
                append_remaining=True,
            )
        rows.append(ranked)
    base_hits = np.asarray(
        [target in set(row[:10]) for target, row in zip(targets, base_rows)], dtype=bool
    )
    current_hits = np.asarray(
        [target in set(row[:10]) for target, row in zip(targets, rows)], dtype=bool
    )
    report = {
        **parameters,
        "selection_mode": "fixed_from_validation",
        "new_hits_over_base": int((current_hits & ~base_hits).sum()),
        "lost_hits_vs_base": int((base_hits & ~current_hits).sum()),
        **exact_metrics(targets, rows),
    }
    return report, rows


def best_trials_by_strategy(
    trials: Sequence[Dict[str, object]],
) -> Dict[str, Dict[str, object]]:
    """Keep an auditable best row for every slate-allocation strategy."""

    output: Dict[str, Dict[str, object]] = {}
    for strategy in sorted({str(row.get("strategy", "unknown")) for row in trials}):
        rows = [row for row in trials if str(row.get("strategy", "unknown")) == strategy]
        if not rows:
            continue
        output[strategy] = dict(
            max(
                rows,
                key=lambda row: (
                    float(row["exact_hit@10"]),
                    float(row["exact_hit@5"]),
                    float(row["exact_hit@1"]),
                    -int(row["lost_hits_vs_base"]),
                ),
            )
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--base_val_candidates", required=True)
    parser.add_argument("--train_candidate_source", action="append", default=[])
    parser.add_argument("--matsci_embeddings", required=True)
    parser.add_argument("--candidate_limit", type=int, default=100)
    parser.add_argument("--source_union_limit", type=int, default=300)
    parser.add_argument("--train_pool_limit", type=int, default=64)
    parser.add_argument("--cross_family_negatives", type=int, default=0)
    parser.add_argument("--max_labels", type=int, default=8)
    parser.add_argument("--matsci_components", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pairwise_weight", type=float, default=0.25)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--gradient_clip", type=float, default=2.0)
    parser.add_argument("--eval_every", type=int, default=2)
    parser.add_argument("--score_chunk_size", type=int, default=8192)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--resume_model", default="")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", required=True)
    parser.add_argument("--output_model", required=True)
    args = parser.parse_args()
    if not args.train_candidate_source:
        parser.error("at least one --train_candidate_source is required")
    if bool(args.eval_only) and not str(args.resume_model):
        parser.error("--eval_only requires --resume_model")
    seed_everything(int(args.seed))

    input_dir = Path(args.input_dir).resolve()
    train_pack = np.load(input_dir / "train.npz", allow_pickle=True)
    val_pack = np.load(input_dir / "val.npz", allow_pickle=True)
    train_x = np.asarray(train_pack["x"], dtype=np.float32)
    val_x = np.asarray(val_pack["x"], dtype=np.float32)
    train_y = np.asarray(train_pack["y_multi_hot"], dtype=np.float32)
    val_y = np.asarray(val_pack["y_multi_hot"], dtype=np.float32)
    train_targets = targets_from_matrix(train_y)
    val_targets = targets_from_matrix(val_y)
    train_meta = pd.read_csv(input_dir / "train_meta.csv", low_memory=False)
    val_meta = pd.read_csv(input_dir / "val_meta.csv", low_memory=False)
    names = [
        str(value)
        for value in json.loads((input_dir / "precursor_names.json").read_text(encoding="utf-8"))
    ]
    label_families = [precursor_family(name) for name in names]
    label_elements, label_groups, label_metals = label_chemistry(names)
    train_seen = np.asarray(train_y.sum(axis=0) > 0)
    length_modes = family_length_modes(train_meta, train_y)
    prior_builder = CandidatePriorBuilder(train_y, train_meta)
    template_builder = TemplatePriorBuilder(train_y, train_meta, names)

    label_matsci, train_matsci, val_matsci, matsci_metadata = load_matsci_pca_views(
        Path(args.matsci_embeddings).resolve(), int(args.matsci_components), int(args.seed)
    )
    label_formula, label_parse = label_structured_features(names)
    label_features = np.hstack([label_formula, label_matsci]).astype(np.float32)
    label_features, _, label_mean, label_std = standardize_from_train(
        label_features, label_features
    )
    train_meta_features, val_meta_features, meta_feature_names = metadata_features(
        train_meta, val_meta
    )
    train_query_raw = np.hstack([train_x, train_matsci, train_meta_features]).astype(np.float32)
    val_query_raw = np.hstack([val_x, val_matsci, val_meta_features]).astype(np.float32)
    train_query, val_query, query_mean, query_std = standardize_from_train(
        train_query_raw, val_query_raw
    )

    train_families = train_meta["family_signature_primary"].fillna("UNK").astype(str).to_numpy()
    val_families = val_meta["family_signature_primary"].fillna("UNK").astype(str).to_numpy()

    def train_pair(row_index: int, candidate: SetKey) -> np.ndarray:
        family = str(train_families[int(row_index)])
        return build_pair_features(
            candidate,
            train_meta.iloc[int(row_index)],
            family,
            int(length_modes.get(family, length_modes["__GLOBAL__"])),
            label_elements,
            label_groups,
            label_metals,
            train_seen,
            prior_builder,
            template_builder,
        )

    def val_pair(row_index: int, candidate: SetKey) -> np.ndarray:
        family = str(val_families[int(row_index)])
        return build_pair_features(
            candidate,
            val_meta.iloc[int(row_index)],
            family,
            int(length_modes.get(family, length_modes["__GLOBAL__"])),
            label_elements,
            label_groups,
            label_metals,
            train_seen,
            prior_builder,
            template_builder,
        )

    train_sources = [
        load_source(path, len(train_targets), int(args.source_union_limit))
        for path in args.train_candidate_source
    ]
    base_rows = load_source(
        args.base_val_candidates, len(val_targets), int(args.candidate_limit)
    )
    registry = CandidateRegistry()
    train_pool = build_training_pools(
        train_targets,
        train_meta,
        train_sources,
        label_families,
        int(args.train_pool_limit),
        int(args.source_union_limit),
        registry,
        train_pair,
        int(args.seed),
        int(args.cross_family_negatives),
    )
    val_query_indices, val_candidate_ids, val_pair_features, val_spans = build_validation_pairs(
        base_rows, val_meta, registry, val_pair
    )
    candidate_labels, candidate_label_mask = candidate_label_tensor(
        registry, int(args.max_labels)
    )
    pair_dim = int(train_pool.pair_features.shape[-1])
    pair_flat = train_pool.pair_features[train_pool.mask]
    pair_mean = pair_flat.mean(axis=0, dtype=np.float64).astype(np.float32)
    pair_std = pair_flat.std(axis=0, dtype=np.float64).astype(np.float32)
    pair_std = np.where(pair_std < 1e-6, 1.0, pair_std).astype(np.float32)
    train_pool.pair_features = np.nan_to_num(
        (train_pool.pair_features - pair_mean) / pair_std
    ).astype(np.float32)
    train_pool.pair_features[~train_pool.mask] = 0.0
    val_pair_features = np.nan_to_num((val_pair_features - pair_mean) / pair_std).astype(
        np.float32
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StructuredEnergyRanker(
        train_query.shape[1],
        label_features.shape[1],
        pair_dim,
        int(args.hidden_dim),
        float(args.dropout),
        torch.from_numpy(label_features),
        torch.from_numpy(candidate_labels),
        torch.from_numpy(candidate_label_mask),
    ).to(device)
    if str(args.resume_model):
        resume_payload = torch.load(
            Path(args.resume_model).resolve(), map_location="cpu", weights_only=False
        )
        resume_state = resume_payload.get("model_state", resume_payload)
        model.load_state_dict(resume_state)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(args.epochs))
    )
    loader_generator = torch.Generator().manual_seed(int(args.seed))
    loader = DataLoader(
        QueryPoolDataset(train_pool),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        generator=loader_generator,
        persistent_workers=bool(int(args.num_workers) > 0),
    )
    train_query_tensor = torch.from_numpy(train_query)
    val_query_for_candidates = val_query[val_query_indices]

    # The energy scale grows with width and training time.  Keep a wide prior
    # interpolation grid so a larger model is not unfairly compared with the
    # fixed base rank at the edge of a too-small search interval.
    alpha_grid = (
        0.0,
        0.025,
        0.05,
        0.1,
        0.2,
        0.4,
        0.8,
        1.6,
        3.2,
        6.4,
        12.8,
        25.6,
    )
    protected_grid = (0, 1, 3, 5, 7, 9, 10)
    minimum_gain_grid = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    history: List[Dict[str, object]] = []
    best_key = None
    best_state = (
        {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        if bool(args.eval_only)
        else None
    )
    best_trial: Dict[str, object] = {}
    best_rows: List[List[SetKey]] = []
    epoch_iterator = () if bool(args.eval_only) else range(1, int(args.epochs) + 1)
    for epoch in epoch_iterator:
        model.train()
        loss_sum = 0.0
        rows_seen = 0
        for query_indices, candidate_ids, mask, pair_features in loader:
            query = train_query_tensor[query_indices].to(device, non_blocking=True)
            candidate_ids = candidate_ids.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            pair_features = pair_features.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                scores = model(query, candidate_ids, pair_features)
                masked_scores = scores.masked_fill(~mask, -1e4)
                target = torch.zeros(len(scores), dtype=torch.long, device=device)
                listwise = nn.functional.cross_entropy(masked_scores.float(), target)
                negative = scores[:, 1:].masked_fill(~mask[:, 1:], -1e4).max(dim=1).values
                pairwise = nn.functional.relu(float(args.margin) - scores[:, 0] + negative).mean()
                loss = listwise + float(args.pairwise_weight) * pairwise
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(args.gradient_clip))
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(scores)
            rows_seen += int(len(scores))
        scheduler.step()
        epoch_row: Dict[str, object] = {
            "epoch": int(epoch),
            "train_loss": float(loss_sum / max(1, rows_seen)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        if epoch % int(args.eval_every) == 0 or epoch == int(args.epochs):
            val_scores = score_candidates(
                model,
                val_query_for_candidates,
                val_candidate_ids,
                val_pair_features,
                device,
                int(args.score_chunk_size),
            )
            trial, ranked, _ = evaluate_grid(
                val_targets,
                base_rows,
                val_scores,
                val_spans,
                label_families,
                alpha_grid,
                protected_grid,
                minimum_gain_grid,
            )
            epoch_row["validation"] = trial
            current_key = (
                float(trial["exact_hit@10"]),
                float(trial["exact_hit@5"]),
                float(trial["exact_hit@1"]),
                -float(epoch_row["train_loss"]),
            )
            if best_key is None or current_key > best_key:
                best_key = current_key
                best_trial = dict(trial)
                best_rows = ranked
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                checkpoint_path = Path(f"{args.output_model}.checkpoint").resolve()
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state": best_state,
                        "epoch": int(epoch),
                        "best_trial": best_trial,
                    },
                    checkpoint_path,
                )
        history.append(epoch_row)
        print(json.dumps(epoch_row, ensure_ascii=False), flush=True)

    if best_state is None:
        raise RuntimeError("no validation checkpoint was evaluated")
    model.load_state_dict(best_state)
    val_scores = score_candidates(
        model,
        val_query_for_candidates,
        val_candidate_ids,
        val_pair_features,
        device,
        int(args.score_chunk_size),
    )
    best_trial, best_rows, final_trials = evaluate_grid(
        val_targets,
        base_rows,
        val_scores,
        val_spans,
        label_families,
        alpha_grid,
        protected_grid,
        minimum_gain_grid,
    )

    base_family_keys = [
        {family_key(candidate, label_families) for candidate in row[:10]} for row in base_rows
    ]
    within_family_oracle = np.asarray(
        [
            family_key(target, label_families) in keys and target in set(row[: int(args.candidate_limit)])
            for target, row, keys in zip(val_targets, base_rows, base_family_keys)
        ],
        dtype=bool,
    )
    report = {
        "protocol": "train_oof_structured_energy_exact_variant_val_formula_group_disjoint",
        "config": vars(args),
        "device": str(device),
        "model_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "label_parse": label_parse,
        "matsci": matsci_metadata,
        "dimensions": {
            "query": int(train_query.shape[1]),
            "label": int(label_features.shape[1]),
            "pair": int(pair_dim),
            "candidate_registry": int(len(registry.keys)),
            "meta_features": int(len(meta_feature_names)),
        },
        "training": {
            "rows_total": int(len(train_targets)),
            "rows_used": int(len(train_pool.query_indices)),
            "target_in_oof_source_pool": int(train_pool.target_in_source_pool),
            "rows_with_same_family_negative": int(train_pool.rows_with_same_family_negative),
            "rows_with_cross_family_negative": int(
                train_pool.rows_with_cross_family_negative
            ),
            "pool_width": int(train_pool.candidate_ids.shape[1]),
        },
        "validation": {
            "rows": int(len(val_targets)),
            "base": exact_metrics(val_targets, base_rows),
            "within_family_top100_oracle_exact_hit@10": float(within_family_oracle.mean()),
            "best": best_trial,
            "best_by_strategy": best_trials_by_strategy(final_trials),
            "top_trials": sorted(
                final_trials,
                key=lambda row: (
                    -float(row["exact_hit@10"]),
                    -float(row["exact_hit@5"]),
                    -float(row["exact_hit@1"]),
                ),
            )[:30],
        },
        "history": history,
    }
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_candidates = Path(args.output_candidates_jsonl).resolve()
    output_candidates.parent.mkdir(parents=True, exist_ok=True)
    with output_candidates.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(best_rows):
            handle.write(
                json.dumps(
                    {"row_index": row_index, "candidate_label_ids": [list(value) for value in row]}
                )
                + "\n"
            )
    output_model = Path(args.output_model).resolve()
    output_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "model_config": {
                "query_dim": int(train_query.shape[1]),
                "label_dim": int(label_features.shape[1]),
                "pair_dim": int(pair_dim),
                "hidden_dim": int(args.hidden_dim),
                "dropout": float(args.dropout),
            },
            "label_features": label_features,
            "candidate_keys": registry.keys,
            "candidate_labels": candidate_labels,
            "candidate_label_mask": candidate_label_mask,
            "normalization": {
                "label_mean": label_mean,
                "label_std": label_std,
                "query_mean": query_mean,
                "query_std": query_std,
                "pair_mean": pair_mean,
                "pair_std": pair_std,
            },
            "best_trial": best_trial,
            "protocol": report["protocol"],
        },
        output_model,
    )
    print(json.dumps(report["validation"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
