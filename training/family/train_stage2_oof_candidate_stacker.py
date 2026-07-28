#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.family.evaluate_stage2_candidate_fusion import load_source  # noqa: E402
from training.family.evaluate_stage2_oof_chemistry_rescore import (  # noqa: E402
    chemistry_features_for_candidate,
    family_length_modes,
    json_set,
    label_chemistry,
)


SetKey = Tuple[int, ...]
MULTILABEL_FEATURE_DIM = 20


class CandidatePriorBuilder:
    """Train-only exact-set and label-frequency priors used by candidate rankers."""

    feature_dim = 8

    def __init__(self, train_y: np.ndarray, train_meta: pd.DataFrame) -> None:
        targets = [tuple(np.flatnonzero(row > 0.5).tolist()) for row in train_y]
        families = train_meta["family_signature_primary"].fillna("UNK").astype(str).to_numpy()
        self.row_count = max(1, len(targets))
        self.set_frequency = Counter(targets)
        self.label_frequency = np.asarray(train_y.sum(axis=0), dtype=np.float32) + 1.0
        self.family_row_count = Counter(families.tolist())
        self.family_set_frequency: Dict[str, Counter[SetKey]] = {}
        self.family_label_frequency: Dict[str, np.ndarray] = {}
        for family in np.unique(families):
            mask = families == family
            self.family_set_frequency[str(family)] = Counter(
                target for target, keep in zip(targets, mask) if bool(keep)
            )
            self.family_label_frequency[str(family)] = (
                np.asarray(train_y[mask].sum(axis=0), dtype=np.float32) + 1.0
            )

    def features(self, candidate: SetKey, family: str) -> np.ndarray:
        labels = np.asarray(candidate, dtype=np.int64)
        global_logs = np.log1p(self.label_frequency[labels]) / math.log1p(self.row_count + 1)
        family_rows = max(1, int(self.family_row_count.get(str(family), 0)))
        family_frequency = self.family_label_frequency.get(str(family), self.label_frequency)
        family_logs = np.log1p(family_frequency[labels]) / math.log1p(family_rows + 1)
        return np.asarray(
            [
                math.log1p(self.set_frequency.get(candidate, 0)) / math.log1p(self.row_count + 1),
                math.log1p(
                    self.family_set_frequency.get(str(family), Counter()).get(candidate, 0)
                )
                / math.log1p(family_rows + 1),
                float(global_logs.mean()),
                float(global_logs.min()),
                float(global_logs.max()),
                float(family_logs.mean()),
                float(family_logs.min()),
                float(family_logs.max()),
            ],
            dtype=np.float32,
        )


def precursor_route_token(raw_name: str) -> tuple[str, tuple[int, ...], int, int]:
    """Periodic-group-invariant precursor role used for train-only route priors."""
    raw = str(raw_name)
    compact = re.sub(r"\s+", "", raw).replace("·", ".")
    upper = compact.upper()
    if "N2H4" in upper:
        role = "hydrazine"
    elif "CO(NH2)2" in upper or "CH4N2O" in upper:
        role = "urea"
    elif "(NH2)2CS" in upper or "CH4N2S" in upper or "NH2CSNH2" in upper:
        role = "thiourea"
    elif "C2H5NS" in upper or "CH3CSNH2" in upper or "C3H7NS" in upper:
        role = "thioamide"
    elif "C6H12O6" in upper:
        role = "glucose"
    elif "C6H9NO" in upper:
        role = "pvp"
    elif "CH3COO" in upper or "C2H3O2" in upper or "OAC" in upper or "(AC)" in upper:
        role = "acetate"
    elif "NO3" in upper:
        role = "nitrate"
    elif "SO4" in upper:
        role = "sulfate"
    elif "PO4" in upper:
        role = "phosphate"
    elif "CO3" in upper or "HCO3" in upper:
        role = "carbonate"
    elif "C2H5OH" in upper or "CH3CH2OH" in upper:
        role = "alcohol"
    elif any(symbol in compact for symbol in ("Cl", "Br", "I", "F")):
        role = "halide"
    elif "OH" in upper:
        role = "hydroxide"
    else:
        role = "other"
    try:
        base_formula = re.split(r"[·.](?=[0-9]*H2O)", raw, maxsplit=1)[0]
        base_formula = re.sub(r"-t[0-9]+$", "", base_formula, flags=re.IGNORECASE)
        composition = Composition(base_formula)
        elements = list(composition.elements)
    except Exception:
        elements = []
    groups = []
    for element in elements:
        if str(element) in {"H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"}:
            continue
        try:
            group = element.group
        except (AttributeError, TypeError):
            group = None
        if group is not None:
            groups.append(int(group))
    if role == "other":
        if len(elements) == 1:
            role = "elemental"
        elif any(str(element) == "C" for element in elements):
            role = "organic"
        elif any(str(element) == "O" for element in elements):
            role = "oxide_or_other_oxygenate"
    hydrated = int("H2O" in upper or re.search(r"[.][0-9]*H2O", upper) is not None)
    return role, tuple(sorted(groups)), hydrated, min(len(elements), 8)


class TemplatePriorBuilder:
    """Train-only periodic-group route-template frequencies.

    Unlike exact label priors, this transfers a route such as hydrated nitrate
    or metal-halide plus hydrazine between elements in the same periodic group.
    """

    feature_dim = 12

    def __init__(
        self,
        train_y: np.ndarray,
        train_meta: pd.DataFrame,
        precursor_names: Sequence[str],
    ) -> None:
        self.label_tokens = [precursor_route_token(value) for value in precursor_names]
        self.row_count = max(1, len(train_y))
        self.global_template: Counter[tuple] = Counter()
        self.family_template: Dict[str, Counter[tuple]] = {}
        self.route_template: Dict[str, Counter[tuple]] = {}
        self.global_token: Counter[tuple] = Counter()
        self.family_token: Dict[str, Counter[tuple]] = {}
        self.route_token: Dict[str, Counter[tuple]] = {}
        self.family_rows: Counter[str] = Counter()
        self.route_rows: Counter[str] = Counter()
        families = train_meta["family_signature_primary"].fillna("UNK").astype(str).to_numpy()
        anions = [json_set(value) for value in train_meta["target_anion_elements"]]
        for row_index, row in enumerate(train_y):
            candidate = tuple(np.flatnonzero(row > 0.5).tolist())
            family = str(families[row_index])
            route = self.route_key(family, anions[row_index])
            template = self.template(candidate)
            self.global_template[template] += 1
            self.family_template.setdefault(family, Counter())[template] += 1
            self.route_template.setdefault(route, Counter())[template] += 1
            self.family_rows[family] += 1
            self.route_rows[route] += 1
            for token in template:
                self.global_token[token] += 1
                self.family_token.setdefault(family, Counter())[token] += 1
                self.route_token.setdefault(route, Counter())[token] += 1

    @staticmethod
    def route_key(family: str, target_anions: set[str]) -> str:
        groups = []
        for symbol in target_anions:
            try:
                group = Element(symbol).group
            except ValueError:
                continue
            if group is not None:
                groups.append(int(group))
        return f"{family}|{','.join(map(str, sorted(set(groups))))}"

    def template(self, candidate: SetKey) -> tuple:
        return tuple(sorted(self.label_tokens[index] for index in candidate))

    @staticmethod
    def normalized_log(value: int, rows: int) -> float:
        return math.log1p(int(value)) / math.log1p(max(2, int(rows) + 1))

    def features(
        self,
        candidate: SetKey,
        family: str,
        target_anions: set[str],
    ) -> np.ndarray:
        route = self.route_key(str(family), target_anions)
        template = self.template(candidate)
        tokens = list(template)
        family_rows = max(1, int(self.family_rows.get(str(family), 0)))
        route_rows = max(1, int(self.route_rows.get(route, 0)))
        family_template = self.family_template.get(str(family), Counter())
        route_template = self.route_template.get(route, Counter())
        family_token = self.family_token.get(str(family), Counter())
        route_token = self.route_token.get(route, Counter())
        token_sources = (
            (self.global_token, self.row_count),
            (family_token, family_rows),
            (route_token, route_rows),
        )
        values = [
            self.normalized_log(self.global_template.get(template, 0), self.row_count),
            self.normalized_log(family_template.get(template, 0), family_rows),
            self.normalized_log(route_template.get(template, 0), route_rows),
        ]
        for counter, rows in token_sources:
            frequencies = [self.normalized_log(counter.get(token, 0), rows) for token in tokens]
            values.extend([
                float(np.mean(frequencies)),
                float(np.min(frequencies)),
                float(np.max(frequencies)),
            ])
        return np.asarray(values, dtype=np.float32)


def sequence_sha256(values: Sequence[str]) -> str:
    payload = "\n".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalized_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def load_matsci_views(
    path: Path,
    input_dir: Path,
    split: str,
    precursor_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and verify row-aligned, label-free MatSciBERT formula representations."""
    with np.load(path, allow_pickle=False) as cache:
        schema = str(cache["schema_version"].item())
        if schema != "stage2_matscibert_embeddings_v1":
            raise ValueError(f"unsupported MatSciBERT cache schema: {schema!r}")
        expected_names = sequence_sha256([str(value) for value in precursor_names])
        if str(cache["precursor_names_sha256"].item()) != expected_names:
            raise ValueError("MatSciBERT cache precursor vocabulary does not match input pack")
        output = []
        for current in ("train", split):
            formulas = (
                pd.read_csv(input_dir / f"{current}_meta.csv", usecols=["formula"])["formula"]
                .fillna("")
                .astype(str)
                .tolist()
            )
            if str(cache[f"{current}_formula_sha256"].item()) != sequence_sha256(formulas):
                raise ValueError(f"MatSciBERT cache row order does not match {current} metadata")
            output.append(np.concatenate([
                np.asarray(cache[f"{current}_query_common_mean"], dtype=np.float32),
                np.asarray(cache[f"{current}_query_role_mean"], dtype=np.float32),
            ], axis=1))
        label_views = np.concatenate([
            np.asarray(cache["precursor_common_mean"], dtype=np.float32),
            np.asarray(cache["precursor_role_mean"], dtype=np.float32),
        ], axis=1)
    return label_views, output[0], output[1]


def load_matsci_multilabel_scores(
    path: Path,
    input_dir: Path,
    split: str,
    precursor_names: Sequence[str],
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as cache:
        schema = str(cache["schema_version"].item())
        if schema != "stage2_matscibert_multilabel_scores_v1":
            raise ValueError(f"unsupported MatSciBERT score cache schema: {schema!r}")
        if str(cache["precursor_names_sha256"].item()) != sequence_sha256(precursor_names):
            raise ValueError("MatSciBERT score cache precursor vocabulary mismatch")
        formulas = (
            pd.read_csv(input_dir / f"{split}_meta.csv", usecols=["formula"])["formula"]
            .fillna("")
            .astype(str)
            .tolist()
        )
        if str(cache[f"{split}_formula_sha256"].item()) != sequence_sha256(formulas):
            raise ValueError(f"MatSciBERT score cache row order does not match {split} metadata")
        return np.asarray(cache[f"{split}_logits"], dtype=np.float32)


def multilabel_candidate_features(candidates: Sequence[SetKey], logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    order = np.argsort(-logits, kind="stable")
    ranks = np.empty(len(logits), dtype=np.int32)
    ranks[order] = np.arange(1, len(logits) + 1, dtype=np.int32)
    rows = []
    for candidate in candidates:
        labels = np.asarray(candidate, dtype=np.int64)
        current_logits = logits[labels]
        current_probabilities = probabilities[labels]
        current_ranks = ranks[labels]
        reciprocal = 1.0 / current_ranks.astype(np.float32)
        rows.append(np.asarray([
            float(current_logits.mean()),
            float(current_logits.max()),
            float(current_logits.min()),
            float(current_logits.sum()),
            float(current_probabilities.mean()),
            float(current_probabilities.max()),
            float(current_probabilities.min()),
            float(np.log(np.maximum(current_probabilities, 1e-8)).sum()),
            float(reciprocal.mean()),
            float(reciprocal.min()),
            float(reciprocal.max()),
            math.log1p(float(current_ranks.max())) / math.log1p(max(len(logits), 1)),
            *[float(np.all(current_ranks <= k)) for k in (1, 3, 5, 10, 20, 50)],
            float(np.mean(current_ranks <= 10)),
            float(np.mean(current_ranks <= 50)),
        ], dtype=np.float32))
    return np.asarray(rows, dtype=np.float32).reshape(-1, MULTILABEL_FEATURE_DIM)


class MatSciFeatureBuilder:
    """Train-only projection from target-formula text to precursor-set text space."""

    def __init__(
        self,
        label_views: np.ndarray,
        train_query_views: np.ndarray,
        train_y: np.ndarray,
        components: int,
        ridge_alpha: float,
        seed: int,
    ) -> None:
        maximum = min(label_views.shape[0] + train_query_views.shape[0] - 1, label_views.shape[1])
        if not 1 <= int(components) <= maximum:
            raise ValueError(f"MatSciBERT components must be between 1 and {maximum}")
        self.components = int(components)
        self.pca = PCA(
            n_components=self.components,
            svd_solver="randomized",
            random_state=int(seed),
        )
        self.pca.fit(np.vstack([label_views, train_query_views]))
        self.label_embedding = normalized_rows(self.pca.transform(label_views))
        train_query = normalized_rows(self.pca.transform(train_query_views))
        set_targets = np.asarray(train_y @ self.label_embedding, dtype=np.float32)
        set_targets = normalized_rows(set_targets)
        self.ridge = Ridge(alpha=float(ridge_alpha)).fit(train_query, set_targets)

    @property
    def feature_dim(self) -> int:
        return self.components * 5 + 9

    def transform_queries(self, query_views: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        direct = normalized_rows(self.pca.transform(query_views))
        projected = normalized_rows(self.ridge.predict(direct))
        return direct, projected

    def candidate_features(
        self,
        candidates: Sequence[SetKey],
        query_direct: np.ndarray,
        query_projected: np.ndarray,
    ) -> np.ndarray:
        rows: List[np.ndarray] = []
        for candidate in candidates:
            labels = np.asarray(candidate, dtype=np.int64)
            individual = self.label_embedding[labels]
            candidate_embedding = normalized_rows(individual.mean(axis=0, keepdims=True))[0]
            projected_label_scores = individual @ query_projected
            direct_label_scores = individual @ query_direct
            scalars = np.asarray([
                float(candidate_embedding @ query_projected),
                float(candidate_embedding @ query_direct),
                float(projected_label_scores.mean()),
                float(projected_label_scores.max()),
                float(projected_label_scores.min()),
                float(direct_label_scores.mean()),
                float(direct_label_scores.max()),
                float(direct_label_scores.min()),
                min(len(candidate), 10) / 10.0,
            ], dtype=np.float32)
            rows.append(np.concatenate([
                candidate_embedding,
                query_projected * candidate_embedding,
                np.abs(query_projected - candidate_embedding),
                query_direct * candidate_embedding,
                np.abs(query_direct - candidate_embedding),
                scalars,
            ]).astype(np.float32))
        return np.asarray(rows, dtype=np.float32).reshape(-1, self.feature_dim)


def append_matsci_features(
    base_features: np.ndarray,
    candidates: Sequence[SetKey],
    builder: MatSciFeatureBuilder | None,
    query_direct: np.ndarray | None,
    query_projected: np.ndarray | None,
) -> np.ndarray:
    if builder is None:
        return base_features
    if query_direct is None or query_projected is None:
        raise ValueError("MatSciBERT query features are required when the builder is enabled")
    extra = builder.candidate_features(candidates, query_direct, query_projected)
    return np.concatenate([base_features, extra], axis=1)


def parse_named_source(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"expert must be NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    return name.strip(), path.strip()


def source_rank_features(rank: int | None) -> List[float]:
    if rank is None:
        return [0.0, 0.0, 0.0]
    return [1.0 / math.log2(int(rank) + 2.0), float(rank <= 10), float(rank <= 50)]


def query_route_features(target_cations: set[str], target_anions: set[str]) -> np.ndarray:
    """Compact observable family identity used to learn family-specific source trust."""
    output = np.zeros(18 + 18 + 3, dtype=np.float32)
    for offset, symbols in ((0, target_cations), (18, target_anions)):
        for symbol in symbols:
            try:
                group = Element(symbol).group
            except ValueError:
                continue
            if group is not None and 1 <= int(group) <= 18:
                output[offset + int(group) - 1] = 1.0
    output[36] = min(len(target_cations), 10) / 10.0
    output[37] = min(len(target_anions), 10) / 10.0
    output[38] = min(len(target_cations | target_anions), 10) / 10.0
    return output


def build_row_candidates_and_features(
    expert_rows: Sequence[Sequence[SetKey]],
    target_cations: set[str],
    target_anions: set[str],
    label_elements: Sequence[set[str]],
    label_groups: Sequence[set[int]],
    label_metals: Sequence[set[str]],
    train_seen: np.ndarray,
    expected_length: int,
    union_limit: int,
    prior_builder: CandidatePriorBuilder | None = None,
    template_prior_builder: TemplatePriorBuilder | None = None,
    family: str = "UNK",
    source_agnostic: bool = False,
    base_aware: bool = False,
) -> tuple[List[SetKey], np.ndarray]:
    rank_maps = [
        {candidate: rank for rank, candidate in enumerate(row, start=1)}
        for row in expert_rows
    ]
    universe = set().union(*(set(values) for values in expert_rows))
    ordered = sorted(
        universe,
        key=lambda candidate: (
            min((mapping.get(candidate, 10**9) for mapping in rank_maps)),
            -sum(candidate in mapping for mapping in rank_maps),
            candidate,
        ),
    )[: int(union_limit)]
    features: List[np.ndarray] = []
    route_features = query_route_features(target_cations, target_anions)
    for candidate in ordered:
        row_features: List[float] = []
        ranks: List[int] = []
        for mapping in rank_maps:
            rank = mapping.get(candidate)
            if not source_agnostic:
                row_features.extend(source_rank_features(rank))
            if rank is not None:
                ranks.append(int(rank))
        if base_aware:
            row_features.extend(source_rank_features(rank_maps[0].get(candidate)))
            alternative_ranks = [
                int(mapping[candidate]) for mapping in rank_maps[1:] if candidate in mapping
            ]
            alternative_count = max(1, len(rank_maps) - 1)
            row_features.extend(
                [
                    float(len(alternative_ranks)) / alternative_count,
                    float(sum(rank <= 10 for rank in alternative_ranks)) / alternative_count,
                    float(sum(rank <= 50 for rank in alternative_ranks)) / alternative_count,
                    1.0 / math.log2(min(alternative_ranks) + 2.0)
                    if alternative_ranks
                    else 0.0,
                    float(np.mean([1.0 / math.log2(rank + 2.0) for rank in alternative_ranks]))
                    if alternative_ranks
                    else 0.0,
                ]
            )
        row_features.extend([
            float(len(ranks)) / max(1, len(rank_maps)),
            float(sum(rank <= 10 for rank in ranks)) / max(1, len(rank_maps)),
            float(sum(rank <= 50 for rank in ranks)) / max(1, len(rank_maps)),
            1.0 / math.log2(min(ranks) + 2.0) if ranks else 0.0,
            float(np.mean([1.0 / math.log2(rank + 2.0) for rank in ranks])) if ranks else 0.0,
            min(len(candidate), 10) / 10.0,
        ])
        chemistry = chemistry_features_for_candidate(
            candidate,
            target_cations,
            target_anions,
            label_elements,
            label_groups,
            label_metals,
            train_seen,
            expected_length,
        )
        parts = [np.asarray(row_features, dtype=np.float32), chemistry, route_features]
        if prior_builder is not None:
            parts.append(prior_builder.features(candidate, str(family)))
        if template_prior_builder is not None:
            parts.append(
                template_prior_builder.features(candidate, str(family), target_anions)
            )
        features.append(np.concatenate(parts))
    feature_dim = (
        (0 if source_agnostic else len(expert_rows) * 3)
        + (8 if base_aware else 0)
        + 12
        + len(route_features)
        + (prior_builder.feature_dim if prior_builder is not None else 0)
        + (template_prior_builder.feature_dim if template_prior_builder is not None else 0)
    )
    return ordered, np.asarray(features, dtype=np.float32).reshape(-1, feature_dim)


def matrix_for_rows(
    row_features: Sequence[np.ndarray],
    row_labels: Sequence[np.ndarray],
    indices: Sequence[int],
    require_positive: bool,
) -> tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    features: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    groups: List[int] = []
    kept_rows: List[int] = []
    for row_index in indices:
        current_labels = row_labels[int(row_index)]
        if require_positive and not bool(np.any(current_labels > 0)):
            continue
        current_features = row_features[int(row_index)]
        if not len(current_features):
            continue
        features.append(current_features)
        labels.append(current_labels)
        groups.append(int(len(current_features)))
        kept_rows.append(int(row_index))
    if not features:
        feature_dim = int(row_features[0].shape[1]) if row_features else 0
        return np.zeros((0, feature_dim), dtype=np.float32), np.zeros(0, dtype=np.int8), [], []
    return np.vstack(features), np.concatenate(labels), groups, kept_rows


def formula_group_folds(
    group_values: Sequence[str],
    n_splits: int,
    seed: int,
) -> List[tuple[np.ndarray, np.ndarray]]:
    """Build deterministic, row-balanced folds without splitting a formula group."""
    values = np.asarray(group_values, dtype=object).astype(str)
    if int(n_splits) < 2:
        raise ValueError("n_splits must be at least 2")
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if len(unique) < int(n_splits):
        raise ValueError(f"only {len(unique)} formula groups are available for {n_splits} folds")
    rng = np.random.default_rng(int(seed))
    tie_break = rng.random(len(unique))
    order = np.lexsort((tie_break, -counts))
    fold_sizes = np.zeros(int(n_splits), dtype=np.int64)
    group_fold = np.full(len(unique), -1, dtype=np.int16)
    for group_index in order:
        fold = int(np.argmin(fold_sizes))
        group_fold[int(group_index)] = fold
        fold_sizes[fold] += int(counts[int(group_index)])
    all_rows = np.arange(len(values), dtype=np.int32)
    output: List[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(int(n_splits)):
        query_mask = group_fold[inverse] == fold
        output.append((all_rows[~query_mask], all_rows[query_mask]))
    return output


def query_balance_weights(
    row_indices: Sequence[int],
    group_values: Sequence[str],
    power: float,
) -> np.ndarray:
    """Return mean-one query weights that reduce repeated-formula domination."""
    if float(power) < 0:
        raise ValueError("group balance power cannot be negative")
    indices = np.asarray(row_indices, dtype=np.int64)
    if not len(indices) or float(power) == 0:
        return np.ones(len(indices), dtype=np.float32)
    values = np.asarray(group_values, dtype=object).astype(str)[indices]
    counts = Counter(values.tolist())
    weights = np.asarray([counts[value] ** (-float(power)) for value in values], dtype=np.float64)
    weights /= max(float(weights.mean()), 1e-12)
    return weights.astype(np.float32)


def candidate_sample_weights(
    kept_rows: Sequence[int],
    groups: Sequence[int],
    group_values: Sequence[str],
    power: float,
) -> np.ndarray:
    query_weights = query_balance_weights(kept_rows, group_values, power)
    return np.repeat(query_weights, np.asarray(groups, dtype=np.int64)).astype(np.float32)


def predict_ranking_scores(model: object, matrix: np.ndarray) -> np.ndarray:
    """Return a single ranking score for either ranker or binary classifier."""
    if hasattr(model, "predict_proba"):
        probability = np.asarray(model.predict_proba(matrix), dtype=np.float32)
        if probability.ndim == 2 and probability.shape[1] >= 2:
            return probability[:, 1]
    return np.asarray(model.predict(matrix), dtype=np.float32).reshape(-1)


def rank_query_rows(
    model: object,
    matrix: np.ndarray,
    groups: Sequence[int],
    row_indices: Sequence[int],
    candidates: Sequence[Sequence[SetKey]],
) -> Dict[int, List[SetKey]]:
    prediction = predict_ranking_scores(model, matrix)
    output: Dict[int, List[SetKey]] = {}
    offset = 0
    for row_index, size in zip(row_indices, groups):
        scores = prediction[offset : offset + size]
        order = np.argsort(-scores, kind="stable")
        output[int(row_index)] = [candidates[int(row_index)][int(index)] for index in order]
        offset += int(size)
    return output


def make_ranker(
    seed: int,
    estimators: int,
    learning_rate: float,
    num_leaves: int,
    min_child_samples: int,
) -> lgb.LGBMRanker:
    return lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[1, 3, 5, 10],
        n_estimators=int(estimators),
        learning_rate=float(learning_rate),
        num_leaves=int(num_leaves),
        min_child_samples=int(min_child_samples),
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        max_bin=127,
        random_state=int(seed),
        n_jobs=-1,
        verbosity=-1,
    )


def make_binary_ranker(
    seed: int,
    estimators: int,
    learning_rate: float,
    num_leaves: int,
    min_child_samples: int,
) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        n_estimators=int(estimators),
        learning_rate=float(learning_rate),
        num_leaves=int(num_leaves),
        min_child_samples=int(min_child_samples),
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        max_bin=127,
        random_state=int(seed),
        n_jobs=-1,
        verbosity=-1,
    )


def make_ranking_model(
    objective: str,
    seed: int,
    estimators: int,
    learning_rate: float,
    num_leaves: int,
    min_child_samples: int,
) -> object:
    factory = make_binary_ranker if objective == "binary" else make_ranker
    return factory(seed, estimators, learning_rate, num_leaves, min_child_samples)


def fit_ranking_model(
    model: object,
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: Sequence[int],
    sample_weight: np.ndarray,
    objective: str,
    positive_weight_power: float,
) -> object:
    """Fit LambdaRank or a query-balanced pointwise binary ranker."""
    weights = np.asarray(sample_weight, dtype=np.float32).copy()
    if objective == "binary":
        offset = 0
        for size in groups:
            size = int(size)
            if size > 1:
                local_positive = np.flatnonzero(labels[offset : offset + size] > 0)
                positive_indices = local_positive + offset
                weights[positive_indices] *= float(size - 1) ** float(positive_weight_power)
            offset += size
        model.fit(matrix, labels, sample_weight=weights)
    else:
        model.fit(matrix, labels, group=groups, sample_weight=weights)
    return model


def exact_metrics(targets: Sequence[SetKey], rows: Sequence[Sequence[SetKey]]) -> Dict[str, float]:
    return {
        f"exact_hit@{k}": float(np.mean([target in set(row[:k]) for target, row in zip(targets, rows)]))
        for k in (1, 3, 5, 10, 20, 50, 100)
    }


def merge_preserved_base_prefix(
    base: Sequence[SetKey], ranked: Sequence[SetKey], prefix: int
) -> List[SetKey]:
    """Keep a trusted base prefix and fill the remainder from a learned ranking."""
    output: List[SetKey] = []
    seen: set[SetKey] = set()
    for candidate in [*list(base[: int(prefix)]), *list(ranked), *list(base)]:
        if candidate and candidate not in seen:
            seen.add(candidate)
            output.append(candidate)
    return output


def group_macro_metrics(
    targets: Sequence[SetKey],
    rows: Sequence[Sequence[SetKey]],
    group_values: Sequence[str],
) -> Dict[str, float]:
    values = np.asarray(group_values, dtype=object).astype(str)
    output: Dict[str, float] = {}
    for k in (1, 3, 5, 10, 20, 50, 100):
        hits = np.asarray(
            [target in set(row[:k]) for target, row in zip(targets, rows)], dtype=np.float32
        )
        group_means = [float(hits[values == group].mean()) for group in np.unique(values)]
        output[f"group_macro_exact_hit@{k}"] = float(np.mean(group_means))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Five-fold OOF candidate-level LambdaRank stacker for frozen Stage2 experts."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--expert", action="append", default=[], help="Repeat NAME=candidates.jsonl")
    parser.add_argument(
        "--aux_expert",
        action="append",
        default=[],
        help=(
            "Optional NAME=train-OOF-candidates.jsonl sources. These train rows are "
            "added to every primary OOF fold and require --source_agnostic."
        ),
    )
    parser.add_argument("--aux_source_limit", type=int, default=100)
    parser.add_argument("--aux_union_limit", type=int, default=600)
    parser.add_argument("--aux_weight", type=float, default=0.25)
    parser.add_argument(
        "--aux_as_score_feature",
        action="store_true",
        help=(
            "Train a source-agnostic ranker only on auxiliary train-OOF rows and append "
            "its leakage-safe score/rank as features to the source-aware primary stacker."
        ),
    )
    parser.add_argument(
        "--families", default="",
        help="Optional comma-separated families used to train a specialist stacker.",
    )
    parser.add_argument("--source_limit", type=int, default=100)
    parser.add_argument("--union_limit", type=int, default=600)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n_estimators", type=int, default=800)
    parser.add_argument("--learning_rate", type=float, default=0.025)
    parser.add_argument("--num_leaves", type=int, default=63)
    parser.add_argument("--min_child_samples", type=int, default=50)
    parser.add_argument(
        "--ranking_objective",
        choices=("lambdarank", "binary"),
        default="lambdarank",
        help="Candidate ranking loss. Binary uses query-balanced positive weighting.",
    )
    parser.add_argument(
        "--binary_positive_weight_power",
        type=float,
        default=1.0,
        help="Positive multiplier is (query_size - 1) raised to this power.",
    )
    parser.add_argument(
        "--matsci_embeddings", default="",
        help="Optional cache produced by build_stage2_matscibert_embeddings.py.",
    )
    parser.add_argument("--matsci_components", type=int, default=32)
    parser.add_argument("--matsci_ridge_alpha", type=float, default=10.0)
    parser.add_argument(
        "--matsci_multilabel_scores", default="",
        help="Optional train-only fine-tuned MatSciBERT label-logit cache.",
    )
    parser.add_argument(
        "--fold_strategy", choices=("formula_group", "row"), default="formula_group"
    )
    parser.add_argument("--group_balance_power", type=float, default=0.0)
    parser.add_argument(
        "--candidate_priors",
        action="store_true",
        help="Add train-only global/family exact-set and label-frequency prior features.",
    )
    parser.add_argument(
        "--template_priors",
        action="store_true",
        help="Add train-only periodic-group route-template frequency features.",
    )
    parser.add_argument(
        "--source_agnostic",
        action="store_true",
        help="Use only source-count-invariant aggregate rank features, allowing richer expert unions at inference.",
    )
    parser.add_argument(
        "--base_aware",
        action="store_true",
        help=(
            "With --source_agnostic, retain explicit rank features for the first "
            "expert, which is treated as the stable base ranking."
        ),
    )
    parser.add_argument(
        "--preserve_base_boundary",
        action="store_true",
        help=(
            "Choose a leading base prefix on train OOF predictions, then rerank only "
            "the remaining Top-10 boundary slots."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", required=True)
    parser.add_argument("--output_model", required=True)
    args = parser.parse_args()
    if not args.expert:
        parser.error("at least one --expert source is required")
    if float(args.binary_positive_weight_power) < 0:
        parser.error("--binary_positive_weight_power must be non-negative")
    if args.aux_expert and not (args.source_agnostic or args.aux_as_score_feature):
        parser.error("--aux_expert requires --source_agnostic because source counts may differ")
    if args.aux_as_score_feature and not args.aux_expert:
        parser.error("--aux_as_score_feature requires --aux_expert sources")
    if args.aux_as_score_feature and args.source_agnostic:
        parser.error("--aux_as_score_feature is intended for a source-aware primary stacker")
    if args.base_aware and not args.source_agnostic:
        parser.error("--base_aware requires --source_agnostic")

    input_dir = Path(args.input_dir).resolve()
    pack = np.load(input_dir / f"{args.split}.npz", allow_pickle=True)
    y = np.asarray(pack["y_multi_hot"], dtype=np.float32)
    targets = [tuple(np.flatnonzero(row > 0.5).tolist()) for row in y]
    train_y = np.asarray(np.load(input_dir / "train.npz", allow_pickle=True)["y_multi_hot"], dtype=np.float32)
    meta = pd.read_csv(input_dir / f"{args.split}_meta.csv", low_memory=False)
    train_meta = pd.read_csv(input_dir / "train_meta.csv", low_memory=False)
    expert_paths = dict(parse_named_source(value) for value in args.expert)
    expert_names = list(expert_paths)
    experts = [
        load_source(path, len(targets), int(args.source_limit))
        for path in expert_paths.values()
    ]
    names = [
        str(value)
        for value in json.loads((input_dir / "precursor_names.json").read_text(encoding="utf-8"))
    ]
    matsci_builder: MatSciFeatureBuilder | None = None
    matsci_direct: np.ndarray | None = None
    matsci_projected: np.ndarray | None = None
    matsci_train_direct: np.ndarray | None = None
    matsci_train_projected: np.ndarray | None = None
    if str(args.matsci_embeddings).strip():
        label_views, train_query_views, query_views = load_matsci_views(
            Path(args.matsci_embeddings).resolve(), input_dir, str(args.split), names
        )
        matsci_builder = MatSciFeatureBuilder(
            label_views,
            train_query_views,
            train_y,
            int(args.matsci_components),
            float(args.matsci_ridge_alpha),
            int(args.seed),
        )
        matsci_train_direct, matsci_train_projected = matsci_builder.transform_queries(
            train_query_views
        )
        matsci_direct, matsci_projected = matsci_builder.transform_queries(query_views)
    matsci_multilabel_scores: np.ndarray | None = None
    matsci_train_multilabel_scores: np.ndarray | None = None
    if str(args.matsci_multilabel_scores).strip():
        matsci_multilabel_scores = load_matsci_multilabel_scores(
            Path(args.matsci_multilabel_scores).resolve(), input_dir, str(args.split), names
        )
        matsci_train_multilabel_scores = load_matsci_multilabel_scores(
            Path(args.matsci_multilabel_scores).resolve(), input_dir, "train", names
        )
    label_elements, label_groups, label_metals = label_chemistry(names)
    train_seen = np.asarray(train_y.sum(axis=0) > 0)
    prior_builder = CandidatePriorBuilder(train_y, train_meta) if args.candidate_priors else None
    template_prior_builder = (
        TemplatePriorBuilder(train_y, train_meta, names) if args.template_priors else None
    )
    length_modes = family_length_modes(train_meta, train_y)
    families = meta["family_signature_primary"].fillna("UNK").astype(str).to_numpy()
    formula_groups = meta["family_group_key"].fillna("UNK").astype(str).to_numpy()
    selected_families = {
        value.strip() for value in str(args.families).split(",") if value.strip()
    }
    active_indices = np.asarray([
        index for index, family in enumerate(families)
        if not selected_families or str(family) in selected_families
    ], dtype=np.int32)
    if not len(active_indices):
        raise RuntimeError("family restriction removed every stacker row")
    cations = [json_set(value) for value in meta["target_cation_elements"]]
    anions = [json_set(value) for value in meta["target_anion_elements"]]

    candidate_rows: List[List[SetKey]] = []
    feature_rows: List[np.ndarray] = []
    transfer_feature_rows: List[np.ndarray] = []
    label_rows: List[np.ndarray] = []
    for row_index, target in enumerate(targets):
        if selected_families and str(families[row_index]) not in selected_families:
            feature_dim = (0 if args.source_agnostic else len(experts) * 3) + (
                8 if args.base_aware else 0
            ) + 12 + 39 + (
                prior_builder.feature_dim if prior_builder is not None else 0
            ) + (
                template_prior_builder.feature_dim
                if template_prior_builder is not None
                else 0
            ) + (
                matsci_builder.feature_dim if matsci_builder is not None else 0
            ) + (MULTILABEL_FEATURE_DIM if matsci_multilabel_scores is not None else 0)
            candidate_rows.append(list(experts[0][row_index]))
            feature_rows.append(np.zeros((len(experts[0][row_index]), feature_dim), dtype=np.float32))
            label_rows.append(np.asarray([
                candidate == target for candidate in experts[0][row_index]
            ], dtype=np.int8))
            continue
        row_candidates, row_features = build_row_candidates_and_features(
            [expert[row_index] for expert in experts],
            cations[row_index], anions[row_index],
            label_elements, label_groups, label_metals, train_seen,
            int(length_modes.get(str(families[row_index]), length_modes["__GLOBAL__"])),
            int(args.union_limit),
            prior_builder,
            template_prior_builder,
            str(families[row_index]),
            bool(args.source_agnostic),
            bool(args.base_aware),
        )
        row_features = append_matsci_features(
            row_features,
            row_candidates,
            matsci_builder,
            None if matsci_direct is None else matsci_direct[row_index],
            None if matsci_projected is None else matsci_projected[row_index],
        )
        if matsci_multilabel_scores is not None:
            row_features = np.concatenate([
                row_features,
                multilabel_candidate_features(
                    row_candidates, matsci_multilabel_scores[row_index]
                ),
            ], axis=1)
        if args.aux_as_score_feature:
            transfer_candidates, transfer_features = build_row_candidates_and_features(
                [expert[row_index] for expert in experts],
                cations[row_index],
                anions[row_index],
                label_elements,
                label_groups,
                label_metals,
                train_seen,
                int(length_modes.get(str(families[row_index]), length_modes["__GLOBAL__"])),
                int(args.union_limit),
                prior_builder,
                template_prior_builder,
                str(families[row_index]),
                True,
                bool(args.base_aware),
            )
            if transfer_candidates != row_candidates:
                raise RuntimeError("source-aware and transfer candidate ordering diverged")
            transfer_features = append_matsci_features(
                transfer_features,
                transfer_candidates,
                matsci_builder,
                None if matsci_direct is None else matsci_direct[row_index],
                None if matsci_projected is None else matsci_projected[row_index],
            )
            if matsci_multilabel_scores is not None:
                transfer_features = np.concatenate(
                    [
                        transfer_features,
                        multilabel_candidate_features(
                            transfer_candidates, matsci_multilabel_scores[row_index]
                        ),
                    ],
                    axis=1,
                )
            transfer_feature_rows.append(transfer_features)
        candidate_rows.append(row_candidates)
        feature_rows.append(row_features)
        label_rows.append(np.asarray([candidate == target for candidate in row_candidates], dtype=np.int8))

    aux_feature_rows: List[np.ndarray] = []
    aux_label_rows: List[np.ndarray] = []
    aux_formula_groups = np.asarray([], dtype=object)
    aux_matrix = np.zeros((0, feature_rows[0].shape[1]), dtype=np.float32)
    aux_labels = np.zeros(0, dtype=np.int8)
    aux_groups: List[int] = []
    aux_kept: List[int] = []
    if args.aux_expert:
        aux_paths = dict(parse_named_source(value) for value in args.aux_expert)
        aux_pack = np.load(input_dir / "train.npz", allow_pickle=True)
        aux_y = np.asarray(aux_pack["y_multi_hot"], dtype=np.float32)
        aux_targets = [tuple(np.flatnonzero(row > 0.5).tolist()) for row in aux_y]
        aux_experts = [
            load_source(path, len(aux_targets), int(args.aux_source_limit))
            for path in aux_paths.values()
        ]
        aux_meta = train_meta
        aux_families = aux_meta["family_signature_primary"].fillna("UNK").astype(str).to_numpy()
        aux_formula_groups = aux_meta["family_group_key"].fillna("UNK").astype(str).to_numpy()
        aux_cations = [json_set(value) for value in aux_meta["target_cation_elements"]]
        aux_anions = [json_set(value) for value in aux_meta["target_anion_elements"]]
        for row_index, target in enumerate(aux_targets):
            row_candidates, row_features = build_row_candidates_and_features(
                [expert[row_index] for expert in aux_experts],
                aux_cations[row_index],
                aux_anions[row_index],
                label_elements,
                label_groups,
                label_metals,
                train_seen,
                int(length_modes.get(str(aux_families[row_index]), length_modes["__GLOBAL__"])),
                int(args.aux_union_limit),
                prior_builder,
                template_prior_builder,
                str(aux_families[row_index]),
                True,
                bool(args.base_aware),
            )
            row_features = append_matsci_features(
                row_features,
                row_candidates,
                matsci_builder,
                None if matsci_train_direct is None else matsci_train_direct[row_index],
                None if matsci_train_projected is None else matsci_train_projected[row_index],
            )
            if matsci_train_multilabel_scores is not None:
                row_features = np.concatenate(
                    [
                        row_features,
                        multilabel_candidate_features(
                            row_candidates, matsci_train_multilabel_scores[row_index]
                        ),
                    ],
                    axis=1,
                )
            aux_feature_rows.append(row_features)
            aux_label_rows.append(
                np.asarray([candidate == target for candidate in row_candidates], dtype=np.int8)
            )
        aux_matrix, aux_labels, aux_groups, aux_kept = matrix_for_rows(
            aux_feature_rows,
            aux_label_rows,
            np.arange(len(aux_targets), dtype=np.int32),
            require_positive=True,
        )
        expected_aux_dim = (
            transfer_feature_rows[0].shape[1]
            if args.aux_as_score_feature
            else feature_rows[0].shape[1]
        )
        if len(aux_matrix) and aux_matrix.shape[1] != expected_aux_dim:
            raise ValueError(
                f"auxiliary feature dimension {aux_matrix.shape[1]} does not match "
                f"expected dimension {expected_aux_dim}"
            )

    transfer_model: object | None = None
    if args.aux_as_score_feature:
        if not len(aux_matrix):
            raise RuntimeError("no positive auxiliary rows are available for transfer scoring")
        transfer_weights = candidate_sample_weights(
            aux_kept,
            aux_groups,
            aux_formula_groups,
            float(args.group_balance_power),
        )
        transfer_model = make_ranking_model(
            str(args.ranking_objective),
            int(args.seed) + 700001,
            int(args.n_estimators),
            float(args.learning_rate),
            int(args.num_leaves),
            int(args.min_child_samples),
        )
        fit_ranking_model(
            transfer_model,
            aux_matrix,
            aux_labels,
            aux_groups,
            transfer_weights,
            str(args.ranking_objective),
            float(args.binary_positive_weight_power),
        )
        transfer_sizes = [len(values) for values in transfer_feature_rows]
        transfer_matrix = np.vstack(transfer_feature_rows)
        all_transfer_scores = predict_ranking_scores(transfer_model, transfer_matrix)
        transfer_offset = 0
        for row_index, size in enumerate(transfer_sizes):
            raw_score = all_transfer_scores[transfer_offset : transfer_offset + size]
            transfer_offset += int(size)
            normalized_score = (raw_score - raw_score.mean()) / max(float(raw_score.std()), 1e-6)
            order = np.argsort(-raw_score, kind="stable")
            ranks = np.empty(len(raw_score), dtype=np.int32)
            ranks[order] = np.arange(1, len(raw_score) + 1, dtype=np.int32)
            transfer_columns = np.column_stack(
                [
                    normalized_score,
                    1.0 / np.log2(ranks.astype(np.float32) + 1.0),
                    ranks <= 10,
                    ranks <= 50,
                ]
            ).astype(np.float32)
            feature_rows[row_index] = np.concatenate(
                [feature_rows[row_index], transfer_columns], axis=1
            )

    if args.fold_strategy == "formula_group":
        local_splits = formula_group_folds(
            formula_groups[active_indices], int(args.folds), int(args.seed)
        )
        splits = [
            (active_indices[train_rows], active_indices[query_rows])
            for train_rows, query_rows in local_splits
        ]
    else:
        splitter = KFold(n_splits=int(args.folds), shuffle=True, random_state=int(args.seed))
        splits = [
            (active_indices[train_rows], active_indices[query_rows])
            for train_rows, query_rows in splitter.split(active_indices)
        ]
    oof_rows: List[List[SetKey]] = [[] for _ in targets]
    fold_reports = []
    for fold, (train_indices, query_indices) in enumerate(splits):
        train_matrix, train_labels, train_groups, train_kept = matrix_for_rows(
            feature_rows, label_rows, train_indices, require_positive=True
        )
        query_matrix, _, query_groups, query_kept = matrix_for_rows(
            feature_rows, label_rows, query_indices, require_positive=False
        )
        primary_train_weights = candidate_sample_weights(
            train_kept, train_groups, formula_groups, float(args.group_balance_power)
        )
        if len(aux_matrix) and not args.aux_as_score_feature:
            current_train_matrix = np.vstack([train_matrix, aux_matrix])
            current_train_labels = np.concatenate([train_labels, aux_labels])
            current_train_groups = [*train_groups, *aux_groups]
            auxiliary_weights = candidate_sample_weights(
                aux_kept,
                aux_groups,
                aux_formula_groups,
                float(args.group_balance_power),
            ) * float(args.aux_weight)
            current_train_weights = np.concatenate(
                [primary_train_weights, auxiliary_weights]
            )
        else:
            current_train_matrix = train_matrix
            current_train_labels = train_labels
            current_train_groups = train_groups
            current_train_weights = primary_train_weights
        model = make_ranking_model(
            str(args.ranking_objective),
            int(args.seed) + fold * 1009,
            int(args.n_estimators),
            float(args.learning_rate),
            int(args.num_leaves),
            int(args.min_child_samples),
        )
        fit_ranking_model(
            model,
            current_train_matrix,
            current_train_labels,
            current_train_groups,
            current_train_weights,
            str(args.ranking_objective),
            float(args.binary_positive_weight_power),
        )
        ranked = rank_query_rows(model, query_matrix, query_groups, query_kept, candidate_rows)
        for row_index in query_indices:
            oof_rows[int(row_index)] = ranked.get(int(row_index), candidate_rows[int(row_index)])
        fold_targets = [targets[int(row)] for row in query_indices]
        fold_ranked = [oof_rows[int(row)] for row in query_indices]
        fold_reports.append({
            "fold": int(fold),
            "n_train_rows": int(len(train_kept)),
            "n_query_rows": int(len(query_indices)),
            **exact_metrics(fold_targets, fold_ranked),
            **group_macro_metrics(
                fold_targets, fold_ranked, formula_groups[np.asarray(query_indices, dtype=np.int64)]
            ),
        })

    raw_oof_rows = [list(row) for row in oof_rows]
    prefix_trials = []
    selected_base_prefix = 0
    if args.preserve_base_boundary:
        best_prefix_key = None
        selected_targets_for_prefix = [targets[int(row)] for row in active_indices]
        for prefix in range(0, 11):
            trial_rows = [
                merge_preserved_base_prefix(
                    experts[0][int(row)], raw_oof_rows[int(row)], prefix
                )
                for row in active_indices
            ]
            trial_metrics = exact_metrics(selected_targets_for_prefix, trial_rows)
            prefix_trials.append({"base_prefix": int(prefix), **trial_metrics})
            key = (
                trial_metrics["exact_hit@10"],
                trial_metrics["exact_hit@5"],
                trial_metrics["exact_hit@1"],
                -int(prefix),
            )
            if best_prefix_key is None or key > best_prefix_key:
                best_prefix_key = key
                selected_base_prefix = int(prefix)
        for row in active_indices:
            oof_rows[int(row)] = merge_preserved_base_prefix(
                experts[0][int(row)], raw_oof_rows[int(row)], selected_base_prefix
            )

    all_indices = active_indices
    full_matrix, full_labels, full_groups, full_kept = matrix_for_rows(
        feature_rows, label_rows, all_indices, require_positive=True
    )
    primary_full_weights = candidate_sample_weights(
        full_kept, full_groups, formula_groups, float(args.group_balance_power)
    )
    if len(aux_matrix) and not args.aux_as_score_feature:
        fit_matrix = np.vstack([full_matrix, aux_matrix])
        fit_labels = np.concatenate([full_labels, aux_labels])
        fit_groups = [*full_groups, *aux_groups]
        auxiliary_full_weights = candidate_sample_weights(
            aux_kept,
            aux_groups,
            aux_formula_groups,
            float(args.group_balance_power),
        ) * float(args.aux_weight)
        fit_weights = np.concatenate([primary_full_weights, auxiliary_full_weights])
    else:
        fit_matrix = full_matrix
        fit_labels = full_labels
        fit_groups = full_groups
        fit_weights = primary_full_weights
    full_model = make_ranking_model(
        str(args.ranking_objective),
        int(args.seed) + 100000,
        int(args.n_estimators),
        float(args.learning_rate),
        int(args.num_leaves),
        int(args.min_child_samples),
    )
    fit_ranking_model(
        full_model,
        fit_matrix,
        fit_labels,
        fit_groups,
        fit_weights,
        str(args.ranking_objective),
        float(args.binary_positive_weight_power),
    )
    oracle = float(np.mean([bool(np.any(label_rows[int(row)] > 0)) for row in active_indices]))
    selected_targets = [targets[int(row)] for row in active_indices]
    selected_ranked = [oof_rows[int(row)] for row in active_indices]
    report = {
        "protocol": (
            f"{args.split}_formula_group_disjoint_oof_candidate_"
            f"{args.ranking_objective}_stacker"
        ),
        "config": vars(args),
        "expert_paths": expert_paths,
        "feature_dim": int(feature_rows[0].shape[1]),
        "matscibert": {
            "enabled": matsci_builder is not None,
            "components": int(args.matsci_components) if matsci_builder is not None else 0,
            "feature_dim": matsci_builder.feature_dim if matsci_builder is not None else 0,
            "projection_fit": "train split only",
            "multilabel_scores_enabled": matsci_multilabel_scores is not None,
            "multilabel_feature_dim": (
                MULTILABEL_FEATURE_DIM if matsci_multilabel_scores is not None else 0
            ),
        },
        "candidate_priors_enabled": prior_builder is not None,
        "template_priors_enabled": template_prior_builder is not None,
        "auxiliary_transfer_score_enabled": transfer_model is not None,
        "source_agnostic": bool(args.source_agnostic),
        "base_aware": bool(args.base_aware),
        "boundary_rescue": {
            "enabled": bool(args.preserve_base_boundary),
            "selected_base_prefix": int(selected_base_prefix),
            "prefix_trials": prefix_trials,
            "raw_oof": exact_metrics(
                [targets[int(row)] for row in active_indices],
                [raw_oof_rows[int(row)] for row in active_indices],
            ),
        },
        "auxiliary": {
            "enabled": bool(args.aux_expert),
            "expert_paths": (
                dict(parse_named_source(value) for value in args.aux_expert)
                if args.aux_expert
                else {}
            ),
            "rows_with_positive": int(len(aux_kept)),
            "candidate_rows": int(len(aux_feature_rows)),
            "weight": float(args.aux_weight),
        },
        "mean_union_candidates": float(np.mean([len(row) for row in candidate_rows])),
        "oracle_union_recall": oracle,
        "selected_rows": int(len(active_indices)),
        "oof": {
            **exact_metrics(selected_targets, selected_ranked),
            **group_macro_metrics(
                selected_targets, selected_ranked, formula_groups[active_indices]
            ),
        },
        "folds": fold_reports,
        "full_train_rows_with_positive": int(len(full_kept)),
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with Path(args.output_candidates_jsonl).resolve().open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(oof_rows):
            handle.write(json.dumps({
                "row_index": row_index,
                "candidate_label_ids": [list(candidate) for candidate in row],
            }) + "\n")
    model_output = Path(args.output_model).resolve()
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": full_model,
        "expert_names": expert_names,
        "source_limit": int(args.source_limit),
        "union_limit": int(args.union_limit),
        "feature_dim": int(feature_rows[0].shape[1]),
        "families": sorted(selected_families),
        "matsci_builder": matsci_builder,
        "matsci_multilabel_enabled": matsci_multilabel_scores is not None,
        "candidate_prior_builder": prior_builder,
        "template_prior_builder": template_prior_builder,
        "auxiliary_transfer_model": transfer_model,
        "source_agnostic": bool(args.source_agnostic),
        "base_aware": bool(args.base_aware),
        "preserve_base_prefix": (
            int(selected_base_prefix) if args.preserve_base_boundary else None
        ),
    }, model_output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
