#!/usr/bin/env python3
"""Deploy the frozen Synthmind family-routed models on a directory of structures.

This adapter intentionally writes outside the immutable release.  It uses:

* the three frozen Stage2 factorized precursor experts;
* chemistry-aware reciprocal-rank fusion for online candidate selection;
* the frozen Stage3 NF + CVAE + Diffusion ensemble with equal sample counts.

The validation-only s9161 miss gate is not relabelled as an online model.  Its
training interface depends on a historical base-candidate stack and
validation-only MatSciBERT PCA features that were not shipped as a supported
new-structure adapter.  Every output row records this limitation explicitly.

The final ``synthesizability_rank_score`` is a transparent ranking proxy, not a
calibrated probability of experimental success.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from chgnet.model.model import CHGNet
from pymatgen.core import Composition, Structure


DEFAULT_SOURCE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = os.environ.get("SYNTHMIND_DATA_ROOT")
STAGE2_REL = Path(
    "06_TRAIN_READY_DATA/04_STAGE2_CANONICAL/"
    "stage2_full_cation_family_canonical_v1"
)
STAGE3_REL = Path(
    "06_TRAIN_READY_DATA/08_STAGE3_FAMILY_FULL/"
    "stage3_full_cation_family_v1"
)
STAGE2_EXPERT_RELS = (
    Path(
        "07_BEST_MODELS/02_STAGE2_FACTORIZED_EXPERTS/"
        "stage2_factorized_h2048_b6_top20_g1_s9140"
    ),
    Path(
        "07_BEST_MODELS/02_STAGE2_FACTORIZED_EXPERTS/"
        "stage2_factorized_h1536_b4_top20_g2_s9151"
    ),
    Path(
        "07_BEST_MODELS/02_STAGE2_FACTORIZED_EXPERTS/"
        "stage2_factorized_h2048_b6_top20_g1_s9152"
    ),
)
STAGE3_MODEL_RELS = {
    "nf": Path(
        "07_BEST_MODELS/04_STAGE3_NF/"
        "stage3_conditional_flow_h1024_s8060/best_model.pt"
    ),
    "cvae": Path(
        "07_BEST_MODELS/05_STAGE3_CVAE/"
        "stage3_hybrid_cvae_h1024_s8040/best_model.pt"
    ),
    "diffusion": Path(
        "07_BEST_MODELS/06_STAGE3_DIFFUSION/"
        "stage3_conditional_diffusion_h1536_s8320/best_model.pt"
    ),
}
IGNORED_REAGENT_ELEMENTS = {
    "H",
    "O",
    "C",
    "N",
    "F",
    "Cl",
    "Br",
    "I",
    "S",
    "P",
    "Se",
    "Te",
}


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_builtin(value), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def compact_json(value: Any) -> str:
    return json.dumps(to_builtin(value), ensure_ascii=False, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_legacy_feature_module(source_root: Path) -> Any:
    path = (
        source_root
        / "scripts/07_infer/best_current_route_predictor/"
        "predict_genome_selected_best_current.py"
    )
    spec = importlib.util.spec_from_file_location(
        "synthmind_frozen_feature_reference", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import feature reference: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_family_symbols(source_root: Path) -> tuple[Any, Any, Any]:
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from synthmind.chemistry.families import (  # type: ignore
        assign_cation_family,
        family_feature_names,
        family_feature_vector,
    )

    return assign_cation_family, family_feature_names, family_feature_vector


def structure_feature_vector(
    structure: Structure,
    feature_cols: Sequence[str],
    legacy: Any,
    max_geom_sites: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, float]]:
    element_vocab = [
        name.replace("feat_frac_el__", "")
        for name in feature_cols
        if name.startswith("feat_frac_el__")
    ]
    composition_features, formula, elements = legacy.composition_feature_dict(
        structure, element_vocab
    )
    a, b, c, alpha, beta, gamma = legacy.lattice_lengths_angles(structure)
    crystal = legacy.approx_crystal_system(a, b, c, alpha, beta, gamma)
    values: dict[str, float] = {name: 0.0 for name in feature_cols}
    values.update(composition_features)
    for system in (
        "triclinic",
        "monoclinic",
        "orthorhombic",
        "tetragonal",
        "trigonal",
        "hexagonal",
        "cubic",
    ):
        values[f"feat_crystal_system__{system}"] = (
            1.0 if crystal == system else 0.0
        )
    values.update(
        {
            "feat_density": float(structure.density),
            "feat_volume": float(structure.volume),
            "feat_nsites": float(len(structure)),
            "feat_nelements": float(len(elements)),
            "feat_band_gap": 0.0,
            "feat_energy_above_hull": 0.0,
            "feat_spacegroup_number": 0.0,
            "feat_lattice_a": a,
            "feat_lattice_b": b,
            "feat_lattice_c": c,
            "feat_lattice_alpha": alpha,
            "feat_lattice_beta": beta,
            "feat_lattice_gamma": gamma,
            "feat_has_summary": 0.0,
        }
    )
    values.update(legacy.geom_features(structure, max_sites=max_geom_sites))
    vector = np.asarray([values.get(name, 0.0) for name in feature_cols], dtype=np.float32)
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    metadata = {
        "formula": formula,
        "target_elements": elements,
        "n_sites": int(len(structure)),
        "crystal_system_approx": crystal,
    }
    return vector, metadata, values


def apply_family_features(
    vector: np.ndarray,
    feature_cols: Sequence[str],
    formula: str,
    assign_cation_family: Any,
    family_feature_names: Any,
    family_feature_vector: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    assignment = assign_cation_family(formula)
    names = list(family_feature_names())
    family_values = np.asarray(family_feature_vector(assignment), dtype=np.float32)
    output = np.asarray(vector, dtype=np.float32).copy()
    index = {name: position for position, name in enumerate(feature_cols)}
    for name, value in zip(names, family_values):
        if name in index:
            output[index[name]] = float(value)
    metadata = assignment.to_dict()
    return output, metadata


def graph_embeddings(
    model: CHGNet,
    structures: Sequence[Structure],
    device: torch.device,
    batch_size: int,
    max_sites: int,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    result = np.zeros((len(structures), 64), dtype=np.float32)
    status = ["not_attempted"] * len(structures)
    energy = np.full(len(structures), np.nan, dtype=np.float32)
    eligible = [i for i, structure in enumerate(structures) if len(structure) <= max_sites]
    for i in range(len(structures)):
        if i not in eligible:
            status[i] = "zero_fallback_too_many_sites"
    if not eligible:
        return result, status, energy

    def consume(indices: Sequence[int]) -> None:
        selected = [structures[i] for i in indices]
        predictions = model.predict_structure(
            selected,
            task="e",
            return_crystal_feas=True,
            batch_size=min(batch_size, len(selected)),
        )
        if isinstance(predictions, dict):
            predictions = [predictions]
        for index, prediction in zip(indices, predictions):
            embedding = prediction["crystal_fea"]
            if hasattr(embedding, "detach"):
                embedding = embedding.detach().float().cpu().numpy()
            embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
            if embedding.shape[0] != 64:
                raise ValueError(
                    f"CHGNet embedding dimension is {embedding.shape[0]}, expected 64"
                )
            result[index] = embedding
            raw_energy = prediction.get("e")
            if raw_energy is not None:
                if hasattr(raw_energy, "detach"):
                    raw_energy = raw_energy.detach().float().cpu().numpy()
                energy[index] = float(np.asarray(raw_energy).reshape(-1)[0])
            status[index] = "ok"

    try:
        with torch.inference_mode():
            consume(eligible)
    except Exception:
        for index in eligible:
            try:
                with torch.inference_mode():
                    consume([index])
            except Exception as exc:
                status[index] = "zero_fallback_error:" + type(exc).__name__
    if device.type == "cuda":
        torch.cuda.synchronize()
    return result, status, energy


def precompute_local_combinations(
    top_labels: int, max_length: int
) -> tuple[np.ndarray, np.ndarray]:
    padded: list[list[int]] = []
    lengths: list[int] = []
    for length in range(1, max_length + 1):
        for combination in itertools.combinations(range(top_labels), length):
            padded.append(list(combination) + [-1] * (max_length - length))
            lengths.append(length)
    return np.asarray(padded, dtype=np.int16), np.asarray(lengths, dtype=np.int8)


def decode_factorized_batch(
    label_logits: np.ndarray,
    length_logits: np.ndarray,
    top_labels: int,
    candidate_limit: int,
    length_weight: float,
    max_length: int,
    local_combinations: np.ndarray,
    combination_lengths: np.ndarray,
) -> tuple[list[list[tuple[int, ...]]], list[list[float]]]:
    top_ids = np.argsort(-label_logits, axis=1)[:, :top_labels]
    top_values = np.take_along_axis(label_logits, top_ids, axis=1)
    length_log_probs = length_logits - np.logaddexp.reduce(
        length_logits, axis=1, keepdims=True
    )
    score_parts: list[np.ndarray] = []
    offsets: list[tuple[int, int, int]] = []
    start = 0
    for length in range(1, max_length + 1):
        mask = combination_lengths == length
        local = local_combinations[mask, :length]
        scores = top_values[:, local].sum(axis=2)
        scores += float(length_weight) * length_log_probs[:, length - 1 : length]
        score_parts.append(scores.astype(np.float32))
        offsets.append((start, start + scores.shape[1], length))
        start += scores.shape[1]
    all_scores = np.concatenate(score_parts, axis=1)
    keep = min(int(candidate_limit), all_scores.shape[1])
    if keep < all_scores.shape[1]:
        selected = np.argpartition(-all_scores, keep - 1, axis=1)[:, :keep]
    else:
        selected = np.broadcast_to(
            np.arange(all_scores.shape[1], dtype=np.int64), all_scores.shape
        ).copy()
    selected_scores = np.take_along_axis(all_scores, selected, axis=1)
    order = np.argsort(-selected_scores, axis=1)
    selected = np.take_along_axis(selected, order, axis=1)
    selected_scores = np.take_along_axis(selected_scores, order, axis=1)

    candidate_rows: list[list[tuple[int, ...]]] = []
    score_rows: list[list[float]] = []
    for row in range(len(label_logits)):
        candidates: list[tuple[int, ...]] = []
        scores: list[float] = []
        for global_index, score in zip(selected[row], selected_scores[row]):
            length = 0
            relative = 0
            for low, high, current_length in offsets:
                if low <= int(global_index) < high:
                    length = current_length
                    relative = int(global_index) - low
                    break
            local = local_combinations[
                combination_lengths == length, :length
            ][relative]
            ids = tuple(sorted(int(top_ids[row, position]) for position in local))
            candidates.append(ids)
            scores.append(float(score))
        candidate_rows.append(candidates)
        score_rows.append(scores)
    return candidate_rows, score_rows


def precursor_elements(names: Sequence[str]) -> set[str]:
    elements: set[str] = set()
    for name in names:
        try:
            composition = Composition(str(name).replace("·", "."))
            elements.update(str(element) for element in composition.elements)
        except Exception:
            continue
    return elements


def chemistry_features(
    names: Sequence[str],
    target_cations: Sequence[str],
    target_required_elements: Sequence[str],
    stage3_vocab: set[str],
) -> dict[str, Any]:
    target = set(str(item) for item in target_cations)
    required = set(str(item) for item in target_required_elements)
    present = precursor_elements(names)
    covered = target & present
    covered_required = required & present
    ignored = IGNORED_REAGENT_ELEMENTS - target - required
    extra = sorted(present - target - required - ignored)
    coverage = len(covered) / max(1, len(target))
    required_coverage = len(covered_required) / max(1, len(required))
    mapped = sum(1 for name in names if name in stage3_vocab)
    return {
        "target_cation_coverage": float(coverage),
        "covered_target_cations": sorted(covered),
        "target_required_element_coverage": float(required_coverage),
        "covered_target_required_elements": sorted(covered_required),
        "extra_cation_like_elements": extra,
        "extra_cation_count": len(extra),
        "stage3_mapped_count": int(mapped),
        "stage3_mapping_fraction": float(mapped / max(1, len(names))),
    }


def complete_candidate_with_elemental_sources(
    candidate: Mapping[str, Any],
    target_cations: Sequence[str],
    target_required_elements: Sequence[str],
    stage3_vocab: set[str],
) -> dict[str, Any]:
    output = dict(candidate)
    names = [str(value) for value in candidate.get("precursors", [])]
    present = precursor_elements(names)
    missing = sorted(set(target_required_elements) - present)
    if not missing:
        output["external_completion_elements"] = []
        return output
    completed_names = [*names, *missing]
    chemistry = chemistry_features(
        completed_names,
        target_cations,
        target_required_elements,
        stage3_vocab,
    )
    output.update(chemistry)
    output["precursors"] = completed_names
    output["external_completion_elements"] = missing
    output["candidate_source"] = (
        str(output.get("candidate_source", "unknown"))
        + "_plus_external_elemental_completion"
    )
    output["fusion_score"] = float(output.get("fusion_score", 0.0)) - 0.03 * len(
        missing
    )
    return output


def build_precursor_prior(
    stage2_dir: Path,
    precursor_names: Sequence[str],
) -> dict[str, Any]:
    with np.load(stage2_dir / "train.npz", allow_pickle=True) as pack:
        frequency = np.asarray(pack["y_multi_hot"], dtype=np.float32).sum(axis=0)
    elements_by_id: list[set[str]] = []
    by_element: dict[str, list[int]] = defaultdict(list)
    for label_id, name in enumerate(precursor_names):
        elements = precursor_elements([str(name)])
        elements_by_id.append(elements)
        for element in elements:
            by_element[element].append(label_id)
    for element, label_ids in by_element.items():
        label_ids.sort(
            key=lambda label_id: (
                -float(frequency[label_id]),
                len(elements_by_id[label_id]),
                str(precursor_names[label_id]),
            )
        )
    return {
        "frequency": frequency.astype(np.float32),
        "elements_by_id": elements_by_id,
        "by_element": dict(by_element),
    }


def composition_prior_candidates(
    target_cations: Sequence[str],
    precursor_names: Sequence[str],
    prior: Mapping[str, Any],
    limit: int,
) -> list[tuple[int, ...]]:
    target = set(str(value) for value in target_cations)
    if not target:
        return []
    frequency = np.asarray(prior["frequency"], dtype=np.float32)
    elements_by_id: Sequence[set[str]] = prior["elements_by_id"]
    by_element: Mapping[str, Sequence[int]] = prior["by_element"]
    ignored = IGNORED_REAGENT_ELEMENTS - target

    eligible_by_target: dict[str, list[int]] = {}
    for element in sorted(target):
        eligible: list[int] = []
        for label_id in by_element.get(element, []):
            present = elements_by_id[label_id]
            extra = present - target - ignored
            if extra:
                continue
            eligible.append(int(label_id))
            if len(eligible) >= 24:
                break
        eligible_by_target[element] = eligible

    scored: dict[tuple[int, ...], float] = {}

    # Direct single precursors that already cover the complete target-cation set.
    direct_pool = set(
        label_id
        for values in eligible_by_target.values()
        for label_id in values
    )
    for label_id in direct_pool:
        if target.issubset(elements_by_id[label_id]):
            candidate = (int(label_id),)
            scored[candidate] = 20.0 + math.log1p(float(frequency[label_id]))

    # Deterministic shifted-frequency constructions provide several alternatives
    # without a combinatorial product over all elements.
    ordered_target = sorted(
        target,
        key=lambda element: (
            len(eligible_by_target.get(element, [])),
            element,
        ),
    )
    for variant in range(max(24, int(limit))):
        selected: set[int] = set()
        covered: set[str] = set()
        for position, element in enumerate(ordered_target):
            if element in covered:
                continue
            choices = eligible_by_target.get(element, [])
            if not choices:
                continue
            choice_index = (variant + position * 3) % min(len(choices), 12)
            label_id = int(choices[choice_index])
            selected.add(label_id)
            covered.update(elements_by_id[label_id] & target)
        if not selected:
            continue
        candidate = tuple(sorted(selected))
        candidate_coverage = set().union(
            *(elements_by_id[label_id] & target for label_id in candidate)
        )
        score = (
            15.0 * len(candidate_coverage) / max(1, len(target))
            + float(np.mean([math.log1p(float(frequency[x])) for x in candidate]))
            - 0.15 * len(candidate)
        )
        scored[candidate] = max(scored.get(candidate, -math.inf), score)

    # Greedy set-cover variants favor labels that cover several target cations.
    union_pool = sorted(
        direct_pool,
        key=lambda label_id: (
            -len(elements_by_id[label_id] & target),
            -float(frequency[label_id]),
            str(precursor_names[label_id]),
        ),
    )[:256]
    for variant in range(12):
        uncovered = set(target)
        selected: list[int] = []
        while uncovered and len(selected) < 6:
            ranked = []
            for label_id in union_pool:
                new = elements_by_id[label_id] & uncovered
                if not new or label_id in selected:
                    continue
                score = (
                    10.0 * len(new)
                    + (0.35 + 0.03 * variant)
                    * math.log1p(float(frequency[label_id]))
                    - 0.10 * len(elements_by_id[label_id])
                )
                ranked.append((score, -label_id, label_id))
            if not ranked:
                break
            ranked.sort(reverse=True)
            chosen = int(ranked[variant % min(len(ranked), 5)][2])
            selected.append(chosen)
            uncovered -= elements_by_id[chosen]
        if selected:
            candidate = tuple(sorted(selected))
            coverage = 1.0 - len(uncovered) / max(1, len(target))
            score = (
                18.0 * coverage
                + float(np.mean([math.log1p(float(frequency[x])) for x in candidate]))
                - 0.15 * len(candidate)
            )
            scored[candidate] = max(scored.get(candidate, -math.inf), score)

    ranked_candidates = sorted(
        scored,
        key=lambda candidate: (
            -len(
                set().union(
                    *(elements_by_id[label_id] & target for label_id in candidate)
                )
            ),
            -float(scored[candidate]),
            len(candidate),
            candidate,
        ),
    )
    return ranked_candidates[: int(limit)]


def fuse_candidates(
    expert_candidates: Sequence[Sequence[tuple[int, ...]]],
    fallback_candidates: Sequence[tuple[int, ...]],
    precursor_names: Sequence[str],
    target_cations: Sequence[str],
    target_required_elements: Sequence[str],
    stage3_vocab: set[str],
    top_n: int,
) -> list[dict[str, Any]]:
    ranks: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for candidates in expert_candidates:
        for rank, candidate in enumerate(candidates, start=1):
            ranks[tuple(candidate)].append(rank)
    expert_count = len(expert_candidates)
    maximum_rrf = expert_count / 20.0
    fallback_rank = {
        tuple(candidate): rank
        for rank, candidate in enumerate(fallback_candidates, start=1)
    }
    all_candidates = set(ranks) | set(fallback_rank)
    rows: list[dict[str, Any]] = []
    for candidate in all_candidates:
        candidate_ranks = ranks.get(candidate, [])
        names = [str(precursor_names[index]) for index in candidate]
        chemistry = chemistry_features(
            names,
            target_cations,
            target_required_elements,
            stage3_vocab,
        )
        rrf = sum(1.0 / (19.0 + rank) for rank in candidate_ranks)
        rrf_norm = min(1.0, rrf / max(maximum_rrf, 1e-8))
        votes = len(candidate_ranks)
        prior_rank = fallback_rank.get(candidate)
        prior_score = (
            20.0 / (19.0 + float(prior_rank))
            if prior_rank is not None
            else 0.0
        )
        score = (
            0.20 * rrf_norm
            + 0.10 * (votes / max(1, expert_count))
            + 0.20 * chemistry["target_cation_coverage"]
            + 0.30 * chemistry["target_required_element_coverage"]
            + 0.10 * chemistry["stage3_mapping_fraction"]
            + 0.10 * prior_score
            - 0.12 * min(2.0, chemistry["extra_cation_count"])
        )
        if votes and prior_rank is not None:
            source = "model_and_composition_prior"
        elif votes:
            source = "factorized_model_ensemble"
        else:
            source = "composition_training_prior_fallback"
        rows.append(
            {
                "label_ids": list(candidate),
                "precursors": names,
                "candidate_source": source,
                "fusion_score": float(score),
                "rrf_score_normalized": float(rrf_norm),
                "expert_votes": int(votes),
                "expert_ranks": candidate_ranks,
                "composition_prior_rank": prior_rank,
                "composition_prior_score": float(prior_score),
                **chemistry,
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["target_required_element_coverage"]),
            -float(row["target_cation_coverage"]),
            int(row["extra_cation_count"]),
            -float(row["stage3_mapping_fraction"]),
            -float(row["fusion_score"]),
            -int(row["expert_votes"]),
            tuple(row["label_ids"]),
        )
    )
    return rows[:top_n]


def load_stage2_models(
    release_root: Path,
    source_root: Path,
    device: torch.device,
) -> tuple[list[Any], list[dict[str, Any]]]:
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from training.family.train_stage2_factorized_generator import (  # type: ignore
        FactorizedSetGenerator,
    )

    models = []
    metadata = []
    for relative in STAGE2_EXPERT_RELS:
        run_dir = release_root / relative
        checkpoint_path = run_dir / "best_factorized_generator.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        model = FactorizedSetGenerator(
            int(checkpoint["x_dim"]),
            int(checkpoint["n_labels"]),
            int(checkpoint["max_set_len"]),
            int(config["hidden"]),
            int(config["blocks"]),
            float(config["dropout"]),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device).eval()
        metrics = json_load(run_dir / "metrics.json")
        models.append(model)
        metadata.append(
            {
                "model_id": run_dir.name,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "top_labels": int(config["top_labels"]),
                "max_enumerated_length": int(config["max_enumerated_length"]),
                "length_score_weight": float(
                    metrics["validation"]["length_score_weight"]
                ),
            }
        )
    return models, metadata


@torch.inference_mode()
def stage2_expert_candidates(
    models: Sequence[Any],
    model_metadata: Sequence[Mapping[str, Any]],
    x_scaled: np.ndarray,
    device: torch.device,
    candidate_limit: int,
) -> list[list[list[tuple[int, ...]]]]:
    output: list[list[list[tuple[int, ...]]]] = []
    tensor = torch.from_numpy(x_scaled.astype(np.float32)).to(device)
    for model, metadata in zip(models, model_metadata):
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            label_logits, length_logits = model(tensor)
        label_np = label_logits.float().cpu().numpy()
        length_np = length_logits.float().cpu().numpy()
        top_labels = int(metadata["top_labels"])
        max_length = int(metadata["max_enumerated_length"])
        local, lengths = precompute_local_combinations(top_labels, max_length)
        candidates, _ = decode_factorized_batch(
            label_np,
            length_np,
            top_labels=top_labels,
            candidate_limit=candidate_limit,
            length_weight=float(metadata["length_score_weight"]),
            max_length=max_length,
            local_combinations=local,
            combination_lengths=lengths,
        )
        output.append(candidates)
    return output


def load_stage3_models(
    release_root: Path,
    source_root: Path,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from training.family.train_stage3_conditional_diffusion import (  # type: ignore
        ConditionalDiffusion,
        cosine_schedule,
        generate_samples as diffusion_generate,
    )
    from training.family.train_stage3_conditional_flow import (  # type: ignore
        ConditionalSynthesisFlow,
        generate_samples as flow_generate,
    )
    from training.family.train_stage3_hybrid_cvae import (  # type: ignore
        HybridCVAE,
        generate_samples as cvae_generate,
    )

    checkpoints = {
        name: torch.load(
            release_root / relative,
            map_location="cpu",
            weights_only=False,
        )
        for name, relative in STAGE3_MODEL_RELS.items()
    }

    flow_checkpoint = checkpoints["nf"]
    flow_config = flow_checkpoint["config"]
    flow = ConditionalSynthesisFlow(
        int(flow_checkpoint["structure_dim"]),
        int(flow_checkpoint["precursor_dim"]),
        int(flow_config["hidden"]),
        int(flow_config["precursor_hidden"]),
        int(flow_config["context_blocks"]),
        int(flow_config["flow_layers"]),
        int(flow_config["coupling_hidden"]),
        float(flow_config["dropout"]),
        int(flow_checkpoint["atmosphere_classes"]),
        int(flow_checkpoint["method_classes"]),
    )
    flow.load_state_dict(flow_checkpoint["state_dict"])

    cvae_checkpoint = checkpoints["cvae"]
    cvae_config = cvae_checkpoint["config"]
    cvae = HybridCVAE(
        int(cvae_checkpoint["structure_dim"]),
        int(cvae_checkpoint["precursor_dim"]),
        int(cvae_config["hidden"]),
        int(cvae_config["precursor_hidden"]),
        int(cvae_config["latent"]),
        int(cvae_config["blocks"]),
        float(cvae_config["dropout"]),
        int(cvae_checkpoint["atmosphere_classes"]),
        int(cvae_checkpoint["method_classes"]),
    )
    cvae.load_state_dict(cvae_checkpoint["state_dict"])

    diffusion_checkpoint = checkpoints["diffusion"]
    diffusion_config = diffusion_checkpoint["config"]
    diffusion = ConditionalDiffusion(
        int(diffusion_checkpoint["structure_dim"]),
        int(diffusion_checkpoint["precursor_dim"]),
        int(diffusion_config["hidden"]),
        int(diffusion_config["precursor_hidden"]),
        int(diffusion_config["blocks"]),
        float(diffusion_config["dropout"]),
        int(diffusion_config["time_dim"]),
        int(diffusion_checkpoint["atmosphere_classes"]),
        int(diffusion_checkpoint["method_classes"]),
    )
    diffusion.load_state_dict(diffusion_checkpoint["state_dict"])

    for model in (flow, cvae, diffusion):
        model.to(device).eval()

    models = {
        "nf": flow,
        "cvae": cvae,
        "diffusion": diffusion,
        "flow_generate": flow_generate,
        "cvae_generate": cvae_generate,
        "diffusion_generate": diffusion_generate,
        "diffusion_alpha_bar": cosine_schedule(
            int(diffusion_config["timesteps"])
        ).to(device),
    }
    metadata = {
        name: {
            "model_id": (release_root / STAGE3_MODEL_RELS[name]).parent.name,
            "checkpoint": str(release_root / STAGE3_MODEL_RELS[name]),
            "checkpoint_sha256": sha256_file(release_root / STAGE3_MODEL_RELS[name]),
            "config": checkpoints[name]["config"],
            "target_stats": checkpoints[name]["target_stats"],
        }
        for name in ("nf", "cvae", "diffusion")
    }
    metadata["_checkpoints"] = checkpoints
    return models, metadata


def categorical_distribution(
    values: np.ndarray, vocabulary: Sequence[str]
) -> tuple[str, float, dict[str, float]]:
    flattened = np.asarray(values, dtype=np.int64).reshape(-1)
    counts = Counter(int(value) for value in flattened)
    total = max(1, len(flattened))
    probabilities = {
        str(vocabulary[index]) if 0 <= index < len(vocabulary) else str(index):
        float(count / total)
        for index, count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
    }
    mode = next(iter(probabilities), "")
    confidence = next(iter(probabilities.values()), 0.0)
    return mode, float(confidence), probabilities


def summarize_stage3(
    continuous_by_model: Mapping[str, np.ndarray],
    discrete_by_model: Mapping[str, np.ndarray],
    atmosphere_vocab: Sequence[str],
    method_vocab: Sequence[str],
    target_stats: Mapping[str, Mapping[str, float]],
) -> list[dict[str, Any]]:
    model_names = list(continuous_by_model)
    all_continuous = np.concatenate(
        [continuous_by_model[name] for name in model_names], axis=1
    )
    all_discrete = np.concatenate(
        [discrete_by_model[name] for name in model_names], axis=1
    )
    rows: list[dict[str, Any]] = []
    temp_scale = max(float(target_stats["temperature_c"]["std"]), 1.0)
    log_time_scale = max(float(target_stats["log1p_time_h"]["std"]), 0.1)
    for row in range(all_continuous.shape[0]):
        temperature = all_continuous[row, :, 0]
        time_h = all_continuous[row, :, 1]
        atmosphere, atmosphere_conf, atmosphere_probs = categorical_distribution(
            all_discrete[row, :, 0], atmosphere_vocab
        )
        method, method_conf, method_probs = categorical_distribution(
            all_discrete[row, :, 1], method_vocab
        )
        model_temp_medians = [
            float(np.median(continuous_by_model[name][row, :, 0]))
            for name in model_names
        ]
        model_time_medians = [
            float(np.median(continuous_by_model[name][row, :, 1]))
            for name in model_names
        ]
        temp_spread = float(np.std(model_temp_medians) / temp_scale)
        log_time_medians = np.log1p(np.clip(model_time_medians, 0.0, None))
        time_spread = float(np.std(log_time_medians) / log_time_scale)
        continuous_consensus = math.exp(-0.5 * (temp_spread + time_spread))
        temperature_iqr = float(
            (np.quantile(temperature, 0.75) - np.quantile(temperature, 0.25))
            / temp_scale
        )
        log_time = np.log1p(np.clip(time_h, 0.0, None))
        log_time_iqr = float(
            (np.quantile(log_time, 0.75) - np.quantile(log_time, 0.25))
            / log_time_scale
        )
        distribution_compactness = math.exp(
            -0.35 * (temperature_iqr + log_time_iqr)
        )
        categorical_consensus = 0.5 * (atmosphere_conf + method_conf)
        ensemble_consensus = float(
            np.clip(
                0.40 * continuous_consensus
                + 0.30 * categorical_consensus
                + 0.30 * distribution_compactness,
                0.0,
                1.0,
            )
        )
        per_model = {
            name: {
                "temperature_c_median": float(
                    np.median(continuous_by_model[name][row, :, 0])
                ),
                "time_h_median": float(
                    np.median(continuous_by_model[name][row, :, 1])
                ),
                "atmosphere_mode": categorical_distribution(
                    discrete_by_model[name][row, :, 0], atmosphere_vocab
                )[0],
                "reaction_method_mode": categorical_distribution(
                    discrete_by_model[name][row, :, 1], method_vocab
                )[0],
            }
            for name in model_names
        }
        rows.append(
            {
                "pred_temperature_c_median": float(np.median(temperature)),
                "pred_temperature_c_p10": float(np.quantile(temperature, 0.10)),
                "pred_temperature_c_p25": float(np.quantile(temperature, 0.25)),
                "pred_temperature_c_p75": float(np.quantile(temperature, 0.75)),
                "pred_temperature_c_p90": float(np.quantile(temperature, 0.90)),
                "pred_time_h_median": float(np.median(time_h)),
                "pred_time_h_p10": float(np.quantile(time_h, 0.10)),
                "pred_time_h_p25": float(np.quantile(time_h, 0.25)),
                "pred_time_h_p75": float(np.quantile(time_h, 0.75)),
                "pred_time_h_p90": float(np.quantile(time_h, 0.90)),
                "pred_atmosphere": atmosphere,
                "pred_atmosphere_vote_fraction": atmosphere_conf,
                "pred_atmosphere_distribution": atmosphere_probs,
                "pred_reaction_method": method,
                "pred_reaction_method_vote_fraction": method_conf,
                "pred_reaction_method_distribution": method_probs,
                "stage3_continuous_consensus": float(continuous_consensus),
                "stage3_categorical_consensus": float(categorical_consensus),
                "stage3_distribution_compactness": float(
                    distribution_compactness
                ),
                "stage3_ensemble_consensus": ensemble_consensus,
                "stage3_per_model_summary": per_model,
            }
        )
    return rows


@torch.inference_mode()
def sample_stage3(
    models: Mapping[str, Any],
    metadata: Mapping[str, Any],
    features: np.ndarray,
    samples_per_model: int,
    device: torch.device,
    row_batch_size: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    tensor = torch.from_numpy(features.astype(np.float32))
    checkpoints = metadata["_checkpoints"]
    flow_checkpoint = checkpoints["nf"]
    cvae_checkpoint = checkpoints["cvae"]
    diffusion_checkpoint = checkpoints["diffusion"]

    flow_continuous, flow_discrete = models["flow_generate"](
        models["nf"],
        tensor,
        int(samples_per_model),
        int(row_batch_size),
        flow_checkpoint["target_stats"],
        device,
        int(seed) + 101,
        float(flow_checkpoint["config"]["base_scale"]),
        float(flow_checkpoint["config"]["categorical_temperature"]),
    )
    cvae_continuous, cvae_discrete = models["cvae_generate"](
        models["cvae"],
        tensor,
        int(samples_per_model),
        int(row_batch_size),
        int(cvae_checkpoint["config"]["latent"]),
        cvae_checkpoint["target_stats"],
        device,
        int(seed) + 202,
        1.0,
        1.0,
        1.0,
    )
    diffusion_continuous, diffusion_discrete = models["diffusion_generate"](
        models["diffusion"],
        tensor,
        int(samples_per_model),
        int(row_batch_size),
        models["diffusion_alpha_bar"],
        int(diffusion_checkpoint["config"]["sampling_steps"]),
        float(diffusion_checkpoint["config"]["ddim_eta"]),
        float(diffusion_checkpoint["config"]["categorical_temperature"]),
        diffusion_checkpoint["target_stats"],
        device,
        int(seed) + 303,
    )
    return (
        {
            "nf": flow_continuous,
            "cvae": cvae_continuous,
            "diffusion": diffusion_continuous,
        },
        {
            "nf": flow_discrete,
            "cvae": cvae_discrete,
            "diffusion": diffusion_discrete,
        },
    )


def domain_score(stage3_scaled: np.ndarray, base_feature_count: int) -> float:
    values = np.abs(stage3_scaled[:base_feature_count])
    values = values[np.isfinite(values)]
    if not len(values):
        return 0.0
    clipped_mean = float(np.mean(np.clip(values, 0.0, 8.0)))
    return float(np.clip(math.exp(-max(0.0, clipped_mean - 0.75) / 2.5), 0.0, 1.0))


def synth_score(
    candidate: Mapping[str, Any],
    condition: Mapping[str, Any],
    domain: float,
    n_sites: int,
    graph_status: str,
) -> tuple[float, dict[str, float], list[str]]:
    vote = float(candidate["expert_votes"]) / 3.0
    stage2_confidence = float(
        np.clip(
            0.25 * vote
            + 0.25 * float(candidate["rrf_score_normalized"])
            + 0.20 * float(candidate["target_cation_coverage"])
            + 0.20 * float(candidate["target_required_element_coverage"])
            + 0.10 * float(candidate["stage3_mapping_fraction"]),
            0.0,
            1.0,
        )
    )
    simplicity = float(1.0 / (1.0 + math.log1p(max(1, n_sites)) / 6.0))
    components = {
        "stage2_precursor_confidence": stage2_confidence,
        "precursor_cation_coverage": float(
            candidate["target_cation_coverage"]
        ),
        "precursor_required_element_coverage": float(
            candidate["target_required_element_coverage"]
        ),
        "stage3_ensemble_consensus": float(
            condition["stage3_ensemble_consensus"]
        ),
        "training_domain_proximity": float(domain),
        "stage3_vocab_mapping": float(candidate["stage3_mapping_fraction"]),
        "structure_simplicity": simplicity,
    }
    score = (
        0.35 * components["stage2_precursor_confidence"]
        + 0.10 * components["precursor_cation_coverage"]
        + 0.10 * components["precursor_required_element_coverage"]
        + 0.20 * components["stage3_ensemble_consensus"]
        + 0.15 * components["training_domain_proximity"]
        + 0.05 * components["stage3_vocab_mapping"]
        + 0.05 * components["structure_simplicity"]
    )
    flags: list[str] = []
    extra_count = int(candidate["extra_cation_count"])
    if extra_count:
        score -= min(0.20, 0.06 * extra_count)
        flags.append("extra_cation_like_elements")
    if float(candidate["target_cation_coverage"]) < 1.0:
        score -= 0.08
        flags.append("incomplete_target_cation_coverage")
    if float(candidate["target_required_element_coverage"]) < 1.0:
        score -= 0.18
        flags.append("incomplete_target_required_element_coverage")
    if float(candidate["stage3_mapping_fraction"]) < 1.0:
        score -= 0.08
        flags.append("stage3_precursor_oov")
    if graph_status != "ok":
        score -= 0.08
        flags.append(graph_status)
    if str(candidate.get("candidate_source", "")).startswith(
        "composition_training_prior_fallback"
    ):
        score -= 0.04
        flags.append("composition_training_prior_fallback")
    external_completion = candidate.get("external_completion_elements", [])
    if external_completion:
        score -= min(0.12, 0.04 * len(external_completion))
        flags.append("external_elemental_completion")
    if float(condition["pred_temperature_c_median"]) <= 0.0:
        score -= 0.10
        flags.append("nonpositive_temperature_median")
    if float(condition["pred_time_h_median"]) < 0.0:
        score -= 0.10
        flags.append("negative_time_median")
    return float(np.clip(score, 0.0, 1.0)), components, flags


def list_structure_files(input_dir: Path) -> list[Path]:
    extensions = {".cif", ".vasp", ".poscar"}
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and (
            path.suffix.lower() in extensions
            or path.name.upper() == "POSCAR"
        )
    )


def write_chunk(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False, compression="gzip")


def merge_chunks(chunks_dir: Path, output_path: Path) -> int:
    files = sorted(chunks_dir.glob("chunk_*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"No completed chunks in {chunks_dir}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    header_written = False
    with gzip.open(output_path, "wt", encoding="utf-8", newline="") as target:
        for path in files:
            with gzip.open(path, "rt", encoding="utf-8", newline="") as source:
                header = source.readline()
                if not header:
                    continue
                if not header_written:
                    target.write(header)
                    header_written = True
                for line in source:
                    target.write(line)
                    rows += 1
    return rows


def write_top100_markdown(frame: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Synthmind GNoME 最可能合成的 100 个候选",
        "",
        "> 排名依据是模型置信、前驱体元素覆盖、三模型条件一致性、训练域接近度和结构复杂度组成的代理分数；不是实验成功概率。",
        "",
        "| 排名 | ID | 化学式 | 代理分数 | 前驱体 | 方法 | 温度 °C（P25–P75） | 时间 h（P25–P75） | 气氛 | 复核标志 |",
        "|---:|---|---|---:|---|---|---|---|---|---|",
    ]
    for _, row in frame.iterrows():
        try:
            precursors = " + ".join(json.loads(row["predicted_precursors"]))
        except Exception:
            precursors = str(row["predicted_precursors"])
        flags = str(row.get("quality_flags", "[]"))
        lines.append(
            "| {rank} | {sample} | {formula} | {score:.4f} | {precursors} | "
            "{method} | {temp:.0f} ({temp_lo:.0f}–{temp_hi:.0f}) | "
            "{time:.2f} ({time_lo:.2f}–{time_hi:.2f}) | {atmosphere} | {flags} |".format(
                rank=int(row["synthesizability_rank"]),
                sample=str(row["sample_id"]),
                formula=str(row["formula"]),
                score=float(row["synthesizability_rank_score"]),
                precursors=precursors.replace("|", "/"),
                method=str(row["pred_reaction_method"]),
                temp=float(row["pred_temperature_c_median"]),
                temp_lo=float(row["pred_temperature_c_p25"]),
                temp_hi=float(row["pred_temperature_c_p75"]),
                time=float(row["pred_time_h_median"]),
                time_lo=float(row["pred_time_h_p25"]),
                time_hi=float(row["pred_time_h_p75"]),
                atmosphere=str(row["pred_atmosphere"]),
                flags=flags.replace("|", "/"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_top100(all_predictions: Path, output_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(all_predictions, low_memory=False)
    valid = frame.loc[frame["prediction_status"].eq("ok")].copy()
    valid["synthesizability_rank_score"] = pd.to_numeric(
        valid["synthesizability_rank_score"], errors="coerce"
    ).fillna(0.0)
    coverage = pd.to_numeric(
        valid["precursor_target_required_element_coverage"], errors="coerce"
    ).fillna(0.0)
    mapping = pd.to_numeric(
        valid["stage3_precursor_mapping_fraction"], errors="coerce"
    ).fillna(0.0)
    flags = valid["quality_flags"].fillna("[]").astype(str)
    eligible_mask = (
        np.isclose(coverage.to_numpy(dtype=float), 1.0)
        & np.isclose(mapping.to_numpy(dtype=float), 1.0)
        & valid["chgnet_feature_status"].eq("ok").to_numpy()
        & ~flags.str.contains("external_elemental_completion", regex=False).to_numpy()
        & ~flags.str.contains("stage3_precursor_oov", regex=False).to_numpy()
        & ~flags.str.contains("incomplete_target", regex=False).to_numpy()
    )
    eligible = valid.loc[eligible_mask].copy()
    if len(eligible) >= min(100, len(valid)):
        selection = eligible
        selection_policy = "strict_quality_eligible"
    else:
        selection = valid
        selection_policy = "all_valid_fallback_due_to_insufficient_strict_rows"
    selection = selection.sort_values(
        [
            "synthesizability_rank_score",
            "stage2_precursor_confidence",
            "stage3_ensemble_consensus",
            "sample_id",
        ],
        ascending=[False, False, False, True],
    ).head(100)
    selection.insert(
        0, "synthesizability_rank", np.arange(1, len(selection) + 1)
    )
    selection.insert(1, "top100_selection_policy", selection_policy)
    selection.insert(2, "strict_quality_eligible_pool_size", int(len(eligible)))
    selection.to_csv(
        output_dir / "top100_most_synthesizable.csv", index=False
    )
    write_top100_markdown(
        selection, output_dir / "top100_most_synthesizable.md"
    )
    return selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--data-root",
        "--release-root",
        dest="data_root",
        default=DEFAULT_DATA_ROOT,
        required=DEFAULT_DATA_ROOT is None,
        help=(
            "Authorized external artifact root. The legacy --release-root "
            "spelling remains accepted; alternatively set SYNTHMIND_DATA_ROOT."
        ),
    )
    parser.add_argument(
        "--source-root",
        default=str(DEFAULT_SOURCE_ROOT),
        help="Synthmind V1.0 Git checkout (defaults to this script's repository).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chgnet-batch-size", type=int, default=32)
    parser.add_argument("--stage3-row-batch-size", type=int, default=16)
    parser.add_argument("--stage3-samples-per-model", type=int, default=64)
    parser.add_argument("--stage2-candidates-per-expert", type=int, default=100)
    parser.add_argument("--top-precursor-sets", type=int, default=10)
    parser.add_argument("--max-geom-sites", type=int, default=300)
    parser.add_argument("--max-chgnet-sites", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    release_root = Path(args.data_root).expanduser().resolve()
    source_root = Path(args.source_root).expanduser().resolve()
    stage2_dir = release_root / STAGE2_REL
    stage3_dir = release_root / STAGE3_REL
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    chunks_dir = output_dir / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(int(args.seed))
    device = torch.device(
        args.device
        if args.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    files = list_structure_files(input_dir)
    if args.limit > 0:
        files = files[: int(args.limit)]
    if not files:
        raise FileNotFoundError(f"No CIF/VASP/POSCAR files in {input_dir}")

    legacy = load_legacy_feature_module(source_root)
    (
        assign_cation_family,
        family_feature_names,
        family_feature_vector,
    ) = load_family_symbols(source_root)

    stage2_feature_cols = [str(x) for x in json_load(stage2_dir / "feature_cols.json")]
    stage2_precursor_names = [
        str(x) for x in json_load(stage2_dir / "precursor_names.json")
    ]
    precursor_prior = build_precursor_prior(stage2_dir, stage2_precursor_names)
    stage2_scaler = json_load(stage2_dir / "scaler.json")
    stage2_mean = np.asarray(stage2_scaler["mean"], dtype=np.float32)
    stage2_std = np.asarray(stage2_scaler["std"], dtype=np.float32)
    stage2_std = np.where(stage2_std > 1e-8, stage2_std, 1.0)

    stage3_schema = json_load(stage3_dir / "schema.json")
    stage3_feature_cols = [str(x) for x in stage3_schema["feature_cols"]]
    stage3_base_count = int(stage3_schema["base_feature_count"])
    stage3_mean = np.asarray(stage3_schema["x_scaler"]["mean"], dtype=np.float32)
    stage3_std = np.asarray(stage3_schema["x_scaler"]["std"], dtype=np.float32)
    stage3_std = np.where(stage3_std > 1e-8, stage3_std, 1.0)
    stage3_precursor_vocab = [str(x) for x in stage3_schema["precursor_vocab"]]
    stage3_vocab_index = {
        value: index for index, value in enumerate(stage3_precursor_vocab)
    }
    stage3_vocab_set = set(stage3_precursor_vocab)
    atmosphere_vocab = [
        str(x)
        for x in stage3_schema["discrete_schema"]["atmosphere_coarse"]["vocab"]
    ]
    method_vocab = [
        str(x)
        for x in stage3_schema["discrete_schema"]["reaction_method"]["vocab"]
    ]

    stage2_models, stage2_metadata = load_stage2_models(
        release_root, source_root, device
    )
    stage3_models, stage3_metadata = load_stage3_models(
        release_root, source_root, device
    )
    print("[INFO] loading CHGNet", flush=True)
    chgnet = CHGNet.load().to(device).eval()

    model_manifest = {
        "release_root": str(release_root),
        "release_zip_sha256": (
            "d5a86bd985f5f462bade6e5661b84fc0b04ec8899f93e450e74df6cf18ca54e2"
        ),
        "stage2": {
            "online_policy": (
                "three_factorized_experts_chemistry_aware_rrf_plus_"
                "training_frequency_composition_fallback"
            ),
            "experts": stage2_metadata,
            "s9161_gate_status": (
                "not_applied_online: validation-only gate depends on an unsupported "
                "historical base-candidate/MatSciBERT-PCA deployment interface"
            ),
        },
        "stage3": {
            "online_policy": "equal_sample_nf_cvae_diffusion",
            "samples_per_model": int(args.stage3_samples_per_model),
            "models": {
                name: {
                    key: value
                    for key, value in stage3_metadata[name].items()
                    if key != "config"
                }
                for name in ("nf", "cvae", "diffusion")
            },
        },
        "feature_policy": {
            "stage2_feature_count": len(stage2_feature_cols),
            "stage3_feature_count": len(stage3_feature_cols),
            "chgnet_embedding_count": 64,
            "missing_external_summary_features": [
                "band_gap",
                "energy_above_hull",
                "spacegroup_number",
            ],
            "missing_external_summary_fill": 0.0,
        },
        "score_semantics": (
            "synthesizability_rank_score is an uncalibrated ranking proxy, "
            "not a probability of experimental synthesis success"
        ),
    }
    json_dump(output_dir / "model_manifest.json", model_manifest)

    processed = 0
    failed = 0
    for chunk_number, start in enumerate(
        range(0, len(files), int(args.batch_size))
    ):
        chunk_path = chunks_dir / f"chunk_{chunk_number:06d}.csv.gz"
        paths = files[start : start + int(args.batch_size)]
        if args.resume and chunk_path.exists():
            existing = pd.read_csv(chunk_path, usecols=["sample_id"])
            if len(existing) == len(paths):
                processed += len(paths)
                print(
                    f"[RESUME] chunk={chunk_number} rows={len(paths)} "
                    f"processed={processed}/{len(files)}",
                    flush=True,
                )
                continue

        structures: list[Structure] = []
        valid_paths: list[Path] = []
        valid_input_indices: list[int] = []
        parse_rows: list[dict[str, Any]] = []
        for offset, path in enumerate(paths):
            input_index = start + offset
            try:
                structure = Structure.from_file(str(path), primitive=False, merge_tol=0.0)
                structures.append(structure)
                valid_paths.append(path)
                valid_input_indices.append(input_index)
            except Exception as exc:
                failed += 1
                parse_rows.append(
                    {
                        "input_index": input_index,
                        "sample_id": path.stem,
                        "input_path": str(path),
                        "input_sha256": sha256_file(path),
                        "prediction_status": "parse_failed",
                        "parse_error": repr(exc),
                        "model_stage2_policy": "not_run",
                        "model_stage3_policy": "not_run",
                        "synthesizability_rank_score": 0.0,
                        "score_is_calibrated_probability": False,
                        "quality_flags": compact_json(["structure_parse_failed"]),
                    }
                )

        chunk_rows = list(parse_rows)
        if structures:
            graph_values, graph_status, chgnet_energy = graph_embeddings(
                chgnet,
                structures,
                device,
                batch_size=int(args.chgnet_batch_size),
                max_sites=int(args.max_chgnet_sites),
            )
            raw_stage2: list[np.ndarray] = []
            raw_stage3: list[np.ndarray] = []
            metadata_rows: list[dict[str, Any]] = []
            family_rows: list[dict[str, Any]] = []
            for structure, path in zip(structures, valid_paths):
                s2_vector, metadata, _ = structure_feature_vector(
                    structure,
                    stage2_feature_cols,
                    legacy,
                    max_geom_sites=int(args.max_geom_sites),
                )
                s2_vector, family = apply_family_features(
                    s2_vector,
                    stage2_feature_cols,
                    metadata["formula"],
                    assign_cation_family,
                    family_feature_names,
                    family_feature_vector,
                )
                s3_vector, _, _ = structure_feature_vector(
                    structure,
                    stage3_feature_cols,
                    legacy,
                    max_geom_sites=int(args.max_geom_sites),
                )
                s3_vector, _ = apply_family_features(
                    s3_vector,
                    stage3_feature_cols,
                    metadata["formula"],
                    assign_cation_family,
                    family_feature_names,
                    family_feature_vector,
                )
                raw_stage2.append(s2_vector)
                raw_stage3.append(s3_vector)
                metadata_rows.append(
                    {
                        **metadata,
                        "sample_id": path.stem,
                        "input_path": str(path),
                        "input_sha256": sha256_file(path),
                    }
                )
                family_rows.append(family)

            x_stage2 = np.vstack(raw_stage2).astype(np.float32)
            graph_indices = [
                index
                for index, name in enumerate(stage2_feature_cols)
                if name.startswith("chgnet_graph_emb_")
            ]
            if len(graph_indices) != 64:
                raise ValueError(
                    f"Expected 64 Stage2 CHGNet columns, got {len(graph_indices)}"
                )
            x_stage2[:, graph_indices] = graph_values
            x_stage2_scaled = (
                (x_stage2 - stage2_mean[None, :]) / stage2_std[None, :]
            ).astype(np.float32)

            expert_output = stage2_expert_candidates(
                stage2_models,
                stage2_metadata,
                x_stage2_scaled,
                device,
                candidate_limit=int(args.stage2_candidates_per_expert),
            )
            fused_rows: list[list[dict[str, Any]]] = []
            for row, (family, metadata) in enumerate(
                zip(family_rows, metadata_rows)
            ):
                target_cations = family.get("target_cation_elements", [])
                target_required_elements = [
                    element
                    for element in metadata["target_elements"]
                    if element not in {"H", "O"}
                ]
                fallback = composition_prior_candidates(
                    target_required_elements,
                    stage2_precursor_names,
                    precursor_prior,
                    limit=max(40, int(args.top_precursor_sets) * 4),
                )
                fused = fuse_candidates(
                    [expert_output[expert][row] for expert in range(len(expert_output))],
                    fallback,
                    stage2_precursor_names,
                    target_cations,
                    target_required_elements,
                    stage3_vocab_set,
                    top_n=int(args.top_precursor_sets),
                )
                if not fused:
                    fused = [
                        {
                            "label_ids": [],
                            "precursors": [],
                            "candidate_source": "no_candidate",
                            "fusion_score": 0.0,
                            "rrf_score_normalized": 0.0,
                            "expert_votes": 0,
                            "expert_ranks": [],
                            "composition_prior_rank": None,
                            "composition_prior_score": 0.0,
                            "target_cation_coverage": 0.0,
                            "covered_target_cations": [],
                            "target_required_element_coverage": 0.0,
                            "covered_target_required_elements": [],
                            "extra_cation_like_elements": [],
                            "extra_cation_count": 0,
                            "stage3_mapped_count": 0,
                            "stage3_mapping_fraction": 0.0,
                        }
                    ]
                fused = [
                    complete_candidate_with_elemental_sources(
                        candidate,
                        target_cations,
                        target_required_elements,
                        stage3_vocab_set,
                    )
                    for candidate in fused
                ]
                fused_rows.append(fused)

            x_stage3_raw = np.vstack(raw_stage3).astype(np.float32)
            x_stage3 = x_stage3_raw.copy()
            x_stage3[:, :stage3_base_count] = (
                x_stage3_raw[:, :stage3_base_count] - stage3_mean[None, :]
            ) / stage3_std[None, :]
            y_set = np.zeros(
                (len(structures), len(stage3_precursor_vocab)), dtype=np.float32
            )
            for row, fused in enumerate(fused_rows):
                for precursor in fused[0]["precursors"]:
                    index = stage3_vocab_index.get(str(precursor))
                    if index is not None:
                        y_set[row, index] = 1.0
            stage3_inputs = np.hstack([x_stage3, y_set]).astype(np.float32)

            continuous, discrete = sample_stage3(
                stage3_models,
                stage3_metadata,
                stage3_inputs,
                samples_per_model=int(args.stage3_samples_per_model),
                device=device,
                row_batch_size=int(args.stage3_row_batch_size),
                seed=int(args.seed) + chunk_number * 1009,
            )
            condition_rows = summarize_stage3(
                continuous,
                discrete,
                atmosphere_vocab,
                method_vocab,
                stage3_metadata["nf"]["target_stats"],
            )

            for row, (path, input_index, metadata, family, fused, condition) in enumerate(
                zip(
                    valid_paths,
                    valid_input_indices,
                    metadata_rows,
                    family_rows,
                    fused_rows,
                    condition_rows,
                )
            ):
                candidate = fused[0]
                proximity = domain_score(x_stage3[row], stage3_base_count)
                score, components, flags = synth_score(
                    candidate,
                    condition,
                    proximity,
                    int(metadata["n_sites"]),
                    graph_status[row],
                )
                chunk_rows.append(
                    {
                        "input_index": input_index,
                        "sample_id": metadata["sample_id"],
                        "input_path": metadata["input_path"],
                        "input_sha256": metadata["input_sha256"],
                        "formula": metadata["formula"],
                        "target_elements": compact_json(metadata["target_elements"]),
                        "target_cation_elements": compact_json(
                            family.get("target_cation_elements", [])
                        ),
                        "target_anion_elements": compact_json(
                            family.get("target_anion_elements", [])
                        ),
                        "family_signature_primary": family.get(
                            "family_signature_primary", ""
                        ),
                        "family_routing_level": family.get(
                            "family_routing_level", ""
                        ),
                        "n_sites": metadata["n_sites"],
                        "crystal_system_approx": metadata[
                            "crystal_system_approx"
                        ],
                        "chgnet_energy_eV_per_atom": float(chgnet_energy[row]),
                        "chgnet_feature_status": graph_status[row],
                        "predicted_precursors": compact_json(
                            candidate["precursors"]
                        ),
                        "precursor_candidate_source": candidate[
                            "candidate_source"
                        ],
                        "precursor_external_completion_elements": compact_json(
                            candidate.get("external_completion_elements", [])
                        ),
                        "top10_precursor_sets": compact_json(fused),
                        "precursor_expert_votes": candidate["expert_votes"],
                        "precursor_rrf_score_normalized": candidate[
                            "rrf_score_normalized"
                        ],
                        "precursor_target_cation_coverage": candidate[
                            "target_cation_coverage"
                        ],
                        "precursor_target_required_element_coverage": candidate[
                            "target_required_element_coverage"
                        ],
                        "precursor_extra_cation_like_elements": compact_json(
                            candidate["extra_cation_like_elements"]
                        ),
                        "stage3_precursor_mapping_fraction": candidate[
                            "stage3_mapping_fraction"
                        ],
                        "pred_reaction_method": condition[
                            "pred_reaction_method"
                        ],
                        "pred_reaction_method_vote_fraction": condition[
                            "pred_reaction_method_vote_fraction"
                        ],
                        "pred_reaction_method_distribution": compact_json(
                            condition["pred_reaction_method_distribution"]
                        ),
                        "pred_temperature_c_median": condition[
                            "pred_temperature_c_median"
                        ],
                        "pred_temperature_c_p10": condition[
                            "pred_temperature_c_p10"
                        ],
                        "pred_temperature_c_p25": condition[
                            "pred_temperature_c_p25"
                        ],
                        "pred_temperature_c_p75": condition[
                            "pred_temperature_c_p75"
                        ],
                        "pred_temperature_c_p90": condition[
                            "pred_temperature_c_p90"
                        ],
                        "pred_time_h_median": condition["pred_time_h_median"],
                        "pred_time_h_p10": condition["pred_time_h_p10"],
                        "pred_time_h_p25": condition["pred_time_h_p25"],
                        "pred_time_h_p75": condition["pred_time_h_p75"],
                        "pred_time_h_p90": condition["pred_time_h_p90"],
                        "pred_atmosphere": condition["pred_atmosphere"],
                        "pred_atmosphere_vote_fraction": condition[
                            "pred_atmosphere_vote_fraction"
                        ],
                        "pred_atmosphere_distribution": compact_json(
                            condition["pred_atmosphere_distribution"]
                        ),
                        "stage3_continuous_consensus": condition[
                            "stage3_continuous_consensus"
                        ],
                        "stage3_categorical_consensus": condition[
                            "stage3_categorical_consensus"
                        ],
                        "stage3_distribution_compactness": condition[
                            "stage3_distribution_compactness"
                        ],
                        "stage3_ensemble_consensus": condition[
                            "stage3_ensemble_consensus"
                        ],
                        "stage3_per_model_summary": compact_json(
                            condition["stage3_per_model_summary"]
                        ),
                        "stage2_precursor_confidence": components[
                            "stage2_precursor_confidence"
                        ],
                        "training_domain_proximity": components[
                            "training_domain_proximity"
                        ],
                        "structure_simplicity": components[
                            "structure_simplicity"
                        ],
                        "synthesizability_rank_score": score,
                        "score_components": compact_json(components),
                        "score_is_calibrated_probability": False,
                        "quality_flags": compact_json(flags),
                        "prediction_status": "ok",
                        "parse_error": "",
                        "model_stage2_policy": (
                            "frozen_factorized3_chemistry_aware_rrf_plus_"
                            "composition_training_prior"
                        ),
                        "model_stage2_gate_status": (
                            "s9161_validation_gate_not_applied_online"
                        ),
                        "model_stage3_policy": (
                            f"frozen_nf_cvae_diffusion_{args.stage3_samples_per_model}"
                            f"+{args.stage3_samples_per_model}"
                            f"+{args.stage3_samples_per_model}"
                        ),
                        "random_seed": int(args.seed) + chunk_number * 1009,
                    }
                )

        chunk_rows.sort(key=lambda row: int(row["input_index"]))
        write_chunk(chunk_path, chunk_rows)
        processed += len(paths)
        print(
            f"[PROGRESS] chunk={chunk_number} rows={len(paths)} "
            f"processed={processed}/{len(files)} parse_failed={failed}",
            flush=True,
        )

    all_predictions = output_dir / "all_predictions.csv.gz"
    merged_rows = merge_chunks(chunks_dir, all_predictions)
    top100 = build_top100(all_predictions, output_dir)
    completed = pd.read_csv(
        all_predictions,
        usecols=["prediction_status", "quality_flags"],
        low_memory=False,
    )
    status_counts = {
        str(key): int(value)
        for key, value in completed["prediction_status"].value_counts().items()
    }
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "input_structure_count": len(files),
        "merged_output_rows": int(merged_rows),
        "prediction_status_counts": status_counts,
        "top100_count": int(len(top100)),
        "stage2_policy": (
            "frozen_factorized3_chemistry_aware_rrf_plus_"
            "composition_training_prior"
        ),
        "stage2_s9161_gate_status": (
            "not applied online; see model_manifest.json"
        ),
        "stage3_policy": (
            "equal-sample frozen NF + CVAE + Diffusion ensemble"
        ),
        "stage3_samples_per_model": int(args.stage3_samples_per_model),
        "ranking_score_semantics": (
            "uncalibrated ranking proxy; not experimental success probability"
        ),
        "elapsed_seconds": time.time() - started,
        "outputs": {
            "all_predictions": str(all_predictions),
            "top100_csv": str(output_dir / "top100_most_synthesizable.csv"),
            "top100_markdown": str(output_dir / "top100_most_synthesizable.md"),
            "model_manifest": str(output_dir / "model_manifest.json"),
            "chunks": str(chunks_dir),
        },
    }
    json_dump(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
