#!/usr/bin/env python3
"""Fine-tune MatSciBERT as an exact precursor-set cross encoder.

This is the heavyweight counterpart to the frozen structured-energy model.
Each example is a pair of texts: the target material description and one exact
precursor-set candidate.  Formula-group-disjoint OOF candidates provide hard
same-family negatives, while validation remains the fixed held-out split and
the frozen test split is never opened.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from training.family.evaluate_stage2_candidate_fusion import load_source
from training.family.evaluate_stage2_score_ensemble import candidate_fingerprint
from training.family.evaluate_stage2_oof_chemistry_rescore import (
    family_length_modes,
    json_set,
    label_chemistry,
)
from training.family.evaluate_stage2_precursor_family_slate import precursor_family
from training.family.train_stage2_oof_candidate_stacker import (
    CandidatePriorBuilder,
    TemplatePriorBuilder,
)
from training.family.train_stage2_structured_energy_ranker import (
    CandidateRegistry,
    PoolData,
    QueryPoolDataset,
    build_pair_features,
    build_training_pools,
    build_validation_pairs,
    apply_fixed_trial,
    best_trials_by_strategy,
    candidate_label_tensor,
    evaluate_grid,
    load_matsci_pca_views,
    merge_candidate_sources,
    seed_everything,
    standardize_from_train,
    targets_from_matrix,
)
from training.family.train_stage2_within_family_variant_ranker import exact_metrics


SetKey = Tuple[int, ...]


def candidate_rank_feature_maps(
    rows: Sequence[Sequence[SetKey]], limit: int
) -> List[Dict[SetKey, np.ndarray]]:
    """Build scale-stable retrieval-rank features for every candidate row."""

    denominator = max(math.log1p(max(1, int(limit))), 1e-6)
    output: List[Dict[SetKey, np.ndarray]] = []
    for row in rows:
        mapping: Dict[SetKey, np.ndarray] = {}
        for rank, candidate in enumerate(dict.fromkeys(row), start=1):
            if rank > int(limit):
                break
            mapping[candidate] = np.asarray(
                [
                    1.0 / math.log2(float(rank) + 1.0),
                    math.log1p(float(rank)) / denominator,
                    float(rank <= 10),
                    float(rank <= 20),
                    float(rank <= 50),
                    float(rank <= 100),
                ],
                dtype=np.float32,
            )
        output.append(mapping)
    return output


def candidate_rank_features(
    maps: Sequence[Dict[SetKey, np.ndarray]], row_index: int, candidate: SetKey
) -> np.ndarray:
    return maps[int(row_index)].get(candidate, np.zeros(6, dtype=np.float32))


def query_texts(meta: pd.DataFrame) -> List[str]:
    rows = []
    for row in meta.itertuples(index=False):
        rows.append(
            "Target material: "
            f"{getattr(row, 'canonical_formula', '') or getattr(row, 'formula', '')}. "
            f"Cation elements: {getattr(row, 'target_cation_elements', '')}. "
            f"Anion elements: {getattr(row, 'target_anion_elements', '')}. "
            f"Periodic family: {getattr(row, 'family_signature_primary', 'UNK')}. "
            f"Synthesis class: {getattr(row, 'synthesis_type', 'unknown')}. "
            f"Dataset class: {getattr(row, 'source_dataset', 'unknown')}."
        )
    return rows


def candidate_text(candidate: SetKey, names: Sequence[str]) -> str:
    if not candidate:
        return "Precursor set: empty."
    formulas = " ; ".join(str(names[int(label)]) for label in candidate)
    return f"Exact precursor set ({len(candidate)} reagents): {formulas}."


def parse_family_filter(raw: str) -> set[str]:
    return {value.strip() for value in str(raw).split(",") if value.strip()}


def distribution_route_key(row: pd.Series, kind: str) -> str:
    """Training-only route bucket used to identify repeated plausible routes."""
    def text_value(column: str, fallback: str) -> str:
        value = row.get(column, fallback)
        return fallback if pd.isna(value) or not str(value).strip() else str(value)

    family = text_value("family_signature_primary", "UNK")
    parts = [] if "nofamily" in str(kind) else [family]
    if "anion" in str(kind):
        anions = json_set(row.get("target_anion_elements", ""))
        periodic_route = TemplatePriorBuilder.route_key(family, anions)
        parts.append(periodic_route.split("|", 1)[-1])
    if "synthesis" in str(kind):
        parts.append(text_value("synthesis_type", "unknown"))
    if "source" in str(kind):
        parts.append(text_value("source_dataset", "unknown"))
    return "|".join(parts) or "GLOBAL"


def build_multi_positive_supervision(
    targets: Sequence[SetKey],
    meta: pd.DataFrame,
    train_pool: PoolData,
    registry: CandidateRegistry,
    route_kind: str,
    minimum_count: int,
    maximum_candidates: int,
    label_elements: Sequence[set[str]] | None = None,
) -> tuple[np.ndarray, Dict[str, object]]:
    """Mark train-only repeated routes as additional positives in each query pool."""
    width = int(train_pool.candidate_ids.shape[1])
    positive = np.zeros((len(targets), width), dtype=bool)
    route_counts: Dict[str, Counter[SetKey]] = defaultdict(Counter)
    route_accessories: Dict[str, set[int]] = defaultdict(set)
    route_lengths: Dict[str, set[int]] = defaultdict(set)
    factorized = "factorized" in str(route_kind)
    if str(route_kind) != "none":
        for row_index, target in enumerate(targets):
            key = distribution_route_key(meta.iloc[int(row_index)], str(route_kind))
            route_counts[key][target] += 1
            if factorized and label_elements is not None:
                target_elements = json_set(
                    meta.iloc[int(row_index)].get("target_cation_elements", "")
                )
                route_lengths[key].add(len(target))
                route_accessories[key].update(
                    int(label)
                    for label in target
                    if not (set(label_elements[int(label)]) & target_elements)
                )

    extra_positive_rows = 0
    extra_positive_pairs = 0
    factorized_positive_pairs = 0
    for pool_index, row_index in enumerate(train_pool.query_indices.tolist()):
        row_index = int(row_index)
        valid = np.flatnonzero(train_pool.mask[int(pool_index)]).tolist()
        if not valid:
            continue
        # build_training_pools guarantees that slot zero is the exact row target.
        positive[row_index, 0] = True
        if str(route_kind) == "none":
            continue
        key = distribution_route_key(meta.iloc[row_index], str(route_kind))
        counter = route_counts.get(key, Counter())
        query_target_elements = json_set(meta.iloc[row_index].get("target_cation_elements", ""))
        allowed_accessories = route_accessories.get(key, set())
        allowed_lengths = route_lengths.get(key, set())
        eligible = []
        for slot in valid[1:]:
            candidate = registry.keys[int(train_pool.candidate_ids[int(pool_index), int(slot)])]
            count = int(counter.get(candidate, 0))
            is_factorized = False
            if (
                factorized
                and label_elements is not None
                and allowed_accessories
                and allowed_lengths
                and min(allowed_lengths) <= len(candidate) <= max(allowed_lengths)
            ):
                candidate_accessories = {
                    int(label)
                    for label in candidate
                    if not (set(label_elements[int(label)]) & query_target_elements)
                }
                target_bearing = len(candidate) - len(candidate_accessories)
                is_factorized = bool(target_bearing > 0 and candidate_accessories) and bool(
                    candidate_accessories <= allowed_accessories
                )
            if count >= int(minimum_count) or is_factorized:
                eligible.append((max(count, int(is_factorized)), -int(slot), int(slot), is_factorized and count < int(minimum_count)))
        eligible.sort(reverse=True)
        limit = max(0, int(maximum_candidates) - 1)
        for _, _, slot, synthetic in eligible[:limit]:
            positive[row_index, int(slot)] = True
            factorized_positive_pairs += int(synthetic)
        extras = int(positive[row_index].sum()) - 1
        extra_positive_rows += int(extras > 0)
        extra_positive_pairs += max(0, extras)
    return positive, {
        "route_kind": str(route_kind),
        "route_buckets": int(len(route_counts)),
        "minimum_count": int(minimum_count),
        "maximum_candidates": int(maximum_candidates),
        "rows_with_extra_positive": int(extra_positive_rows),
        "extra_positive_pairs": int(extra_positive_pairs),
        "factorized": bool(factorized),
        "factorized_positive_pairs": int(factorized_positive_pairs),
    }


def evaluate_specialist_grid(
    targets: Sequence[SetKey],
    base_rows: Sequence[Sequence[SetKey]],
    raw_scores: np.ndarray,
    spans: Sequence[Tuple[int, int]],
    label_families: Sequence[str],
    active_indices: np.ndarray,
    alphas: Sequence[float],
    protected_prefixes: Sequence[int],
    minimum_gains: Sequence[float],
    candidate_windows: Sequence[int] = (20, 50, 100),
) -> tuple[Dict[str, object], List[List[SetKey]], List[Dict[str, object]]]:
    """Tune a specialist on selected validation rows while preserving all others."""
    indices = np.asarray(active_indices, dtype=np.int64)
    if len(indices) == len(targets):
        return evaluate_grid(
            targets,
            base_rows,
            raw_scores,
            spans,
            label_families,
            alphas,
            protected_prefixes,
            minimum_gains,
            candidate_windows,
        )
    specialist_scores: List[np.ndarray] = []
    specialist_spans: List[Tuple[int, int]] = []
    offset = 0
    for row_index in indices:
        start, end = spans[int(row_index)]
        values = raw_scores[int(start) : int(end)]
        specialist_scores.append(values)
        specialist_spans.append((offset, offset + len(values)))
        offset += len(values)
    flat_scores = (
        np.concatenate(specialist_scores).astype(np.float32)
        if specialist_scores
        else np.zeros(0, dtype=np.float32)
    )
    specialist_targets = [targets[int(index)] for index in indices]
    specialist_base = [base_rows[int(index)] for index in indices]
    trial, specialist_rows, trials = evaluate_grid(
        specialist_targets,
        specialist_base,
        flat_scores,
        specialist_spans,
        label_families,
        alphas,
        protected_prefixes,
        minimum_gains,
        candidate_windows,
    )
    rows = [list(row) for row in base_rows]
    for row_index, row in zip(indices, specialist_rows):
        rows[int(row_index)] = list(row)
    full_metrics = exact_metrics(targets, rows)
    trial = {
        **trial,
        **full_metrics,
        "specialist_rows": int(len(indices)),
        "specialist_exact_hit@10": float(
            np.mean(
                [target in set(row[:10]) for target, row in zip(specialist_targets, specialist_rows)]
            )
        ),
    }
    return trial, rows, trials


class MatSciCrossEncoder(nn.Module):
    def __init__(
        self,
        model_path: str,
        pair_dim: int,
        query_dim: int,
        dropout: float,
        freeze_bottom_layers: int,
        pooling: str = "auto",
        gradient_checkpointing: bool = False,
        attention_implementation: str = "",
        freeze_embeddings: bool = False,
    ) -> None:
        super().__init__()
        model_kwargs = {"local_files_only": True}
        if str(attention_implementation):
            model_kwargs["attn_implementation"] = str(attention_implementation)
        self.encoder = AutoModel.from_pretrained(str(model_path), **model_kwargs)
        hidden = int(self.encoder.config.hidden_size)
        layers = getattr(getattr(self.encoder, "encoder", None), "layer", None)
        if layers is None:
            layers = getattr(self.encoder, "layers", None)
        if layers is None:
            layers = getattr(getattr(self.encoder, "model", None), "layers", None)
        if layers is not None:
            for layer in list(layers)[: int(freeze_bottom_layers)]:
                for parameter in layer.parameters():
                    parameter.requires_grad = False
        if bool(freeze_embeddings):
            embeddings = self.encoder.get_input_embeddings()
            if embeddings is not None:
                for parameter in embeddings.parameters():
                    parameter.requires_grad = False
        if bool(gradient_checkpointing):
            self.encoder.gradient_checkpointing_enable()
            if hasattr(self.encoder.config, "use_cache"):
                self.encoder.config.use_cache = False
        requested_pooling = str(pooling)
        if requested_pooling == "auto":
            decoder_like = bool(getattr(self.encoder.config, "is_decoder", False)) or str(
                getattr(self.encoder.config, "model_type", "")
            ).lower() in {"qwen2", "qwen3", "llama", "mistral", "gemma", "gemma2"}
            requested_pooling = "last" if decoder_like else "cls"
        self.pooling = requested_pooling
        self.pair_encoder = nn.Sequential(
            nn.Linear(int(pair_dim), hidden // 2),
            nn.GELU(),
            nn.LayerNorm(hidden // 2),
            nn.Dropout(float(dropout)),
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(int(query_dim), hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.LayerNorm(hidden // 2),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self,
        tokens: Dict[str, torch.Tensor],
        pair_features: torch.Tensor,
        query_features: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.encoder(**tokens).last_hidden_state
        attention = tokens.get("attention_mask")
        if self.pooling == "mean":
            if attention is None:
                encoded = hidden.mean(dim=1)
            else:
                weights = attention.to(hidden.dtype).unsqueeze(-1)
                encoded = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        elif self.pooling == "last":
            if attention is None:
                encoded = hidden[:, -1]
            else:
                positions = torch.arange(hidden.shape[1], device=hidden.device)[None, :]
                last = positions.masked_fill(attention == 0, -1).argmax(dim=1)
                encoded = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
        else:
            encoded = hidden[:, 0]
        pair = self.pair_encoder(pair_features)
        query = self.query_encoder(query_features)
        return self.head(torch.cat([encoded, pair, query], dim=-1)).squeeze(-1)


def tokenize_pairs(
    tokenizer,
    query_indices: torch.Tensor,
    candidate_ids: torch.Tensor,
    query_rows: Sequence[str],
    candidate_rows: Sequence[str],
    max_length: int,
    device: torch.device,
    pair_mode: str = "tokenizer_pair",
) -> Dict[str, torch.Tensor]:
    query_flat = query_indices.reshape(-1).tolist()
    candidate_flat = candidate_ids.reshape(-1).tolist()
    queries = [query_rows[int(index)] for index in query_flat]
    candidates = [candidate_rows[int(index)] for index in candidate_flat]
    if str(pair_mode) == "explicit":
        combined = [
            f"{query}\nCandidate synthesis route: {candidate}"
            for query, candidate in zip(queries, candidates)
        ]
        tokens = tokenizer(
            combined,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
    else:
        tokens = tokenizer(
            queries,
            candidates,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
    return {key: value.to(device, non_blocking=True) for key, value in tokens.items()}


def score_validation(
    model: MatSciCrossEncoder,
    tokenizer,
    query_indices: np.ndarray,
    candidate_ids: np.ndarray,
    pair_features: np.ndarray,
    query_features: np.ndarray,
    query_rows: Sequence[str],
    candidate_rows: Sequence[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
    pair_mode: str = "tokenizer_pair",
) -> np.ndarray:
    # Formula-group validation commonly contains many literature rows for the
    # same material.  When their query text/features and candidate features are
    # identical, the deterministic eval-mode transformer score is identical as
    # well.  Score each exact pair once and broadcast it to duplicate rows.
    # This changes only compute cost, never candidate order or supervision.
    unique_lookup: Dict[tuple[object, ...], int] = {}
    unique_positions: List[int] = []
    inverse = np.empty(len(candidate_ids), dtype=np.int64)
    query_signature_cache: Dict[int, tuple[str, bytes]] = {}
    for position, (query_index, candidate_id) in enumerate(
        zip(query_indices.tolist(), candidate_ids.tolist())
    ):
        query_index = int(query_index)
        query_signature = query_signature_cache.get(query_index)
        if query_signature is None:
            query_signature = (
                str(query_rows[query_index]),
                # Repeated rows of an identical canonical formula can differ
                # by ~1e-6 solely from floating composition normalization.
                # Quantize only the deduplication signature; model inputs stay
                # untouched for the representative pair.
                np.round(
                    np.asarray(query_features[query_index], dtype=np.float32), 4
                ).tobytes(),
            )
            query_signature_cache[query_index] = query_signature
        key = (
            query_signature[0],
            query_signature[1],
            int(candidate_id),
            np.asarray(pair_features[position], dtype=np.float32).tobytes(),
        )
        unique_index = unique_lookup.get(key)
        if unique_index is None:
            unique_index = len(unique_positions)
            unique_lookup[key] = unique_index
            unique_positions.append(int(position))
        inverse[int(position)] = int(unique_index)
    unique_positions_array = np.asarray(unique_positions, dtype=np.int64)
    unique_scores = np.zeros(len(unique_positions_array), dtype=np.float32)
    print(
        json.dumps(
            {
                "validation_pairs": int(len(candidate_ids)),
                "unique_validation_pairs": int(len(unique_positions_array)),
                "deduplication_ratio": float(
                    len(unique_positions_array) / max(1, len(candidate_ids))
                ),
            }
        ),
        flush=True,
    )
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(unique_positions_array), int(batch_size)):
            end = min(len(unique_positions_array), start + int(batch_size))
            positions = unique_positions_array[start:end]
            selected_queries = query_indices[positions]
            query_tensor = torch.from_numpy(selected_queries).reshape(-1, 1)
            candidate_tensor = torch.from_numpy(candidate_ids[positions]).reshape(-1, 1)
            tokens = tokenize_pairs(
                tokenizer,
                query_tensor,
                candidate_tensor,
                query_rows,
                candidate_rows,
                int(max_length),
                device,
                str(pair_mode),
            )
            pair = torch.from_numpy(pair_features[positions]).to(device)
            query = torch.from_numpy(
                query_features[selected_queries]
            ).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                scores = model(tokens, pair, query)
            unique_scores[start:end] = scores.float().cpu().numpy()
    return unique_scores[inverse]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--evaluation_split", choices=("val", "test"), default="val")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--matsci_embeddings", required=True)
    parser.add_argument("--matsci_components", type=int, default=64)
    parser.add_argument("--base_val_candidates", required=True)
    parser.add_argument("--train_candidate_source", action="append", default=[])
    parser.add_argument("--candidate_limit", type=int, default=100)
    parser.add_argument(
        "--candidate_windows",
        default="20,50,100",
        help="Comma-separated global reranking windows evaluated on validation.",
    )
    parser.add_argument("--source_union_limit", type=int, default=200)
    parser.add_argument("--train_pool_limit", type=int, default=12)
    parser.add_argument("--cross_family_negatives", type=int, default=0)
    parser.add_argument(
        "--include_rank_features",
        action="store_true",
        help="Append aggregate retrieval-rank features to every query-candidate pair.",
    )
    parser.add_argument(
        "--train_aux_rank_source",
        action="append",
        default=[],
        help="Optional train OOF source used only as additional candidate-rank features.",
    )
    parser.add_argument(
        "--val_aux_rank_source",
        action="append",
        default=[],
        help="Validation counterpart of --train_aux_rank_source, in matching order.",
    )
    parser.add_argument("--aux_rank_limit", type=int, default=100)
    parser.add_argument("--train_row_limit", type=int, default=0)
    parser.add_argument(
        "--multi_positive_route_kind",
        choices=(
            "none",
            "family",
            "family_anion",
            "family_synthesis",
            "family_source",
            "family_anion_synthesis",
            "family_anion_source",
            "family_anion_synthesis_source",
            "anion_synthesis_nofamily",
            "anion_synthesis_source_nofamily",
            "anion_synthesis_nofamily_factorized",
        ),
        default="none",
    )
    parser.add_argument("--multi_positive_min_count", type=int, default=2)
    parser.add_argument("--multi_positive_max_candidates", type=int, default=4)
    parser.add_argument("--multi_positive_weight", type=float, default=0.0)
    parser.add_argument(
        "--train_families",
        default="",
        help="Optional comma-separated target families used for specialist fine-tuning.",
    )
    parser.add_argument(
        "--val_families",
        default="",
        help="Optional comma-separated target families reranked by the specialist.",
    )
    parser.add_argument(
        "--train_anion_signature",
        default="",
        help="Optional exact sorted anion signature, for example C+O.",
    )
    parser.add_argument("--train_source_dataset", default="")
    parser.add_argument("--train_synthesis_type", default="")
    parser.add_argument("--max_length", type=int, default=96)
    parser.add_argument(
        "--text_pair_mode",
        choices=("tokenizer_pair", "explicit"),
        default="tokenizer_pair",
    )
    parser.add_argument("--freeze_bottom_layers", type=int, default=6)
    parser.add_argument("--pooling", choices=("auto", "cls", "mean", "last"), default="auto")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--freeze_embeddings", action="store_true")
    parser.add_argument(
        "--attention_implementation",
        choices=("", "eager", "sdpa", "flash_attention_2"),
        default="",
    )
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--batch_queries", type=int, default=10)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--encoder_learning_rate", type=float, default=2e-5)
    parser.add_argument("--head_learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.08)
    parser.add_argument("--pairwise_weight", type=float, default=0.2)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--resume_model", default="")
    parser.add_argument(
        "--recompute_resume_normalization",
        action="store_true",
        help="Recompute pair normalization instead of restoring it from a resume checkpoint.",
    )
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument(
        "--fixed_trial_json",
        default="",
        help="Validation report whose selected slate policy is replayed without evaluation search.",
    )
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", required=True)
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--output_scores_npz", default="")
    args = parser.parse_args()
    if not args.train_candidate_source:
        parser.error("at least one --train_candidate_source is required")
    if bool(args.eval_only) and not str(args.resume_model):
        parser.error("--eval_only requires --resume_model")
    if str(args.evaluation_split) == "test" and not bool(args.eval_only):
        parser.error("test evaluation is allowed only with --eval_only")
    if str(args.evaluation_split) == "test" and not str(args.fixed_trial_json).strip():
        parser.error("test evaluation requires --fixed_trial_json")
    if not 0.0 <= float(args.multi_positive_weight) <= 1.0:
        parser.error("--multi_positive_weight must be between 0 and 1")
    if int(args.multi_positive_max_candidates) < 1:
        parser.error("--multi_positive_max_candidates must be at least 1")
    if len(args.train_aux_rank_source) != len(args.val_aux_rank_source):
        parser.error("train and validation auxiliary rank sources must have equal counts")
    if args.train_aux_rank_source and not bool(args.include_rank_features):
        parser.error("auxiliary rank sources require --include_rank_features")
    seed_everything(int(args.seed))
    resume_payload = None
    if str(args.resume_model):
        resume_payload = torch.load(
            Path(args.resume_model).resolve(), map_location="cpu", weights_only=False
        )

    input_dir = Path(args.input_dir).resolve()
    train_pack = np.load(input_dir / "train.npz", allow_pickle=True)
    val_pack = np.load(
        input_dir / f"{str(args.evaluation_split)}.npz", allow_pickle=True
    )
    train_y = np.asarray(train_pack["y_multi_hot"], dtype=np.float32)
    val_y = np.asarray(val_pack["y_multi_hot"], dtype=np.float32)
    train_x = np.asarray(train_pack["x"], dtype=np.float32)
    val_x = np.asarray(val_pack["x"], dtype=np.float32)
    train_targets = targets_from_matrix(train_y)
    val_targets = targets_from_matrix(val_y)
    train_meta = pd.read_csv(input_dir / "train_meta.csv", low_memory=False)
    val_meta = pd.read_csv(
        input_dir / f"{str(args.evaluation_split)}_meta.csv", low_memory=False
    )
    _, train_matsci, val_matsci, matsci_metadata = load_matsci_pca_views(
        Path(args.matsci_embeddings).resolve(),
        int(args.matsci_components),
        int(args.seed),
        str(args.evaluation_split),
    )
    train_query_dense, val_query_dense, query_mean, query_std = standardize_from_train(
        np.hstack([train_x, train_matsci]).astype(np.float32),
        np.hstack([val_x, val_matsci]).astype(np.float32),
    )
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
    train_families = train_meta["family_signature_primary"].fillna("UNK").astype(str).to_numpy()
    val_families = val_meta["family_signature_primary"].fillna("UNK").astype(str).to_numpy()
    selected_train_families = parse_family_filter(args.train_families)
    selected_val_families = parse_family_filter(args.val_families)
    included_val_indices = (
        {
            int(index)
            for index, family in enumerate(val_families)
            if str(family) in selected_val_families
        }
        if selected_val_families
        else None
    )
    requested_anion_signature = "+".join(
        sorted(
            value.strip()
            for value in str(args.train_anion_signature).split("+")
            if value.strip()
        )
    )
    filter_training_rows = bool(
        selected_train_families
        or requested_anion_signature
        or str(args.train_source_dataset).strip()
        or str(args.train_synthesis_type).strip()
    )
    included_train_indices = None
    if filter_training_rows:
        included_train_indices = set()
        for index, family in enumerate(train_families):
            row = train_meta.iloc[int(index)]
            if selected_train_families and str(family) not in selected_train_families:
                continue
            if requested_anion_signature:
                signature = "+".join(
                    sorted(json_set(row.get("target_anion_elements", "")))
                )
                if signature != requested_anion_signature:
                    continue
            if str(args.train_source_dataset).strip() and str(
                row.get("source_dataset", "")
            ) != str(args.train_source_dataset).strip():
                continue
            if str(args.train_synthesis_type).strip() and str(
                row.get("synthesis_type", "")
            ) != str(args.train_synthesis_type).strip():
                continue
            included_train_indices.add(int(index))
        if not included_train_indices:
            raise RuntimeError("training metadata filters removed every row")

    def train_pair(row_index: int, candidate: SetKey) -> np.ndarray:
        family = str(train_families[int(row_index)])
        features = build_pair_features(
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
        if bool(args.include_rank_features):
            features = np.concatenate(
                [features, candidate_rank_features(train_rank_maps, row_index, candidate)]
            )
            for maps in train_aux_rank_maps:
                features = np.concatenate(
                    [features, candidate_rank_features(maps, row_index, candidate)]
                )
        return features.astype(np.float32)

    val_pair_feature_cache: Dict[tuple[object, ...], np.ndarray] = {}

    def val_pair(row_index: int, candidate: SetKey) -> np.ndarray:
        family = str(val_families[int(row_index)])
        row = val_meta.iloc[int(row_index)]
        rank_features = (
            candidate_rank_features(val_rank_maps, row_index, candidate)
            if bool(args.include_rank_features)
            else None
        )
        aux_rank_features = (
            [candidate_rank_features(maps, row_index, candidate) for maps in val_aux_rank_maps]
            if bool(args.include_rank_features)
            else []
        )
        cache_key = (
            family,
            str(row.get("canonical_formula", row.get("formula", ""))),
            str(row.get("target_cation_elements", "")),
            str(row.get("target_anion_elements", "")),
            str(row.get("source_dataset", "")),
            str(row.get("synthesis_type", "")),
            candidate,
            None if rank_features is None else rank_features.tobytes(),
            tuple(values.tobytes() for values in aux_rank_features),
        )
        cached = val_pair_feature_cache.get(cache_key)
        if cached is not None:
            return cached
        features = build_pair_features(
            candidate,
            row,
            family,
            int(length_modes.get(family, length_modes["__GLOBAL__"])),
            label_elements,
            label_groups,
            label_metals,
            train_seen,
            prior_builder,
            template_builder,
        )
        if rank_features is not None:
            features = np.concatenate([features, rank_features, *aux_rank_features])
        val_pair_feature_cache[cache_key] = features
        return features

    train_source_paths = [str(Path(path).resolve()) for path in args.train_candidate_source]
    train_sources = [
        load_source(path, len(train_targets), int(args.source_union_limit))
        for path in train_source_paths
    ]
    train_source_cache = dict(zip(train_source_paths, train_sources))
    base_rows = load_source(
        args.base_val_candidates, len(val_targets), int(args.candidate_limit)
    )
    train_rank_maps: List[Dict[SetKey, np.ndarray]] = []
    val_rank_maps: List[Dict[SetKey, np.ndarray]] = []
    train_aux_rank_maps: List[List[Dict[SetKey, np.ndarray]]] = []
    val_aux_rank_maps: List[List[Dict[SetKey, np.ndarray]]] = []
    if bool(args.include_rank_features):
        train_merged_rows = [
            merge_candidate_sources(train_sources, row_index, int(args.source_union_limit))
            for row_index in range(len(train_targets))
        ]
        train_rank_maps = candidate_rank_feature_maps(
            train_merged_rows, int(args.source_union_limit)
        )
        val_rank_maps = candidate_rank_feature_maps(base_rows, int(args.candidate_limit))
        for train_path, val_path in zip(
            args.train_aux_rank_source, args.val_aux_rank_source
        ):
            resolved_train_path = str(Path(train_path).resolve())
            cached_train_rows = train_source_cache.get(resolved_train_path)
            train_aux_rows = (
                [row[: int(args.aux_rank_limit)] for row in cached_train_rows]
                if cached_train_rows is not None
                else load_source(train_path, len(train_targets), int(args.aux_rank_limit))
            )
            val_aux_rows = load_source(
                val_path, len(val_targets), int(args.aux_rank_limit)
            )
            train_aux_rank_maps.append(
                candidate_rank_feature_maps(train_aux_rows, int(args.aux_rank_limit))
            )
            val_aux_rank_maps.append(
                candidate_rank_feature_maps(val_aux_rows, int(args.aux_rank_limit))
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
        included_train_indices,
    )
    multi_positive_mask, multi_positive_metadata = build_multi_positive_supervision(
        train_targets,
        train_meta,
        train_pool,
        registry,
        str(args.multi_positive_route_kind),
        int(args.multi_positive_min_count),
        int(args.multi_positive_max_candidates),
        label_elements,
    )
    print(
        json.dumps({"multi_positive_supervision": multi_positive_metadata}),
        flush=True,
    )
    val_query_indices, val_candidate_ids, val_pair_features, val_spans = build_validation_pairs(
        base_rows, val_meta, registry, val_pair, included_val_indices
    )
    # Validate that all registry entries fit the same maximum label count used by
    # the structured models, even though this cross encoder consumes text.
    candidate_label_tensor(registry, max_labels=8)
    pair_flat = train_pool.pair_features[train_pool.mask]
    normalization_source = "current_training_pool"
    restored_mean = None if resume_payload is None else resume_payload.get("pair_mean")
    restored_std = None if resume_payload is None else resume_payload.get("pair_std")
    if (
        resume_payload is not None
        and not bool(args.recompute_resume_normalization)
        and restored_mean is not None
        and restored_std is not None
        and np.asarray(restored_mean).shape == pair_flat.shape[1:]
        and np.asarray(restored_std).shape == pair_flat.shape[1:]
    ):
        pair_mean = np.asarray(restored_mean, dtype=np.float32)
        pair_std = np.asarray(restored_std, dtype=np.float32)
        normalization_source = "resume_checkpoint"
    else:
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

    train_query_rows = query_texts(train_meta)
    val_query_rows = query_texts(val_meta)
    candidate_rows = [candidate_text(candidate, names) for candidate in registry.keys]
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MatSciCrossEncoder(
        str(args.model_path),
        int(train_pool.pair_features.shape[-1]),
        int(train_query_dense.shape[1]),
        float(args.dropout),
        int(args.freeze_bottom_layers),
        str(args.pooling),
        bool(args.gradient_checkpointing),
        str(args.attention_implementation),
        bool(args.freeze_embeddings),
    ).to(device)
    if resume_payload is not None:
        model.load_state_dict(resume_payload.get("model_state", resume_payload))
    encoder_parameters = []
    head_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_parameters.append(parameter)
        else:
            head_parameters.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": float(args.encoder_learning_rate)},
            {"params": head_parameters, "lr": float(args.head_learning_rate)},
        ],
        weight_decay=float(args.weight_decay),
    )

    dataset = QueryPoolDataset(train_pool)
    if selected_train_families:
        specialist_pool_indices = [
            index
            for index, query_index in enumerate(train_pool.query_indices.tolist())
            if str(train_families[int(query_index)]) in selected_train_families
        ]
        if not specialist_pool_indices:
            raise RuntimeError("train family filter removed every cross-encoder pool row")
        dataset = Subset(dataset, specialist_pool_indices)
    if int(args.train_row_limit) > 0 and int(args.train_row_limit) < len(dataset):
        rng = np.random.default_rng(int(args.seed))
        indices = np.sort(
            rng.choice(len(dataset), size=int(args.train_row_limit), replace=False)
        ).tolist()
        dataset = Subset(dataset, indices)
    generator = torch.Generator().manual_seed(int(args.seed))
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_queries),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        generator=generator,
        persistent_workers=bool(int(args.num_workers) > 0),
    )
    update_steps = math.ceil(
        len(loader) * int(args.epochs) / max(1, int(args.gradient_accumulation))
    )
    warmup_steps = int(round(update_steps * float(args.warmup_ratio)))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=max(1, update_steps)
    )
    alpha_grid = (0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6)
    protected_grid = (0, 1, 3, 5, 7, 9, 10)
    minimum_gain_grid = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    candidate_window_grid = tuple(
        sorted(
            {
                int(value.strip())
                for value in str(args.candidate_windows).split(",")
                if value.strip() and int(value.strip()) > 0
            }
        )
    )
    if not candidate_window_grid:
        raise ValueError("--candidate_windows must contain at least one positive integer")
    active_val_indices = np.asarray(
        [
            index
            for index, family in enumerate(val_families)
            if not selected_val_families or str(family) in selected_val_families
        ],
        dtype=np.int64,
    )
    if not len(active_val_indices):
        raise RuntimeError("validation family filter removed every cross-encoder row")
    history: List[Dict[str, object]] = []
    best_key = None
    best_epoch = 0
    best_state = None
    best_trial: Dict[str, object] = {}
    best_rows: List[List[SetKey]] = []
    best_grid_trials: List[Dict[str, object]] = []
    best_val_scores: np.ndarray | None = None
    optimizer.zero_grad(set_to_none=True)
    train_query_tensor = torch.from_numpy(train_query_dense)
    multi_positive_tensor = torch.from_numpy(multi_positive_mask)
    if bool(args.eval_only):
        best_val_scores = score_validation(
            model,
            tokenizer,
            val_query_indices,
            val_candidate_ids,
            val_pair_features,
            val_query_dense,
            val_query_rows,
            candidate_rows,
            int(args.max_length),
            int(args.eval_batch_size),
            device,
            str(args.text_pair_mode),
        )
        if str(args.fixed_trial_json).strip():
            if len(active_val_indices) != len(val_targets):
                raise ValueError("fixed slate replay currently requires all evaluation rows")
            fixed_report = json.loads(
                Path(args.fixed_trial_json).resolve().read_text(encoding="utf-8")
            )
            fixed_trial = fixed_report.get("validation", {}).get("best")
            if not isinstance(fixed_trial, dict):
                fixed_trial = fixed_report.get("best")
            if not isinstance(fixed_trial, dict):
                raise ValueError("fixed trial report must contain validation.best or best")
            best_trial, best_rows = apply_fixed_trial(
                val_targets,
                base_rows,
                best_val_scores,
                val_spans,
                label_families,
                fixed_trial,
            )
            best_grid_trials = [dict(best_trial)]
        else:
            best_trial, best_rows, best_grid_trials = evaluate_specialist_grid(
                val_targets,
                base_rows,
                best_val_scores,
                val_spans,
                label_families,
                active_val_indices,
                alpha_grid,
                protected_grid,
                minimum_gain_grid,
                candidate_window_grid,
            )
        best_state = {
            name: value.detach().cpu().clone() for name, value in model.state_dict().items()
        }
        best_key = (
            float(best_trial["exact_hit@10"]),
            float(best_trial["exact_hit@5"]),
            float(best_trial["exact_hit@1"]),
        )
    epoch_iterator = () if bool(args.eval_only) else range(1, int(args.epochs) + 1)
    for epoch in epoch_iterator:
        model.train()
        loss_sum = 0.0
        rows_seen = 0
        epoch_started = time.time()
        for step, (query_indices, candidate_ids, mask, pair_features) in enumerate(loader, start=1):
            batch, candidates = candidate_ids.shape
            query_grid = query_indices[:, None].expand(-1, candidates)
            tokens = tokenize_pairs(
                tokenizer,
                query_grid,
                candidate_ids,
                train_query_rows,
                candidate_rows,
                int(args.max_length),
                device,
                str(args.text_pair_mode),
            )
            pair = pair_features.reshape(batch * candidates, -1).to(device, non_blocking=True)
            query = (
                train_query_tensor[query_indices]
                .to(device, non_blocking=True)[:, None, :]
                .expand(-1, candidates, -1)
                .reshape(batch * candidates, -1)
            )
            mask = mask.to(device, non_blocking=True)
            positive_mask = multi_positive_tensor[query_indices].to(
                device, non_blocking=True
            )
            positive_mask = positive_mask & mask
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                scores = model(tokens, pair, query).reshape(batch, candidates)
                masked_scores = scores.masked_fill(~mask, -1e4)
                target = torch.zeros(batch, dtype=torch.long, device=device)
                exact_listwise = nn.functional.cross_entropy(masked_scores.float(), target)
                positive_scores = masked_scores.float().masked_fill(~positive_mask, -1e4)
                multi_positive = -(
                    torch.logsumexp(positive_scores, dim=1)
                    - torch.logsumexp(masked_scores.float(), dim=1)
                ).mean()
                listwise = (
                    (1.0 - float(args.multi_positive_weight)) * exact_listwise
                    + float(args.multi_positive_weight) * multi_positive
                )
                negative_mask = mask[:, 1:] & ~positive_mask[:, 1:]
                negative = scores[:, 1:].masked_fill(~negative_mask, -1e4).max(dim=1).values
                pairwise = nn.functional.relu(float(args.margin) - scores[:, 0] + negative).mean()
                loss = (listwise + float(args.pairwise_weight) * pairwise) / max(
                    1, int(args.gradient_accumulation)
                )
            loss.backward()
            if step % int(args.gradient_accumulation) == 0 or step == len(loader):
                nn.utils.clip_grad_norm_(model.parameters(), float(args.gradient_clip))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += float(loss.detach().cpu()) * batch * max(
                1, int(args.gradient_accumulation)
            )
            rows_seen += int(batch)
            if int(args.log_every_steps) > 0 and (
                step % int(args.log_every_steps) == 0 or step == len(loader)
            ):
                print(
                    json.dumps(
                        {
                            "epoch": int(epoch),
                            "step": int(step),
                            "steps": int(len(loader)),
                            "train_loss_running": float(loss_sum / max(1, rows_seen)),
                            "elapsed_minutes": float((time.time() - epoch_started) / 60.0),
                        }
                    ),
                    flush=True,
                )

        val_scores = score_validation(
            model,
            tokenizer,
            val_query_indices,
            val_candidate_ids,
            val_pair_features,
            val_query_dense,
            val_query_rows,
            candidate_rows,
            int(args.max_length),
            int(args.eval_batch_size),
            device,
            str(args.text_pair_mode),
        )
        trial, ranked, grid_trials = evaluate_specialist_grid(
            val_targets,
            base_rows,
            val_scores,
            val_spans,
            label_families,
            active_val_indices,
            alpha_grid,
            protected_grid,
            minimum_gain_grid,
            candidate_window_grid,
        )
        epoch_row: Dict[str, object] = {
            "epoch": int(epoch),
            "train_loss": float(loss_sum / max(1, rows_seen)),
            "encoder_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "head_learning_rate": float(optimizer.param_groups[1]["lr"]),
            "validation": trial,
        }
        history.append(epoch_row)
        print(json.dumps(epoch_row, ensure_ascii=False), flush=True)
        key = (
            float(trial["exact_hit@10"]),
            float(trial["exact_hit@5"]),
            float(trial["exact_hit@1"]),
            -float(epoch_row["train_loss"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = int(epoch)
            best_trial = dict(trial)
            best_rows = ranked
            best_grid_trials = grid_trials
            best_val_scores = val_scores.copy()
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
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
        elif int(args.early_stopping_patience) > 0 and (
            int(epoch) - int(best_epoch) >= int(args.early_stopping_patience)
        ):
            print(
                json.dumps(
                    {
                        "early_stopping": True,
                        "best_epoch": int(best_epoch),
                        "stopped_after_epoch": int(epoch),
                    }
                ),
                flush=True,
            )
            break

    if best_state is None:
        raise RuntimeError("no cross-encoder checkpoint was evaluated")
    if best_val_scores is None:
        raise RuntimeError("no validation score vector was retained")
    report = {
        "protocol": (
            f"train_oof_transformer_cross_encoder_{str(args.evaluation_split)}_"
            "formula_group_disjoint"
        ),
        "config": vars(args),
        "device": str(device),
        "model_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "matsci": matsci_metadata,
        "dimensions": {
            "query_dense": int(train_query_dense.shape[1]),
            "pair": int(train_pool.pair_features.shape[-1]),
            "rank_features": (
                6 * (1 + len(args.train_aux_rank_source))
                if bool(args.include_rank_features)
                else 0
            ),
        },
        "normalization_source": normalization_source,
        "training": {
            "rows_total": int(len(train_targets)),
            "rows_in_pool": int(len(train_pool.query_indices)),
            "rows_used": int(len(dataset)),
            "target_in_oof_source_pool": int(train_pool.target_in_source_pool),
            "rows_with_same_family_negative": int(train_pool.rows_with_same_family_negative),
            "rows_with_cross_family_negative": int(
                train_pool.rows_with_cross_family_negative
            ),
            "pool_width": int(train_pool.candidate_ids.shape[1]),
            "selected_families": sorted(selected_train_families),
            "multi_positive": multi_positive_metadata,
            "multi_positive_weight": float(args.multi_positive_weight),
        },
        "validation": {
            "rows": int(len(val_targets)),
            "specialist_rows": int(len(active_val_indices)),
            "selected_families": sorted(selected_val_families),
            "base": exact_metrics(val_targets, base_rows),
            "best": best_trial,
            "best_by_strategy": best_trials_by_strategy(best_grid_trials),
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
            "pair_mean": pair_mean,
            "pair_std": pair_std,
            "query_mean": query_mean,
            "query_std": query_std,
            "best_trial": best_trial,
            "candidate_keys": registry.keys,
            "protocol": report["protocol"],
        },
        output_model,
    )
    output_scores = (
        Path(args.output_scores_npz).resolve()
        if str(args.output_scores_npz)
        else output_json.with_name("val_scores.npz")
    )
    output_scores.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_scores,
        raw_scores=best_val_scores.astype(np.float32),
        spans=np.asarray(val_spans, dtype=np.int64),
        row_indices=val_query_indices.astype(np.int64),
        candidate_hashes=np.asarray(
            [candidate_fingerprint(registry.keys[int(value)]) for value in val_candidate_ids],
            dtype=np.uint64,
        ),
    )
    print(json.dumps(report["validation"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
