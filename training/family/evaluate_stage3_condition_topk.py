#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.family.train_stage3_gpu_multitask import (  # noqa: E402
    Stage3MultiTaskNet,
    inverse_continuous,
    predict,
)


TOP_K = (1, 3, 5, 10, 20, 50)


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
    return value


def load_pack(input_dir: Path, split: str) -> Dict[str, np.ndarray]:
    return {
        key: value
        for key, value in np.load(input_dir / f"{split}.npz", allow_pickle=True).items()
    }


def model_predictions(
    checkpoint_path: Path,
    pack: Mapping[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    structure_dim = int(checkpoint["structure_dim"])
    precursor_dim = int(checkpoint["precursor_dim"])
    config = checkpoint["config"]
    model = Stage3MultiTaskNet(
        structure_dim=structure_dim,
        precursor_dim=precursor_dim,
        hidden=int(config["hidden"]),
        precursor_hidden=int(config["precursor_hidden"]),
        blocks=int(config["blocks"]),
        dropout=float(config["dropout"]),
        atmosphere_classes=int(checkpoint["atmosphere_classes"]),
        method_classes=int(checkpoint["method_classes"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    features = np.hstack(
        [
            np.asarray(pack["x"], dtype=np.float32)[:, :structure_dim],
            np.asarray(pack["y_set"], dtype=np.float32),
        ]
    ).astype(np.float32)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    return predict(model, loader, device), checkpoint


def raw_quantiles(prediction: Mapping[str, np.ndarray], stats: Mapping[str, Any]) -> np.ndarray:
    normalized = np.asarray(prediction["quantiles"], dtype=np.float32)
    raw = np.zeros_like(normalized)
    for quantile_index in range(normalized.shape[2]):
        raw[:, :, quantile_index] = inverse_continuous(
            normalized[:, :, quantile_index], stats
        )
    raw[:, 0, :] = np.clip(raw[:, 0, :], 0.0, 3000.0)
    raw[:, 1, :] = np.clip(raw[:, 1, :], 0.0, 10000.0)
    return raw


def normalized_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norm, 1e-8)


def build_neighbor_index(
    train: Mapping[str, np.ndarray],
    query: Mapping[str, np.ndarray],
    structure_dim: int,
    precursor_weight: float,
    neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_structure = np.asarray(train["x"], dtype=np.float32)[:, :structure_dim]
    query_structure = np.asarray(query["x"], dtype=np.float32)[:, :structure_dim]
    train_precursor = np.asarray(train["y_set"], dtype=np.float32)
    query_precursor = np.asarray(query["y_set"], dtype=np.float32)
    train_features = np.hstack(
        [normalized_rows(train_structure), precursor_weight * normalized_rows(train_precursor)]
    )
    query_features = np.hstack(
        [normalized_rows(query_structure), precursor_weight * normalized_rows(query_precursor)]
    )
    index = NearestNeighbors(
        n_neighbors=min(int(neighbors), len(train_features)),
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    index.fit(train_features)
    distance, indices = index.kneighbors(query_features, return_distance=True)
    return indices, 1.0 - distance


def neural_candidates(
    row: int,
    quantiles: np.ndarray,
    atmosphere_prob: np.ndarray,
    method_prob: np.ndarray,
    limit: int,
) -> list[tuple[float, float, int, int, float, str]]:
    atmosphere_order = np.argsort(-atmosphere_prob[row])
    method_order = np.argsort(-method_prob[row])
    # Median-first ordering makes the first four entries cover every known
    # coarse atmosphere while keeping the strongest continuous point estimate.
    quantile_pairs = [
        (1, 1, 0.0),
        (0, 1, 0.7),
        (2, 1, 0.7),
        (1, 0, 0.7),
        (1, 2, 0.7),
        (0, 0, 1.2),
        (2, 2, 1.2),
        (0, 2, 1.4),
        (2, 0, 1.4),
    ]
    candidates: list[tuple[float, float, int, int, float, str]] = []
    for temp_q, time_q, quantile_penalty in quantile_pairs:
        for atm_rank, atmosphere in enumerate(atmosphere_order):
            # Missing class is retained because it is a real prediction for
            # samples whose atmosphere is unrecorded, but known-atmosphere
            # classes remain fully covered within a small beam.
            for method_rank, method in enumerate(method_order[:3]):
                score = (
                    math.log(max(float(atmosphere_prob[row, atmosphere]), 1e-12))
                    + 0.20 * math.log(max(float(method_prob[row, method]), 1e-12))
                    - 0.35 * quantile_penalty
                    - 0.02 * atm_rank
                    - 0.01 * method_rank
                )
                candidates.append(
                    (
                        float(quantiles[row, 0, temp_q]),
                        float(quantiles[row, 1, time_q]),
                        int(atmosphere),
                        int(method),
                        score,
                        "neural",
                    )
                )
    candidates.sort(key=lambda item: item[4], reverse=True)
    return candidates[:limit]


def neighbor_candidates(
    row: int,
    train: Mapping[str, np.ndarray],
    neighbor_indices: np.ndarray,
    similarities: np.ndarray,
    atmosphere_prob: np.ndarray,
    method_prob: np.ndarray,
    limit: int,
) -> list[tuple[float, float, int, int, float, str]]:
    continuous = np.asarray(train["y_cond_continuous_raw"], dtype=np.float32)
    continuous_mask = np.asarray(train["y_cond_continuous_mask"], dtype=np.float32) > 0.5
    discrete = np.asarray(train["y_cond_discrete"], dtype=np.int64)
    discrete_mask = np.asarray(train["y_cond_discrete_mask"], dtype=np.float32) > 0.5
    output: list[tuple[float, float, int, int, float, str]] = []
    seen: set[tuple[float, float, int, int]] = set()
    for train_row, similarity in zip(neighbor_indices[row], similarities[row]):
        train_row = int(train_row)
        if not bool(continuous_mask[train_row, 0] and continuous_mask[train_row, 1]):
            continue
        atmosphere = int(discrete[train_row, 0]) if discrete_mask[train_row, 0] else 0
        method = int(discrete[train_row, 1])
        key = (
            round(float(continuous[train_row, 0]), 3),
            round(float(continuous[train_row, 1]), 3),
            atmosphere,
            method,
        )
        if key in seen:
            continue
        seen.add(key)
        score = (
            2.0 * float(similarity)
            + 0.35 * math.log(max(float(atmosphere_prob[row, atmosphere]), 1e-12))
            + 0.08 * math.log(max(float(method_prob[row, method]), 1e-12))
        )
        output.append((*key, score, "neighbor"))
        if len(output) >= limit:
            break
    output.sort(key=lambda item: item[4], reverse=True)
    return output


def reciprocal_rank_fusion(
    sources: Sequence[Sequence[tuple[float, float, int, int, float, str]]],
    limit: int,
    rrf_k: float,
) -> list[tuple[float, float, int, int, float, str]]:
    scores: Dict[tuple[float, float, int, int], float] = {}
    origins: Dict[tuple[float, float, int, int], set[str]] = {}
    for source in sources:
        for rank, candidate in enumerate(source, start=1):
            key = (
                round(float(candidate[0]), 3),
                round(float(candidate[1]), 3),
                int(candidate[2]),
                int(candidate[3]),
            )
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            origins.setdefault(key, set()).add(str(candidate[5]))
    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)[:limit]
    return [(*key, scores[key], "+".join(sorted(origins[key]))) for key in ordered]


def hit(
    candidate: tuple[float, float, int, int, float, str],
    row: int,
    pack: Mapping[str, np.ndarray],
    method_inclusive: bool,
    missing_aware: bool,
) -> bool:
    continuous = np.asarray(pack["y_cond_continuous_raw"])
    continuous_mask = np.asarray(pack["y_cond_continuous_mask"]) > 0.5
    discrete = np.asarray(pack["y_cond_discrete"])
    discrete_mask = np.asarray(pack["y_cond_discrete_mask"]) > 0.5
    if not missing_aware and not bool(continuous_mask[row, 0] and continuous_mask[row, 1]):
        return False
    checks: list[bool] = []
    if continuous_mask[row, 0]:
        checks.append(abs(float(candidate[0]) - float(continuous[row, 0])) <= 200.0)
    if continuous_mask[row, 1]:
        checks.append(abs(float(candidate[1]) - float(continuous[row, 1])) <= 48.0)
    if discrete_mask[row, 0]:
        checks.append(int(candidate[2]) == int(discrete[row, 0]))
    if method_inclusive and discrete_mask[row, 1]:
        checks.append(int(candidate[3]) == int(discrete[row, 1]))
    return bool(checks) and all(checks)


def evaluate(
    ranked: Sequence[Sequence[tuple[float, float, int, int, float, str]]],
    pack: Mapping[str, np.ndarray],
    method_inclusive: bool,
    missing_aware: bool,
) -> Dict[str, Any]:
    continuous_mask = np.asarray(pack["y_cond_continuous_mask"]) > 0.5
    if missing_aware:
        eligible = np.ones(len(ranked), dtype=bool)
    else:
        eligible = continuous_mask[:, 0] & continuous_mask[:, 1]
    rows = np.flatnonzero(eligible)
    result: Dict[str, Any] = {"n": int(len(rows))}
    for k in TOP_K:
        hits = sum(
            any(hit(candidate, int(row), pack, method_inclusive, missing_aware) for candidate in ranked[int(row)][:k])
            for row in rows
        )
        result[f"hit@{k}"] = float(hits / max(1, len(rows)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage3 condition-tuple Top-K generation.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--neighbors", type=int, default=200)
    parser.add_argument("--candidate_limit", type=int, default=100)
    parser.add_argument("--precursor_weight", type=float, default=1.5)
    parser.add_argument("--rrf_k", type=float, default=20.0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    train = load_pack(input_dir, "train")
    query = load_pack(input_dir, args.split)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    prediction, checkpoint = model_predictions(
        checkpoint_path, query, device, int(args.batch_size)
    )
    quantiles = raw_quantiles(prediction, checkpoint["target_stats"])
    atmosphere_prob = torch.softmax(torch.from_numpy(prediction["atmosphere"]), dim=1).numpy()
    method_prob = torch.softmax(torch.from_numpy(prediction["method"]), dim=1).numpy()
    neighbor_indices, similarities = build_neighbor_index(
        train,
        query,
        int(checkpoint["structure_dim"]),
        float(args.precursor_weight),
        int(args.neighbors),
    )

    neural_ranked = []
    neighbor_ranked = []
    fused_ranked = []
    for row in range(len(np.asarray(query["x"]))):
        neural = neural_candidates(
            row, quantiles, atmosphere_prob, method_prob, int(args.candidate_limit)
        )
        neighbor = neighbor_candidates(
            row,
            train,
            neighbor_indices,
            similarities,
            atmosphere_prob,
            method_prob,
            int(args.candidate_limit),
        )
        fused = reciprocal_rank_fusion(
            [neural, neighbor], int(args.candidate_limit), float(args.rrf_k)
        )
        neural_ranked.append(neural)
        neighbor_ranked.append(neighbor)
        fused_ranked.append(fused)

    ranked_by_name = {
        "neural_quantile_cartesian": neural_ranked,
        "observed_neighbor": neighbor_ranked,
        "rrf_neural_neighbor": fused_ranked,
    }
    metrics: Dict[str, Any] = {
        "protocol": {
            "primary": "missing-aware relaxed condition tuple; temperature <=200 C, time <=48 h, known coarse atmosphere exact; method excluded for historical comparability",
            "secondary_method_inclusive": "primary plus reaction method exact",
            "strict_comparable_denominator": "rows with both temperature and time recorded; missing atmosphere ignored",
            "split": args.split,
        },
        "config": vars(args),
        "rows": int(len(np.asarray(query["x"]))),
        "models": {},
    }
    for name, ranked in ranked_by_name.items():
        metrics["models"][name] = {
            "missing_aware_relaxed": evaluate(ranked, query, False, True),
            "strict_comparable_relaxed": evaluate(ranked, query, False, False),
            "missing_aware_method_inclusive": evaluate(ranked, query, True, True),
            "strict_comparable_method_inclusive": evaluate(ranked, query, True, False),
        }

    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(to_builtin(metrics), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.output_candidates_jsonl:
        output_candidates = Path(args.output_candidates_jsonl).resolve()
        output_candidates.parent.mkdir(parents=True, exist_ok=True)
        sample_ids = np.asarray(query["sample_id"]).astype(str)
        with output_candidates.open("w", encoding="utf-8") as handle:
            for row, sample_id in enumerate(sample_ids):
                record = {
                    "sample_id": sample_id,
                    "candidates": [
                        {
                            "temperature_c": candidate[0],
                            "time_h": candidate[1],
                            "atmosphere_id": candidate[2],
                            "method_id": candidate[3],
                            "score": candidate[4],
                            "source": candidate[5],
                        }
                        for candidate in fused_ranked[row]
                    ],
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(to_builtin(metrics), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
