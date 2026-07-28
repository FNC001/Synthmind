#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


SetKey = Tuple[int, ...]


def keys_from_matrix(values: np.ndarray) -> List[SetKey]:
    return [tuple(np.flatnonzero(row > 0.5).tolist()) for row in values]


def unique_in_order(values: Iterable[SetKey], limit: int) -> List[SetKey]:
    result: List[SetKey] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
        if len(result) >= limit:
            break
    return result


def hit_rate(targets: Sequence[SetKey], candidates: Sequence[Sequence[SetKey]], k: int) -> float:
    return float(np.mean([target in set(row[:k]) for target, row in zip(targets, candidates)]))


def frequency_candidates(
    train_keys: Sequence[SetKey],
    train_groups: Sequence[str],
    query_groups: Sequence[str],
    limit: int,
) -> List[List[SetKey]]:
    global_order = [key for key, _ in Counter(train_keys).most_common()]
    counters: Dict[str, Counter[SetKey]] = defaultdict(Counter)
    for key, group in zip(train_keys, train_groups):
        counters[str(group)][key] += 1
    result = []
    for group in query_groups:
        local = [key for key, _ in counters.get(str(group), Counter()).most_common()]
        result.append(unique_in_order([*local, *global_order], limit))
    return result


def cosine_knn_indices(
    train_x: np.ndarray,
    query_x: np.ndarray,
    neighbors: int,
    device: str,
    chunk_size: int,
) -> np.ndarray:
    resolved = torch.device(device if torch.cuda.is_available() else "cpu")
    train = torch.from_numpy(np.nan_to_num(train_x).astype(np.float32)).to(resolved)
    train = torch.nn.functional.normalize(train, dim=1)
    outputs = []
    for start in range(0, len(query_x), chunk_size):
        query = torch.from_numpy(np.nan_to_num(query_x[start : start + chunk_size]).astype(np.float32)).to(resolved)
        query = torch.nn.functional.normalize(query, dim=1)
        indices = torch.topk(query @ train.T, k=min(neighbors, len(train)), dim=1).indices
        outputs.append(indices.cpu().numpy())
    return np.vstack(outputs)


def knn_candidates(
    neighbor_indices: np.ndarray,
    train_keys: Sequence[SetKey],
    train_groups: Sequence[str],
    query_groups: Sequence[str],
    limit: int,
    same_family_first: bool,
) -> List[List[SetKey]]:
    result = []
    for row, group in zip(neighbor_indices, query_groups):
        ordered = [int(index) for index in row]
        if same_family_first:
            same = [index for index in ordered if str(train_groups[index]) == str(group)]
            other = [index for index in ordered if str(train_groups[index]) != str(group)]
            ordered = [*same, *other]
        result.append(unique_in_order((train_keys[index] for index in ordered), limit))
    return result


def evaluate_candidates(targets: Sequence[SetKey], candidates: Sequence[Sequence[SetKey]]) -> Dict[str, Any]:
    return {
        "top1": hit_rate(targets, candidates, 1),
        "top3": hit_rate(targets, candidates, 3),
        "top5": hit_rate(targets, candidates, 5),
        "top10": hit_rate(targets, candidates, 10),
        "top20": hit_rate(targets, candidates, 20),
        "top50": hit_rate(targets, candidates, 50),
        "mean_candidates": float(np.mean([len(row) for row in candidates])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Stage2 exact-set Top-K ceilings on validation only.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--neighbors", type=int, default=1000)
    parser.add_argument("--candidate_limit", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk_size", type=int, default=256)
    args = parser.parse_args()
    input_dir = Path(args.input_dir).resolve()
    packs = {split: np.load(input_dir / f"{split}.npz", allow_pickle=True) for split in ("train", "val")}
    meta = {split: pd.read_csv(input_dir / f"{split}_meta.csv", low_memory=False) for split in ("train", "val")}
    train_keys = keys_from_matrix(packs["train"]["y_multi_hot"])
    val_keys = keys_from_matrix(packs["val"]["y_multi_hot"])
    train_family = meta["train"]["family_signature_primary"].fillna("UNK").astype(str).tolist()
    val_family = meta["val"]["family_signature_primary"].fillna("UNK").astype(str).tolist()
    train_source = meta["train"]["source_dataset"].fillna("UNK").astype(str).tolist()
    val_source = meta["val"]["source_dataset"].fillna("UNK").astype(str).tolist()
    train_family_source = [f"{family}||{source}" for family, source in zip(train_family, train_source)]
    val_family_source = [f"{family}||{source}" for family, source in zip(val_family, val_source)]

    train_key_set = set(train_keys)
    labels_seen = np.asarray(packs["train"]["y_multi_hot"]).sum(axis=0) > 0
    val_y = np.asarray(packs["val"]["y_multi_hot"])
    all_labels_seen = np.asarray([bool(np.all(labels_seen[row > 0.5])) for row in val_y])
    exact_set_seen = np.asarray([key in train_key_set for key in val_keys])

    family_frequency = frequency_candidates(
        train_keys, train_family, val_family, int(args.candidate_limit)
    )
    family_source_frequency = frequency_candidates(
        train_keys, train_family_source, val_family_source, int(args.candidate_limit)
    )
    neighbors = cosine_knn_indices(
        np.asarray(packs["train"]["x"], dtype=np.float32),
        np.asarray(packs["val"]["x"], dtype=np.float32),
        int(args.neighbors),
        str(args.device),
        int(args.chunk_size),
    )
    knn = knn_candidates(
        neighbors, train_keys, train_family, val_family, int(args.candidate_limit), False
    )
    family_knn = knn_candidates(
        neighbors, train_keys, train_family, val_family, int(args.candidate_limit), True
    )
    hybrid = [
        unique_in_order([*family_knn_row, *family_freq_row, *family_source_row, *knn_row], int(args.candidate_limit))
        for family_knn_row, family_freq_row, family_source_row, knn_row in zip(
            family_knn, family_frequency, family_source_frequency, knn
        )
    ]
    report = {
        "protocol": "validation_only_formula_disjoint_exact_precursor_set",
        "n_train": len(train_keys),
        "n_val": len(val_keys),
        "n_unique_train_sets": len(train_key_set),
        "label_coverage": {
            "all_val_labels_seen_in_train": float(all_labels_seen.mean()),
            "exact_val_set_seen_in_train": float(exact_set_seen.mean()),
            "unseen_label_rows": int((~all_labels_seen).sum()),
            "unseen_exact_set_rows": int((~exact_set_seen).sum()),
        },
        "retrieval": {
            "family_frequency": evaluate_candidates(val_keys, family_frequency),
            "family_source_frequency": evaluate_candidates(val_keys, family_source_frequency),
            "cosine_knn": evaluate_candidates(val_keys, knn),
            "family_first_cosine_knn": evaluate_candidates(val_keys, family_knn),
            "hybrid": evaluate_candidates(val_keys, hybrid),
        },
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
