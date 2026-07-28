#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, jaccard_score


def load_pack(path: Path) -> Dict[str, np.ndarray]:
    pack = np.load(path, allow_pickle=True)
    return {key: pack[key] for key in pack.files}


def normalized_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-8, None)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    exact = np.all(y_true == y_pred, axis=1)
    return {
        "n": int(len(y_true)),
        "exact_accuracy": float(exact.mean()) if len(exact) else 0.0,
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "samples_jaccard": float(
            jaccard_score(y_true, y_pred, average="samples", zero_division=0)
        ),
        "mean_pred_labels": float(y_pred.sum(axis=1).mean()),
    }


def predict_family_knn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    family_train: np.ndarray,
    x_query: np.ndarray,
    family_query: np.ndarray,
    n_neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    output = np.zeros((len(x_query), y_train.shape[1]), dtype=np.int8)
    nearest_similarity = np.full(len(x_query), np.nan, dtype=np.float32)
    support = np.zeros(len(x_query), dtype=np.int32)

    train_norm = normalized_rows(x_train)
    query_norm = normalized_rows(x_query)
    train_by_family: Dict[str, np.ndarray] = {}
    for family in np.unique(family_train.astype(str)):
        train_by_family[str(family)] = np.flatnonzero(family_train.astype(str) == str(family))

    for family in np.unique(family_query.astype(str)):
        query_indices = np.flatnonzero(family_query.astype(str) == str(family))
        train_indices = train_by_family.get(str(family))
        if train_indices is None or len(train_indices) == 0:
            continue
        support[query_indices] = int(len(train_indices))
        k = min(int(n_neighbors), len(train_indices))
        similarities = query_norm[query_indices] @ train_norm[train_indices].T
        nearest_similarity[query_indices] = similarities.max(axis=1)
        local_top = np.argpartition(-similarities, kth=k - 1, axis=1)[:, :k]
        for local_row, (row_index, neighbor_local) in enumerate(zip(query_indices, local_top)):
            neighbor_indices = train_indices[neighbor_local]
            weights = np.clip(similarities[local_row, neighbor_local], 0.0, None) + 1e-6
            label_scores = (y_train[neighbor_indices] * weights[:, None]).sum(axis=0)
            cardinalities = y_train[neighbor_indices].sum(axis=1)
            cardinality = int(np.clip(np.rint(np.average(cardinalities, weights=weights)), 1, y_train.shape[1]))
            best_labels = np.argpartition(-label_scores, kth=cardinality - 1)[:cardinality]
            output[row_index, best_labels] = 1
    return output, nearest_similarity, support


def labels_from_multihot(values: np.ndarray, names: Sequence[str]) -> List[str]:
    return [str(names[index]) for index in np.flatnonzero(values) if index < len(names)]


def parse_label_set(value: Any) -> set[str]:
    text = str(value)
    for loader in (json.loads, ast.literal_eval):
        try:
            result = loader(text)
            if isinstance(result, list):
                return {str(item) for item in result}
        except Exception:
            pass
    return set()


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune and evaluate same-cation-family KNN retrieval.")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--global_prediction_csv", default="")
    parser.add_argument("--base_feature_count", type=int, default=195)
    parser.add_argument("--neighbor_grid", default="1,3,5,10,20")
    args = parser.parse_args()

    dataset = Path(args.dataset_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    packs = {split: load_pack(dataset / f"{split}.npz") for split in ("train", "val", "test")}
    metadata = {
        split: pd.read_csv(dataset / f"{split}_meta.csv") for split in ("train", "val", "test")
    }
    names = json.loads((dataset / "precursor_names.json").read_text(encoding="utf-8"))

    x_train = np.asarray(packs["train"]["x"], dtype=np.float32)[:, : int(args.base_feature_count)]
    y_train = (np.asarray(packs["train"]["y_multi_hot"]) > 0).astype(np.int8)
    family_train = metadata["train"]["family_signature_primary"].astype(str).to_numpy()
    neighbor_grid = [int(value) for value in str(args.neighbor_grid).split(",") if value.strip()]

    validation_results: Dict[str, Any] = {}
    predictions: Dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for neighbors in neighbor_grid:
        pred, similarity, support = predict_family_knn(
            x_train,
            y_train,
            family_train,
            np.asarray(packs["val"]["x"], dtype=np.float32)[:, : int(args.base_feature_count)],
            metadata["val"]["family_signature_primary"].astype(str).to_numpy(),
            neighbors,
        )
        score = metrics((packs["val"]["y_multi_hot"] > 0).astype(np.int8), pred)
        score["unseen_family_rows"] = int(np.sum(support == 0))
        validation_results[str(neighbors)] = score
        predictions[neighbors] = (pred, similarity, support)

    best_neighbors = max(neighbor_grid, key=lambda value: validation_results[str(value)]["samples_f1"])
    test_pred, similarity, support = predict_family_knn(
        x_train,
        y_train,
        family_train,
        np.asarray(packs["test"]["x"], dtype=np.float32)[:, : int(args.base_feature_count)],
        metadata["test"]["family_signature_primary"].astype(str).to_numpy(),
        best_neighbors,
    )
    y_test = (packs["test"]["y_multi_hot"] > 0).astype(np.int8)
    test_metrics = metrics(y_test, test_pred)
    test_metrics["unseen_family_rows"] = int(np.sum(support == 0))

    out = metadata["test"].copy()
    out["true_labels"] = [json.dumps(labels_from_multihot(row, names), ensure_ascii=False) for row in y_test]
    out["pred_labels"] = [json.dumps(labels_from_multihot(row, names), ensure_ascii=False) for row in test_pred]
    out["family_knn_neighbors"] = int(best_neighbors)
    out["family_knn_similarity"] = similarity
    out["family_train_support"] = support
    out.to_csv(output / "family_knn_test_predictions.csv", index=False)

    result: Dict[str, Any] = {
        "selected_neighbors": int(best_neighbors),
        "validation_grid": validation_results,
        "test_metrics": test_metrics,
    }
    if str(args.global_prediction_csv).strip():
        global_df = pd.read_csv(Path(args.global_prediction_csv).expanduser().resolve())
        if len(global_df) != len(out):
            raise ValueError("global prediction row count does not match test split")
        global_sets = [parse_label_set(value) for value in global_df["pred_labels"]]
        knn_sets = [parse_label_set(value) for value in out["pred_labels"]]
        true_sets = [parse_label_set(value) for value in out["true_labels"]]
        result["candidate_union_oracle"] = {
            "exact_hit_at_2": float(
                np.mean([truth == global_pred or truth == knn for truth, global_pred, knn in zip(true_sets, global_sets, knn_sets)])
            ),
            "note": "Diagnostic only: either the global top1 or family-KNN top1 exactly matches.",
        }

    (output / "family_knn_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
