#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synthmind.chemistry.families import family_feature_names, family_feature_vector, assign_cation_family  # noqa: E402
from training.family.build_full_database_split import (  # noqa: E402
    SOURCE_SPLITS,
    attach_family_metadata,
    build_group_folds,
    sha256_file,
    write_json,
)


def parse_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = "" if value is None else str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "[]"}:
        return []
    for loader in (json.loads, ast.literal_eval):
        try:
            result = loader(text)
            if isinstance(result, (list, tuple, set)):
                return [str(item).strip() for item in result if str(item).strip()]
        except Exception:
            pass
    return [item.strip() for item in text.split(";") if item.strip()]


def atmosphere_coarse(value: Any) -> str:
    text = "" if value is None else str(value).strip().lower()
    if not text or text in {"nan", "none", "unknown", "<unk_or_missing>"}:
        return "<UNK_OR_MISSING>"
    if any(token in text for token in ("air", "oxygen", "o2", "oxid")):
        return "air_or_oxidizing"
    if any(token in text for token in ("redu", "hydrogen", "h2", "forming gas", "carbon monoxide")):
        return "reducing"
    if "vacuum" in text:
        return "vacuum"
    if any(token in text for token in ("argon", " ar", "nitrogen", " n2", "inert", "helium")):
        return "inert"
    return "<UNK_OR_MISSING>"


def standardize_base_features(
    values: np.ndarray, family_values: np.ndarray, train_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    means = np.nanmean(values[train_indices], axis=0).astype(np.float32)
    means = np.where(np.isfinite(means), means, 0.0).astype(np.float32)
    filled = np.where(np.isfinite(values), values, means[None, :]).astype(np.float32)
    stds = np.std(filled[train_indices], axis=0).astype(np.float32)
    stds = np.where(np.isfinite(stds) & (stds > 1e-8), stds, 1.0).astype(np.float32)
    scaled = ((filled - means[None, :]) / stds[None, :]).astype(np.float32)
    return (
        np.hstack([values, family_values]).astype(np.float32),
        np.hstack([scaled, family_values]).astype(np.float32),
        means,
        stds,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full-database Stage3 cation-family dataset.")
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--source_schema", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--n_folds", type=int, default=10)
    parser.add_argument("--val_fold", type=int, default=5)
    parser.add_argument("--test_fold", type=int, default=8)
    parser.add_argument(
        "--precursor_column", default="predicted_precursor_set_chem_checked"
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir).expanduser().resolve()
    schema_path = Path(args.source_schema).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    base_feature_cols = list(source_schema["feature_cols"])
    schema_precursor_vocab = [str(item) for item in source_schema["precursor_vocab"]]

    frames: List[pd.DataFrame] = []
    input_files: List[Path] = []
    for old_split in SOURCE_SPLITS:
        path = source_dir / f"{old_split}.csv"
        frame = pd.read_csv(path, low_memory=False)
        frame["old_split"] = old_split
        frames.append(frame)
        input_files.append(path)
    meta = pd.concat(frames, ignore_index=True)
    if meta["sample_id"].astype(str).duplicated().any():
        raise ValueError("duplicate Stage3 sample_id values after full-database merge")
    if "id" not in meta:
        meta["id"] = meta["sample_id"]

    # The precursor set is an inference-time input, not a prediction target.  It is
    # therefore safe (and necessary) to define its vocabulary from the complete
    # database.  The legacy Stage3 schema omitted a non-trivial number of valid
    # precursors, which silently erased useful process-conditioning information.
    observed_precursors = {
        label
        for value in meta[str(args.precursor_column)].tolist()
        for label in parse_list(value)
    }
    precursor_vocab = [
        *schema_precursor_vocab,
        *sorted(observed_precursors.difference(schema_precursor_vocab)),
    ]
    precursor_to_index = {name: index for index, name in enumerate(precursor_vocab)}

    meta, family_matrix, added_family_cols = attach_family_metadata(meta)
    meta["split_fold"] = build_group_folds(meta, seed=int(args.seed), n_folds=int(args.n_folds))
    meta["split"] = "train"
    meta.loc[meta["split_fold"] == int(args.val_fold), "split"] = "val"
    meta.loc[meta["split_fold"] == int(args.test_fold), "split"] = "test"
    split_indices = {
        split: np.flatnonzero(meta["split"].to_numpy() == split) for split in SOURCE_SPLITS
    }

    missing_features = [column for column in base_feature_cols if column not in meta]
    if missing_features:
        raise ValueError(f"Stage3 source lacks schema features: {missing_features[:20]}")
    base_values = meta[base_feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    x_raw, x, means, stds = standardize_base_features(
        base_values, family_matrix, split_indices["train"]
    )

    y_set = np.zeros((len(meta), len(precursor_vocab)), dtype=np.float32)
    oov_counts: Dict[str, int] = {}
    for row_index, value in enumerate(meta[str(args.precursor_column)].tolist()):
        for label in parse_list(value):
            label_index = precursor_to_index.get(label)
            if label_index is None:
                oov_counts[label] = oov_counts.get(label, 0) + 1
            else:
                y_set[row_index, label_index] = 1.0

    continuous_names = ["temperature_c", "time_h"]
    y_cont_raw = np.zeros((len(meta), 2), dtype=np.float32)
    y_cont_mask = np.zeros((len(meta), 2), dtype=np.float32)
    continuous_schema: Dict[str, Dict[str, float]] = {}
    for column_index, column in enumerate(continuous_names):
        values = pd.to_numeric(meta[column], errors="coerce").to_numpy(np.float32)
        mask_column = "has_temperature_c" if column == "temperature_c" else "has_time_h"
        mask = np.isfinite(values)
        if mask_column in meta:
            mask &= pd.to_numeric(meta[mask_column], errors="coerce").fillna(0).to_numpy() > 0.5
        train_mask = mask & (meta["split"].to_numpy() == "train")
        mean = float(np.mean(values[train_mask]))
        std = float(np.std(values[train_mask]))
        if not np.isfinite(std) or std <= 1e-8:
            std = 1.0
        y_cont_raw[:, column_index] = np.where(mask, values, 0.0)
        y_cont_mask[:, column_index] = mask.astype(np.float32)
        continuous_schema[column] = {
            "mean": mean,
            "std": std,
            "median": float(np.median(values[train_mask])),
        }
    y_cont_norm = y_cont_raw.copy()
    for column_index, column in enumerate(continuous_names):
        stats = continuous_schema[column]
        valid = y_cont_mask[:, column_index] > 0.5
        y_cont_norm[valid, column_index] = (
            y_cont_raw[valid, column_index] - stats["mean"]
        ) / stats["std"]

    atmosphere_vocab = ["<UNK_OR_MISSING>", "air_or_oxidizing", "inert", "reducing", "vacuum"]
    method_vocab = sorted(meta["reaction_method"].fillna("other").astype(str).unique().tolist())
    atmosphere_labels = meta["atmosphere"].map(atmosphere_coarse)
    y_discrete = np.column_stack(
        [
            atmosphere_labels.map({value: index for index, value in enumerate(atmosphere_vocab)}),
            meta["reaction_method"].fillna("other").astype(str).map(
                {value: index for index, value in enumerate(method_vocab)}
            ),
        ]
    ).astype(np.int64)
    y_discrete_mask = np.column_stack(
        [
            (atmosphere_labels != "<UNK_OR_MISSING>").astype(np.float32),
            np.ones(len(meta), dtype=np.float32),
        ]
    )

    for split, indices in split_indices.items():
        split_meta = meta.iloc[indices].reset_index(drop=True)
        split_meta.to_csv(output_dir / f"{split}_meta.csv", index=False)
        np.savez_compressed(
            output_dir / f"{split}.npz",
            x_raw=x_raw[indices],
            x=x[indices],
            y_set=y_set[indices],
            y_cond_continuous=y_cont_norm[indices],
            y_cond_continuous_raw=y_cont_raw[indices],
            y_cond_continuous_mask=y_cont_mask[indices],
            y_cond_discrete=y_discrete[indices],
            y_cond_discrete_mask=y_discrete_mask[indices],
            sample_id=split_meta["sample_id"].astype(str).to_numpy(),
        )

    output_schema = {
        "schema_version": "stage3_full_cation_family_v1",
        "family_schema_version": "target_cation_family_v1",
        "feature_cols": [*base_feature_cols, *added_family_cols],
        "base_feature_count": len(base_feature_cols),
        "family_feature_count": len(added_family_cols),
        "precursor_vocab": precursor_vocab,
        "precursor_vocab_base_count": len(schema_precursor_vocab),
        "precursor_vocab_added_from_full_input": len(precursor_vocab) - len(schema_precursor_vocab),
        "continuous_cols": continuous_names,
        "continuous_schema": continuous_schema,
        "discrete_cols": ["atmosphere_coarse", "reaction_method"],
        "discrete_schema": {
            "atmosphere_coarse": {"vocab": atmosphere_vocab, "missing_index": 0},
            "reaction_method": {"vocab": method_vocab},
        },
        "x_scaler": {"mean": means.tolist(), "std": stds.tolist()},
        "precursor_column": str(args.precursor_column),
        "precursor_oov_unique": len(oov_counts),
        "precursor_oov_occurrences": int(sum(oov_counts.values())),
        "precursor_oov_top100": dict(sorted(oov_counts.items(), key=lambda item: -item[1])[:100]),
    }
    write_json(output_dir / "schema.json", output_schema)

    group_sets = {
        split: set(meta.loc[meta["split"] == split, "family_group_key"].astype(str))
        for split in SOURCE_SPLITS
    }
    manifest = {
        "split_version": "stage3_full_database_cation_family_v1",
        "seed": int(args.seed),
        "val_fold": int(args.val_fold),
        "test_fold": int(args.test_fold),
        "split_rows": {split: int(len(indices)) for split, indices in split_indices.items()},
        "formula_group_intersections": {
            "train_val": len(group_sets["train"] & group_sets["val"]),
            "train_test": len(group_sets["train"] & group_sets["test"]),
            "val_test": len(group_sets["val"] & group_sets["test"]),
        },
        "input_sha256": {str(path): sha256_file(path) for path in [*input_files, schema_path]},
    }
    write_json(output_dir / "split_manifest.json", manifest)
    print(json.dumps({"manifest": manifest, "schema_summary": {
        "n_features": len(output_schema["feature_cols"]),
        "n_precursors": len(precursor_vocab),
        "precursor_oov_occurrences": output_schema["precursor_oov_occurrences"],
    }}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
