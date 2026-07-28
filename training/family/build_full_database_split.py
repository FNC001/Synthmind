#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synthmind.chemistry.families import (  # noqa: E402
    SCHEMA_VERSION,
    assign_cation_family,
    family_feature_names,
    family_feature_vector,
)


SOURCE_SPLITS = ("train", "val", "test")
COPY_ARTIFACTS = (
    "action_vocab.json",
    "action_to_id.json",
    "precursor_names.json",
    "label_cols.json",
    "label_names.json",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_list(values: Sequence[str]) -> str:
    return json.dumps(list(values), ensure_ascii=False)


def choose_group_key(row: Mapping[str, Any]) -> str:
    for key in ("split_group", "doi", "synth_uid", "material_id", "formula", "id"):
        value = row.get(key)
        if value is not None and str(value).strip() and str(value).lower() != "nan":
            return str(value).strip()
    raise ValueError(f"cannot determine split group for row id={row.get('id')!r}")


def _valid_token(value: Any) -> bool:
    text = "" if value is None else str(value).strip()
    return bool(text and text.lower() not in {"nan", "none", "null"})


def attach_leakage_groups(meta: pd.DataFrame) -> pd.DataFrame:
    """Keep every canonical target composition in exactly one split."""
    keys: List[str] = []
    for row in meta.to_dict("records"):
        if _valid_token(row.get("canonical_formula")):
            token = "formula::" + str(row["canonical_formula"]).strip()
        elif _valid_token(row.get("material_id")):
            token = "material::" + str(row["material_id"]).strip().lower()
        else:
            reaction_value = (
                row.get("split_group")
                if _valid_token(row.get("split_group"))
                else row.get("doi")
            )
            if _valid_token(reaction_value):
                token = "reaction::" + str(reaction_value).strip().lower()
            else:
                token = "sample::" + str(row.get("id", len(keys)))
        keys.append("leakgrp_v1__" + hashlib.sha1(token.encode("utf-8")).hexdigest()[:16])
    out = meta.copy()
    out["family_group_key"] = keys
    return out


def load_source(source_dir: Path) -> tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    metadata: List[pd.DataFrame] = []
    arrays: Dict[str, List[np.ndarray]] = {}
    expected_keys: set[str] | None = None

    for old_split in SOURCE_SPLITS:
        meta_path = source_dir / f"{old_split}_meta.csv"
        npz_path = source_dir / f"{old_split}.npz"
        if not meta_path.exists() or not npz_path.exists():
            raise FileNotFoundError(f"missing source split files: {meta_path}, {npz_path}")

        meta = pd.read_csv(meta_path)
        pack_file = np.load(npz_path, allow_pickle=True)
        pack = {key: pack_file[key] for key in pack_file.files}
        if expected_keys is None:
            expected_keys = set(pack)
        elif set(pack) != expected_keys:
            raise ValueError(
                f"NPZ key mismatch for {old_split}: expected={sorted(expected_keys)}, "
                f"got={sorted(pack)}"
            )
        if any(value.shape[0] != len(meta) for value in pack.values()):
            shapes = {key: value.shape for key, value in pack.items()}
            raise ValueError(f"row mismatch for {old_split}: meta={len(meta)}, arrays={shapes}")

        meta = meta.copy()
        meta["old_split"] = old_split
        meta["old_row_index"] = np.arange(len(meta), dtype=np.int64)
        metadata.append(meta)
        for key, value in pack.items():
            arrays.setdefault(key, []).append(value)

    merged_meta = pd.concat(metadata, ignore_index=True)
    merged_arrays = {key: np.concatenate(values, axis=0) for key, values in arrays.items()}
    if merged_meta["id"].astype(str).duplicated().any():
        duplicated = merged_meta.loc[
            merged_meta["id"].astype(str).duplicated(keep=False), "id"
        ].astype(str).head(20).tolist()
        raise ValueError(f"duplicate sample ids in full database: {duplicated}")
    return merged_meta, merged_arrays


def attach_family_metadata(meta: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, List[str]]:
    if "formula" not in meta.columns:
        raise ValueError("metadata must contain formula")

    rows: List[Dict[str, Any]] = []
    vectors: List[List[float]] = []
    failures: List[str] = []
    for index, formula in enumerate(meta["formula"].tolist()):
        try:
            assignment = assign_cation_family(formula)
        except Exception as exc:
            failures.append(f"row={index} formula={formula!r}: {exc}")
            continue
        item = assignment.to_dict()
        for key in ("target_elements", "target_cation_elements", "target_anion_elements"):
            item[key] = json_list(item[key])
        rows.append(item)
        vectors.append(family_feature_vector(assignment))

    if failures:
        preview = "\n".join(failures[:20])
        raise ValueError(f"family parsing failed for {len(failures)} rows:\n{preview}")

    family_df = pd.DataFrame(rows).drop(columns=["input_formula"])
    out = pd.concat([meta.reset_index(drop=True), family_df], axis=1)
    out = attach_leakage_groups(out)
    return out, np.asarray(vectors, dtype=np.float32), family_feature_names()


def mark_quality(meta: pd.DataFrame, gold_meta_path: Path | None, relaxed_weight: float) -> pd.DataFrame:
    out = meta.copy()
    gold_ids: set[str] = set()
    if gold_meta_path is not None and gold_meta_path.exists():
        gold_meta = pd.read_csv(gold_meta_path, usecols=["id"])
        gold_ids = set(gold_meta["id"].astype(str))
    is_gold = out["id"].astype(str).isin(gold_ids)
    out["quality_tier"] = np.where(is_gold, "gold", "relaxed")
    out["quality_weight"] = np.where(is_gold, 1.0, float(relaxed_weight))
    return out


def build_group_folds(meta: pd.DataFrame, seed: int, n_folds: int) -> pd.Series:
    work = meta.copy()
    source = work.get("source_dataset", pd.Series("UNKNOWN", index=work.index)).fillna("UNKNOWN")
    method_column = "reaction_method" if "reaction_method" in work.columns else "synthesis_type"
    method = work.get(method_column, pd.Series("UNKNOWN", index=work.index)).fillna("UNKNOWN")
    work["_raw_stratum"] = (
        work["family_signature_primary"].astype(str)
        + "||"
        + source.astype(str)
        + "||"
        + method.astype(str)
    )
    minimum = int(n_folds)
    family_source = work["family_signature_primary"].astype(str) + "||" + source.astype(str)
    family = work["family_signature_primary"].astype(str)

    def group_support(labels: pd.Series) -> Dict[str, int]:
        frame = pd.DataFrame(
            {"label": labels.astype(str), "group": work["family_group_key"].astype(str)}
        ).drop_duplicates()
        return frame.groupby("label")["group"].nunique().astype(int).to_dict()

    raw_support = group_support(work["_raw_stratum"])
    family_source_support = group_support(family_source)
    family_support = group_support(family)

    strata: List[str] = []
    for index in range(len(work)):
        raw = str(work["_raw_stratum"].iloc[index])
        fs = str(family_source.iloc[index])
        family_value = str(family.iloc[index])
        if int(raw_support.get(raw, 0)) >= minimum:
            strata.append("RAW||" + raw)
        elif int(family_source_support.get(fs, 0)) >= minimum:
            strata.append("FAMILY_SOURCE||" + fs)
        elif int(family_support.get(family_value, 0)) >= minimum:
            strata.append("FAMILY||" + family_value)
        else:
            strata.append("RARE")

    strata_series = pd.Series(strata, index=work.index, dtype=str)
    preliminary_support = group_support(strata_series)
    strata_series = strata_series.map(
        lambda value: value if preliminary_support.get(str(value), 0) >= minimum else "RARE"
    )
    final_support = group_support(strata_series)
    if min(final_support.values()) < minimum:
        raise RuntimeError(f"rare-stratum collapse failed: {final_support}")

    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=int(seed))
    folds = np.full(len(work), -1, dtype=np.int16)
    dummy = np.zeros(len(work), dtype=np.int8)
    for fold, (_, fold_indices) in enumerate(
        splitter.split(
            dummy,
            strata_series.to_numpy(),
            groups=work["family_group_key"].astype(str).to_numpy(),
        )
    ):
        folds[fold_indices] = int(fold)
    if (folds < 0).any():
        raise RuntimeError("some rows were not assigned a fold")
    return pd.Series(folds, index=meta.index, dtype=int)


def append_and_standardize_features(
    arrays: Dict[str, np.ndarray],
    family_features: np.ndarray,
    train_indices: np.ndarray,
) -> Dict[str, Any]:
    if "x_raw" not in arrays:
        raise KeyError("source NPZ files must contain x_raw to refit preprocessing without leakage")
    raw = np.asarray(arrays["x_raw"], dtype=np.float32)
    if raw.ndim != 2 or family_features.shape[0] != raw.shape[0]:
        raise ValueError(f"invalid feature shapes: x_raw={raw.shape}, family={family_features.shape}")
    augmented_raw = np.hstack([raw, family_features]).astype(np.float32)

    # Fit preprocessing on the original continuous/descriptor features only.
    # Sparse family indicators stay in their natural 0/1 scale; standardizing
    # rare families would create very large values and destabilize training.
    base_means = np.nanmean(raw[train_indices], axis=0).astype(np.float32)
    base_means = np.where(np.isfinite(base_means), base_means, 0.0).astype(np.float32)
    filled_base = np.where(np.isfinite(raw), raw, base_means[None, :]).astype(np.float32)
    base_stds = np.std(filled_base[train_indices], axis=0).astype(np.float32)
    base_stds = np.where(
        np.isfinite(base_stds) & (base_stds > 1e-8), base_stds, 1.0
    ).astype(np.float32)
    scaled_base = ((filled_base - base_means[None, :]) / base_stds[None, :]).astype(np.float32)
    scaled = np.hstack([scaled_base, family_features]).astype(np.float32)
    means = np.concatenate(
        [base_means, np.zeros(family_features.shape[1], dtype=np.float32)]
    )
    stds = np.concatenate(
        [base_stds, np.ones(family_features.shape[1], dtype=np.float32)]
    )

    arrays = dict(arrays)
    arrays["x_raw"] = augmented_raw
    arrays["x"] = scaled
    return {"arrays": arrays, "mean": means, "std": stds}


def family_counts(meta: pd.DataFrame) -> Dict[str, int]:
    return {
        str(key): int(value)
        for key, value in meta["family_signature_primary"].value_counts().sort_index().items()
    }


def split_summary(meta: pd.DataFrame) -> Dict[str, Any]:
    return {
        "n_rows": int(len(meta)),
        "n_groups": int(meta["family_group_key"].nunique()),
        "n_families": int(meta["family_signature_primary"].nunique()),
        "n_formulas": int(meta["formula"].nunique()),
        "source_dataset": {
            str(key): int(value) for key, value in meta["source_dataset"].value_counts().items()
        },
        "synthesis_type": {
            str(key): int(value)
            for key, value in meta["synthesis_type"].fillna("UNKNOWN").value_counts().items()
        },
        "quality_tier": {
            str(key): int(value) for key, value in meta["quality_tier"].value_counts().items()
        },
        "family_counts": family_counts(meta),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge the complete Stage2 database and create a leakage-safe cation-family split."
    )
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gold_meta", default="")
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--n_folds", type=int, default=10)
    parser.add_argument("--val_fold", type=int, default=0)
    parser.add_argument("--test_fold", type=int, default=1)
    parser.add_argument("--relaxed_weight", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    meta, arrays = load_source(source_dir)
    meta, family_features, added_feature_names = attach_family_metadata(meta)
    gold_meta = Path(args.gold_meta).expanduser().resolve() if str(args.gold_meta).strip() else None
    meta = mark_quality(meta, gold_meta, relaxed_weight=float(args.relaxed_weight))
    meta["split_fold"] = build_group_folds(meta, seed=int(args.seed), n_folds=int(args.n_folds))
    meta["split"] = "train"
    meta.loc[meta["split_fold"] == int(args.val_fold), "split"] = "val"
    meta.loc[meta["split_fold"] == int(args.test_fold), "split"] = "test"

    split_indices = {
        split: np.flatnonzero(meta["split"].to_numpy() == split)
        for split in SOURCE_SPLITS
    }
    standardized = append_and_standardize_features(arrays, family_features, split_indices["train"])
    arrays = standardized["arrays"]

    for split, indices in split_indices.items():
        split_meta = meta.iloc[indices].reset_index(drop=True)
        split_meta.to_csv(output_dir / f"{split}_meta.csv", index=False)
        assignment_cols = [
            "id",
            "material_id",
            "formula",
            "family_group_key",
            "family_signature_primary",
            "family_id_primary",
            "family_routing_level",
            "target_cation_elements",
            "target_anion_elements",
            "quality_tier",
            "quality_weight",
            "old_split",
            "split",
        ]
        split_meta[[column for column in assignment_cols if column in split_meta]].to_csv(
            output_dir / f"family_assignments_{split}.csv", index=False
        )
        np.savez_compressed(
            output_dir / f"{split}.npz",
            **{key: value[indices] for key, value in arrays.items()},
        )

    for artifact in COPY_ARTIFACTS:
        source = source_dir / artifact
        if source.exists():
            shutil.copy2(source, output_dir / artifact)

    old_feature_cols_path = source_dir / "feature_cols.json"
    if old_feature_cols_path.exists():
        old_feature_cols = json.loads(old_feature_cols_path.read_text(encoding="utf-8"))
    else:
        old_feature_cols = [f"feature_{i}" for i in range(arrays["x_raw"].shape[1] - len(added_feature_names))]
    feature_cols = [*list(old_feature_cols), *added_feature_names]
    write_json(output_dir / "feature_cols.json", feature_cols)
    write_json(
        output_dir / "scaler.json",
        {
            "mean": standardized["mean"].tolist(),
            "std": standardized["std"].tolist(),
            "fit_split": "train",
            "feature_cols": feature_cols,
        },
    )

    source_summary_path = source_dir / "summary.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    source_schema = dict(source_summary.get("schema", {}))
    source_schema.update(
        {
            "n_features": int(arrays["x"].shape[1]),
            "feature_cols_path": str((output_dir / "feature_cols.json").resolve()),
            "family_schema_version": SCHEMA_VERSION,
            "family_feature_count": int(len(added_feature_names)),
            "family_feature_names": added_feature_names,
        }
    )
    summary = {
        "config": vars(args),
        "schema": source_schema,
        "full_database": {
            "n_rows": int(len(meta)),
            "n_unique_ids": int(meta["id"].astype(str).nunique()),
            "n_unique_groups": int(meta["family_group_key"].nunique()),
            "n_primary_families": int(meta["family_signature_primary"].nunique()),
            "family_schema_version": SCHEMA_VERSION,
        },
        "splits": {
            split: split_summary(meta.loc[meta["split"] == split]) for split in SOURCE_SPLITS
        },
    }
    write_json(output_dir / "summary.json", summary)

    group_sets = {
        split: set(meta.loc[meta["split"] == split, "family_group_key"].astype(str))
        for split in SOURCE_SPLITS
    }
    leakage = {
        "train_val": len(group_sets["train"] & group_sets["val"]),
        "train_test": len(group_sets["train"] & group_sets["test"]),
        "val_test": len(group_sets["val"] & group_sets["test"]),
    }
    if any(leakage.values()):
        raise RuntimeError(f"group leakage detected: {leakage}")

    def cross_split_counts(column: str) -> Dict[str, int]:
        value_sets = {
            split: {
                str(value).strip().lower()
                for value in meta.loc[meta["split"] == split, column].tolist()
                if _valid_token(value)
            }
            for split in SOURCE_SPLITS
        }
        return {
            "train_val": len(value_sets["train"] & value_sets["val"]),
            "train_test": len(value_sets["train"] & value_sets["test"]),
            "val_test": len(value_sets["val"] & value_sets["test"]),
        }

    overlap_audit = {
        "canonical_formula": cross_split_counts("canonical_formula"),
        "material_id": cross_split_counts("material_id"),
        "doi": cross_split_counts("doi"),
    }

    input_files = [
        *[source_dir / f"{split}.npz" for split in SOURCE_SPLITS],
        *[source_dir / f"{split}_meta.csv" for split in SOURCE_SPLITS],
    ]
    if gold_meta is not None and gold_meta.exists():
        input_files.append(gold_meta)
    manifest = {
        "split_version": "full_database_cation_family_v1",
        "family_schema_version": SCHEMA_VERSION,
        "seed": int(args.seed),
        "n_folds": int(args.n_folds),
        "val_fold": int(args.val_fold),
        "test_fold": int(args.test_fold),
        "group_key": "canonical_formula_then_material_id_then_reaction_group",
        "stratification": ["family_signature_primary", "source_dataset", "synthesis_type"],
        "leakage_group_intersections": leakage,
        "cross_split_overlap_audit": overlap_audit,
        "input_sha256": {str(path): sha256_file(path) for path in input_files},
        "split_row_counts": {split: int(len(indices)) for split, indices in split_indices.items()},
    }
    write_json(output_dir / "split_manifest.json", manifest)
    print(json.dumps({"summary": summary["full_database"], "manifest": manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
