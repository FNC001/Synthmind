#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
27_build_stage3_condition_dataset_v5_mixed.py

Mixed-type extension template of 27_build_stage3_condition_dataset_v4.py.

Goal
----
Keep your original v4 feature-building / merge / split logic unchanged, and
add mixed-type target export on top of it.

First-version targets
---------------------
continuous_cols:
- target_temperature_c_clean
- target_time_h_clean

discrete_cols:
- target_atmosphere_coarse
- synthesis_type

Output NPZ keys
---------------
- x
- y_set
- y_cond_continuous
- y_cond_continuous_mask
- y_cond_discrete
- y_cond_discrete_mask
- sample_id
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Utils
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def normalize_key(s: str) -> str:
    return str(s).strip().lower()


def parse_precursor_value(v: Any) -> List[str]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                obj = json.loads(s)
                if isinstance(obj, list):
                    return [str(x).strip() for x in obj if str(x).strip()]
            except Exception:
                pass
        for sep in [";", "|", ","]:
            if sep in s:
                return [x.strip() for x in s.split(sep) if x.strip()]
        return [s]
    return [str(v).strip()]


def dedup_keep_order(values: Sequence[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def collect_precursors_from_row(row: pd.Series, precursor_cols: Sequence[str]) -> List[str]:
    vals: List[str] = []
    for c in precursor_cols:
        if c in row.index:
            vals.extend(parse_precursor_value(row[c]))
    return dedup_keep_order(vals)


def build_precursor_vocab(train_df: pd.DataFrame, precursor_cols: Sequence[str]) -> List[str]:
    vocab: set = set()
    for _, row in train_df.iterrows():
        vocab.update(collect_precursors_from_row(row, precursor_cols))
    return sorted(vocab)


def precursor_to_multihot(values: List[str], vocab_index: Dict[str, int]) -> np.ndarray:
    y = np.zeros(len(vocab_index), dtype=np.float32)
    for name in values:
        idx = vocab_index.get(name)
        if idx is not None:
            y[idx] = 1.0
    return y


def infer_feature_cols(df: pd.DataFrame, exclude_cols: Sequence[str]) -> List[str]:
    exact = {normalize_key(x) for x in exclude_cols if not x.endswith("_")}
    prefixes = tuple(normalize_key(x) for x in exclude_cols if x.endswith("_"))
    cols: List[str] = []
    for c in df.columns:
        nk = normalize_key(c)
        if nk in exact:
            continue
        if prefixes and nk.startswith(prefixes):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def coerce_float_array(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    if not cols:
        return np.zeros((len(df), 0), dtype=np.float32)
    arr = df.loc[:, cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def norm_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "null", "na", "n/a"}:
        return ""
    return s


def safe_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        if pd.isna(x):
            return None
        return float(x)

    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "null", "na", "n/a"}:
        return None

    s = (
        s.replace("°c", "")
         .replace("℃", "")
         .replace("hours", "")
         .replace("hour", "")
         .replace("hrs", "")
         .replace("hr", "")
         .replace("h", "")
         .replace(",", "")
         .strip()
    )
    try:
        return float(s)
    except Exception:
        return None


# ============================================================
# Schema helpers
# ============================================================

def build_continuous_stats(train_df: pd.DataFrame, continuous_cols: List[str]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for col in continuous_cols:
        vals = [safe_float(x) for x in train_df[col].tolist()] if col in train_df.columns else []
        vals = [v for v in vals if v is not None]
        if len(vals) == 0:
            stats[col] = {"mean": 0.0, "std": 1.0, "median": 0.0}
        else:
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            if not np.isfinite(std) or std < 1e-8:
                std = 1.0
            stats[col] = {
                "mean": mean,
                "std": std,
                "median": float(np.median(vals)),
            }
    return stats


def build_vocab(train_df: pd.DataFrame, col: str) -> Tuple[List[str], Dict[str, int]]:
    vals = [norm_str(x) for x in train_df[col].tolist()] if col in train_df.columns else []
    vals = [x for x in vals if x]
    vocab = ["<UNK_OR_MISSING>"] + sorted(set(vals))
    stoi = {tok: i for i, tok in enumerate(vocab)}
    return vocab, stoi


def build_discrete_vocabs(
    train_df: pd.DataFrame,
    discrete_cols: List[str],
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, int]]]:
    vocabs: Dict[str, List[str]] = {}
    stoi_all: Dict[str, Dict[str, int]] = {}
    for col in discrete_cols:
        vocab, stoi = build_vocab(train_df, col)
        vocabs[col] = vocab
        stoi_all[col] = stoi
    return vocabs, stoi_all


# ============================================================
# Target encoders
# ============================================================

def encode_continuous_targets(
    df: pd.DataFrame,
    continuous_cols: List[str],
    cont_stats: Dict[str, Dict[str, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(df)
    n_cont = len(continuous_cols)

    y = np.zeros((n, n_cont), dtype=np.float32)
    m = np.zeros((n, n_cont), dtype=np.float32)

    for j, col in enumerate(continuous_cols):
        mean = cont_stats[col]["mean"]
        std = cont_stats[col]["std"]
        for i, val in enumerate(df[col].tolist() if col in df.columns else [None] * n):
            fv = safe_float(val)
            if fv is None:
                continue
            y[i, j] = (float(fv) - mean) / std
            m[i, j] = 1.0

    return y, m


def encode_discrete_targets(
    df: pd.DataFrame,
    discrete_cols: List[str],
    discrete_stoi: Dict[str, Dict[str, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(df)
    n_disc = len(discrete_cols)

    y = np.zeros((n, n_disc), dtype=np.int64)
    m = np.zeros((n, n_disc), dtype=np.float32)

    for j, col in enumerate(discrete_cols):
        stoi = discrete_stoi[col]
        for i, val in enumerate(df[col].tolist() if col in df.columns else [None] * n):
            s = norm_str(val)
            if not s:
                continue
            y[i, j] = stoi.get(s, 0)
            m[i, j] = 1.0

    return y, m


# ============================================================
# Checks
# ============================================================

def check_required_columns(df: pd.DataFrame, cols: List[str], df_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{df_name} missing required columns: {missing}")


def verify_shapes(split_name: str, df: pd.DataFrame, x: np.ndarray, y_set: np.ndarray) -> None:
    n = len(df)
    if x.shape[0] != n:
        raise ValueError(f"[{split_name}] x rows {x.shape[0]} != df rows {n}")
    if y_set.shape[0] != n:
        raise ValueError(f"[{split_name}] y_set rows {y_set.shape[0]} != df rows {n}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--feature_dir",
        type=str,
        default="/Users/wyc/SynPred/data/interim/features/structdesc_features_stage3_v2",
    )
    parser.add_argument(
        "--continuous_cols",
        type=str,
        default="target_temperature_c_clean,target_time_h_clean",
    )
    parser.add_argument(
        "--discrete_cols",
        type=str,
        default="target_atmosphere_coarse,synthesis_type",
    )
    parser.add_argument(
        "--precursor_cols",
        type=str,
        default="target_main_precursors,target_aux_precursors",
    )
    parser.add_argument("--sample_id_col", type=str, default="id")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    continuous_cols = parse_csv_list(args.continuous_cols)
    discrete_cols = parse_csv_list(args.discrete_cols)
    precursor_cols = parse_csv_list(args.precursor_cols)
    sample_id_col = args.sample_id_col
    feature_dir = Path(args.feature_dir)

    # ========================================================
    # Load splits from feature CSVs and build x / y_set
    # ========================================================

    NON_FEATURE_PREFIXES = ("id", "synth_uid", "source_dataset", "record_index",
                            "material_id", "formula", "mp_formula", "synth_formula",
                            "parent_formula", "doi", "dois", "split_group",
                            "poscar_path", "summary_json_path", "provenance_json_path",
                            "reaction_string", "synthesis_text",
                            "main_precursors", "aux_precursors",
                            "target_", "has_target", "condition_source")
    exclude_cols = list(NON_FEATURE_PREFIXES) + precursor_cols + [sample_id_col]

    split_dfs: Dict[str, pd.DataFrame] = {}
    split_xs: Dict[str, np.ndarray] = {}
    feature_cols_used: Optional[List[str]] = None

    for split in ["train", "val", "test"]:
        candidates = [
            feature_dir / f"stage3_{split}_raw.csv",
            feature_dir / f"stage3_{split}.csv",
        ]
        csv_path = None
        for p in candidates:
            if p.exists():
                csv_path = p
                break
        if csv_path is None:
            raise FileNotFoundError(f"No feature CSV found for split={split} in {feature_dir}")

        df = pd.read_csv(csv_path)
        split_dfs[split] = df

        if feature_cols_used is None:
            feature_cols_used = infer_feature_cols(df, exclude_cols)
        split_xs[split] = coerce_float_array(df, feature_cols_used)

    train_df = split_dfs["train"]
    val_df = split_dfs["val"]
    test_df = split_dfs["test"]
    train_x = split_xs["train"]
    val_x = split_xs["val"]
    test_x = split_xs["test"]

    precursor_vocab = build_precursor_vocab(train_df, precursor_cols)
    vocab_index = {v: i for i, v in enumerate(precursor_vocab)}

    train_yset = np.stack([
        precursor_to_multihot(collect_precursors_from_row(row, precursor_cols), vocab_index)
        for _, row in train_df.iterrows()
    ]).astype(np.float32)
    val_yset = np.stack([
        precursor_to_multihot(collect_precursors_from_row(row, precursor_cols), vocab_index)
        for _, row in val_df.iterrows()
    ]).astype(np.float32)
    test_yset = np.stack([
        precursor_to_multihot(collect_precursors_from_row(row, precursor_cols), vocab_index)
        for _, row in test_df.iterrows()
    ]).astype(np.float32)

    # ========================================================
    # Validation
    # ========================================================

    required_cols = continuous_cols + discrete_cols + [sample_id_col]
    check_required_columns(train_df, required_cols, "train_df")
    check_required_columns(val_df, required_cols, "val_df")
    check_required_columns(test_df, required_cols, "test_df")

    verify_shapes("train", train_df, train_x, train_yset)
    verify_shapes("val", val_df, val_x, val_yset)
    verify_shapes("test", test_df, test_x, test_yset)

    # ========================================================
    # Build schema from TRAIN only
    # ========================================================

    continuous_schema = build_continuous_stats(train_df, continuous_cols)
    discrete_vocabs, discrete_stoi = build_discrete_vocabs(train_df, discrete_cols)

    # ========================================================
    # Encode targets
    # ========================================================

    train_yc, train_yc_mask = encode_continuous_targets(train_df, continuous_cols, continuous_schema)
    val_yc, val_yc_mask = encode_continuous_targets(val_df, continuous_cols, continuous_schema)
    test_yc, test_yc_mask = encode_continuous_targets(test_df, continuous_cols, continuous_schema)

    train_yd, train_yd_mask = encode_discrete_targets(train_df, discrete_cols, discrete_stoi)
    val_yd, val_yd_mask = encode_discrete_targets(val_df, discrete_cols, discrete_stoi)
    test_yd, test_yd_mask = encode_discrete_targets(test_df, discrete_cols, discrete_stoi)

    # ========================================================
    # Save split
    # ========================================================

    def save_split(
        split_name: str,
        df: pd.DataFrame,
        x: np.ndarray,
        y_set: np.ndarray,
        y_cont: np.ndarray,
        y_cont_mask: np.ndarray,
        y_disc: np.ndarray,
        y_disc_mask: np.ndarray,
    ) -> Dict[str, Any]:
        payload = {
            "x": x.astype(np.float32),
            "y_set": y_set.astype(np.float32),
            "y_cond_continuous": y_cont.astype(np.float32),
            "y_cond_continuous_mask": y_cont_mask.astype(np.float32),
            "y_cond_discrete": y_disc.astype(np.int64),
            "y_cond_discrete_mask": y_disc_mask.astype(np.float32),
            "sample_id": np.asarray([norm_str(x) for x in df[sample_id_col].tolist()], dtype=object),
        }
        np.savez_compressed(output_dir / f"{split_name}.npz", **payload)

        return {
            "n_rows": int(len(df)),
            "x_shape": list(x.shape),
            "y_set_shape": list(y_set.shape),
            "y_cond_continuous_shape": list(y_cont.shape),
            "y_cond_discrete_shape": list(y_disc.shape),
            "continuous_nonmissing_counts": {
                col: int(y_cont_mask[:, j].sum()) for j, col in enumerate(continuous_cols)
            },
            "discrete_nonmissing_counts": {
                col: int(y_disc_mask[:, j].sum()) for j, col in enumerate(discrete_cols)
            },
        }

    train_summary = save_split("train", train_df, train_x, train_yset, train_yc, train_yc_mask, train_yd, train_yd_mask)
    val_summary = save_split("val", val_df, val_x, val_yset, val_yc, val_yc_mask, val_yd, val_yd_mask)
    test_summary = save_split("test", test_df, test_x, test_yset, test_yc, test_yc_mask, test_yd, test_yd_mask)

    # ========================================================
    # Schema + summary
    # ========================================================

    schema = {
        "continuous_cols": continuous_cols,
        "discrete_cols": discrete_cols,
        "continuous_schema": continuous_schema,
        "discrete_schema": {
            col: {
                "vocab": discrete_vocabs[col],
                "n_classes": len(discrete_vocabs[col]),
                "missing_index": 0,
            }
            for col in discrete_cols
        },
        "feature_cols": feature_cols_used or [],
        "precursor_vocab": precursor_vocab,
        "sample_id_col": sample_id_col,
        "data": {
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
            "x_dim": len(feature_cols_used or []),
            "y_set_dim": len(precursor_vocab),
            "n_continuous_heads": len(continuous_cols),
            "n_discrete_heads": len(discrete_cols),
        },
    }

    export_summary = {
        "continuous_cols": continuous_cols,
        "discrete_cols": discrete_cols,
        "splits": {
            "train": train_summary,
            "val": val_summary,
            "test": test_summary,
        },
    }

    write_json(output_dir / "schema.json", schema)
    write_json(output_dir / "condition_schema.json", {
        "used_precursor_cols": precursor_cols,
        "discrete_cols": discrete_cols,
        "continuous_cols": continuous_cols,
        "continuous_schema": continuous_schema,
        "discrete_schema": schema["discrete_schema"],
    })
    write_json(output_dir / "export_summary.json", export_summary)

    print(json.dumps({
        "output_dir": str(output_dir),
        "artifacts": {
            "schema": str(output_dir / "schema.json"),
            "summary": str(output_dir / "export_summary.json"),
            "train_npz": str(output_dir / "train.npz"),
            "val_npz": str(output_dir / "val.npz"),
            "test_npz": str(output_dir / "test.npz"),
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
