#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
27_build_stage3_condition_dataset.py

Mixed-type stage3 condition dataset builder.

Key behavior in this patched version
-----------------------------------
1) Defaults are aligned to SynPred:
   - splits_dir: /Users/wyc/SynPred/data/interim/splits/structdesc_splits
   - output_dir: /Users/wyc/SynPred/data/interim/generative/stage3_condition_dataset/hybrid_mixed_v1
   - stage2_feature_dir: /Users/wyc/SynPred/data/interim/features/stage2_hybrid_features

2) Supports stage3 train split fallback names:
   - stage3_train.jsonl
   - stage3_gold_train_holdout.jsonl

3) Removes obvious target-leakage feature columns automatically before writing the final dataset.

4) Drops discrete targets that have fewer than 2 valid classes in the train split,
   exporting a continuous-only dataset when necessary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_SPLITS_DIR = "/Users/wyc/SynPred/data/interim/splits/structdesc_splits"
DEFAULT_OUTPUT_DIR = "/Users/wyc/SynPred/data/interim/generative/stage3_condition_dataset/hybrid_mixed_v1"
DEFAULT_STAGE2_FEATURE_DIR = "/Users/wyc/SynPred/data/interim/features/stage2_hybrid_features"

TARGET_LEAKAGE_KEYWORDS = [
    "temp",
    "temperature",
    "time_h",
    "target_time_h",
    "target_temperature",
    "atmosphere",
    "solvent",
    "synthesis",
]
TARGET_LEAKAGE_EXACT = {
    "temperature_c_op",
    "time_h_op",
    "temperature_c_fallback",
    "time_h_fallback",
    "temperature_c",
    "time_h",
    "target_temperature_c",
    "target_time_h",
    "target_atmosphere",
    "target_solvent",
    "target_atmosphere_coarse",
    "synthesis_type",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)



def read_jsonl(path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)



def load_table(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() in {".jsonl", ".jl"}:
        return read_jsonl(p)
    raise ValueError(f"Unsupported table format: {p}")



def parse_str_list(text: str) -> List[str]:
    if text is None:
        return []
    text = str(text).strip()
    if not text:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]



def normalize_key(s: str) -> str:
    return str(s).strip().lower()



def normalize_str_value(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    s = str(v).strip()
    if s.lower() in {"", "nan", "none", "null", "na", "n/a"}:
        return ""
    return s



def find_first_existing(cols: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    norm_map = {normalize_key(c): c for c in cols}
    for cand in candidates:
        hit = norm_map.get(normalize_key(cand))
        if hit is not None:
            return hit
    return None



def merge_split_with_source(split_df: pd.DataFrame, source_df: Optional[pd.DataFrame], join_keys: Sequence[str]) -> pd.DataFrame:
    if source_df is None:
        return split_df.copy()
    left_key = find_first_existing(split_df.columns, join_keys)
    right_key = find_first_existing(source_df.columns, join_keys)
    if left_key and right_key:
        return split_df.merge(source_df, left_on=left_key, right_on=right_key, how="left", suffixes=("", "__src"))
    return split_df.copy()



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
    seen = set()
    out = []
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
    vocab = set()
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



def infer_feature_cols(df: pd.DataFrame, explicit_cols: Sequence[str], exclude_cols: Sequence[str]) -> List[str]:
    if explicit_cols:
        return [c for c in explicit_cols if c in df.columns]
    ex = {normalize_key(x) for x in exclude_cols}
    cols: List[str] = []
    for c in df.columns:
        if normalize_key(c) in ex:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols



def _looks_like_target_leakage(col: str) -> bool:
    c = str(col).strip().lower()
    if c in TARGET_LEAKAGE_EXACT:
        return True
    return any(k in c for k in TARGET_LEAKAGE_KEYWORDS)



def filter_leakage_feature_cols(cols: Sequence[str]) -> Tuple[List[str], List[str]]:
    kept: List[str] = []
    dropped: List[str] = []
    for c in cols:
        if _looks_like_target_leakage(c):
            dropped.append(c)
        else:
            kept.append(c)
    return kept, dropped



def coerce_float_array(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    if not cols:
        return np.zeros((len(df), 0), dtype=np.float32)
    arr = df.loc[:, cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr



def fit_continuous_stats(train_series: pd.Series, clip_percentile: float = 0.0) -> Dict[str, float]:
    vals = pd.to_numeric(train_series, errors="coerce")
    finite = vals[np.isfinite(vals)]
    if len(finite) == 0:
        return {"median": 0.0, "mean": 0.0, "std": 1.0, "clip_lo": float("-inf"), "clip_hi": float("inf")}
    clip_lo = float(np.percentile(finite, clip_percentile)) if clip_percentile > 0 else float(np.min(finite))
    clip_hi = float(np.percentile(finite, 100.0 - clip_percentile)) if clip_percentile > 0 else float(np.max(finite))
    clipped = np.clip(finite.to_numpy(dtype=np.float32), clip_lo, clip_hi)
    median = float(np.median(clipped))
    mean = float(np.mean(clipped))
    std = float(np.std(clipped, ddof=1)) if len(clipped) > 1 else 1.0
    if not np.isfinite(std) or std <= 0:
        std = 1.0
    return {"median": median, "mean": mean, "std": std, "clip_lo": clip_lo, "clip_hi": clip_hi}



def encode_continuous(series: pd.Series, stats: Dict[str, float], normalize: bool) -> Tuple[np.ndarray, np.ndarray]:
    vals = pd.to_numeric(series, errors="coerce")
    mask = vals.notna().astype(np.float32).to_numpy()
    filled = vals.fillna(stats["median"]).to_numpy(dtype=np.float32)
    filled = np.clip(filled, stats["clip_lo"], stats["clip_hi"]).astype(np.float32)
    if normalize:
        filled = ((filled - stats["mean"]) / stats["std"]).astype(np.float32)
    return filled, mask



def build_discrete_vocab(train_series: pd.Series) -> Tuple[List[str], Dict[str, int]]:
    vals = [normalize_str_value(x) for x in train_series.tolist()]
    vals = [x for x in vals if x]
    vocab = ["<UNK_OR_MISSING>"] + sorted(set(vals))
    stoi = {tok: i for i, tok in enumerate(vocab)}
    return vocab, stoi



def encode_discrete(series: pd.Series, stoi: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    vals = [normalize_str_value(x) for x in series.tolist()]
    y = np.zeros(len(vals), dtype=np.int64)
    mask = np.zeros(len(vals), dtype=np.float32)
    for i, s in enumerate(vals):
        if s:
            y[i] = stoi.get(s, 0)
            mask[i] = 1.0
    return y, mask



def resolve_features_for_split(
    split_name: str,
    split_df: pd.DataFrame,
    feature_dir: Optional[Path],
    feature_file_template: Optional[str],
    join_keys: Sequence[str],
    feature_cols: Sequence[str],
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    feat_df: Optional[pd.DataFrame] = None
    if feature_file_template:
        feat_df = pd.read_csv(Path(feature_file_template.format(split=split_name)))
    elif feature_dir:
        candidates = [
            feature_dir / f"stage2_{split_name}_hybrid.csv",
            feature_dir / f"stage2_{split_name}.csv",
            feature_dir / f"{split_name}.csv",
        ]
        for p in candidates:
            if p.exists():
                feat_df = pd.read_csv(p)
                break
    if feat_df is None:
        raise FileNotFoundError(f"No feature csv found for split={split_name}")

    left_key = find_first_existing(split_df.columns, join_keys)
    right_key = find_first_existing(feat_df.columns, join_keys)
    if left_key and right_key:
        merged = split_df.merge(feat_df, left_on=left_key, right_on=right_key, how="left", suffixes=("", "__feat"))
    elif len(split_df) == len(feat_df):
        merged = pd.concat([split_df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)
    else:
        raise ValueError(
            f"Cannot align split={split_name} with features. No shared id key and row counts differ: {len(split_df)} vs {len(feat_df)}"
        )

    exclude = list(join_keys) + ["main_precursors", "aux_precursors", "split"]
    used_feature_cols = infer_feature_cols(merged, feature_cols, exclude)
    used_feature_cols, dropped_leakage_cols = filter_leakage_feature_cols(used_feature_cols)
    if dropped_leakage_cols:
        print(f"[Warn] split={split_name} dropped {len(dropped_leakage_cols)} leakage-like feature cols")
    x = coerce_float_array(merged, used_feature_cols)
    return x, used_feature_cols, merged



def build_split_arrays(
    split_name: str,
    merged_df: pd.DataFrame,
    precursor_cols: Sequence[str],
    precursor_vocab: Sequence[str],
    continuous_cols: Sequence[str],
    discrete_cols: Sequence[str],
    continuous_stats: Dict[str, Dict[str, float]],
    discrete_vocabs: Dict[str, Dict[str, Any]],
    normalize_continuous: bool,
    id_col: Optional[str],
) -> Dict[str, np.ndarray]:
    vocab_index = {v: i for i, v in enumerate(precursor_vocab)}
    y_set = np.stack([
        precursor_to_multihot(collect_precursors_from_row(row, precursor_cols), vocab_index)
        for _, row in merged_df.iterrows()
    ]).astype(np.float32)

    cont_vals = []
    cont_masks = []
    for c in continuous_cols:
        vals, mask = encode_continuous(merged_df[c], continuous_stats[c], normalize_continuous)
        cont_vals.append(vals)
        cont_masks.append(mask)

    disc_vals = []
    disc_masks = []
    for c in discrete_cols:
        vals, mask = encode_discrete(merged_df[c], discrete_vocabs[c]["stoi"])
        disc_vals.append(vals)
        disc_masks.append(mask)

    out: Dict[str, np.ndarray] = {
        "y_set": y_set,
        "y_cond_discrete": np.stack(disc_vals, axis=1).astype(np.int64) if disc_vals else np.zeros((len(merged_df), 0), dtype=np.int64),
        "y_cond_discrete_mask": np.stack(disc_masks, axis=1).astype(np.float32) if disc_masks else np.zeros((len(merged_df), 0), dtype=np.float32),
        "y_cond_continuous": np.stack(cont_vals, axis=1).astype(np.float32),
        "y_cond_continuous_mask": np.stack(cont_masks, axis=1).astype(np.float32),
    }

    if id_col and id_col in merged_df.columns:
        out["sample_id"] = merged_df[id_col].astype(str).to_numpy()
    else:
        out["sample_id"] = np.array([f"{split_name}_{i}" for i in range(len(merged_df))], dtype=object)
    return out



def _resolve_split_file(splits_dir: Path, split: str) -> Path:
    candidates = []
    if split == "train":
        candidates = [
            splits_dir / "stage3_train.jsonl",
            splits_dir / "stage3_gold_train_holdout.jsonl",
        ]
    elif split == "val":
        candidates = [splits_dir / "stage3_val.jsonl"]
    elif split == "test":
        candidates = [splits_dir / "stage3_test.jsonl"]
    else:
        raise ValueError(f"Unknown split: {split}")

    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing split file for split={split}; tried: {[str(x) for x in candidates]}")



def main() -> None:
    parser = argparse.ArgumentParser(description="Build mixed-type stage3 condition dataset from stage3 split jsonl files.")
    parser.add_argument("--splits_dir", type=str, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source_table", type=str, default="")
    parser.add_argument("--stage2_feature_dir", type=str, default=DEFAULT_STAGE2_FEATURE_DIR)
    parser.add_argument("--feature_file_template", type=str, default="")
    parser.add_argument(
        "--join_keys",
        type=str,
        default="row_id,sample_id,id,synth_uid,entry_id,reaction_id,material_id,record_index",
    )
    parser.add_argument("--main_precursor_col", type=str, default="main_precursors")
    parser.add_argument("--aux_precursor_col", type=str, default="aux_precursors")
    parser.add_argument(
        "--continuous_cols",
        type=str,
        default="temperature_c,time_h",
        help="Continuous stage3 targets.",
    )
    parser.add_argument(
        "--discrete_cols",
        type=str,
        default="target_atmosphere_coarse,synthesis_type",
        help="Discrete stage3 targets.",
    )
    parser.add_argument("--feature_cols", type=str, default="")
    parser.add_argument("--normalize_continuous", action="store_true")
    parser.add_argument(
        "--clip_percentile",
        type=float,
        default=1.0,
        help="Winsorize each continuous target using train percentiles before normalization. 0 disables clipping.",
    )
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    join_keys = parse_str_list(args.join_keys)
    feature_cols = parse_str_list(args.feature_cols)
    continuous_cols_requested = parse_str_list(args.continuous_cols)
    discrete_cols_requested = parse_str_list(args.discrete_cols)
    precursor_cols = [args.main_precursor_col, args.aux_precursor_col]

    source_df = load_table(args.source_table)
    feature_dir = Path(args.stage2_feature_dir) if args.stage2_feature_dir else None
    feature_file_template = args.feature_file_template.strip() or None

    merged_by_split: Dict[str, pd.DataFrame] = {}
    used_feature_cols: Optional[List[str]] = None
    id_col_final: Optional[str] = None
    split_sizes: Dict[str, int] = {}

    for split in ["train", "val", "test"]:
        split_path = _resolve_split_file(splits_dir, split)
        split_df = read_jsonl(split_path)
        split_df = merge_split_with_source(split_df, source_df, join_keys)
        x, used_cols_this, merged_df = resolve_features_for_split(
            split_name=split,
            split_df=split_df,
            feature_dir=feature_dir,
            feature_file_template=feature_file_template,
            join_keys=join_keys,
            feature_cols=feature_cols,
        )
        merged_by_split[split] = merged_df
        np.save(output_dir / f"{split}_x.npy", x.astype(np.float32))
        if used_feature_cols is None:
            used_feature_cols = used_cols_this
        id_col_final = id_col_final or find_first_existing(merged_df.columns, join_keys)
        split_sizes[split] = int(len(merged_df))

    train_df = merged_by_split["train"]
    for c in precursor_cols:
        if c not in train_df.columns:
            raise ValueError(f"Missing precursor column in train split: {c}")

    continuous_cols = [c for c in continuous_cols_requested if c in train_df.columns]
    discrete_cols_raw = [c for c in discrete_cols_requested if c in train_df.columns]

    if not continuous_cols:
        raise ValueError("No requested continuous stage3 target columns found in train split.")

    discrete_cols: List[str] = []
    dropped_discrete_cols: List[Dict[str, Any]] = []
    for c in discrete_cols_raw:
        vals = [normalize_str_value(x) for x in train_df[c].tolist()]
        vals = [x for x in vals if x]
        uniq = sorted(set(vals))
        if len(uniq) >= 2:
            discrete_cols.append(c)
        else:
            dropped_discrete_cols.append({
                "col": c,
                "reason": "single_valid_class_in_train",
                "unique_valid_values": uniq,
                "n_unique_valid": len(uniq),
            })

    if not discrete_cols:
        print("[Warn] No usable discrete targets remain after train-set filtering; exporting continuous-only dataset.")

    precursor_vocab = build_precursor_vocab(train_df, precursor_cols)
    continuous_stats = {
        c: fit_continuous_stats(train_df[c], clip_percentile=float(args.clip_percentile))
        for c in continuous_cols
    }
    discrete_vocabs: Dict[str, Dict[str, Any]] = {}
    for c in discrete_cols:
        vocab, stoi = build_discrete_vocab(train_df[c])
        discrete_vocabs[c] = {"vocab": vocab, "stoi": stoi}

    for split in ["train", "val", "test"]:
        arrays = build_split_arrays(
            split_name=split,
            merged_df=merged_by_split[split],
            precursor_cols=precursor_cols,
            precursor_vocab=precursor_vocab,
            continuous_cols=continuous_cols,
            discrete_cols=discrete_cols,
            continuous_stats=continuous_stats,
            discrete_vocabs=discrete_vocabs,
            normalize_continuous=args.normalize_continuous,
            id_col=id_col_final,
        )
        x = np.load(output_dir / f"{split}_x.npy")
        np.savez_compressed(output_dir / f"{split}.npz", x=x, **arrays)
        (output_dir / f"{split}_x.npy").unlink(missing_ok=True)

    schema = {
        "config": {
            "splits_dir": str(splits_dir),
            "output_dir": str(output_dir),
            "source_table": args.source_table,
            "stage2_feature_dir": str(feature_dir) if feature_dir else "",
            "feature_file_template": feature_file_template or "",
            "join_keys": join_keys,
            "main_precursor_col": args.main_precursor_col,
            "aux_precursor_col": args.aux_precursor_col,
            "used_precursor_cols": precursor_cols,
            "continuous_cols_requested": continuous_cols_requested,
            "discrete_cols_requested": discrete_cols_requested,
            "normalize_continuous": bool(args.normalize_continuous),
            "clip_percentile": float(args.clip_percentile),
        },
        "data": {
            "n_train": split_sizes["train"],
            "n_val": split_sizes["val"],
            "n_test": split_sizes["test"],
            "x_dim": len(used_feature_cols or []),
            "y_set_dim": len(precursor_vocab),
            "n_discrete_heads": len(discrete_cols),
            "n_continuous_heads": len(continuous_cols),
        },
        "feature_cols": used_feature_cols or [],
        "sample_id_col": id_col_final,
        "precursor_vocab": precursor_vocab,
        "discrete_schema": {
            c: {
                "vocab": discrete_vocabs[c]["vocab"],
                "n_classes": len(discrete_vocabs[c]["vocab"]),
                "missing_index": 0,
            }
            for c in discrete_cols
        },
        "continuous_schema": {c: continuous_stats[c] for c in continuous_cols},
        "dropped_discrete_cols": dropped_discrete_cols,
    }

    write_json(output_dir / "schema.json", schema)
    write_json(
        output_dir / "condition_schema.json",
        {
            "used_precursor_cols": precursor_cols,
            "discrete_cols": discrete_cols,
            "continuous_cols": continuous_cols,
            "discrete_schema": schema["discrete_schema"],
            "continuous_schema": schema["continuous_schema"],
            "dropped_discrete_cols": dropped_discrete_cols,
        },
    )
    print(json.dumps(schema, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
