#!/usr/bin/env python3
import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def read_jsonl(path: str | Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_feature_cols(df: pd.DataFrame) -> List[str]:
    return [
        c for c in df.columns
        if c.startswith("feat_")
        or c.startswith("graph_emb_")
        or "_graph_emb_" in c
    ]


def detect_meta_cols(df: pd.DataFrame) -> List[str]:
    preferred = [
        "id",
        "material_id",
        "formula",
        "doi",
        "split_group",
        "source_dataset",
        "synthesis_type",
        "main_precursors",
    ]
    return [c for c in preferred if c in df.columns]


def parse_list_field(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    if isinstance(v, float) and np.isnan(v):
        return []
    s = str(v).strip()
    if not s:
        return []
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(x) for x in obj if str(x).strip()]
    except Exception:
        pass
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list):
            return [str(x) for x in obj if str(x).strip()]
    except Exception:
        pass
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    if ";" in s:
        return [x.strip() for x in s.split(";") if x.strip()]
    return [s]


def fit_standardizer(train_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def transform_standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def precursor_list_to_multihot(labels: List[str], precursor_to_idx: Dict[str, int]) -> np.ndarray:
    y = np.zeros(len(precursor_to_idx), dtype=np.float32)
    for lab in labels:
        idx = precursor_to_idx.get(lab)
        if idx is not None:
            y[idx] = 1.0
    return y


def precursor_list_to_key(labels: List[str]) -> Tuple[str, ...]:
    return tuple(sorted(set(labels)))


def load_all_stage3_meta(project_root: Path) -> pd.DataFrame:
    split_names = ["train", "val", "test", "gold_train_holdout"]
    dfs = []
    for split in split_names:
        split_jsonl = project_root / "data" / "interim" / "splits" / "structdesc_splits" / f"stage3_{split}.jsonl"
        if not split_jsonl.exists():
            raise FileNotFoundError(f"Missing split jsonl: {split_jsonl}")
        df = read_jsonl(split_jsonl)
        df["_source_split"] = split
        dfs.append(df)

    merged = pd.concat(dfs, axis=0, ignore_index=True)
    keep_cols = [c for c in [
        "id",
        "material_id",
        "formula",
        "doi",
        "split_group",
        "source_dataset",
        "synthesis_type",
        "main_precursors",
        "_source_split",
    ] if c in merged.columns]
    merged = merged[keep_cols].drop_duplicates(subset=["id"], keep="first").copy()
    return merged


def select_mode_input_paths(
    mode_input_root: str,
    train_mode: str,
    target_mode: str,
) -> dict[str, Path]:
    root = Path(mode_input_root)
    file_prefix = target_mode

    if train_mode == "relaxed_only":
        train_dir = root / "relaxed_only" / "train"
        val_dir = root / "relaxed_only" / "val"
        test_dir = root / "relaxed_only" / "test"
        gold_dir = root / "gold_only" / "train"
        return {
            "train": train_dir / f"{file_prefix}_train.csv",
            "val": val_dir / f"{file_prefix}_val.csv",
            "test": test_dir / f"{file_prefix}_test.csv",
            "gold_train_holdout": gold_dir / f"{file_prefix}_gold_train_holdout.csv",
        }

    elif train_mode == "gold_only":
        train_dir = root / "gold_only" / "train"
        val_dir = root / "gold_only" / "val"
        test_dir = root / "gold_only" / "test"
        return {
            "train": train_dir / f"{file_prefix}_gold_train_holdout.csv",
            "val": val_dir / f"{file_prefix}_val.csv",
            "test": test_dir / f"{file_prefix}_test.csv",
            "gold_train_holdout": train_dir / f"{file_prefix}_gold_train_holdout.csv",
        }

    elif train_mode == "curriculum_phase1":
        train_dir = root / "curriculum" / "phase1_train"
        val_dir = root / "curriculum" / "val"
        test_dir = root / "curriculum" / "test"
        gold_dir = root / "curriculum" / "phase2_train"
        return {
            "train": train_dir / f"{file_prefix}_train.csv",
            "val": val_dir / f"{file_prefix}_val.csv",
            "test": test_dir / f"{file_prefix}_test.csv",
            "gold_train_holdout": gold_dir / f"{file_prefix}_gold_train_holdout.csv",
        }

    elif train_mode == "curriculum_phase2":
        train_dir = root / "curriculum" / "phase2_train"
        val_dir = root / "curriculum" / "val"
        test_dir = root / "curriculum" / "test"
        return {
            "train": train_dir / f"{file_prefix}_gold_train_holdout.csv",
            "val": val_dir / f"{file_prefix}_val.csv",
            "test": test_dir / f"{file_prefix}_test.csv",
            "gold_train_holdout": train_dir / f"{file_prefix}_gold_train_holdout.csv",
        }

    else:
        raise ValueError(f"Unsupported train_mode: {train_mode}")


def select_legacy_input_paths(project_root: Path, input_mode: str) -> dict[str, Path]:
    if input_mode == "descriptor":
        stage3_dir = project_root / "data" / "interim" / "features" / "structdesc_features_stage3_v2"
        suffix = "raw"
    elif input_mode == "hybrid":
        stage3_dir = project_root / "data" / "interim" / "features" / "stage3_hybrid_features"
        suffix = "hybrid"
    else:
        raise ValueError(f"Unsupported input_mode: {input_mode}")

    return {
        "train": stage3_dir / f"stage3_train_{suffix}.csv",
        "val": stage3_dir / f"stage3_val_{suffix}.csv",
        "test": stage3_dir / f"stage3_test_{suffix}.csv",
        "gold_train_holdout": stage3_dir / f"stage3_gold_train_holdout_{suffix}.csv",
    }


def load_stage3_mode_splits(paths: dict[str, Path], stage3_meta_union: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for split_name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing input file for split={split_name}: {path}")
        feat_df = load_csv(path).drop_duplicates(subset=["id"], keep="first").copy()
        merged = feat_df.merge(stage3_meta_union, on="id", how="left", suffixes=("", "_meta"))

        for col in ["material_id", "formula", "doi", "split_group", "source_dataset", "synthesis_type", "main_precursors"]:
            meta_col = f"{col}_meta"
            if meta_col in merged.columns:
                if col in merged.columns:
                    merged[col] = merged[col].where(merged[col].notna(), merged[meta_col])
                else:
                    merged[col] = merged[meta_col]
                merged = merged.drop(columns=[meta_col])

        if "_source_split_meta" in merged.columns:
            merged = merged.drop(columns=["_source_split_meta"])
        if "_source_split" in merged.columns:
            merged = merged.drop(columns=["_source_split"])

        out[split_name] = merged
    return out


def load_stage3_legacy_splits(project_root: Path, input_mode: str, stage3_meta_union: pd.DataFrame) -> dict[str, pd.DataFrame]:
    paths = select_legacy_input_paths(project_root, input_mode)
    return load_stage3_mode_splits(paths, stage3_meta_union)


def build_split_pairs(
    df: pd.DataFrame,
    feature_cols: List[str],
    meta_cols: List[str],
    precursor_names: List[str],
    n_neg_per_pos: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    precursor_to_idx = {p: i for i, p in enumerate(precursor_names)}

    x_struct = df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)

    true_lists: List[List[str]] = []
    true_keys: List[Tuple[str, ...]] = []
    true_multihots: List[np.ndarray] = []

    for _, row in df.iterrows():
        labs = parse_list_field(row.get("main_precursors", []))
        true_lists.append(labs)
        key = precursor_list_to_key(labs)
        true_keys.append(key)
        true_multihots.append(precursor_list_to_multihot(labs, precursor_to_idx))

    key_to_multihot: Dict[Tuple[str, ...], np.ndarray] = {}
    for labs, key, mh in zip(true_lists, true_keys, true_multihots):
        if len(key) > 0 and key not in key_to_multihot:
            key_to_multihot[key] = mh.copy()

    pool_keys = list(key_to_multihot.keys())

    x_struct_rows = []
    precursor_rows = []
    y_rows = []
    meta_rows: List[Dict[str, Any]] = []

    for i, (_, row) in enumerate(df.iterrows()):
        base_meta = {c: row[c] if c in row.index else "" for c in meta_cols}
        true_list = true_lists[i]
        true_key = true_keys[i]
        true_mh = true_multihots[i]

        x_struct_rows.append(x_struct[i])
        precursor_rows.append(true_mh)
        y_rows.append(1)
        meta_rows.append({
            **base_meta,
            "pair_label": 1,
            "pair_type": "positive",
            "candidate_precursors": json.dumps(true_list, ensure_ascii=False),
            "candidate_precursor_key": json.dumps(list(true_key), ensure_ascii=False),
        })

        available_neg_keys = [k for k in pool_keys if k != true_key]
        if not available_neg_keys:
            continue

        if len(available_neg_keys) <= n_neg_per_pos:
            sampled_keys = available_neg_keys
        else:
            sampled_idx = rng.choice(len(available_neg_keys), size=n_neg_per_pos, replace=False)
            sampled_keys = [available_neg_keys[j] for j in sampled_idx]

        for neg_key in sampled_keys:
            neg_mh = key_to_multihot[neg_key]
            neg_list = list(neg_key)

            x_struct_rows.append(x_struct[i])
            precursor_rows.append(neg_mh)
            y_rows.append(0)
            meta_rows.append({
                **base_meta,
                "pair_label": 0,
                "pair_type": "negative",
                "candidate_precursors": json.dumps(neg_list, ensure_ascii=False),
                "candidate_precursor_key": json.dumps(neg_list, ensure_ascii=False),
            })

    x_struct_arr = np.vstack(x_struct_rows).astype(np.float32)
    precursor_arr = np.vstack(precursor_rows).astype(np.float32)
    x_joint_arr = np.concatenate([x_struct_arr, precursor_arr], axis=1).astype(np.float32)
    y_arr = np.array(y_rows, dtype=np.int64)
    meta_df = pd.DataFrame(meta_rows)

    return {
        "x_struct": x_struct_arr,
        "precursor_y": precursor_arr,
        "x_joint": x_joint_arr,
        "y": y_arr,
        "meta": meta_df,
    }


def save_split(out_dir: Path, split: str, pack: Dict[str, Any]) -> Dict[str, Any]:
    npz_path = out_dir / f"{split}.npz"
    meta_path = out_dir / f"{split}_meta.csv"

    np.savez_compressed(
        npz_path,
        x_struct=pack["x_struct"],
        precursor_y=pack["precursor_y"],
        x_joint=pack["x_joint"],
        y=pack["y"],
    )
    pack["meta"].to_csv(meta_path, index=False)

    y = pack["y"]
    return {
        "split": split,
        "n_rows": int(len(y)),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "positive_rate": float((y == 1).mean()) if len(y) else 0.0,
        "n_struct_features": int(pack["x_struct"].shape[1]),
        "n_precursor_features": int(pack["precursor_y"].shape[1]),
        "n_joint_features": int(pack["x_joint"].shape[1]),
        "npz_path": str(npz_path),
        "meta_csv_path": str(meta_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compatibility dataset for (structure, precursor_set) -> compatible?")
    parser.add_argument("--project_root", type=str, default="/Users/wyc/MP_exp_doi")
    parser.add_argument("--input_mode", type=str, default="descriptor", choices=["descriptor", "hybrid"])
    parser.add_argument(
        "--mode_input_root",
        type=str,
        default="",
        help="Optional stage3 training_modes root, e.g. /.../data/interim/training_modes/stage3_temperature",
    )
    parser.add_argument(
        "--target_mode",
        type=str,
        default="temperature",
        choices=["temperature", "time_bucket", "atmosphere_coarse", "solvent", "synthesis_type"],
        help="Used only when --mode_input_root is provided, to resolve canonical task-view filenames.",
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default="relaxed_only",
        choices=["relaxed_only", "gold_only", "curriculum_phase1", "curriculum_phase2"],
        help="Used only when --mode_input_root is provided.",
    )
    parser.add_argument(
        "--precursor_vocab_json",
        type=str,
        default="/Users/wyc/MP_exp_doi/data/interim/features/structdesc_features/meta/precursor_vocab.json",
        help="Stable precursor schema from structdesc feature builder.",
    )
    parser.add_argument("--n_neg_per_pos", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    rng = np.random.default_rng(args.seed)

    precursor_names = load_json(args.precursor_vocab_json)
    if not isinstance(precursor_names, list) or len(precursor_names) == 0:
        raise ValueError(f"Invalid precursor vocab json: {args.precursor_vocab_json}")

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        if args.mode_input_root:
            out_dir = (
                project_root
                / "data"
                / "interim"
                / "generative"
                / "stage35_compat_dataset"
                / "task_view"
                / args.target_mode
                / args.train_mode
                / f"neg{args.n_neg_per_pos}"
            )
        else:
            out_dir = (
                project_root
                / "data"
                / "interim"
                / "generative"
                / "stage35_compat_dataset"
                / args.input_mode
                / f"neg{args.n_neg_per_pos}"
            )
    ensure_dir(out_dir)

    stage3_meta_union = load_all_stage3_meta(project_root)

    if args.mode_input_root:
        input_paths = select_mode_input_paths(
            mode_input_root=args.mode_input_root,
            train_mode=args.train_mode,
            target_mode=args.target_mode,
        )
        split_dfs = load_stage3_mode_splits(input_paths, stage3_meta_union)
        stage3_source = args.mode_input_root
    else:
        input_paths = select_legacy_input_paths(project_root, args.input_mode)
        split_dfs = load_stage3_mode_splits(input_paths, stage3_meta_union)
        stage3_source = str(input_paths["train"].parent)

    feature_cols = detect_feature_cols(split_dfs["train"])
    if not feature_cols:
        raise ValueError("No structure feature columns detected.")
    meta_cols = detect_meta_cols(split_dfs["train"])

    split_names = ["train", "val", "test", "gold_train_holdout"]
    train_pack = build_split_pairs(
        df=split_dfs["train"],
        feature_cols=feature_cols,
        meta_cols=meta_cols,
        precursor_names=precursor_names,
        n_neg_per_pos=args.n_neg_per_pos,
        rng=rng,
    )

    mean, std = fit_standardizer(train_pack["x_joint"])

    summary: Dict[str, Any] = {
        "config": {
            "project_root": str(project_root),
            "input_mode": args.input_mode,
            "mode_input_root": args.mode_input_root,
            "target_mode": args.target_mode,
            "train_mode": args.train_mode if args.mode_input_root else "",
            "precursor_vocab_json": args.precursor_vocab_json,
            "n_neg_per_pos": args.n_neg_per_pos,
            "seed": args.seed,
            "output_dir": str(out_dir),
            "stage3_source": stage3_source,
        },
        "input_paths": {k: str(v) for k, v in input_paths.items()},
        "schema": {
            "n_struct_features": len(feature_cols),
            "n_precursor_features": len(precursor_names),
            "n_joint_features": len(feature_cols) + len(precursor_names),
            "feature_cols_path": str(out_dir / "feature_cols.json"),
            "precursor_names_path": str(out_dir / "precursor_names.json"),
            "meta_cols": meta_cols,
        },
        "splits": {},
    }

    write_json(out_dir / "feature_cols.json", feature_cols)
    write_json(out_dir / "precursor_names.json", precursor_names)
    np.save(out_dir / "joint_feature_mean.npy", mean)
    np.save(out_dir / "joint_feature_std.npy", std)

    for split in split_names:
        pack = train_pack if split == "train" else build_split_pairs(
            df=split_dfs[split],
            feature_cols=feature_cols,
            meta_cols=meta_cols,
            precursor_names=precursor_names,
            n_neg_per_pos=args.n_neg_per_pos,
            rng=rng,
        )
        pack["x_joint"] = transform_standardize(pack["x_joint"], mean, std)
        summary["splits"][split] = save_split(out_dir, split, pack)

    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
