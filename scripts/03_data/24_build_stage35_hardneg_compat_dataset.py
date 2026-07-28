#!/usr/bin/env python3
import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str | Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


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


def precursor_list_to_key(labels: List[str]) -> Tuple[str, ...]:
    return tuple(sorted(set(str(x) for x in labels if str(x).strip())))


def precursor_list_to_multihot(labels: List[str], precursor_to_idx: Dict[str, int]) -> np.ndarray:
    y = np.zeros(len(precursor_to_idx), dtype=np.float32)
    for lab in labels:
        idx = precursor_to_idx.get(lab)
        if idx is not None:
            y[idx] = 1.0
    return y


def fit_standardizer(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def transform_standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def select_legacy_feature_paths(
    project_root: Path,
    input_mode: str,
    split: str,
) -> Path:
    if input_mode == "descriptor":
        feat_path = project_root / "data" / "interim" / "features" / "structdesc_features_stage3_v2" / f"stage3_{split}_raw.csv"
    elif input_mode == "hybrid":
        feat_path = project_root / "data" / "interim" / "features" / "stage3_hybrid_features" / f"stage3_{split}_hybrid.csv"
    else:
        raise ValueError(f"Unsupported input_mode: {input_mode}")
    return feat_path


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


def load_stage3_split_legacy(project_root: Path, input_mode: str, split: str) -> pd.DataFrame:
    split_jsonl = project_root / "data" / "interim" / "splits" / "structdesc_splits" / f"stage3_{split}.jsonl"
    if not split_jsonl.exists():
        raise FileNotFoundError(f"Missing split jsonl: {split_jsonl}")
    split_meta = read_jsonl(split_jsonl)

    keep_cols = [c for c in [
        "id",
        "material_id",
        "formula",
        "doi",
        "split_group",
        "source_dataset",
        "synthesis_type",
        "main_precursors",
    ] if c in split_meta.columns]
    split_meta = split_meta[keep_cols].drop_duplicates(subset=["id"], keep="first").copy()

    feat_path = select_legacy_feature_paths(project_root, input_mode, split)
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing feature file: {feat_path}")

    feat_df = load_csv(str(feat_path)).drop_duplicates(subset=["id"], keep="first").copy()
    merged = feat_df.merge(split_meta, on="id", how="left", suffixes=("", "_meta"))

    for col in ["material_id", "formula", "doi", "split_group", "source_dataset", "synthesis_type", "main_precursors"]:
        meta_col = f"{col}_meta"
        if meta_col in merged.columns:
            if col in merged.columns:
                merged[col] = merged[col].where(merged[col].notna(), merged[meta_col])
            else:
                merged[col] = merged[meta_col]
            merged = merged.drop(columns=[meta_col])

    return merged


def load_stage3_split_mode(
    project_root: Path,
    csv_path: Path,
    split: str,
) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing mode-input csv for split={split}: {csv_path}")

    feat_df = load_csv(str(csv_path)).drop_duplicates(subset=["id"], keep="first").copy()

    split_jsonl = project_root / "data" / "interim" / "splits" / "structdesc_splits" / f"stage3_{split}.jsonl"
    if not split_jsonl.exists():
        raise FileNotFoundError(f"Missing split jsonl: {split_jsonl}")
    split_meta = read_jsonl(split_jsonl)

    keep_cols = [c for c in [
        "id",
        "material_id",
        "formula",
        "doi",
        "split_group",
        "source_dataset",
        "synthesis_type",
        "main_precursors",
    ] if c in split_meta.columns]
    split_meta = split_meta[keep_cols].drop_duplicates(subset=["id"], keep="first").copy()

    merged = feat_df.merge(split_meta, on="id", how="left", suffixes=("", "_meta"))
    for col in ["material_id", "formula", "doi", "split_group", "source_dataset", "synthesis_type", "main_precursors"]:
        meta_col = f"{col}_meta"
        if meta_col in merged.columns:
            if col in merged.columns:
                merged[col] = merged[col].where(merged[col].notna(), merged[meta_col])
            else:
                merged[col] = merged[meta_col]
            merged = merged.drop(columns=[meta_col])

    return merged


def load_hard_negative_candidates(samples_csv: Path) -> pd.DataFrame:
    if not samples_csv.exists():
        raise FileNotFoundError(f"Missing samples csv: {samples_csv}")
    df = pd.read_csv(samples_csv)
    needed = {"id", "pred_labels", "true_labels"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Hard-negative samples csv missing columns: {sorted(missing)}")
    return df


def build_candidate_pool_from_samples(samples_df: pd.DataFrame) -> Dict[str, List[Tuple[Tuple[str, ...], List[str]]]]:
    pool: Dict[str, List[Tuple[Tuple[str, ...], List[str]]]] = {}
    for sid, g in samples_df.groupby("id", sort=False):
        seen = set()
        items: List[Tuple[Tuple[str, ...], List[str]]] = []
        true_labels = parse_list_field(g.iloc[0]["true_labels"])
        true_key = precursor_list_to_key(true_labels)

        for _, row in g.iterrows():
            pred_labels = parse_list_field(row["pred_labels"])
            pred_key = precursor_list_to_key(pred_labels)
            if len(pred_key) == 0:
                continue
            if pred_key == true_key:
                continue
            if pred_key in seen:
                continue
            seen.add(pred_key)
            items.append((pred_key, list(pred_key)))
        pool[str(sid)] = items
    return pool


def build_split_pairs(
    df: pd.DataFrame,
    feature_cols: List[str],
    meta_cols: List[str],
    precursor_names: List[str],
    n_random_neg_per_pos: int,
    n_hard_neg_per_pos: int,
    hardneg_pool: Dict[str, List[Tuple[Tuple[str, ...], List[str]]]],
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
    random_pool_keys = list(key_to_multihot.keys())

    x_struct_rows = []
    precursor_rows = []
    y_rows = []
    meta_rows: List[Dict[str, Any]] = []

    for i, (_, row) in enumerate(df.iterrows()):
        base_meta = {c: row[c] if c in row.index else "" for c in meta_cols}
        sid = str(row["id"]) if "id" in row.index else str(i)

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

        hard_items = hardneg_pool.get(sid, [])
        if n_hard_neg_per_pos > 0 and hard_items:
            if len(hard_items) <= n_hard_neg_per_pos:
                chosen_hard = hard_items
            else:
                idx = rng.choice(len(hard_items), size=n_hard_neg_per_pos, replace=False)
                chosen_hard = [hard_items[int(j)] for j in idx]

            for neg_key, neg_list in chosen_hard:
                neg_mh = precursor_list_to_multihot(neg_list, precursor_to_idx)
                x_struct_rows.append(x_struct[i])
                precursor_rows.append(neg_mh)
                y_rows.append(0)
                meta_rows.append({
                    **base_meta,
                    "pair_label": 0,
                    "pair_type": "hard_negative",
                    "candidate_precursors": json.dumps(neg_list, ensure_ascii=False),
                    "candidate_precursor_key": json.dumps(list(neg_key), ensure_ascii=False),
                })

        available_random_neg_keys = [k for k in random_pool_keys if k != true_key]
        if n_random_neg_per_pos > 0 and available_random_neg_keys:
            if len(available_random_neg_keys) <= n_random_neg_per_pos:
                chosen_random = available_random_neg_keys
            else:
                idx = rng.choice(len(available_random_neg_keys), size=n_random_neg_per_pos, replace=False)
                chosen_random = [available_random_neg_keys[int(j)] for j in idx]

            for neg_key in chosen_random:
                neg_list = list(neg_key)
                neg_mh = key_to_multihot[neg_key]
                x_struct_rows.append(x_struct[i])
                precursor_rows.append(neg_mh)
                y_rows.append(0)
                meta_rows.append({
                    **base_meta,
                    "pair_label": 0,
                    "pair_type": "random_negative",
                    "candidate_precursors": json.dumps(neg_list, ensure_ascii=False),
                    "candidate_precursor_key": json.dumps(list(neg_key), ensure_ascii=False),
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
    meta = pack["meta"]
    pair_counts = meta["pair_type"].value_counts().to_dict() if "pair_type" in meta.columns else {}

    return {
        "split": split,
        "n_rows": int(len(y)),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "positive_rate": float((y == 1).mean()) if len(y) else 0.0,
        "pair_type_dist": {str(k): int(v) for k, v in pair_counts.items()},
        "n_struct_features": int(pack["x_struct"].shape[1]),
        "n_precursor_features": int(pack["precursor_y"].shape[1]),
        "n_joint_features": int(pack["x_joint"].shape[1]),
        "npz_path": str(npz_path),
        "meta_csv_path": str(meta_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hard-negative compatibility dataset.")
    parser.add_argument("--project_root", type=str, default="/Users/wyc/MP_exp_doi")
    parser.add_argument("--input_mode", type=str, default="hybrid", choices=["descriptor", "hybrid"])
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
        help="Used only when --mode_input_root is provided.",
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
    )
    parser.add_argument("--n_random_neg_per_pos", type=int, default=2)
    parser.add_argument("--n_hard_neg_per_pos", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--samples_split",
        type=str,
        default="train",
        choices=["train", "val", "test", "gold_train_holdout"],
    )
    parser.add_argument(
        "--samples_csv",
        type=str,
        default="",
        help="Optional explicit samples csv. If empty, infer from runs/generative/stage2/cvae_hybrid_v1/samples_<split>/<split>_samples.csv",
    )
    parser.add_argument("--output_dir", type=str, default="")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    rng = np.random.default_rng(args.seed)

    if args.samples_csv:
        samples_csv = Path(args.samples_csv)
    else:
        samples_csv = (
            project_root
            / "runs"
            / "generative"
            / "stage2"
            / "cvae_hybrid_v1"
            / f"samples_{args.samples_split}"
            / f"{args.samples_split}_samples.csv"
        )

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        if args.mode_input_root:
            out_dir = (
                project_root
                / "data"
                / "interim"
                / "generative"
                / "stage35_hardneg_compat_dataset"
                / "task_view"
                / args.target_mode
                / args.train_mode
                / f"rand{args.n_random_neg_per_pos}_hard{args.n_hard_neg_per_pos}"
            )
        else:
            out_dir = (
                project_root
                / "data"
                / "interim"
                / "generative"
                / "stage35_hardneg_compat_dataset"
                / args.input_mode
                / f"rand{args.n_random_neg_per_pos}_hard{args.n_hard_neg_per_pos}"
            )
    ensure_dir(out_dir)

    precursor_names = load_json(args.precursor_vocab_json)
    if not isinstance(precursor_names, list) or len(precursor_names) == 0:
        raise ValueError(f"Invalid precursor vocab json: {args.precursor_vocab_json}")

    samples_df = load_hard_negative_candidates(samples_csv)
    hardneg_pool = build_candidate_pool_from_samples(samples_df)

    split_names = ["train", "val", "test", "gold_train_holdout"]

    if args.mode_input_root:
        mode_paths = select_mode_input_paths(
            mode_input_root=args.mode_input_root,
            train_mode=args.train_mode,
            target_mode=args.target_mode,
        )
        split_dfs = {
            split: load_stage3_split_mode(project_root, mode_paths[split], split)
            for split in split_names
        }
        input_paths = {k: str(v) for k, v in mode_paths.items()}
        stage3_source = args.mode_input_root
    else:
        split_dfs = {
            split: load_stage3_split_legacy(project_root, args.input_mode, split)
            for split in split_names
        }
        input_paths = {k: f"legacy::{args.input_mode}::{k}" for k in split_names}
        stage3_source = str(project_root / "data" / "interim" / "features")

    feature_cols = detect_feature_cols(split_dfs["train"])
    if not feature_cols:
        raise ValueError("No structure feature columns detected.")
    meta_cols = detect_meta_cols(split_dfs["train"])

    train_pack = build_split_pairs(
        df=split_dfs["train"],
        feature_cols=feature_cols,
        meta_cols=meta_cols,
        precursor_names=precursor_names,
        n_random_neg_per_pos=args.n_random_neg_per_pos,
        n_hard_neg_per_pos=args.n_hard_neg_per_pos,
        hardneg_pool=hardneg_pool,
        rng=rng,
    )

    mean, std = fit_standardizer(train_pack["x_joint"])

    summary: Dict[str, Any] = {
        "config": {
            "project_root": str(project_root),
            "input_mode": args.input_mode,
            "mode_input_root": args.mode_input_root,
            "target_mode": args.target_mode if args.mode_input_root else "",
            "train_mode": args.train_mode if args.mode_input_root else "",
            "precursor_vocab_json": args.precursor_vocab_json,
            "n_random_neg_per_pos": args.n_random_neg_per_pos,
            "n_hard_neg_per_pos": args.n_hard_neg_per_pos,
            "seed": args.seed,
            "samples_split": args.samples_split,
            "samples_csv": str(samples_csv),
            "output_dir": str(out_dir),
            "stage3_source": stage3_source,
        },
        "input_paths": input_paths,
        "schema": {
            "n_struct_features": len(feature_cols),
            "n_precursor_features": len(precursor_names),
            "n_joint_features": len(feature_cols) + len(precursor_names),
            "feature_cols_path": str(out_dir / "feature_cols.json"),
            "precursor_names_path": str(out_dir / "precursor_names.json"),
            "meta_cols": meta_cols,
        },
        "samples_summary": {
            "n_sample_rows": int(len(samples_df)),
            "n_structure_ids_with_candidates": int(len(hardneg_pool)),
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
            n_random_neg_per_pos=args.n_random_neg_per_pos,
            n_hard_neg_per_pos=args.n_hard_neg_per_pos,
            hardneg_pool=hardneg_pool,
            rng=rng,
        )
        pack["x_joint"] = transform_standardize(pack["x_joint"], mean, std)
        summary["splits"][split] = save_split(out_dir, split, pack)

    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
