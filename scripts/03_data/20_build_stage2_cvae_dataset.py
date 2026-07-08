#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any

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


def detect_feature_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c.startswith("feat_") or c.startswith("graph_emb_") or "_graph_emb_" in c
    ]


def detect_label_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("label_prec__")]


def detect_meta_cols(df: pd.DataFrame) -> list[str]:
    preferred = [
        "id",
        "material_id",
        "formula",
        "doi",
        "split_group",
        "source_dataset",
        "synthesis_type",
    ]
    return [c for c in preferred if c in df.columns]


def label_cols_to_names(label_cols: list[str]) -> list[str]:
    return [c.replace("label_prec__", "", 1) for c in label_cols]


def select_input_paths(base_mode: str, base_dir: str) -> dict[str, Path]:
    base = Path(base_dir)
    if base_mode == "descriptor":
        in_dir = base / "data" / "interim" / "features" / "structdesc_features"
        suffix = "ml"
    elif base_mode == "hybrid":
        in_dir = base / "data" / "interim" / "features" / "stage2_hybrid_features"
        suffix = "hybrid"
    else:
        raise ValueError(f"Unsupported input_mode: {base_mode}")

    paths = {
        "train": in_dir / f"stage2_train_{suffix}.csv",
        "val": in_dir / f"stage2_val_{suffix}.csv",
        "test": in_dir / f"stage2_test_{suffix}.csv",
    }

    holdout_path = in_dir / f"stage2_gold_train_holdout_{suffix}.csv"
    if holdout_path.exists():
        paths["gold_train_holdout"] = holdout_path

    return paths


def select_mode_input_paths(
    mode_input_root: str,
    train_mode: str,
    input_mode: str,
) -> dict[str, Path]:
    root = Path(mode_input_root)

    if input_mode == "descriptor":
        suffix = "ml"
    elif input_mode == "hybrid":
        suffix = "hybrid"
    else:
        raise ValueError(f"Unsupported input_mode: {input_mode}")

    if train_mode == "relaxed_only":
        in_dir = root / "relaxed_only"
    elif train_mode == "gold_only":
        in_dir = root / "gold_only"
    elif train_mode == "curriculum_phase1":
        in_dir = root / "curriculum" / "phase1"
    elif train_mode == "curriculum_phase2":
        in_dir = root / "curriculum" / "phase2"
    else:
        raise ValueError(f"Unsupported train_mode: {train_mode}")

    paths = {
        "train": in_dir / f"stage2_train_{suffix}.csv",
        "val": in_dir / f"stage2_val_{suffix}.csv",
        "test": in_dir / f"stage2_test_{suffix}.csv",
    }

    holdout_path = in_dir / f"stage2_gold_train_holdout_{suffix}.csv"
    if holdout_path.exists():
        paths["gold_train_holdout"] = holdout_path

    return paths


def load_splits(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    required_splits = {"train", "val", "test"}
    out: dict[str, pd.DataFrame] = {}

    for split_name, path in paths.items():
        if not path.exists():
            if split_name in required_splits:
                raise FileNotFoundError(f"Missing required input file for split={split_name}: {path}")
            print(f"[WARN] Optional split missing, skip: {split_name}: {path}")
            continue
        out[split_name] = load_csv(str(path))

    return out


def fit_standardizer(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def transform_standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def make_metadata_frame(df: pd.DataFrame, meta_cols: list[str]) -> pd.DataFrame:
    if not meta_cols:
        return pd.DataFrame(index=df.index)
    out = df[meta_cols].copy()
    for c in out.columns:
        out[c] = out[c].fillna("")
    return out


def build_arrays(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_cols: list[str],
    meta_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    x = df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    y = df[label_cols].fillna(0).to_numpy(dtype=np.float32)
    meta = make_metadata_frame(df, meta_cols)
    return x, y, meta


def save_split_npz(
    out_dir: Path,
    split_name: str,
    x_raw: np.ndarray,
    x_std: np.ndarray,
    y: np.ndarray,
) -> Path:
    out_path = out_dir / f"{split_name}.npz"
    np.savez_compressed(
        out_path,
        x_raw=x_raw,
        x=x_std,
        y=y,
    )
    return out_path


def save_split_meta(
    out_dir: Path,
    split_name: str,
    meta_df: pd.DataFrame,
) -> Path:
    out_path = out_dir / f"{split_name}_meta.csv"
    meta_df.to_csv(out_path, index=False)
    return out_path


def summarize_split(
    split_name: str,
    x_std: np.ndarray,
    y: np.ndarray,
    npz_path: Path,
    meta_path: Path,
) -> dict[str, Any]:
    label_count_per_row = y.sum(axis=1)
    return {
        "split": split_name,
        "n_rows": int(x_std.shape[0]),
        "n_features": int(x_std.shape[1]),
        "n_labels": int(y.shape[1]),
        "mean_labels_per_sample": float(label_count_per_row.mean()) if len(label_count_per_row) else 0.0,
        "max_labels_per_sample": float(label_count_per_row.max()) if len(label_count_per_row) else 0.0,
        "npz_path": str(npz_path),
        "meta_csv_path": str(meta_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stage2 CVAE dataset from descriptor or hybrid features.")
    parser.add_argument(
        "--project_root",
        type=str,
        default="/Users/wyc/MP_exp_doi",
        help="Project root directory.",
    )
    parser.add_argument(
        "--input_mode",
        type=str,
        default="descriptor",
        choices=["descriptor", "hybrid"],
        help="Use descriptor-only stage2 features or descriptor+graph hybrid features.",
    )
    parser.add_argument(
        "--mode_input_root",
        type=str,
        default="",
        help="Optional canonical mode root, e.g. .../data/interim/model_inputs/stage2_cvae_modes/stage2_hybrid_cgcnn",
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default="relaxed_only",
        choices=["relaxed_only", "gold_only", "curriculum_phase1", "curriculum_phase2"],
        help="Training-data mode used only when --mode_input_root is provided.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Optional explicit output dir.",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root)

    if args.mode_input_root:
        input_paths = select_mode_input_paths(
            mode_input_root=args.mode_input_root,
            train_mode=args.train_mode,
            input_mode=args.input_mode,
        )
        if args.output_dir:
            out_dir = Path(args.output_dir)
        else:
            out_dir = (
                project_root
                / "data"
                / "interim"
                / "generative"
                / "stage2_cvae_dataset"
                / args.input_mode
                / args.train_mode
            )
    else:
        input_paths = select_input_paths(args.input_mode, str(project_root))
        if args.output_dir:
            out_dir = Path(args.output_dir)
        else:
            out_dir = project_root / "data" / "interim" / "generative" / "stage2_cvae_dataset" / args.input_mode

    ensure_dir(out_dir)

    split_dfs = load_splits(input_paths)

    train_df = split_dfs["train"]
    feature_cols = detect_feature_cols(train_df)
    label_cols = detect_label_cols(train_df)
    meta_cols = detect_meta_cols(train_df)

    if not feature_cols:
        raise ValueError("No feature columns detected.")
    if not label_cols:
        raise ValueError("No label columns detected.")

    for split_name, df in split_dfs.items():
        missing_feat = [c for c in feature_cols if c not in df.columns]
        missing_label = [c for c in label_cols if c not in df.columns]
        if missing_feat:
            raise ValueError(f"Split {split_name} missing feature columns: {missing_feat[:10]}")
        if missing_label:
            raise ValueError(f"Split {split_name} missing label columns: {missing_label[:10]}")

    arrays: dict[str, dict[str, Any]] = {}
    for split_name, df in split_dfs.items():
        x_raw, y, meta = build_arrays(df, feature_cols, label_cols, meta_cols)
        arrays[split_name] = {
            "x_raw": x_raw,
            "y": y,
            "meta": meta,
        }

    mean, std = fit_standardizer(arrays["train"]["x_raw"])

    summary: dict[str, Any] = {
        "config": {
            "project_root": str(project_root),
            "input_mode": args.input_mode,
            "mode_input_root": args.mode_input_root,
            "train_mode": args.train_mode if args.mode_input_root else "",
            "output_dir": str(out_dir),
        },
        "input_paths": {k: str(v) for k, v in input_paths.items()},
        "feature_schema": {
            "n_features": len(feature_cols),
            "n_labels": len(label_cols),
            "n_meta_cols": len(meta_cols),
            "feature_cols_path": str(out_dir / "feature_cols.json"),
            "label_cols_path": str(out_dir / "label_cols.json"),
            "label_names_path": str(out_dir / "label_names.json"),
            "meta_cols": meta_cols,
        },
        "splits": {},
    }

    write_json(out_dir / "feature_cols.json", feature_cols)
    write_json(out_dir / "label_cols.json", label_cols)
    write_json(out_dir / "label_names.json", label_cols_to_names(label_cols))
    np.save(out_dir / "feature_mean.npy", mean)
    np.save(out_dir / "feature_std.npy", std)

    for split_name, pack in arrays.items():
        x_std = transform_standardize(pack["x_raw"], mean, std)
        npz_path = save_split_npz(out_dir, split_name, pack["x_raw"], x_std, pack["y"])
        meta_path = save_split_meta(out_dir, split_name, pack["meta"])

        summary["splits"][split_name] = summarize_split(
            split_name=split_name,
            x_std=x_std,
            y=pack["y"],
            npz_path=npz_path,
            meta_path=meta_path,
        )

    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
