#!/usr/bin/env python3
import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def detect_label_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("label_prec__")]


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


def label_cols_to_names(label_cols: List[str]) -> List[str]:
    return [c.replace("label_prec__", "", 1) for c in label_cols]


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


def infer_precursor_names_from_splits(split_dfs: Dict[str, pd.DataFrame]) -> List[str]:
    if "train" in split_dfs:
        label_cols = detect_label_cols(split_dfs["train"])
        if label_cols:
            return label_cols_to_names(label_cols)

    all_names = set()
    for df in split_dfs.values():
        all_names.update(label_cols_to_names(detect_label_cols(df)))
    if all_names:
        return sorted(all_names)

    precursor_names = set()
    for df in split_dfs.values():
        if "main_precursors" not in df.columns:
            continue
        for v in df["main_precursors"].tolist():
            precursor_names.update(parse_list_field(v))
    return sorted(x for x in precursor_names if str(x).strip())


def resolve_precursor_names(
    precursor_vocab_json: str,
    project_root: Path,
    split_dfs: Dict[str, pd.DataFrame],
) -> tuple[List[str], str]:
    candidates: List[Path] = []

    if precursor_vocab_json:
        candidates.append(Path(precursor_vocab_json).expanduser())

    candidates.extend([
        project_root / "data" / "interim" / "features" / "structdesc_features" / "meta" / "precursor_vocab.json",
        project_root / "data" / "interim" / "features" / "structdesc_features" / "precursor_vocab.json",
        project_root / "data" / "interim" / "generative" / "stage2_ar_dataset" / "hybrid" / "precursor_names.json",
        project_root / "data" / "interim" / "generative" / "stage2_setpred_dataset" / "hybrid" / "precursor_names.json",
        project_root / "data" / "interim" / "generative" / "stage2_ar_dataset" / "descriptor" / "precursor_names.json",
        project_root / "data" / "interim" / "generative" / "stage2_setpred_dataset" / "descriptor" / "precursor_names.json",
    ])

    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            obj = load_json(path)
            if isinstance(obj, list) and len(obj) > 0:
                return [str(x) for x in obj], str(path)

    inferred = infer_precursor_names_from_splits(split_dfs)
    if inferred:
        return inferred, "inferred_from_input_splits"

    raise FileNotFoundError(
        "Could not resolve precursor vocab. Checked provided path and project_root candidates, "
        "and also failed to infer precursor names from input splits."
    )


def make_precursor_multihot_from_main_precursors(
    df: pd.DataFrame,
    precursor_to_idx: Dict[str, int],
) -> np.ndarray:
    y = np.zeros((len(df), len(precursor_to_idx)), dtype=np.float32)
    if "main_precursors" not in df.columns:
        return y

    for i, v in enumerate(df["main_precursors"].tolist()):
        labs = parse_list_field(v)
        for lab in labs:
            idx = precursor_to_idx.get(lab)
            if idx is not None:
                y[i, idx] = 1.0
    return y


def make_precursor_multihot_from_label_cols(
    df: pd.DataFrame,
    precursor_names: List[str],
) -> Optional[np.ndarray]:
    label_cols = [f"label_prec__{p}" for p in precursor_names if f"label_prec__{p}" in df.columns]
    if len(label_cols) != len(precursor_names):
        return None
    return df[label_cols].fillna(0).to_numpy(dtype=np.float32)


def fit_standardizer(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def transform_standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def _add_optional_split(paths: dict[str, Path], split_name: str, path: Path) -> None:
    if path.exists():
        paths[split_name] = path


def _prefer_existing(primary: Path, fallback: Path) -> Path:
    if primary.exists():
        return primary
    return fallback


def select_legacy_input_paths(
    project_root: Path,
    input_mode: str,
    target_mode: str,
) -> dict[str, Path]:
    if input_mode == "descriptor":
        stage3_dir = project_root / "data" / "interim" / "features" / "structdesc_features_stage3_v2"
        stage3_suffix = "raw"
    elif input_mode == "hybrid":
        stage3_dir = project_root / "data" / "interim" / "features" / "stage3_hybrid_features"
        stage3_suffix = "hybrid"
    else:
        raise ValueError(f"Unsupported input_mode: {input_mode}")

    paths = {
        "train": stage3_dir / f"stage3_train_{stage3_suffix}.csv",
        "val": stage3_dir / f"stage3_val_{stage3_suffix}.csv",
        "test": stage3_dir / f"stage3_test_{stage3_suffix}.csv",
    }

    gold_holdout_path = stage3_dir / f"stage3_gold_train_holdout_{stage3_suffix}.csv"
    _add_optional_split(paths, "gold_train_holdout", gold_holdout_path)
    return paths


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

        paths = {
            "train": train_dir / f"{file_prefix}_train.csv",
            "val": val_dir / f"{file_prefix}_val.csv",
            "test": test_dir / f"{file_prefix}_test.csv",
        }
        _add_optional_split(paths, "gold_train_holdout", gold_dir / f"{file_prefix}_gold_train_holdout.csv")
        return paths

    if train_mode == "gold_only":
        train_dir = root / "gold_only" / "train"
        val_dir = root / "gold_only" / "val"
        test_dir = root / "gold_only" / "test"

        gold_train = train_dir / f"{file_prefix}_gold_train_holdout.csv"
        regular_train = train_dir / f"{file_prefix}_train.csv"

        paths = {
            "train": _prefer_existing(gold_train, regular_train),
            "val": val_dir / f"{file_prefix}_val.csv",
            "test": test_dir / f"{file_prefix}_test.csv",
        }
        _add_optional_split(paths, "gold_train_holdout", gold_train)
        return paths

    if train_mode == "curriculum_phase1":
        train_dir = root / "curriculum" / "phase1_train"
        val_dir = root / "curriculum" / "val"
        test_dir = root / "curriculum" / "test"
        gold_dir = root / "curriculum" / "phase2_train"

        paths = {
            "train": train_dir / f"{file_prefix}_train.csv",
            "val": val_dir / f"{file_prefix}_val.csv",
            "test": test_dir / f"{file_prefix}_test.csv",
        }
        _add_optional_split(paths, "gold_train_holdout", gold_dir / f"{file_prefix}_gold_train_holdout.csv")
        return paths

    if train_mode == "curriculum_phase2":
        train_dir = root / "curriculum" / "phase2_train"
        val_dir = root / "curriculum" / "val"
        test_dir = root / "curriculum" / "test"

        gold_train = train_dir / f"{file_prefix}_gold_train_holdout.csv"
        regular_train = train_dir / f"{file_prefix}_train.csv"

        paths = {
            "train": _prefer_existing(gold_train, regular_train),
            "val": val_dir / f"{file_prefix}_val.csv",
            "test": test_dir / f"{file_prefix}_test.csv",
        }
        _add_optional_split(paths, "gold_train_holdout", gold_train)
        return paths

    raise ValueError(f"Unsupported train_mode: {train_mode}")


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


def build_one_split(
    df: pd.DataFrame,
    feature_cols: List[str],
    precursor_names: List[str],
    target_mode: str,
    meta_cols: List[str],
) -> Dict[str, Any]:
    x_struct = df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)

    precursor_y = make_precursor_multihot_from_label_cols(df, precursor_names)
    if precursor_y is None:
        precursor_to_idx = {p: i for i, p in enumerate(precursor_names)}
        precursor_y = make_precursor_multihot_from_main_precursors(df, precursor_to_idx)

    x_joint = np.concatenate([x_struct, precursor_y], axis=1).astype(np.float32)

    if target_mode == "temperature":
        if "target_temperature_c_clean" in df.columns:
            target_col = "target_temperature_c_clean"
        elif "target_temperature_c" in df.columns:
            target_col = "target_temperature_c"
        else:
            raise ValueError("No temperature target column found.")
        y_target = df[target_col].to_numpy(dtype=np.float32)
    elif target_mode == "time_bucket":
        if "target_time_bucket" in df.columns:
            target_col = "target_time_bucket"
        elif "time_bucket" in df.columns:
            target_col = "time_bucket"
        else:
            raise ValueError("No time_bucket target column found.")
        y_target = df[target_col].astype("string").fillna(pd.NA).to_numpy()
    else:
        raise ValueError(f"Unsupported target_mode: {target_mode}")

    meta = df[meta_cols].copy() if meta_cols else pd.DataFrame(index=df.index)
    if not meta.empty:
        for c in meta.columns:
            meta[c] = meta[c].fillna("")

    return {
        "x_struct": x_struct,
        "precursor_y": precursor_y,
        "x_joint": x_joint,
        "y_target": y_target,
        "meta": meta,
        "target_col": target_col,
    }


def save_split(
    out_dir: Path,
    split_name: str,
    pack: Dict[str, Any],
    target_mode: str,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    npz_path = out_dir / f"{split_name}.npz"
    meta_path = out_dir / f"{split_name}_meta.csv"

    if target_mode == "temperature":
        y_arr = pack["y_target"].astype(np.float32)
        np.savez_compressed(
            npz_path,
            x_struct=pack["x_struct"],
            precursor_y=pack["precursor_y"],
            x_joint=pack["x_joint"],
            y=y_arr,
        )
        n_target_nonnull = int(np.isfinite(y_arr).sum())
        target_summary = {
            "n_target_nonnull": n_target_nonnull,
            "target_min": float(np.nanmin(y_arr)) if n_target_nonnull > 0 else None,
            "target_max": float(np.nanmax(y_arr)) if n_target_nonnull > 0 else None,
            "target_mean": float(np.nanmean(y_arr)) if n_target_nonnull > 0 else None,
        }
    else:
        y_str = pd.Series(pack["y_target"], dtype="string")
        if class_names is None:
            nonnull = sorted(y_str.dropna().unique().tolist())
            class_names = [str(x) for x in nonnull]
        class_to_idx = {c: i for i, c in enumerate(class_names)}
        y_idx = np.full(len(y_str), -1, dtype=np.int64)
        for i, v in enumerate(y_str.tolist()):
            if pd.notna(v):
                y_idx[i] = class_to_idx[str(v)]
        np.savez_compressed(
            npz_path,
            x_struct=pack["x_struct"],
            precursor_y=pack["precursor_y"],
            x_joint=pack["x_joint"],
            y=y_idx,
        )
        vc = y_str.dropna().value_counts().to_dict()
        target_summary = {
            "n_target_nonnull": int(y_str.notna().sum()),
            "target_dist": {str(k): int(v) for k, v in vc.items()},
        }

    pack["meta"].to_csv(meta_path, index=False)

    return {
        "split": split_name,
        "n_rows": int(pack["x_joint"].shape[0]),
        "n_struct_features": int(pack["x_struct"].shape[1]),
        "n_precursor_features": int(pack["precursor_y"].shape[1]),
        "n_joint_features": int(pack["x_joint"].shape[1]),
        "npz_path": str(npz_path),
        "meta_csv_path": str(meta_path),
        **target_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stage3.5 joint dataset: (structure, precursor_set) -> target.")
    parser.add_argument("--project_root", type=str, default="/Users/wyc/SynPred")
    parser.add_argument(
        "--input_mode",
        type=str,
        default="descriptor",
        choices=["descriptor", "hybrid"],
        help="Legacy mode only: use descriptor-only stage3 data or stage3 hybrid data as structure features.",
    )
    parser.add_argument(
        "--target_mode",
        type=str,
        default="temperature",
        choices=["temperature", "time_bucket"],
    )
    parser.add_argument(
        "--mode_input_root",
        type=str,
        default="",
        help="Optional stage3 training_modes root, e.g. /.../data/interim/training_modes/stage3_temperature",
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
        default="",
        help="Optional precursor vocab json. If omitted or missing, the script will try project_root candidates and then infer from input splits.",
    )
    parser.add_argument("--output_dir", type=str, default="", help="Optional explicit output dir.")
    args = parser.parse_args()

    project_root = Path(args.project_root)

    if args.mode_input_root:
        input_paths = select_mode_input_paths(
            mode_input_root=args.mode_input_root,
            train_mode=args.train_mode,
            target_mode=args.target_mode,
        )
        out_dir = Path(args.output_dir) if args.output_dir else (
            project_root / "data" / "interim" / "generative" / "stage35_joint_dataset" / "task_view" / args.target_mode / args.train_mode
        )
        stage3_source = args.mode_input_root
    else:
        input_paths = select_legacy_input_paths(
            project_root=project_root,
            input_mode=args.input_mode,
            target_mode=args.target_mode,
        )
        out_dir = Path(args.output_dir) if args.output_dir else (
            project_root / "data" / "interim" / "generative" / "stage35_joint_dataset" / args.input_mode / args.target_mode
        )
        stage3_source = str(
            project_root / "data" / "interim" / "features" /
            ("structdesc_features_stage3_v2" if args.input_mode == "descriptor" else "stage3_hybrid_features")
        )

    ensure_dir(out_dir)

    split_dfs = load_splits(input_paths)
    split_names = list(split_dfs.keys())

    precursor_names, precursor_vocab_source = resolve_precursor_names(
        precursor_vocab_json=args.precursor_vocab_json,
        project_root=project_root,
        split_dfs=split_dfs,
    )

    feature_cols = detect_feature_cols(split_dfs["train"])
    if not feature_cols:
        raise ValueError("No structure feature columns detected.")

    meta_cols = detect_meta_cols(split_dfs["train"])

    train_pack = build_one_split(
        split_dfs["train"],
        feature_cols=feature_cols,
        precursor_names=precursor_names,
        target_mode=args.target_mode,
        meta_cols=meta_cols,
    )
    mean, std = fit_standardizer(train_pack["x_joint"])

    summary: Dict[str, Any] = {
        "config": {
            "project_root": str(project_root),
            "input_mode": args.input_mode,
            "target_mode": args.target_mode,
            "mode_input_root": args.mode_input_root,
            "train_mode": args.train_mode if args.mode_input_root else "",
            "precursor_vocab_json": args.precursor_vocab_json,
            "precursor_vocab_source": precursor_vocab_source,
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

    class_names: Optional[List[str]] = None
    if args.target_mode == "time_bucket":
        train_target = pd.Series(train_pack["y_target"], dtype="string")
        class_names = [str(x) for x in sorted(train_target.dropna().unique().tolist())]
        write_json(out_dir / "target_class_names.json", class_names)
        summary["schema"]["target_class_names_path"] = str(out_dir / "target_class_names.json")

    for split in split_names:
        pack = train_pack if split == "train" else build_one_split(
            split_dfs[split],
            feature_cols=feature_cols,
            precursor_names=precursor_names,
            target_mode=args.target_mode,
            meta_cols=meta_cols,
        )
        pack["x_joint_raw"] = pack["x_joint"].copy()
        pack["x_joint"] = transform_standardize(pack["x_joint"], mean, std)

        summary["splits"][split] = save_split(
            out_dir=out_dir,
            split_name=split,
            pack=pack,
            target_mode=args.target_mode,
            class_names=class_names,
        )

    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
