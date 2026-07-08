#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


STOP_TOKEN = "<stop>"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


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
    ]
    return [c for c in preferred if c in df.columns]


def label_cols_to_names(label_cols: List[str]) -> List[str]:
    return [c.replace("label_prec__", "", 1) for c in label_cols]


def select_input_paths(input_mode: str, project_root: Path) -> Dict[str, Path]:
    if input_mode == "descriptor":
        in_dir = project_root / "data" / "interim" / "features" / "structdesc_features"
        suffix = "ml"
    elif input_mode == "hybrid":
        in_dir = project_root / "data" / "interim" / "features" / "stage2_hybrid_features"
        suffix = "hybrid"
    else:
        raise ValueError(f"Unsupported input_mode: {input_mode}")

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
) -> Dict[str, Path]:
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
    elif train_mode in ("curriculum_phase1", "curriculum_phase2"):
        in_dir = root / "curriculum"
    else:
        raise ValueError(f"Unsupported train_mode: {train_mode}")

    def _find(name: str, subdirs: list) -> Path:
        """Search for a file in in_dir directly, then in subdirs."""
        flat = in_dir / name
        if flat.exists():
            return flat
        for sd in subdirs:
            p = in_dir / sd / name
            if p.exists():
                return p
        return flat  # fallback (will trigger FileNotFoundError later)

    # Resolve train file
    if train_mode == "curriculum_phase1":
        train_path = _find(f"stage2_train_{suffix}.csv", ["phase1_train", "train"])
    elif train_mode == "curriculum_phase2":
        train_path = _find(f"stage2_gold_train_holdout_{suffix}.csv", ["phase2_train", "train"])
    elif train_mode == "gold_only":
        # gold_only uses holdout as train
        candidate = _find(f"stage2_train_{suffix}.csv", ["train"])
        if not candidate.exists():
            candidate = _find(f"stage2_gold_train_holdout_{suffix}.csv", ["train"])
        train_path = candidate
    else:
        train_path = _find(f"stage2_train_{suffix}.csv", ["train"])

    val_path = _find(f"stage2_val_{suffix}.csv", ["val"])
    test_path = _find(f"stage2_test_{suffix}.csv", ["test"])

    paths: Dict[str, Path] = {
        "train": train_path,
        "val": val_path,
        "test": test_path,
    }

    # Check for holdout as separate optional split
    if train_mode != "gold_only":
        holdout = _find(f"stage2_gold_train_holdout_{suffix}.csv", ["train"])
        if holdout.exists():
            paths["gold_train_holdout"] = holdout

    return paths


def load_splits(paths: Dict[str, Path]) -> Dict[str, pd.DataFrame]:
    required_splits = {"train", "val", "test"}
    split_dfs: Dict[str, pd.DataFrame] = {}
    for split, path in paths.items():
        if not path.exists():
            if split in required_splits:
                raise FileNotFoundError(f"Missing required input file for split={split}: {path}")
            print(f"[WARN] Optional split missing, skip: {split}: {path}")
            continue
        split_dfs[split] = load_csv(str(path))
    return split_dfs


def fit_standardizer(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def transform_standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def make_meta_df(df: pd.DataFrame, meta_cols: List[str]) -> pd.DataFrame:
    if not meta_cols:
        return pd.DataFrame(index=df.index)
    out = df[meta_cols].copy()
    for c in out.columns:
        out[c] = out[c].fillna("")
    return out


def build_action_vocab(precursor_names: List[str]) -> List[str]:
    return precursor_names + [STOP_TOKEN]


def encode_reference_trajectories(
    y_multi_hot: np.ndarray,
    precursor_names: List[str],
    action_to_id: Dict[str, int],
    max_set_len: int,
) -> Dict[str, np.ndarray]:
    stop_id = action_to_id[STOP_TOKEN]
    n = y_multi_hot.shape[0]

    traj_actions = np.full((n, max_set_len + 1), stop_id, dtype=np.int64)
    traj_mask = np.zeros((n, max_set_len + 1), dtype=np.int64)
    set_len = np.zeros(n, dtype=np.int64)

    for i in range(n):
        active_idx = np.where(y_multi_hot[i] > 0)[0].tolist()
        actions = [action_to_id[precursor_names[j]] for j in active_idx]
        set_len[i] = len(actions)

        if actions:
            traj_actions[i, : len(actions)] = actions
        traj_actions[i, len(actions)] = stop_id
        traj_mask[i, : len(actions) + 1] = 1

    return {
        "traj_actions": traj_actions,
        "traj_mask": traj_mask,
        "set_len": set_len,
    }


def summarize_split(
    split_name: str,
    x: np.ndarray,
    y_multi_hot: np.ndarray,
    traj_pack: Dict[str, np.ndarray],
    npz_path: Path,
    meta_path: Path,
) -> Dict[str, Any]:
    set_len = traj_pack["set_len"]
    return {
        "split": split_name,
        "n_rows": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "n_labels": int(y_multi_hot.shape[1]),
        "max_set_len": int(set_len.max()) if len(set_len) else 0,
        "mean_set_len": float(set_len.mean()) if len(set_len) else 0.0,
        "max_traj_len": int(traj_pack["traj_actions"].shape[1]),
        "npz_path": str(npz_path),
        "meta_csv_path": str(meta_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stage2 GFlowNet dataset.")
    parser.add_argument("--project_root", type=str, default="/Users/wyc/SynPred")
    parser.add_argument("--input_mode", type=str, default="hybrid", choices=["descriptor", "hybrid"])
    parser.add_argument("--mode_input_root", type=str, default="", help="Optional canonical mode root")
    parser.add_argument(
        "--train_mode",
        type=str,
        default="relaxed_only",
        choices=["relaxed_only", "gold_only", "curriculum_phase1", "curriculum_phase2"],
        help="Training-data mode used only when --mode_input_root is provided.",
    )
    parser.add_argument("--output_dir", type=str, default="")
    args = parser.parse_args()

    project_root = Path(args.project_root)

    if args.mode_input_root:
        input_paths = select_mode_input_paths(
            mode_input_root=args.mode_input_root,
            train_mode=args.train_mode,
            input_mode=args.input_mode,
        )
        out_dir = Path(args.output_dir) if args.output_dir else (
            project_root / "data" / "interim" / "generative" / "stage2_gflownet_dataset" / args.input_mode / args.train_mode
        )
    else:
        input_paths = select_input_paths(args.input_mode, project_root)
        out_dir = Path(args.output_dir) if args.output_dir else (
            project_root / "data" / "interim" / "generative" / "stage2_gflownet_dataset" / args.input_mode
        )

    ensure_dir(out_dir)

    split_dfs = load_splits(input_paths)

    train_df = split_dfs["train"]
    feature_cols = detect_feature_cols(train_df)
    label_cols = detect_label_cols(train_df)
    meta_cols = detect_meta_cols(train_df)

    if not feature_cols:
        raise ValueError("No feature columns detected.")
    if not label_cols:
        raise ValueError("No precursor label columns detected.")

    for split_name, df in split_dfs.items():
        missing_feat = [c for c in feature_cols if c not in df.columns]
        missing_label = [c for c in label_cols if c not in df.columns]
        if missing_feat:
            raise ValueError(f"Split {split_name} missing feature columns: {missing_feat[:10]}")
        if missing_label:
            raise ValueError(f"Split {split_name} missing label columns: {missing_label[:10]}")

    precursor_names = label_cols_to_names(label_cols)
    action_vocab = build_action_vocab(precursor_names)
    action_to_id = {tok: i for i, tok in enumerate(action_vocab)}
    stop_id = action_to_id[STOP_TOKEN]

    arrays: Dict[str, Dict[str, Any]] = {}
    global_max_set_len = 0

    for split_name, df in split_dfs.items():
        x_raw = df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
        y_multi_hot = df[label_cols].fillna(0).to_numpy(dtype=np.float32)
        meta_df = make_meta_df(df, meta_cols)
        set_len = y_multi_hot.sum(axis=1).astype(np.int64)
        if len(set_len):
            global_max_set_len = max(global_max_set_len, int(set_len.max()))

        arrays[split_name] = {
            "x_raw": x_raw,
            "y_multi_hot": y_multi_hot,
            "meta": meta_df,
            "set_len": set_len,
        }

    mean, std = fit_standardizer(arrays["train"]["x_raw"])

    summary: Dict[str, Any] = {
        "config": {
            "project_root": str(project_root),
            "input_mode": args.input_mode,
            "mode_input_root": args.mode_input_root,
            "train_mode": args.train_mode if args.mode_input_root else "",
            "output_dir": str(out_dir),
        },
        "input_paths": {k: str(v) for k, v in input_paths.items()},
        "schema": {
            "n_features": int(len(feature_cols)),
            "n_precursors": int(len(precursor_names)),
            "action_vocab_size": int(len(action_vocab)),
            "global_max_set_len": int(global_max_set_len),
            "max_traj_len": int(global_max_set_len + 1),
            "feature_cols_path": str(out_dir / "feature_cols.json"),
            "label_cols_path": str(out_dir / "label_cols.json"),
            "precursor_names_path": str(out_dir / "precursor_names.json"),
            "action_vocab_path": str(out_dir / "action_vocab.json"),
            "action_to_id_path": str(out_dir / "action_to_id.json"),
            "special_tokens": {
                "stop": STOP_TOKEN,
                "stop_id": int(stop_id),
            },
            "meta_cols": meta_cols,
        },
        "splits": {},
    }

    write_json(out_dir / "feature_cols.json", feature_cols)
    write_json(out_dir / "label_cols.json", label_cols)
    write_json(out_dir / "precursor_names.json", precursor_names)
    write_json(out_dir / "action_vocab.json", action_vocab)
    write_json(out_dir / "action_to_id.json", action_to_id)
    np.save(out_dir / "feature_mean.npy", mean)
    np.save(out_dir / "feature_std.npy", std)

    for split_name, pack in arrays.items():
        x = transform_standardize(pack["x_raw"], mean, std)
        traj_pack = encode_reference_trajectories(
            y_multi_hot=pack["y_multi_hot"],
            precursor_names=precursor_names,
            action_to_id=action_to_id,
            max_set_len=global_max_set_len,
        )

        npz_path = out_dir / f"{split_name}.npz"
        meta_path = out_dir / f"{split_name}_meta.csv"

        np.savez_compressed(
            npz_path,
            x_raw=pack["x_raw"],
            x=x,
            y_multi_hot=pack["y_multi_hot"],
            traj_actions=traj_pack["traj_actions"],
            traj_mask=traj_pack["traj_mask"],
            set_len=traj_pack["set_len"],
        )
        pack["meta"].to_csv(meta_path, index=False)

        summary["splits"][split_name] = summarize_split(
            split_name=split_name,
            x=x,
            y_multi_hot=pack["y_multi_hot"],
            traj_pack=traj_pack,
            npz_path=npz_path,
            meta_path=meta_path,
        )

    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
