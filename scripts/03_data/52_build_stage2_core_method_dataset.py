#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set

import numpy as np
import pandas as pd


CORE_METHODS = {"solid_state", "solution", "melt_arc"}
PAD_TOKEN = "<pad>"
PAD_ID = 0


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_builtin(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    return {k: arr[k] for k in arr.files}


def get_y(pack: Dict[str, np.ndarray]) -> np.ndarray:
    for key in ["y_multi_hot", "y", "labels", "targets"]:
        if key in pack:
            return (np.asarray(pack[key]) > 0).astype(np.float32)
    raise KeyError(f"No label matrix found in keys={list(pack)}")


def build_slots(y_multi_hot: np.ndarray, n_slots: int) -> Dict[str, np.ndarray]:
    n = y_multi_hot.shape[0]
    slot_targets = np.full((n, n_slots), PAD_ID, dtype=np.int64)
    slot_mask = np.zeros((n, n_slots), dtype=np.int64)
    set_len = np.zeros(n, dtype=np.int64)
    overflow = np.zeros(n, dtype=np.int64)
    for i in range(n):
        active = np.where(y_multi_hot[i] > 0)[0].tolist()
        set_len[i] = len(active)
        if len(active) > n_slots:
            overflow[i] = 1
            active = active[:n_slots]
        if active:
            slot_targets[i, : len(active)] = np.asarray([j + 1 for j in active], dtype=np.int64)
            slot_mask[i, : len(active)] = 1
    return {"slot_targets": slot_targets, "slot_mask": slot_mask, "set_len": set_len, "overflow": overflow}


def label_indices_for_core(input_dir: Path, names: Sequence[str], keep_all_labels: bool) -> List[int]:
    if keep_all_labels:
        return list(range(len(names)))
    used: Set[int] = set()
    for split in ["train", "val", "test"]:
        pack = load_npz(input_dir / f"{split}.npz")
        meta = pd.read_csv(input_dir / f"{split}_meta.csv")
        mask = meta["reaction_method"].astype(str).isin(CORE_METHODS).to_numpy()
        y = get_y(pack)
        if mask.any():
            used |= set(np.where(y[mask].sum(axis=0) > 0)[0].tolist())
    return sorted(used)


def save_split(input_dir: Path, output_dir: Path, split: str, label_idx: Sequence[int], n_slots: int) -> Dict[str, Any]:
    pack = load_npz(input_dir / f"{split}.npz")
    meta = pd.read_csv(input_dir / f"{split}_meta.csv")
    mask = meta["reaction_method"].astype(str).isin(CORE_METHODS).to_numpy()
    rows = np.where(mask)[0]
    y = get_y(pack)[rows][:, label_idx]
    x = np.asarray(pack["x"], dtype=np.float32)[rows]
    x_raw = np.asarray(pack.get("x_raw", pack["x"]), dtype=np.float32)[rows]
    slots = build_slots(y, n_slots)
    np.savez_compressed(
        output_dir / f"{split}.npz",
        x_raw=x_raw,
        x=x,
        y_multi_hot=y.astype(np.float32),
        slot_targets=slots["slot_targets"],
        slot_mask=slots["slot_mask"],
        set_len=slots["set_len"],
        overflow=slots["overflow"],
    )
    out_meta = meta.loc[rows].copy().reset_index(drop=False).rename(columns={"index": "original_sample_index"})
    out_meta["core_sample_index"] = np.arange(len(out_meta), dtype=int)
    out_meta.to_csv(output_dir / f"{split}_meta.csv", index=False)
    return {
        "n_rows": int(len(rows)),
        "original_rows": int(len(meta)),
        "method_distribution": out_meta["reaction_method"].value_counts().to_dict(),
        "average_precursor_set_size": float(y.sum(axis=1).mean()) if len(y) else 0.0,
        "max_precursor_set_size": int(y.sum(axis=1).max()) if len(y) else 0,
    }


def label_counts(input_dir: Path, split: str, label_idx: Sequence[int]) -> np.ndarray:
    pack = load_npz(input_dir / f"{split}.npz")
    meta = pd.read_csv(input_dir / f"{split}_meta.csv")
    mask = meta["reaction_method"].astype(str).isin(CORE_METHODS).to_numpy()
    return get_y(pack)[mask][:, label_idx].sum(axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Stage2 core-method-only dataset for solid_state, solution, and melt_arc.")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--ontology_csv", required=True)
    ap.add_argument("--n_slots", type=int, default=7)
    ap.add_argument("--keep_all_labels", action="store_true", help="Keep original label dimension instead of dropping labels unused by core splits.")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    old_names = [str(x) for x in load_json(input_dir / "precursor_names.json")]
    label_idx = label_indices_for_core(input_dir, old_names, bool(args.keep_all_labels))
    names = [old_names[i] for i in label_idx]

    for fname in ["feature_cols.json", "feature_mean.npy", "feature_std.npy"]:
        src = input_dir / fname
        if src.exists():
            if src.suffix == ".npy":
                np.save(output_dir / fname, np.load(src))
            else:
                (output_dir / fname).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    write_json(output_dir / "precursor_names.json", names)
    write_json(output_dir / "label_cols.json", [f"label_prec__{p}" for p in names])
    write_json(output_dir / "slot_vocab.json", [PAD_TOKEN] + names)
    write_json(output_dir / "slot_to_id.json", {tok: i for i, tok in enumerate([PAD_TOKEN] + names)})
    write_json(output_dir / "old_label_indices.json", [int(x) for x in label_idx])

    summary: Dict[str, Any] = {
        "core_methods": sorted(CORE_METHODS),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "all_method_label_count": int(len(old_names)),
        "core_label_count": int(len(names)),
        "dropped_label_count": int(len(old_names) - len(names)),
        "splits": {},
    }
    for split in ["train", "val", "test"]:
        summary["splits"][split] = save_split(input_dir, output_dir, split, label_idx, int(args.n_slots))

    train_pos = label_counts(input_dir, "train", label_idx)
    test_pos = label_counts(input_dir, "test", label_idx)
    val_pos = label_counts(input_dir, "val", label_idx)
    oov_label_idx = np.where((test_pos > 0) & (train_pos == 0))[0]
    train_seen = set(np.where(train_pos > 0)[0].tolist())
    test_pack = load_npz(output_dir / "test.npz")
    test_y = get_y(test_pack)
    oov_rows = int(np.sum([any((j not in train_seen) for j in np.where(test_y[i] > 0)[0]) for i in range(test_y.shape[0])]))
    summary["oov_label_count_test_vs_train"] = int(len(oov_label_idx))
    summary["oov_row_count_test_vs_train"] = int(oov_rows)
    summary["val_oov_label_count_vs_train"] = int(np.sum((val_pos > 0) & (train_pos == 0)))

    ont = pd.read_csv(args.ontology_csv)
    fam_lookup = dict(zip(ont["canonical_precursor"].astype(str), ont["precursor_family"].astype(str)))
    fam_counter: Counter = Counter()
    for j in np.where((train_pos + val_pos + test_pos) > 0)[0]:
        fam_counter[fam_lookup.get(names[int(j)], "unknown")] += int(train_pos[j] + val_pos[j] + test_pos[j])
    summary["family_distribution_by_label_occurrence"] = dict(fam_counter.most_common())

    write_json(output_dir / "summary.json", summary)
    lines = ["# Core Method Stage2 Dataset", "", "## Methods", ""]
    for m in sorted(CORE_METHODS):
        lines.append(f"- {m}")
    lines += [
        "",
        "## Counts",
        "",
        f"- all-method label count: {len(old_names)}",
        f"- core label count: {len(names)}",
        f"- dropped labels: {len(old_names) - len(names)}",
        f"- test OOV label count vs train: {summary['oov_label_count_test_vs_train']}",
        f"- test OOV row count vs train: {summary['oov_row_count_test_vs_train']}",
        "",
        "## Splits",
    ]
    for split, rec in summary["splits"].items():
        lines.append(f"- {split}: n={rec['n_rows']}, method_distribution={rec['method_distribution']}, avg_set_size={rec['average_precursor_set_size']:.3f}")
    lines += ["", "## Family Distribution"]
    for fam, n in summary["family_distribution_by_label_occurrence"].items():
        lines.append(f"- {fam}: {n}")
    (output_dir / "core_method_dataset_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
