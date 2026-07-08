#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


PAD_TOKEN = "<pad>"
PAD_ID = 0


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
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


def collapse_y(y_old: np.ndarray, old_to_new: Dict[int, int], n_new: int) -> np.ndarray:
    y_new = np.zeros((y_old.shape[0], n_new), dtype=np.float32)
    old_pos = np.where(y_old > 0)
    if len(old_pos[0]):
        new_cols = np.asarray([old_to_new[int(j)] for j in old_pos[1]], dtype=np.int64)
        y_new[old_pos[0], new_cols] = 1.0
    return y_new


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Build canonical Stage2 v4 dataset from v2 and alias patch.")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--patch_csv", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_slots", type=int, default=7)
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    old_names = [str(x) for x in load_json(input_dir / "precursor_names.json")]
    patch_df = pd.read_csv(args.patch_csv) if Path(args.patch_csv).exists() else pd.DataFrame(columns=["raw_label", "patched_label"])
    patch = dict(zip(patch_df["raw_label"].astype(str), patch_df["patched_label"].astype(str)))
    new_name_for_old = [patch.get(x, x) for x in old_names]
    new_names = sorted(set(new_name_for_old))
    new_idx = {p: i for i, p in enumerate(new_names)}
    old_to_new = {i: new_idx[new_name_for_old[i]] for i in range(len(old_names))}

    for fname in ["feature_cols.json", "feature_mean.npy", "feature_std.npy"]:
        src = input_dir / fname
        if src.exists():
            if src.suffix == ".npy":
                np.save(output_dir / fname, np.load(src))
            else:
                (output_dir / fname).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    write_json(output_dir / "precursor_names.json", new_names)
    write_json(output_dir / "label_cols.json", [f"label_prec__{p}" for p in new_names])
    write_json(output_dir / "slot_vocab.json", [PAD_TOKEN] + new_names)
    write_json(output_dir / "slot_to_id.json", {tok: i for i, tok in enumerate([PAD_TOKEN] + new_names)})
    alias_rows = [{"original_precursor": old, "canonical_v4_precursor": new_name_for_old[i], "was_patched": old != new_name_for_old[i]} for i, old in enumerate(old_names)]
    pd.DataFrame(alias_rows).to_csv(output_dir / "precursor_alias_report_v4.csv", index=False)

    summary: Dict[str, Any] = {
        "v2_label_count": len(old_names),
        "v4_label_count": len(new_names),
        "merged_alias_count": len(old_names) - len(new_names),
        "n_patch_rows": int(len(patch_df)),
        "splits": {},
    }
    for split in ["train", "val", "test"]:
        pack = load_npz(input_dir / f"{split}.npz")
        y_old = np.asarray(pack["y_multi_hot"], dtype=np.float32)
        y_new = collapse_y(y_old, old_to_new, len(new_names))
        slots = build_slots(y_new, int(args.n_slots))
        np.savez_compressed(
            output_dir / f"{split}.npz",
            x_raw=np.asarray(pack["x_raw"], dtype=np.float32),
            x=np.asarray(pack["x"], dtype=np.float32),
            y_multi_hot=y_new,
            slot_targets=slots["slot_targets"],
            slot_mask=slots["slot_mask"],
            set_len=slots["set_len"],
            overflow=slots["overflow"],
        )
        meta = pd.read_csv(input_dir / f"{split}_meta.csv")
        meta.to_csv(output_dir / f"{split}_meta.csv", index=False)
        summary["splits"][split] = {
            "n_rows": int(y_new.shape[0]),
            "mean_old_set_len": float(y_old.sum(axis=1).mean()),
            "mean_new_set_len": float(y_new.sum(axis=1).mean()),
            "rows_changed_set_len": int(np.sum(y_old.sum(axis=1) != y_new.sum(axis=1))),
        }

    # Remaining OOV relative to v4 train positives.
    y_train = load_npz(output_dir / "train.npz")["y_multi_hot"]
    train_pos = set(np.where(y_train.sum(axis=0) > 0)[0].tolist())
    rows = []
    y_test = load_npz(output_dir / "test.npz")["y_multi_hot"]
    for j in np.where((y_test.sum(axis=0) > 0) & (y_train.sum(axis=0) == 0))[0]:
        rows.append({"label": new_names[int(j)], "test_count": int(y_test[:, j].sum())})
    rem = pd.DataFrame(rows).sort_values("test_count", ascending=False) if rows else pd.DataFrame(columns=["label", "test_count"])
    rem.to_csv(output_dir / "remaining_oov_labels_v4.csv", index=False)
    summary["remaining_oov_label_count"] = int(len(rem))
    write_json(output_dir / "summary.json", summary)
    report = [
        "# Canonical v4 Report",
        "",
        f"- v2 label count: {len(old_names)}",
        f"- v4 label count: {len(new_names)}",
        f"- merged alias count: {len(old_names) - len(new_names)}",
        f"- patch rows: {len(patch_df)}",
        f"- remaining OOV labels in test vs train: {len(rem)}",
    ]
    (output_dir / "canonical_v4_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
