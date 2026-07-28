#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


PAD_TOKEN = "<pad>"
PAD_ID = 0

DOT_CHARS = ["▪", "•", "·", "∙", "⋅", "･", "．"]
SUBSCRIPT = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    return {k: arr[k] for k in arr.files}


def canonical_precursor(name: str) -> str:
    s = unicodedata.normalize("NFKC", str(name or "")).strip()
    s = s.translate(SUBSCRIPT)
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    for ch in DOT_CHARS:
        s = s.replace(ch, "·")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"(?i)\((?:s|l|g|aq)\)$", "", s)
    s = re.sub(r"(?i)\[(?:s|l|g|aq)\]$", "", s)
    s = s.replace("H20", "H2O")
    s = s.replace("h2o", "H2O")
    s = s.replace("H₂O", "H2O")
    s = re.sub(r"\.([0-9]+)H2O$", r"·\1H2O", s)
    s = re.sub(r"\.H2O$", r"·H2O", s)
    s = re.sub(r"-([0-9]+)H2O$", r"·\1H2O", s)
    s = re.sub(r"(?i)hydrate$", "", s)
    s = re.sub(r"(?i)deionizedwater|distilledwater", "H2O", s)
    aliases = {
        "water": "H2O",
        "Water": "H2O",
        "DIwater": "H2O",
        "deionizedH2O": "H2O",
        "ethanol": "C2H5OH",
        "EtOH": "C2H5OH",
        "methanol": "CH3OH",
        "MeOH": "CH3OH",
        "isopropanol": "(CH3)2CHOH",
        "IPA": "(CH3)2CHOH",
        "ammonia": "NH3",
        "ammoniumhydroxide": "NH4OH",
    }
    return aliases.get(s, s)


def build_mapping(precursors: List[str]) -> tuple[List[str], Dict[int, int], Dict[str, List[str]]]:
    canon_to_originals: Dict[str, List[str]] = defaultdict(list)
    for p in precursors:
        canon_to_originals[canonical_precursor(p)].append(str(p))
    canonical_names = sorted(canon_to_originals)
    canon_idx = {p: i for i, p in enumerate(canonical_names)}
    old_to_new = {i: canon_idx[canonical_precursor(p)] for i, p in enumerate(precursors)}
    return canonical_names, old_to_new, dict(canon_to_originals)


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
    return {
        "slot_targets": slot_targets,
        "slot_mask": slot_mask,
        "set_len": set_len,
        "overflow": overflow,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a Stage2 dataset with canonicalized/merged precursor labels.")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_slots", type=int, default=7)
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    precursor_names = [str(x) for x in load_json(input_dir / "precursor_names.json")]
    canonical_names, old_to_new, canon_to_originals = build_mapping(precursor_names)

    for fname in ["feature_cols.json", "feature_mean.npy", "feature_std.npy"]:
        src = input_dir / fname
        if src.exists():
            dst = output_dir / fname
            if src.suffix == ".npy":
                np.save(dst, np.load(src))
            else:
                dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    label_cols = [f"label_prec__{p}" for p in canonical_names]
    write_json(output_dir / "label_cols.json", label_cols)
    write_json(output_dir / "precursor_names.json", canonical_names)
    write_json(output_dir / "slot_vocab.json", [PAD_TOKEN] + canonical_names)
    write_json(output_dir / "slot_to_id.json", {tok: i for i, tok in enumerate([PAD_TOKEN] + canonical_names)})
    write_json(output_dir / "canonical_to_originals.json", canon_to_originals)

    summary: Dict[str, Any] = {
        "config": vars(args),
        "n_original_precursors": len(precursor_names),
        "n_canonical_precursors": len(canonical_names),
        "n_merged_labels": len(precursor_names) - len(canonical_names),
        "largest_merge_groups": [
            {"canonical": k, "n": len(v), "examples": v[:20]}
            for k, v in sorted(canon_to_originals.items(), key=lambda kv: len(kv[1]), reverse=True)[:30]
            if len(v) > 1
        ],
        "splits": {},
    }

    for split in ["train", "val", "test"]:
        pack = load_npz(input_dir / f"{split}.npz")
        y_new = collapse_y(np.asarray(pack["y_multi_hot"], dtype=np.float32), old_to_new, len(canonical_names))
        slots = build_slots(y_new, int(args.n_slots))
        out_pack = {
            "x_raw": np.asarray(pack["x_raw"], dtype=np.float32),
            "x": np.asarray(pack["x"], dtype=np.float32),
            "y_multi_hot": y_new,
            "slot_targets": slots["slot_targets"],
            "slot_mask": slots["slot_mask"],
            "set_len": slots["set_len"],
            "overflow": slots["overflow"],
        }
        np.savez_compressed(output_dir / f"{split}.npz", **out_pack)
        meta_path = input_dir / f"{split}_meta.csv"
        if meta_path.exists():
            pd.read_csv(meta_path).to_csv(output_dir / f"{split}_meta.csv", index=False)
        old_lens = np.asarray(pack["y_multi_hot"]).sum(axis=1)
        new_lens = y_new.sum(axis=1)
        summary["splits"][split] = {
            "n_rows": int(y_new.shape[0]),
            "mean_old_set_len": float(old_lens.mean()),
            "mean_new_set_len": float(new_lens.mean()),
            "rows_changed_set_len": int(np.sum(old_lens != new_lens)),
            "max_new_set_len": int(new_lens.max()) if len(new_lens) else 0,
            "label_positive_counts_top": {
                canonical_names[int(i)]: int(v)
                for i, v in sorted(enumerate(y_new.sum(axis=0).astype(int)), key=lambda kv: kv[1], reverse=True)[:20]
            },
        }

    write_json(output_dir / "summary.json", summary)
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
