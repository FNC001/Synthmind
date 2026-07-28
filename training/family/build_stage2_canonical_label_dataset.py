#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synthmind.chemistry.precursor_normalization import (  # noqa: E402
    CANONICALIZATION_VERSION,
    PrecursorNormalization,
    normalize_many,
)


SPLITS = ("train", "val", "test")
GENERATED_LABEL_FILES = {
    "action_to_id.json",
    "action_vocab.json",
    "label_cols.json",
    "label_names.json",
    "precursor_names.json",
    "precursor_canonicalization.json",
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def representative_label(
    label_ids: Sequence[int],
    names: Sequence[str],
    records: Sequence[PrecursorNormalization],
    train_frequency: np.ndarray,
) -> int:
    def preference(label_id: int) -> tuple[float, int, int, str]:
        raw = str(names[label_id])
        record = records[label_id]
        typography_penalty = raw.count("((") + raw.count("_()") + raw.count(".")
        return (
            -float(train_frequency[label_id]),
            int(typography_penalty),
            len(raw),
            record.normalized_text,
        )

    return min((int(value) for value in label_ids), key=preference)


def remap_multihot(y: np.ndarray, old_to_new: np.ndarray, new_count: int) -> np.ndarray:
    output = np.zeros((len(y), int(new_count)), dtype=np.float32)
    for old_id, new_id in enumerate(old_to_new.tolist()):
        output[:, int(new_id)] = np.maximum(output[:, int(new_id)], y[:, int(old_id)])
    return output


def build_trajectories(y: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stop_id = int(y.shape[1])
    actions = np.full((len(y), int(width)), stop_id, dtype=np.int64)
    mask = np.zeros((len(y), int(width)), dtype=np.int64)
    set_len = np.zeros(len(y), dtype=np.int64)
    for row_index, row in enumerate(y):
        labels = np.flatnonzero(row > 0.5).astype(np.int64)
        if len(labels) + 1 > int(width):
            raise ValueError(
                f"canonical target set at row {row_index} needs {len(labels) + 1} trajectory slots; "
                f"available width is {width}"
            )
        actions[row_index, : len(labels)] = labels
        mask[row_index, : len(labels) + 1] = 1
        set_len[row_index] = int(len(labels))
    return actions, mask, set_len


def copy_nonlabel_artifacts(input_dir: Path, output_dir: Path) -> None:
    for source in input_dir.iterdir():
        if not source.is_file() or source.name in GENERATED_LABEL_FILES:
            continue
        if source.name in {*(f"{split}.npz" for split in SPLITS), "summary.json", "split_manifest.json"}:
            continue
        shutil.copy2(source, output_dir / source.name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Stage2 pack whose formula-equivalent precursor spellings share one label."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    names = [
        str(value)
        for value in json.loads((input_dir / "precursor_names.json").read_text(encoding="utf-8"))
    ]
    packs: Dict[str, np.lib.npyio.NpzFile] = {
        split: np.load(input_dir / f"{split}.npz", allow_pickle=True) for split in SPLITS
    }
    train_y = np.asarray(packs["train"]["y_multi_hot"], dtype=np.float32)
    if train_y.shape[1] != len(names):
        raise ValueError(f"label mismatch: train_y={train_y.shape}, names={len(names)}")

    records = normalize_many(names)
    groups_by_key: Dict[str, List[int]] = defaultdict(list)
    for old_id, record in enumerate(records):
        groups_by_key[record.canonical_key].append(int(old_id))
    ordered_groups = sorted(groups_by_key.values(), key=lambda values: min(values))
    train_frequency = np.asarray(train_y.sum(axis=0), dtype=np.float64)

    old_to_new = np.full(len(names), -1, dtype=np.int64)
    representatives: List[int] = []
    group_records: List[Dict[str, Any]] = []
    for new_id, old_ids in enumerate(ordered_groups):
        representative = representative_label(old_ids, names, records, train_frequency)
        representatives.append(int(representative))
        for old_id in old_ids:
            old_to_new[int(old_id)] = int(new_id)
        group_records.append({
            "canonical_label_id": int(new_id),
            "canonical_key": records[old_ids[0]].canonical_key,
            "canonical_formula": records[old_ids[0]].canonical_formula,
            "representative_old_label_id": int(representative),
            "representative_name": names[representative],
            "old_label_ids": [int(value) for value in old_ids],
            "aliases": [names[int(value)] for value in old_ids],
            "train_frequency": int(train_frequency[old_ids].sum()),
            "normalization_status": records[old_ids[0]].status,
        })
    if (old_to_new < 0).any():
        raise RuntimeError("some original precursor labels were not mapped")

    canonical_names = [names[label_id] for label_id in representatives]
    trajectory_width = max(int(packs[split]["traj_actions"].shape[1]) for split in SPLITS)
    split_label_counts: Dict[str, Dict[str, int]] = {}
    for split in SPLITS:
        source = packs[split]
        remapped_y = remap_multihot(
            np.asarray(source["y_multi_hot"], dtype=np.float32), old_to_new, len(canonical_names)
        )
        actions, mask, set_len = build_trajectories(remapped_y, trajectory_width)
        arrays = {key: np.asarray(source[key]) for key in source.files}
        arrays.update({
            "y_multi_hot": remapped_y,
            "traj_actions": actions,
            "traj_mask": mask,
            "set_len": set_len,
        })
        np.savez_compressed(output_dir / f"{split}.npz", **arrays)
        split_label_counts[split] = {
            "rows": int(len(remapped_y)),
            "positive_assignments_raw": int(np.asarray(source["y_multi_hot"]).sum()),
            "positive_assignments_canonical": int(remapped_y.sum()),
            "rows_with_collapsed_duplicate_aliases": int(
                np.sum(np.asarray(source["y_multi_hot"]).sum(axis=1) > remapped_y.sum(axis=1))
            ),
        }

    copy_nonlabel_artifacts(input_dir, output_dir)
    write_json(output_dir / "precursor_names.json", canonical_names)
    write_json(output_dir / "label_names.json", canonical_names)
    write_json(output_dir / "label_cols.json", [f"label_prec__{name}" for name in canonical_names])
    action_vocab = [*canonical_names, "<stop>"]
    write_json(output_dir / "action_vocab.json", action_vocab)
    write_json(output_dir / "action_to_id.json", {name: index for index, name in enumerate(action_vocab)})

    merged_groups = [group for group in group_records if len(group["old_label_ids"]) > 1]
    canonicalization = {
        "version": CANONICALIZATION_VERSION,
        "source_dir": str(input_dir),
        "source_precursor_names_sha256": sha256_file(input_dir / "precursor_names.json"),
        "original_label_count": int(len(names)),
        "canonical_label_count": int(len(canonical_names)),
        "labels_removed_as_aliases": int(len(names) - len(canonical_names)),
        "composition_parsed_count": int(sum(record.status == "composition" for record in records)),
        "text_fallback_count": int(sum(record.status != "composition" for record in records)),
        "merged_group_count": int(len(merged_groups)),
        "old_to_canonical_label_id": old_to_new.tolist(),
        "groups": group_records,
        "normalization_records": [record.to_dict() for record in records],
        "split_assignment_audit": split_label_counts,
        "metric_policy": {
            "raw_label_metric": "retain for historical comparison",
            "canonical_metric": "formula-equivalent spellings count as the same chemical precursor",
            "family_equivalence": "periodic-family routing remains a feature and does not merge LiCl with NaCl",
        },
    }
    write_json(output_dir / "precursor_canonicalization.json", canonicalization)
    pd.DataFrame(merged_groups).to_csv(output_dir / "precursor_alias_groups.csv", index=False)

    source_summary_path = input_dir / "summary.json"
    summary = json.loads(source_summary_path.read_text(encoding="utf-8")) if source_summary_path.exists() else {}
    summary["precursor_canonicalization"] = {
        key: canonicalization[key]
        for key in (
            "version", "original_label_count", "canonical_label_count",
            "labels_removed_as_aliases", "composition_parsed_count",
            "text_fallback_count", "merged_group_count", "split_assignment_audit",
        )
    }
    if isinstance(summary.get("schema"), dict):
        summary["schema"]["n_labels"] = int(len(canonical_names))
        summary["schema"]["precursor_canonicalization_version"] = CANONICALIZATION_VERSION
    write_json(output_dir / "summary.json", summary)

    manifest_path = input_dir / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest["label_canonicalization"] = {
        "version": CANONICALIZATION_VERSION,
        "source_precursor_names_sha256": canonicalization["source_precursor_names_sha256"],
        "original_label_count": int(len(names)),
        "canonical_label_count": int(len(canonical_names)),
        "split_membership_unchanged": True,
    }
    write_json(output_dir / "split_manifest.json", manifest)

    print(json.dumps({
        "output_dir": str(output_dir),
        "original_label_count": len(names),
        "canonical_label_count": len(canonical_names),
        "labels_removed_as_aliases": len(names) - len(canonical_names),
        "merged_group_count": len(merged_groups),
        "split_assignment_audit": split_label_counts,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
