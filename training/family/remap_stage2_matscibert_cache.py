#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np


def sequence_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(str(value) for value in values).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remap the label axis of a Stage2 MatSciBERT embedding cache to canonical labels."
    )
    parser.add_argument("--input_cache", required=True)
    parser.add_argument("--canonical_input_dir", required=True)
    parser.add_argument("--output_cache", required=True)
    args = parser.parse_args()

    input_cache = Path(args.input_cache).expanduser().resolve()
    input_dir = Path(args.canonical_input_dir).expanduser().resolve()
    output_cache = Path(args.output_cache).expanduser().resolve()
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    canonicalization = json.loads(
        (input_dir / "precursor_canonicalization.json").read_text(encoding="utf-8")
    )
    names = [
        str(value)
        for value in json.loads((input_dir / "precursor_names.json").read_text(encoding="utf-8"))
    ]
    representatives = np.asarray(
        [int(group["representative_old_label_id"]) for group in canonicalization["groups"]],
        dtype=np.int64,
    )
    if len(representatives) != len(names):
        raise ValueError("canonical group count does not match precursor vocabulary")

    with np.load(input_cache, allow_pickle=False) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    for key in ("precursor_common_mean", "precursor_role_mean"):
        if key not in arrays:
            raise KeyError(f"embedding cache is missing {key}")
        arrays[key] = np.asarray(arrays[key][representatives], dtype=np.float32)
    arrays["precursor_names_sha256"] = np.asarray(sequence_sha256(names))
    arrays["canonicalization_version"] = np.asarray(canonicalization["version"])
    np.savez_compressed(output_cache, **arrays)

    report = {
        "input_cache": str(input_cache),
        "output_cache": str(output_cache),
        "canonicalization_version": canonicalization["version"],
        "original_label_count": canonicalization["original_label_count"],
        "canonical_label_count": len(names),
        "label_embedding_policy": "select_frequency_preferred_representative_spelling",
        "query_embeddings_unchanged": True,
    }
    output_cache.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
