#!/usr/bin/env python3
"""Index dropped/excluded/audit artifacts without modifying source assets."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


TOKENS = ("dropped", "drop_", "excluded", "discard", "reject", "audit")
OUTPUT_REL = "03_CLEANED_AND_MERGED_DATA/06_DROPPED_AND_AUDITS/DROPPED_AUDITS_INDEX.tsv"
PIPELINE_TOP_LEVEL = {
    "02_RAW_DATA",
    "03_CLEANED_AND_MERGED_DATA",
    "04_SPLITS",
    "05_FEATURES_AND_EMBEDDINGS",
    "06_TRAIN_READY_DATA",
    "07_BEST_MODELS",
    "08_GENERATED_OUTPUTS",
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def line_count(path: Path) -> int | None:
    if path.suffix.lower() not in {".csv", ".tsv", ".jsonl", ".log", ".md", ".txt"}:
        return None
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_root", type=Path)
    args = parser.parse_args()
    root = args.release_root.resolve()
    output = root / OUTPUT_REL
    rows = []
    for path in root.rglob("*"):
        if not path.is_file() or path == output:
            continue
        rel = path.relative_to(root).as_posix()
        if rel.split("/", 1)[0] not in PIPELINE_TOP_LEVEL:
            continue
        lowered = rel.lower()
        if not any(token in lowered for token in TOKENS):
            continue
        rows.append((rel, path))
    rows.sort(key=lambda item: item[0].encode("utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("sha256\tbytes\tlines_or_blank\trelative_path\n")
        for rel, path in rows:
            lines = line_count(path)
            handle.write(f"{sha256(path)}\t{path.stat().st_size}\t{'' if lines is None else lines}\t{rel}\n")
    print(f"indexed={len(rows)} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
