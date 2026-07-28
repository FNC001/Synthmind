#!/usr/bin/env python3
"""Build a deterministically ordered full-file inventory for a synthmind release.

The generated SHA256SUMS deliberately excludes itself. FILE_INDEX.tsv lists
all payload files but not itself; SHA256SUMS additionally covers FILE_INDEX.tsv.
This avoids circular hashes while still protecting the inventory.
RELEASE_SUMMARY.json contains a build timestamp, so rebuilding the inventory
intentionally changes the control-file bytes even when the payload is unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


MANIFEST_DIR = "00_OVERVIEW_AND_MANIFEST"
INDEX_REL = f"{MANIFEST_DIR}/FILE_INDEX.tsv"
HASH_REL = f"{MANIFEST_DIR}/SHA256SUMS"
SUMMARY_REL = f"{MANIFEST_DIR}/RELEASE_SUMMARY.json"
EXCLUDED_FROM_INDEX = {INDEX_REL, HASH_REL}
EXCLUDED_FROM_SUMMARY_STATS = {INDEX_REL, HASH_REL, SUMMARY_REL}
FORBIDDEN_NAMES = {
    ".DS_Store",
    "__MACOSX",
    "__pycache__",
    ".pytest_cache",
    ".ipynb_checkpoints",
    ".claude",
}
FORBIDDEN_SUFFIXES = (
    ".pyc",
    ".pyo",
    ".swp",
    ".swo",
    ".tmp",
    ".temp",
    ".bak",
    ".orig",
    ".rej",
    ".old",
    ".save",
    "-bak",
    "~",
)


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def relative_files(root: Path) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise SystemExit(f"Refusing to freeze a release containing a symlink: {path}")
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if "\n" in rel or "\r" in rel or "\t" in rel:
            raise SystemExit(f"Unsupported control character in path: {rel!r}")
        rows.append((rel, path))
    return sorted(rows, key=lambda item: item[0].encode("utf-8"))


def assert_clean_tree(root: Path) -> None:
    problems: list[str] = []
    if stat.S_IMODE(root.stat().st_mode) != 0o755:
        problems.append(f". [mode={stat.S_IMODE(root.stat().st_mode):04o}, expected=0755]")
    for path in root.rglob("*"):
        name = path.name
        if name in FORBIDDEN_NAMES or name.startswith("._") or ".bak" in name or name.endswith(FORBIDDEN_SUFFIXES):
            problems.append(path.relative_to(root).as_posix())
            continue
        actual_mode = stat.S_IMODE(path.stat().st_mode)
        expected_mode = 0o755 if path.is_dir() or (path.is_file() and path.suffix == ".sh") else 0o644
        if actual_mode != expected_mode:
            problems.append(
                f"{path.relative_to(root).as_posix()} [mode={actual_mode:04o}, expected={expected_mode:04o}]"
            )
    if problems:
        preview = "\n".join(problems[:30])
        raise SystemExit(f"Tree hygiene/permission policy failed before freeze:\n{preview}")


def write_summary(root: Path) -> None:
    # SUMMARY_REL is excluded from these statistics to avoid a self-referential
    # byte count. It is still included in FILE_INDEX.tsv and SHA256SUMS.
    rows = [(rel, path) for rel, path in relative_files(root) if rel not in EXCLUDED_FROM_SUMMARY_STATS]
    top_counts: dict[str, int] = defaultdict(int)
    top_bytes: dict[str, int] = defaultdict(int)
    total_bytes = 0
    for rel, path in rows:
        size = path.stat().st_size
        top = rel.split("/", 1)[0]
        top_counts[top] += 1
        top_bytes[top] += size
        total_bytes += size

    summary = {
        "schema": "synthmind_release_summary_v1",
        "release": root.name,
        "status": "frozen_payload_ready_for_hashing",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "payload_file_count_excluding_inventory_files_and_summary": len(rows),
        "payload_logical_bytes_excluding_inventory_files_and_summary": total_bytes,
        "top_level": {
            key: {"files": top_counts[key], "logical_bytes": top_bytes[key]}
            for key in sorted(top_counts)
        },
        "integrity_contract": {
            "file_index": INDEX_REL,
            "sha256_manifest": HASH_REL,
            "sha256_manifest_excludes_itself": True,
            "file_index_excludes_itself": True,
            "source_archives_are_preserved_as_immutable_provenance_snapshots": True,
            "symlinks_allowed": False,
            "macos_metadata_or_pyc_allowed": False,
            "directory_mode": "0755",
            "regular_file_mode": "0644",
            "shell_entry_mode": "0755",
        },
    }
    output = root / SUMMARY_REL
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output.chmod(0o644)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_root", type=Path)
    args = parser.parse_args()
    root = args.release_root.resolve()
    if not (root / MANIFEST_DIR).is_dir():
        raise SystemExit(f"Not a synthmind release root: {root}")

    assert_clean_tree(root)
    write_summary(root)

    payload = [(rel, path) for rel, path in relative_files(root) if rel not in EXCLUDED_FROM_INDEX]
    index_path = root / INDEX_REL
    with index_path.open("w", encoding="utf-8", newline="\n") as index:
        index.write("sha256\tbytes\tmode\trelative_path\n")
        hash_rows: list[tuple[str, str]] = []
        for rel, path in payload:
            digest = sha256(path)
            mode = f"{stat.S_IMODE(path.stat().st_mode):04o}"
            index.write(f"{digest}\t{path.stat().st_size}\t{mode}\t{rel}\n")
            hash_rows.append((digest, rel))
    index_path.chmod(0o644)

    hash_rows.append((sha256(index_path), INDEX_REL))
    hash_rows.sort(key=lambda item: item[1].encode("utf-8"))
    hash_path = root / HASH_REL
    with hash_path.open("w", encoding="utf-8", newline="\n") as handle:
        for digest, rel in hash_rows:
            handle.write(f"{digest}  {rel}\n")
    hash_path.chmod(0o644)

    print(
        json.dumps(
            {
                "status": "PASS",
                "release_root": str(root),
                "hashed_files": len(hash_rows),
                "file_index": str(index_path),
                "sha256_manifest": str(hash_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
