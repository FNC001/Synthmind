#!/usr/bin/env python3
"""Verify every frozen release file and reject missing or unexpected files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import stat
from pathlib import Path, PurePosixPath


HASH_REL = "00_OVERVIEW_AND_MANIFEST/SHA256SUMS"
INDEX_REL = "00_OVERVIEW_AND_MANIFEST/FILE_INDEX.tsv"
FORBIDDEN_NAMES = {
    ".DS_Store", "__MACOSX", "__pycache__", ".pytest_cache",
    ".ipynb_checkpoints", ".claude",
}
FORBIDDEN_SUFFIXES = (
    ".pyc", ".pyo", ".swp", ".swo", ".tmp", ".temp", ".bak",
    ".orig", ".rej", ".old", ".save", "-bak", "~",
)


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_relative_path(value: str, source: str) -> None:
    pure = PurePosixPath(value)
    if (
        not value
        or value == "."
        or "\\" in value
        or pure.is_absolute()
        or ".." in pure.parts
        or pure.as_posix() != value
    ):
        raise SystemExit(f"Unsafe or non-canonical path in {source}: {value!r}")


def parse_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw:
            continue
        if len(raw) < 67 or raw[64:66] != "  ":
            raise SystemExit(f"Malformed SHA256SUMS line {line_number}")
        digest, rel = raw[:64], raw[66:]
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise SystemExit(f"Invalid SHA-256 at line {line_number}")
        if rel in entries:
            raise SystemExit(f"Duplicate manifest path: {rel}")
        validate_relative_path(rel, f"SHA256SUMS line {line_number}")
        entries[rel] = digest
    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_root", type=Path)
    args = parser.parse_args()
    root = args.release_root.resolve()
    manifest = root / HASH_REL
    if not manifest.is_file():
        raise SystemExit(f"Missing manifest: {manifest}")

    expected = parse_manifest(manifest)
    actual_paths: set[str] = set()
    symlinks: list[str] = []
    forbidden: list[str] = []
    for path in root.rglob("*"):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            symlinks.append(rel)
        elif path.is_file() and rel != HASH_REL:
            actual_paths.add(rel)
        name = path.name
        if name in FORBIDDEN_NAMES or name.startswith("._") or ".bak" in name or name.endswith(FORBIDDEN_SUFFIXES):
            forbidden.append(rel)

    missing = sorted(set(expected) - actual_paths)
    unexpected = sorted(actual_paths - set(expected))
    mismatches: list[dict[str, str]] = []
    for rel in sorted(set(expected) & actual_paths, key=lambda item: item.encode("utf-8")):
        actual = sha256(root / rel)
        if actual != expected[rel]:
            mismatches.append({"path": rel, "expected": expected[rel], "actual": actual})

    permission_mismatches: list[dict[str, str]] = []
    index_errors: list[str] = []
    index_rows: set[str] = set()
    index_path = root / INDEX_REL
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                rel = row["relative_path"]
                validate_relative_path(rel, "FILE_INDEX.tsv")
                if rel in index_rows:
                    index_errors.append(f"duplicate index path: {rel}")
                    continue
                index_rows.add(rel)
                path = root / rel
                if not path.is_file():
                    index_errors.append(f"index path is missing: {rel}")
                    continue
                if rel not in expected or row["sha256"] != expected[rel]:
                    index_errors.append(f"index/manifest hash disagreement: {rel}")
                if row["bytes"] != str(path.stat().st_size):
                    index_errors.append(f"index byte count disagreement: {rel}")
                actual_mode = f"{stat.S_IMODE(path.stat().st_mode):04o}"
                policy_mode = "0755" if path.suffix == ".sh" else "0644"
                if actual_mode != row["mode"] or actual_mode != policy_mode:
                    permission_mismatches.append(
                        {"path": rel, "index": row["mode"], "policy": policy_mode, "actual": actual_mode}
                    )
        expected_index_rows = set(expected) - {INDEX_REL}
        if index_rows != expected_index_rows:
            index_errors.append(
                f"index coverage mismatch: missing={len(expected_index_rows - index_rows)} "
                f"unexpected={len(index_rows - expected_index_rows)}"
            )
    else:
        index_errors.append(f"missing index: {INDEX_REL}")

    directory_mode_mismatches = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_dir() and stat.S_IMODE(path.stat().st_mode) != 0o755
    ]
    if stat.S_IMODE(root.stat().st_mode) != 0o755:
        directory_mode_mismatches.insert(0, ".")
    manifest_mode_mismatch = stat.S_IMODE(manifest.stat().st_mode) != 0o644
    index_mode_mismatch = not index_path.is_file() or stat.S_IMODE(index_path.stat().st_mode) != 0o644

    status = "PASS" if not (
        missing
        or unexpected
        or mismatches
        or symlinks
        or forbidden
        or permission_mismatches
        or directory_mode_mismatches
        or manifest_mode_mismatch
        or index_mode_mismatch
        or index_errors
    ) else "FAIL"
    report = {
        "status": status,
        "release_root": str(root),
        "verified_files": len(expected) - len(missing),
        "manifest_entries": len(expected),
        "missing": missing,
        "unexpected": unexpected,
        "hash_mismatches": mismatches,
        "symlinks": symlinks,
        "forbidden_auxiliary_entries": forbidden,
        "file_permission_mismatches": permission_mismatches,
        "directory_mode_mismatches": directory_mode_mismatches,
        "sha256_manifest_mode_mismatch": manifest_mode_mismatch,
        "file_index_mode_mismatch": index_mode_mismatch,
        "index_errors": index_errors,
        "manifest_sha256": sha256(manifest),
    }
    print(json.dumps(report, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
