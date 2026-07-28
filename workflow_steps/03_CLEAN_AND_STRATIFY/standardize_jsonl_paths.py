#!/usr/bin/env python3
"""Create strict-JSON, package-relative copies without altering raw checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PATH_TARGETS = {
    "poscar_path": "02_RAW_DATA/02_MATERIALS_PROJECT_ARCHIVE/mp_full_archive_export/poscar",
    "summary_json_path": "02_RAW_DATA/02_MATERIALS_PROJECT_ARCHIVE/mp_full_archive_export/summary_json",
    "provenance_json_path": "02_RAW_DATA/02_MATERIALS_PROJECT_ARCHIVE/mp_full_archive_export/provenance_json",
    "doi_json_path": "02_RAW_DATA/02_MATERIALS_PROJECT_ARCHIVE/mp_full_archive_export/doi_json",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def replace_nonfinite(value: Any, counter: list[int]) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        counter[0] += 1
        return None
    if isinstance(value, list):
        return [replace_nonfinite(item, counter) for item in value]
    if isinstance(value, dict):
        return {key: replace_nonfinite(item, counter) for key, item in value.items()}
    return value


def standardize_one(root: Path, source: Path, output: Path) -> dict[str, Any]:
    rows = 0
    changed_paths = 0
    nonfinite = [0]
    nonstandard_constants = [0]
    missing_targets: dict[str, int] = {key: 0 for key in PATH_TARGETS}
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8") as reader, output.open("w", encoding="utf-8") as writer:
        for line_number, line in enumerate(reader, start=1):
            if not line.strip():
                continue
            try:
                def replace_constant(_token: str) -> None:
                    nonstandard_constants[0] += 1
                    return None

                row = json.loads(line, parse_constant=replace_constant)
            except Exception as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSONL") from exc
            row = replace_nonfinite(row, nonfinite)
            for field, target_dir in PATH_TARGETS.items():
                raw = row.get(field)
                if not isinstance(raw, str) or not raw.strip():
                    continue
                source_field = f"source_{field}"
                row.setdefault(source_field, raw)
                relative = str(Path(target_dir) / Path(raw).name)
                row[field] = relative
                changed_paths += 1
                if not (root / relative).is_file():
                    missing_targets[field] += 1
            writer.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
            rows += 1
    return {
        "input": source.resolve().relative_to(root).as_posix(),
        "output": output.resolve().relative_to(root).as_posix(),
        "rows": rows,
        "changed_path_fields": changed_paths,
        "nonfinite_values_replaced": nonfinite[0] + nonstandard_constants[0],
        "nonstandard_json_constants_replaced": nonstandard_constants[0],
        "missing_relative_targets": missing_targets,
        "input_sha256": digest(source),
        "output_sha256": digest(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--input", action="append", required=True, help="NAME=relative/path.jsonl")
    args = parser.parse_args()
    root = args.release_root.resolve()
    output_dir = args.output_dir.resolve()
    reports = []
    for item in args.input:
        name, separator, relative = item.partition("=")
        if not separator or not name or not relative:
            raise SystemExit(f"Invalid --input {item!r}; expected NAME=relative/path.jsonl")
        source = root / relative
        if not source.is_file():
            raise SystemExit(f"Missing input: {source}")
        reports.append(standardize_one(root, source, output_dir / f"{name}.jsonl"))
    missing = sum(sum(report["missing_relative_targets"].values()) for report in reports)
    manifest = {
        "schema": "synthmind_portable_standardized_jsonl_v1",
        "raw_files_modified": False,
        "source_path_fields_preserved": True,
        "standard_json_allow_nan": False,
        "files": reports,
        "total_missing_relative_targets": missing,
        "status": "PASS" if missing == 0 else "FAIL",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "manifest": str(manifest_path)}, ensure_ascii=False))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
