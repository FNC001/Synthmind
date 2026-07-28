#!/usr/bin/env python3
"""Audit that all active software surfaces use the synthmind name."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path


PUBLIC_NAME = "synthmind"
EXPECTED_RELEASE = "synthmind_optimal_full_lineage_20260723_v1"
TEXT_SUFFIXES = {
    "", ".cfg", ".csv", ".ini", ".json", ".jsonl", ".log", ".md",
    ".py", ".rst", ".sh", ".tex", ".toml", ".tsv", ".txt", ".yaml", ".yml",
}


def retired_spellings() -> tuple[str, ...]:
    base = "syn" + "pred"
    return (base, base.capitalize(), base.upper(), "SYN" + "_PRED")


def active_text_files(root: Path) -> list[Path]:
    paths: set[Path] = set()
    for name in ("README.md", "RUN_ALL_VALIDATIONS.sh"):
        path = root / name
        if path.is_file():
            paths.add(path)

    overview = root / "00_OVERVIEW_AND_MANIFEST"
    overview_exclusions = {
        "BRAND_MIGRATION_MANIFEST.json",
    }
    for path in overview.rglob("*"):
        if not path.is_file() or "source_archives" in path.relative_to(overview).parts:
            continue
        if path.name in overview_exclusions:
            continue
        paths.add(path)

    source = root / "01_SOURCE_CODE"
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(source)
        if rel.parts and rel.parts[0] == "99_HISTORICAL_REFERENCE":
            continue
        if rel.as_posix().endswith("research/specs/split_manifest_v1.json"):
            continue
        if path.name == Path(__file__).name:
            continue
        paths.add(path)

    guides = root / "10_METHODS_AND_GUIDES"
    for path in guides.rglob("*"):
        if path.is_file() and path.name != "06_BRANDING_AND_COMPATIBILITY.md":
            paths.add(path)

    audits = root / "11_TESTS_AND_AUDITS"
    for path in (
        audits / "README.md",
        audits / "verify_release.py",
        audits / "TEST_SUMMARY.json",
    ):
        if path.is_file():
            paths.add(path)
    final_evidence = audits / "02_FINAL_FREEZE_VALIDATION"
    if final_evidence.is_dir():
        paths.update(path for path in final_evidence.rglob("*") if path.is_file())

    for path in root.rglob("README.md"):
        rel = path.relative_to(root)
        if rel.parts[:2] == ("01_SOURCE_CODE", "99_HISTORICAL_REFERENCE"):
            continue
        if rel.parts[:2] == ("00_OVERVIEW_AND_MANIFEST", "source_archives"):
            continue
        paths.add(path)
    return sorted(paths)


def text_violations(root: Path) -> list[dict[str, object]]:
    spellings = retired_spellings()
    violations: list[dict[str, object]] = []
    for path in active_text_files(root):
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        hits = {token: text.count(token) for token in spellings if token in text}
        title_case = text.count("Synthmind")
        uppercase_display = len(re.findall(r"(?<![A-Za-z0-9_])SYNTHMIND(?![A-Za-z0-9_])", text))
        if title_case:
            hits["noncanonical_title_case"] = title_case
        if uppercase_display:
            hits["noncanonical_uppercase_display"] = uppercase_display
        if hits:
            violations.append({"path": path.relative_to(root).as_posix(), "hits": hits})
    return violations


def path_violations(root: Path) -> list[str]:
    folded = ("syn" + "pred").replace("_", "")
    violations = []
    for path in root.rglob("*"):
        normalized = path.name.lower().replace("_", "").replace("-", "")
        if folded in normalized or "Synthmind" in path.name or "SYNTHMIND" in path.name:
            violations.append(path.relative_to(root).as_posix())
    return sorted(violations)


def docx_violations(root: Path) -> dict[str, object]:
    core = root / "01_SOURCE_CODE/00_SHARED_CORE/synthmind_source"
    expected = core / "synthmind_全流程算法原理与复现指南.docx"
    if not expected.is_file():
        return {"status": "FAIL", "reason": "renamed Word guide is missing", "path": str(expected)}
    hits: dict[str, int] = {}
    with zipfile.ZipFile(expected) as archive:
        for name in archive.namelist():
            if not name.endswith((".xml", ".rels")):
                continue
            data = archive.read(name).decode("utf-8", errors="ignore")
            count = sum(data.count(token) for token in retired_spellings())
            if count:
                hits[name] = count
    return {
        "status": "PASS" if not hits else "FAIL",
        "path": expected.relative_to(root).as_posix(),
        "xml_parts_with_violations": hits,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.release_root.resolve()
    output = args.output or root / "11_TESTS_AND_AUDITS/02_FINAL_FREEZE_VALIDATION/15_branding_audit.json"
    core = root / "01_SOURCE_CODE/00_SHARED_CORE/synthmind_source"
    manifest_path = root / "00_OVERVIEW_AND_MANIFEST/BRAND_MIGRATION_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}

    text_hits = text_violations(root)
    path_hits = path_violations(root)
    word = docx_violations(root)
    namespace = {
        "core_present": core.is_dir(),
        "package_present": (core / "synthmind/__init__.py").is_file(),
        "chemistry_present": (core / "synthmind/chemistry").is_dir(),
        "research_present": (core / "synthmind/research").is_dir(),
        "retired_namespace_absent": not (core / ("syn" + "pred")).exists(),
    }
    distribution = core / "pyproject.toml"
    distribution_ok = distribution.is_file() and 'name = "synthmind"' in distribution.read_text(encoding="utf-8")
    passed = (
        root.name == EXPECTED_RELEASE
        and not text_hits
        and not path_hits
        and word.get("status") == "PASS"
        and all(namespace.values())
        and distribution_ok
        and manifest.get("public_software_name") == PUBLIC_NAME
        and manifest.get("migration_state") in {"ready_for_validation", "validated"}
    )
    payload = {
        "schema": "synthmind_branding_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "release": root.name,
        "public_software_name": PUBLIC_NAME,
        "active_text_legacy_name_violations": text_hits,
        "legacy_named_path_violations": path_hits,
        "word_guide": word,
        "namespace": namespace,
        "distribution_name_ok": distribution_ok,
        "provenance_residual_policy_recorded": bool(manifest.get("preserved_residual_policy")),
        "note": "Historical provenance residuals are allowed only in the migration manifest's declared scopes.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "report": str(output)}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
