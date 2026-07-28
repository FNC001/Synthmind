#!/usr/bin/env python3
"""Finalize synthmind status documents from a real full validation report."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_RELEASE = "synthmind_optimal_full_lineage_20260723_v1"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def check_by_id(report: dict[str, Any], check_id: str) -> dict[str, Any]:
    return next(item for item in report["checks"] if item["id"] == check_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_root", type=Path)
    args = parser.parse_args()
    root = args.release_root.resolve()
    if root.name != EXPECTED_RELEASE:
        raise SystemExit(f"Unexpected release root: {root}")
    evidence = root / "11_TESTS_AND_AUDITS/02_FINAL_FREEZE_VALIDATION"
    report_path = evidence / "FINAL_VALIDATION_REPORT.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("overall_status") != "PASS"
        or report.get("checks_total") != 16
        or report.get("checks_passed") != 16
        or report.get("checks_failed") != 0
        or report.get("checks_skipped") != 0
    ):
        raise SystemExit("Refusing to finalize: full 16/16 no-skip validation has not passed")
    release_audit = json.loads((evidence / "release_audit.json").read_text(encoding="utf-8"))
    if release_audit.get("audit_status") != "PASS":
        raise SystemExit("Refusing to finalize: expanded release audit did not pass")
    brand_audit = json.loads((evidence / "15_branding_audit.json").read_text(encoding="utf-8"))
    if brand_audit.get("status") != "PASS":
        raise SystemExit("Refusing to finalize: branding audit did not pass")

    compile_count = check_by_id(report, "02_python_compile")["observed"]["compiled_files"]
    shell_count = check_by_id(report, "03_shell_syntax")["observed"]["shell_files"]
    unit_log = (evidence / "04_unit_tests.log").read_text(encoding="utf-8")
    match = re.search(r"Ran (\d+) tests in ", unit_log)
    if not match:
        raise SystemExit("Unable to determine unit-test count")
    unit_count = int(match.group(1))
    readme_count = check_by_id(report, "13_readme_coverage")["observed"]["directories_checked"]

    status = {
        "schema": "synthmind_release_status_v3",
        "release": EXPECTED_RELEASE,
        "status": "frozen_validation_release",
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_semantic_validation": {
            "status": "PASS",
            "checks_total": 16,
            "checks_passed": 16,
            "checks_failed": 0,
            "checks_skipped": 0,
            "evidence": "11_TESTS_AND_AUDITS/02_FINAL_FREEZE_VALIDATION/FINAL_VALIDATION_REPORT.json",
        },
        "completed_checks": [
            f"{release_audit['checks_total']}/{release_audit['checks_total']} expanded release semantic checks PASS",
            "three frozen validation metrics exact reproduction PASS",
            "Stage2 final candidate SHA reproduction PASS",
            f"{compile_count} Python files compile PASS",
            f"{shell_count} shell files syntax PASS",
            f"{unit_count}/{unit_count} chemistry and research unit tests PASS",
            "synthmind active-surface branding audit PASS with zero legacy-name violations",
            "23 portable JSONL files and 362529 rows verified with zero escaping or missing targets",
            "structure scope indexes independently rebuilt and byte-identical",
            "62689/62689 local structures match the frozen per-file SHA manifest",
            "three immutable source ZIP files pass declared SHA-256 and internal CRC",
            f"{readme_count} directories through depth 3 have README coverage",
            "tree hygiene, Unix permission policy and no-symlink policy PASS",
        ],
        "archive_integrity_contract": {
            "internal_file_index": "00_OVERVIEW_AND_MANIFEST/FILE_INDEX.tsv",
            "internal_sha256_manifest": "00_OVERVIEW_AND_MANIFEST/SHA256SUMS",
            "external_zip_sha256_and_clean_extract_attestation_required": True,
            "external_attestation_location": "sibling files beside the final ZIP",
            "why_external": "Archive SHA and post-extraction evidence cannot be stored inside the archive without circular hashing.",
        },
        "truth_limits": [
            "All three selected final metrics are validation metrics; the selected final models were not evaluated on test in this freeze.",
            "The current test split is not pristine because historical non-final models accessed it.",
            "strict_end_to_end is a mixed frozen Stage3-input coupled metric with 1896/2422 true-precursor fallback rows, not online final Stage2-to-Stage3 accuracy.",
            "DOI and reused literature text overlap across splits; the release is not literature-independent evaluation.",
            "Historical paths retaining the former spelling are provenance only; current software, release and Python namespaces are synthmind.",
            "Metric replay proves frozen predictions and samples, not byte-identical GPU retraining across hardware.",
        ],
    }
    write_json(root / "00_OVERVIEW_AND_MANIFEST/RELEASE_STATUS.json", status)

    summary_path = root / "11_TESTS_AND_AUDITS/TEST_SUMMARY.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["schema"] = "synthmind_final_test_summary_v2"
    summary["release"] = EXPECTED_RELEASE
    summary["overall_status"] = "PASS"
    summary["final_freeze_validation"] = {
        "status": "PASS",
        "checks_total": 16,
        "checks_passed": 16,
        "checks_failed": 0,
        "checks_skipped": 0,
        "python_files_compiled": compile_count,
        "shell_files_syntax_checked": shell_count,
        "chemistry_and_research_unit_tests": unit_count,
        "directories_with_readme_policy_checked": readme_count,
        "evidence": "11_TESTS_AND_AUDITS/02_FINAL_FREEZE_VALIDATION/FINAL_VALIDATION_REPORT.json",
    }
    summary["release_integrity_audit"] = {
        "status": "PASS",
        "checks_total": release_audit["checks_total"],
        "checks_failed": release_audit["checks_failed"],
        "evidence": "11_TESTS_AND_AUDITS/02_FINAL_FREEZE_VALIDATION/release_audit.json",
    }
    summary["branding_migration"] = {
        "status": "PASS",
        "public_name": "synthmind",
        "active_python_namespace": "synthmind",
        "active_surface_legacy_name_violations": 0,
        "historical_provenance_residuals_declared": True,
        "evidence": [
            "00_OVERVIEW_AND_MANIFEST/BRAND_MIGRATION_MANIFEST.json",
            "11_TESTS_AND_AUDITS/02_FINAL_FREEZE_VALIDATION/15_branding_audit.json",
        ],
    }
    summary["code_quality_checks"] = {
        "python_compile": {"status": "PASS", "files": compile_count},
        "shell_syntax": {"status": "PASS", "files": shell_count},
        "chemistry_and_research_unit_tests": {"status": "PASS", "tests": unit_count},
        "numbered_directory_validation": "PASS",
        "metric_reproduction": "PASS",
        "branding_audit": "PASS",
        "evidence": [
            "11_TESTS_AND_AUDITS/02_FINAL_FREEZE_VALIDATION/02_python_compile.json",
            "11_TESTS_AND_AUDITS/02_FINAL_FREEZE_VALIDATION/03_shell_syntax.json",
            "11_TESTS_AND_AUDITS/02_FINAL_FREEZE_VALIDATION/04_unit_tests.log",
            "11_TESTS_AND_AUDITS/02_FINAL_FREEZE_VALIDATION/05_numbered_steps.log",
            "11_TESTS_AND_AUDITS/02_FINAL_FREEZE_VALIDATION/10_metric_reproduction.log",
            "11_TESTS_AND_AUDITS/02_FINAL_FREEZE_VALIDATION/15_branding_audit.json",
        ],
    }
    write_json(summary_path, summary)

    migration_path = root / "00_OVERVIEW_AND_MANIFEST/BRAND_MIGRATION_MANIFEST.json"
    migration = json.loads(migration_path.read_text(encoding="utf-8"))
    migration["migration_state"] = "validated"
    migration["validation"] = {
        "status": "PASS",
        "checks_total": 16,
        "checks_passed": 16,
        "checks_failed": 0,
        "checks_skipped": 0,
        "active_surface_legacy_name_violations": 0,
        "evidence": "11_TESTS_AND_AUDITS/02_FINAL_FREEZE_VALIDATION/FINAL_VALIDATION_REPORT.json",
    }
    write_json(migration_path, migration)
    print(json.dumps({"status": "FINALIZED", "release": str(root), "unit_tests": unit_count}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
