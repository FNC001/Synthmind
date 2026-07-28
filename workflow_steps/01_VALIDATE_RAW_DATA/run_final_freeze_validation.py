#!/usr/bin/env python3
"""Run the final, evidence-producing validation suite for the synthmind release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
import tokenize
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


EXPECTED_SCOPE = {
    "true_precursor_fallback": {"n": 1896, "precursor": 1679, "condition": 1431, "coupled": 1288},
    "stage2_v5_repaired_top1": {"n": 526, "precursor": 163, "condition": 416, "coupled": 118},
}
FORBIDDEN_NAMES = {
    ".DS_Store", "__MACOSX", "__pycache__", ".pytest_cache",
    ".ipynb_checkpoints", ".claude",
}
FORBIDDEN_SUFFIXES = (
    ".pyc", ".pyo", ".swp", ".swo", ".tmp", ".temp", ".bak",
    ".orig", ".rej", ".old", ".save", "-bak", "~",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


class Suite:
    def __init__(self, root: Path, output: Path, python: str, skip_heavy: bool) -> None:
        self.root = root
        self.output = output
        self.python = python
        self.skip_heavy = skip_heavy
        self.checks: list[dict[str, Any]] = []
        self.output.mkdir(parents=True, exist_ok=True)
        self.env = os.environ.copy()
        self.env.update({
            "PYTHON": python,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": "/tmp/synthmind_final_validation_pycache",
            "RELEASE_ROOT": str(root),
        })

    def custom(self, check_id: str, func: Callable[[], dict[str, Any]]) -> None:
        started = utc_now()
        begin = time.monotonic()
        status = "PASS"
        error = None
        observed: dict[str, Any] = {}
        try:
            observed = func()
        except Exception as exc:  # noqa: BLE001 - audit must record any failure
            status = "FAIL"
            error = f"{type(exc).__name__}: {exc}"
        duration = time.monotonic() - begin
        log = self.output / f"{check_id}.json"
        payload = {"status": status, "observed": observed, "error": error}
        log.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.checks.append({
            "id": check_id,
            "scope": "custom",
            "started_at_utc": started,
            "duration_s": duration,
            "exit_code": 0 if status == "PASS" else 1,
            "status": status,
            "evidence_path": str(log),
            "evidence_sha256": sha256(log),
            "observed": observed,
            "error": error,
        })

    def command(
        self,
        check_id: str,
        argv: list[str],
        cwd: Path | None = None,
        required_output_pattern: str | None = None,
    ) -> None:
        if self.skip_heavy and check_id in {"10_metric_reproduction", "11_structure_full_sha", "12_source_zip_crc"}:
            self.checks.append({
                "id": check_id, "scope": "command", "command": argv,
                "status": "SKIP", "reason": "--skip-heavy", "exit_code": None,
            })
            return
        started = utc_now()
        begin = time.monotonic()
        error = None
        try:
            result = subprocess.run(
                argv,
                cwd=str(cwd or self.root),
                env=self.env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            exit_code = result.returncode
            output_text = result.stdout
        except Exception as exc:  # noqa: BLE001 - audit must record launch failures
            exit_code = 127
            output_text = f"{type(exc).__name__}: {exc}\n"
            error = output_text.strip()
        if required_output_pattern and not re.search(required_output_pattern, output_text):
            error = f"required output pattern not found: {required_output_pattern!r}"
        duration = time.monotonic() - begin
        log = self.output / f"{check_id}.log"
        log.write_text(output_text, encoding="utf-8")
        self.checks.append({
            "id": check_id,
            "scope": "command",
            "command": argv,
            "working_directory": str(cwd or self.root),
            "started_at_utc": started,
            "duration_s": duration,
            "exit_code": exit_code,
            "status": "PASS" if exit_code == 0 and error is None else "FAIL",
            "required_output_pattern": required_output_pattern,
            "error": error,
            "evidence_path": str(log),
            "evidence_sha256": sha256(log),
        })


def environment_check(requested_python: str) -> dict[str, Any]:
    requested = Path(requested_python).resolve()
    running = Path(sys.executable).resolve()
    if running != requested:
        raise RuntimeError(f"Validator interpreter mismatch: running={running}, requested={requested}")
    packages = {}
    for name in ("numpy", "pandas", "torch", "lightgbm", "scikit-learn", "scipy", "chgnet", "pymatgen"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu: dict[str, Any] = {}
    try:
        import torch  # type: ignore
        gpu = {
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "torch_cuda_version": torch.version.cuda,
            "device_count": int(torch.cuda.device_count()),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    except Exception as exc:  # noqa: BLE001
        gpu = {"torch_probe_error": str(exc)}
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "requested_python": str(requested),
        "interpreter_identity_match": True,
        "python_version": sys.version,
        "packages": packages,
        "gpu": gpu,
    }


def compile_check(root: Path) -> dict[str, Any]:
    files = sorted((root / "01_SOURCE_CODE").rglob("*.py"))
    failures = []
    for path in files:
        try:
            with tokenize.open(path) as handle:
                compile(handle.read(), str(path), "exec")
        except Exception as exc:  # noqa: BLE001
            failures.append({"path": str(path.relative_to(root)), "error": str(exc)})
    if failures:
        raise RuntimeError(f"Python syntax failures: {failures[:10]}")
    return {"compiled_files": len(files), "failures": 0}


def shell_syntax_check(root: Path) -> dict[str, Any]:
    files = sorted(root.rglob("*.sh"))
    failures = []
    for path in files:
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True, check=False)
        if result.returncode:
            failures.append({"path": str(path.relative_to(root)), "output": result.stderr})
    if failures:
        raise RuntimeError(f"Shell syntax failures: {failures[:10]}")
    return {"shell_files": len(files), "failures": 0}


def portable_manifest_check(root: Path) -> dict[str, Any]:
    manifests = [
        root / "03_CLEANED_AND_MERGED_DATA/00_PORTABLE_STANDARDIZED_INPUTS/manifest.json",
        root / "04_SPLITS/00_PORTABLE_SPLITS/01_BASE_GROUP_SPLIT/manifest.json",
        root / "04_SPLITS/00_PORTABLE_SPLITS/02_ROUTE_GROUP_SPLIT/manifest.json",
    ]
    totals = {"files": 0, "rows": 0, "changed_path_fields": 0, "missing_targets": 0}
    for manifest in manifests:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if data["status"] != "PASS":
            raise RuntimeError(f"Manifest not PASS: {manifest}")
        for item in data["files"]:
            resolved: dict[str, Path] = {}
            for key in ("input", "output"):
                relative = Path(item[key])
                if relative.is_absolute() or ".." in relative.parts:
                    raise RuntimeError(f"Unsafe {key} in {manifest}: {item[key]}")
                candidate = (root / relative).resolve()
                try:
                    candidate.relative_to(root)
                except ValueError as exc:
                    raise RuntimeError(f"Escaping {key} in {manifest}: {item[key]}") from exc
                resolved[key] = candidate
            source, output = resolved["input"], resolved["output"]
            if sha256(source) != item["input_sha256"] or sha256(output) != item["output_sha256"]:
                raise RuntimeError(f"Hash mismatch recorded by {manifest}: {item['output']}")
            with output.open("rb") as handle:
                rows = sum(1 for line in handle if line.strip())
            if rows != item["rows"]:
                raise RuntimeError(f"Row mismatch: {item['output']}")
            totals["files"] += 1
            totals["rows"] += rows
            totals["changed_path_fields"] += item["changed_path_fields"]
            totals["missing_targets"] += sum(item["missing_relative_targets"].values())
    expected = {"files": 23, "rows": 362529, "changed_path_fields": 932060, "missing_targets": 0}
    if totals != expected:
        raise RuntimeError(f"Portable totals differ: {totals} != {expected}")
    return {"manifests": [str(p.relative_to(root)) for p in manifests], **totals}


def structure_scope_rebuild_check(root: Path, python: str) -> dict[str, Any]:
    frozen_dir = root / "11_TESTS_AND_AUDITS/03_STRUCTURE_SCOPE_INDEXES"
    frozen_audit = root / "11_TESTS_AND_AUDITS/STRUCTURE_SCOPE_COVERAGE_AUDIT.json"
    temp_root = Path(tempfile.mkdtemp(prefix="synthmind_structure_scope_rebuild_"))
    generated_dir = temp_root / "indexes"
    generated_audit = temp_root / "STRUCTURE_SCOPE_COVERAGE_AUDIT.json"
    env = os.environ.copy()
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPYCACHEPREFIX": str(temp_root / "pycache")})
    try:
        result = subprocess.run(
            [
                python,
                "-B",
                str(root / "01_SOURCE_CODE/01_VALIDATE_RAW_DATA/build_structure_scope_indexes.py"),
                str(root),
                "--output-dir",
                str(generated_dir),
                "--audit-path",
                str(generated_audit),
            ],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(f"Structure scope rebuild failed: {result.stdout[-2000:]}")
        names = [
            "strict_3451_material_ids.txt",
            "strict_3451_structure_mapping.tsv",
            "final_8478_material_ids.txt",
            "final_8478_structure_mapping.tsv",
        ]
        compared: dict[str, str] = {}
        for name in names:
            frozen = frozen_dir / name
            rebuilt = generated_dir / name
            frozen_sha, rebuilt_sha = sha256(frozen), sha256(rebuilt)
            if frozen_sha != rebuilt_sha:
                raise RuntimeError(f"Rebuilt structure scope differs: {name}")
            compared[name] = rebuilt_sha
        if sha256(frozen_audit) != sha256(generated_audit):
            raise RuntimeError("Rebuilt STRUCTURE_SCOPE_COVERAGE_AUDIT.json differs from frozen audit")
        compared["STRUCTURE_SCOPE_COVERAGE_AUDIT.json"] = sha256(generated_audit)
        return {"files_rebuilt_and_identical": len(compared), "sha256": compared}
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def structure_sha_check(root: Path) -> dict[str, Any]:
    manifest = root / "11_TESTS_AND_AUDITS/01_STRUCTURE_ARCHIVE_AUDIT/poscar_file_manifest_sha256.tsv"
    poscar = root / "02_RAW_DATA/02_MATERIALS_PROJECT_ARCHIVE/mp_full_archive_export/poscar"
    metadata = poscar.parent / "mp_full_archive_metadata.csv"
    checked = 0
    metadata_backed_ids: set[str] = set()
    conventional_ids: set[str] = set()
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            path = poscar / row["relative_path"]
            if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
                raise RuntimeError(f"Structure identity mismatch: {row['relative_path']}")
            if path.suffix == ".vasp":
                metadata_backed_ids.add(path.stem.split("_", 1)[0])
            elif path.name.startswith("POSCAR_") and path.name.endswith("_conventional"):
                conventional_ids.add(path.name[len("POSCAR_"):-len("_conventional")])
            else:
                raise RuntimeError(f"Unknown structure evidence tier: {path.name}")
            checked += 1
    if checked != 62689:
        raise RuntimeError(f"Expected 62689 structures, checked {checked}")
    with metadata.open("r", encoding="utf-8-sig", newline="") as handle:
        metadata_ids = {
            row["material_id"].strip()
            for row in csv.DictReader(handle)
            if row.get("material_id", "").strip()
        }
    summary_ids = {path.stem for path in (poscar.parent / "summary_json").glob("*.json") if path.is_file()}
    provenance_ids = {path.stem for path in (poscar.parent / "provenance_json").glob("*.json") if path.is_file()}
    doi_ids = {path.stem for path in (poscar.parent / "doi_json").glob("*.json") if path.is_file()}
    expected_tiers = {"mp_metadata_backed": 49283, "conventional_supplement_no_upstream_metadata": 13406}
    observed_tiers = {
        "mp_metadata_backed": len(metadata_backed_ids),
        "conventional_supplement_no_upstream_metadata": len(conventional_ids),
    }
    if (
        observed_tiers != expected_tiers
        or metadata_ids != metadata_backed_ids
        or summary_ids != metadata_ids
        or provenance_ids != metadata_ids
        or len(doi_ids) != 30119
        or not doi_ids <= metadata_ids
        or conventional_ids & (metadata_ids | summary_ids | provenance_ids | doi_ids)
    ):
        raise RuntimeError(
            f"Structure evidence tiers changed: {observed_tiers}; "
            f"metadata_only={len(metadata_ids - metadata_backed_ids)} "
            f"structure_only={len(metadata_backed_ids - metadata_ids)} "
            f"summary_delta={len(summary_ids ^ metadata_ids)} "
            f"provenance_delta={len(provenance_ids ^ metadata_ids)} "
            f"doi_outside_metadata={len(doi_ids - metadata_ids)}"
        )
    parse_audit_path = root / "11_TESTS_AND_AUDITS/01_STRUCTURE_ARCHIVE_AUDIT/audit_summary.json"
    parse_audit = json.loads(parse_audit_path.read_text(encoding="utf-8"))
    expected_parse = {
        "file_count": 62689,
        "parseable_file_count": 62689,
        "unparseable_file_count": 0,
        "unique_material_id_count": 62689,
        "duplicate_material_id_group_count": 0,
    }
    if any(parse_audit.get(key) != value for key, value in expected_parse.items()):
        raise RuntimeError(f"Structure parse audit changed: {parse_audit}")
    attach_path = root / (
        "03_CLEANED_AND_MERGED_DATA/02_MERGED_WITH_STRUCTURES/"
        "merged_20260609_with_structures/structure_attach_summary.json"
    )
    attach = json.loads(attach_path.read_text(encoding="utf-8"))
    attach_counts = attach["validation"]["counts"]
    expected_attach = {
        "poscar_parsed": 13406,
        "cif_poscar_comp_match": 13405,
        "cif_poscar_comp_mismatch": 1,
        "mp_formula_matches_poscar": 17125,
        "parent_formula_matches_poscar": 17125,
    }
    mismatch = attach["validation"]["samples"].get("cif_poscar_mismatches", [])
    if attach["inputs"].get("unique_new_material_ids") != 13406 or attach_counts != expected_attach or [item.get("material_id") for item in mismatch] != ["mp-bcol"]:
        raise RuntimeError(f"Conventional attach evidence changed: counts={attach_counts}, mismatches={mismatch}")
    scope_path = root / "11_TESTS_AND_AUDITS/STRUCTURE_SCOPE_COVERAGE_AUDIT.json"
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    final_tiers = scope["final_training_scope"]["stage2_stage3_union"]["source_evidence_tiers"]
    if final_tiers != {"mp_metadata_backed": 2530, "conventional_supplement_no_upstream_metadata": 5948}:
        raise RuntimeError(f"Final-training structure evidence tiers changed: {final_tiers}")
    return {
        "structures_verified": checked,
        "source_evidence_tiers": observed_tiers,
        "metadata_identity_match": True,
        "summary_identity_match": True,
        "provenance_identity_match": True,
        "doi_snapshot_structures": len(doi_ids),
        "parse_audit": expected_parse,
        "conventional_attach_validation": {**attach_counts, "mismatch_material_id": "mp-bcol"},
        "final_training_source_evidence_tiers": final_tiers,
        "manifest": str(manifest.relative_to(root)),
        "scope_audit": str(scope_path.relative_to(root)),
    }


def hygiene_check(root: Path) -> dict[str, Any]:
    symlinks, forbidden, mode_mismatches = [], [], []
    for path in root.rglob("*"):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            symlinks.append(rel)
        name = path.name
        if name in FORBIDDEN_NAMES or name.startswith("._") or ".bak" in name or name.endswith(FORBIDDEN_SUFFIXES):
            forbidden.append(rel)
        if path.is_dir():
            expected_mode = 0o755
        elif path.is_file():
            expected_mode = 0o755 if path.suffix == ".sh" else 0o644
        else:
            continue
        actual_mode = stat.S_IMODE(path.stat().st_mode)
        if actual_mode != expected_mode:
            mode_mismatches.append({"path": rel, "expected": f"{expected_mode:04o}", "actual": f"{actual_mode:04o}"})
    if symlinks or forbidden or mode_mismatches:
        raise RuntimeError(
            f"symlinks={symlinks[:10]} forbidden={forbidden[:30]} modes={mode_mismatches[:30]}"
        )
    return {"symlinks": 0, "forbidden_auxiliary_entries": 0, "permission_mismatches": 0}


def readme_check(root: Path) -> dict[str, Any]:
    checked, missing = 0, []
    for directory in sorted(path for path in root.rglob("*") if path.is_dir()):
        rel = directory.relative_to(root)
        if len(rel.parts) > 3:
            continue
        checked += 1
        if not (directory / "README.md").is_file():
            missing.append(rel.as_posix())
    if missing:
        raise RuntimeError(f"Missing README.md through depth 3: {missing}")
    return {"directories_checked": checked, "missing": 0, "policy": "all directories through depth 3"}


def disclosure_check(root: Path) -> dict[str, Any]:
    disclosure = json.loads((root / "09_ACCURACY_EVALUATION/02_TEST_LOCKBOX/TEST_SPLIT_DISCLOSURE.json").read_text())
    selected = json.loads((root / "00_OVERVIEW_AND_MANIFEST/SELECTED_MODEL_MANIFEST.json").read_text())
    history = root / "09_ACCURACY_EVALUATION/06_HISTORICAL_NONFINAL_TEST_RESULTS"
    expected_dirs = [item["run"] for item in disclosure["historical_access"] if "destination" in item]
    missing = [name for name in expected_dirs if not (history / name).is_dir()]
    passed = (
        disclosure["test_split_pristine"] is False
        and selected["test_split_pristine"] is False
        and selected["final_selected_models_test_evaluated"] is False
        and not missing
    )
    if not passed:
        raise RuntimeError(f"Test disclosure mismatch; missing={missing}")
    return {"test_split_pristine": False, "final_models_test_evaluated": False, "historical_dirs": len(expected_dirs)}


def scope_check(root: Path) -> dict[str, Any]:
    import numpy as np  # type: ignore

    data_dir = root / "06_TRAIN_READY_DATA/08_STAGE3_FAMILY_FULL/stage3_full_cation_family_v1"
    arrays = np.load(data_dir / "val.npz", allow_pickle=True)
    continuous = arrays["y_cond_continuous_raw"]
    continuous_mask = arrays["y_cond_continuous_mask"].astype(bool)
    discrete = arrays["y_cond_discrete"]
    discrete_mask = arrays["y_cond_discrete_mask"].astype(bool)
    with (data_dir / "val_meta.csv").open("r", encoding="utf-8", newline="") as handle:
        meta = list(csv.DictReader(handle))
    candidates = []
    path = root / "08_GENERATED_OUTPUTS/05_STAGE3_RANKED_CONDITIONS/final_condition_candidates.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for expected_row, line in enumerate(handle):
            item = json.loads(line)
            if item["row_index"] != expected_row:
                raise RuntimeError("Condition candidate row_index is not contiguous")
            candidates.append(item["candidates"])
    if len(meta) != 2422 or len(candidates) != 2422:
        raise RuntimeError("Stage3 validation rows do not align")

    result: dict[str, dict[str, int]] = {}
    for index, row in enumerate(meta):
        source = row["precursor_input_source"]
        bucket = result.setdefault(source, {"n": 0, "precursor": 0, "condition": 0, "coupled": 0})
        bucket["n"] += 1
        precursor = float(row["precursor_exact"]) > 0.5
        bucket["precursor"] += int(precursor)
        condition = False
        for candidate in candidates[index][:10]:
            if continuous_mask[index, 0] and abs(float(candidate["temperature_c"]) - float(continuous[index, 0])) > 200:
                continue
            if continuous_mask[index, 1] and abs(float(candidate["time_h"]) - float(continuous[index, 1])) > 48:
                continue
            if discrete_mask[index, 0] and int(candidate["atmosphere_coarse"]) != int(discrete[index, 0]):
                continue
            if discrete_mask[index, 1] and int(candidate["reaction_method"]) != int(discrete[index, 1]):
                continue
            condition = True
            break
        bucket["condition"] += int(condition)
        bucket["coupled"] += int(condition and precursor)
    if result != EXPECTED_SCOPE:
        raise RuntimeError(f"Scope audit mismatch: {result} != {EXPECTED_SCOPE}")
    return result


def source_zip_check(root: Path) -> dict[str, Any]:
    archive_dir = root / "00_OVERVIEW_AND_MANIFEST/source_archives"
    expected = {
        "raw_full_local_snapshot.zip": "5ad71f46818c2d97850f90de5a65640d13a6f101923095c25e3d5fc472272cce",
        "direct_aligned_json_20260608.zip": "26ed4b29b0cafc8f92055e6b87507614389abc518353fa3a4f907340d9e23632",
        "final_database_csv_20260608_clean.zip": "d44a5bf8f369d70f438da34b634c874e0dc294ec353a3caa2ddb3884e8cd5433",
    }
    for name, digest in expected.items():
        path = archive_dir / name
        if sha256(path) != digest:
            raise RuntimeError(f"Source archive hash mismatch: {name}")
        result = subprocess.run(["unzip", "-tq", str(path)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.returncode:
            raise RuntimeError(f"Source archive CRC failed: {name}: {result.stdout[-1000:]}")
    return {"archives_crc_and_sha_passed": len(expected)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-heavy", action="store_true")
    args = parser.parse_args()
    root = args.release_root.resolve()
    output = (args.output_dir or root / "11_TESTS_AND_AUDITS/02_FINAL_FREEZE_VALIDATION").resolve()
    suite = Suite(root, output, args.python, args.skip_heavy)
    started = utc_now()

    suite.custom("01_environment", lambda: environment_check(args.python))
    suite.custom("02_python_compile", lambda: compile_check(root))
    suite.custom("03_shell_syntax", lambda: shell_syntax_check(root))
    core = root / "01_SOURCE_CODE/00_SHARED_CORE/synthmind_source"
    suite.command(
        "04_unit_tests",
        [args.python, "-B", str(root / "01_SOURCE_CODE/01_VALIDATE_RAW_DATA/run_core_unit_tests.py"), "--core", str(core)],
        cwd=core,
        required_output_pattern=r"Ran 77 tests in ",
    )
    suite.command("05_numbered_steps", ["bash", str(root / "RUN_ALL_VALIDATIONS.sh")])
    suite.custom("06_portable_manifests", lambda: portable_manifest_check(root))
    suite.custom("07_test_split_disclosure", lambda: disclosure_check(root))
    suite.custom("08_structure_scope_rebuild", lambda: structure_scope_rebuild_check(root, args.python))
    suite.custom("09_stage3_scope_recompute", lambda: scope_check(root))
    metric_work = Path(tempfile.mkdtemp(prefix="synthmind_final_freeze_metric_"))
    try:
        suite.command("10_metric_reproduction", ["bash", str(root / "01_SOURCE_CODE/09_EVALUATE_THREE_METRICS/run_step.sh"), str(metric_work)])
    finally:
        shutil.rmtree(metric_work, ignore_errors=True)
    if args.skip_heavy:
        suite.checks.append({"id": "11_structure_full_sha", "status": "SKIP", "reason": "--skip-heavy"})
    else:
        suite.custom("11_structure_full_sha", lambda: structure_sha_check(root))
    if args.skip_heavy:
        suite.checks.append({"id": "12_source_zip_crc", "status": "SKIP", "reason": "--skip-heavy"})
    else:
        suite.custom("12_source_zip_crc", lambda: source_zip_check(root))
    suite.custom("13_readme_coverage", lambda: readme_check(root))
    suite.custom("14_tree_hygiene", lambda: hygiene_check(root))
    suite.command(
        "15_branding_audit",
        [
            args.python,
            "-B",
            str(root / "01_SOURCE_CODE/01_VALIDATE_RAW_DATA/audit_synthmind_branding.py"),
            str(root),
            "--output",
            str(output / "15_branding_audit.json"),
        ],
        required_output_pattern=r'"status": "PASS"',
    )
    suite.command(
        "16_release_audit",
        [args.python, "-B", str(root / "11_TESTS_AND_AUDITS/verify_release.py"), "--release-root", str(root), "--output", str(output / "release_audit.json")],
    )

    failed = [item for item in suite.checks if item["status"] == "FAIL"]
    skipped = [item for item in suite.checks if item["status"] == "SKIP"]
    report = {
        "schema": "synthmind_final_freeze_validation_v2",
        "release": root.name,
        "run_id": f"synthmind-final-freeze-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "host": socket.gethostname(),
        "working_directory": str(root),
        "overall_status": "PASS" if not failed and not skipped else ("FAIL" if failed else "PASS_WITH_SKIPS"),
        "checks_total": len(suite.checks),
        "checks_passed": sum(item["status"] == "PASS" for item in suite.checks),
        "checks_failed": len(failed),
        "checks_skipped": len(skipped),
        "checks": suite.checks,
    }
    report_path = output / "FINAL_VALIDATION_REPORT.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["overall_status"], "report": str(report_path)}, ensure_ascii=False))
    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
