#!/usr/bin/env python3
"""Build strict and final-training structure coverage indexes from frozen data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def material_id_from_name(name: str) -> str:
    if name.startswith("POSCAR_") and name.endswith("_conventional"):
        return name[len("POSCAR_"):-len("_conventional")]
    if name.endswith(".vasp"):
        return name[:-len(".vasp")].split("_", 1)[0]
    raise ValueError(name)


def structure_source_tier(path: Path) -> str:
    """Return the evidence tier encoded by the frozen archive layout.

    ``*.vasp`` records are the 49,283 structures that have a matching row in
    ``mp_full_archive_metadata.csv``.  ``POSCAR_*_conventional`` records are a
    separately imported local supplement: they are parseable and ID-aligned,
    but this package does not contain upstream MP summary/provenance evidence
    for them.
    """
    if path.suffix == ".vasp":
        return "mp_metadata_backed"
    if path.name.startswith("POSCAR_") and path.name.endswith("_conventional"):
        return "conventional_supplement_no_upstream_metadata"
    raise ValueError(f"Unrecognized structure filename: {path.name}")


def load_archive_metadata_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "material_id" not in reader.fieldnames:
            raise RuntimeError(f"material_id column not found in {path}")
        return {row["material_id"].strip() for row in reader if row.get("material_id", "").strip()}


def json_stem_ids(directory: Path) -> set[str]:
    return {path.stem for path in directory.glob("*.json") if path.is_file()}


def load_meta_ids(directory: Path) -> set[str]:
    result: set[str] = set()
    for split in ("train", "val", "test"):
        with (directory / f"{split}_meta.csv").open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                value = row.get("material_id", "").strip()
                if value:
                    result.add(value)
    return result


def load_jsonl_ids(path: Path) -> set[str]:
    result: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = str(json.loads(line).get("material_id", "")).strip()
            if value:
                result.add(value)
    return result


def write_mapping(
    path: Path,
    ids: set[str],
    archive: dict[str, Path],
    root: Path,
    flags: dict[str, tuple[bool, bool]],
    flag_names: tuple[str, str],
) -> None:
    cache: dict[Path, str] = {}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow([
            "material_id", flag_names[0], flag_names[1], "source_evidence_tier",
            "bytes", "sha256", "relative_path",
        ])
        for material_id in sorted(ids):
            source = archive[material_id]
            digest = cache.setdefault(source, sha256(source))
            first, second = flags[material_id]
            writer.writerow([
                material_id, str(first).lower(), str(second).lower(), structure_source_tier(source),
                source.stat().st_size, digest, source.relative_to(root).as_posix(),
            ])


def tier_counts(ids: set[str], archive: dict[str, Path]) -> dict[str, int]:
    counts = {
        "mp_metadata_backed": 0,
        "conventional_supplement_no_upstream_metadata": 0,
    }
    for material_id in ids:
        counts[structure_source_tier(archive[material_id])] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--audit-path", type=Path)
    args = parser.parse_args()
    root = args.release_root.resolve()
    output = (args.output_dir or root / "11_TESTS_AND_AUDITS/03_STRUCTURE_SCOPE_INDEXES").resolve()
    output.mkdir(parents=True, exist_ok=True)
    poscar = root / "02_RAW_DATA/02_MATERIALS_PROJECT_ARCHIVE/mp_full_archive_export/poscar"
    archive: dict[str, Path] = {}
    for path in poscar.iterdir():
        if path.is_file():
            material_id = material_id_from_name(path.name)
            if material_id in archive:
                raise RuntimeError(f"Duplicate material_id in archive: {material_id}")
            archive[material_id] = path
    metadata_ids = load_archive_metadata_ids(poscar.parent / "mp_full_archive_metadata.csv")
    metadata_backed_ids = {item for item, path in archive.items() if structure_source_tier(path) == "mp_metadata_backed"}
    conventional_ids = set(archive) - metadata_backed_ids
    if metadata_ids != metadata_backed_ids:
        raise RuntimeError(
            "Structure/metadata identity mismatch: "
            f"metadata_only={len(metadata_ids - metadata_backed_ids)} "
            f"structure_only={len(metadata_backed_ids - metadata_ids)}"
        )
    summary_ids = json_stem_ids(poscar.parent / "summary_json")
    provenance_ids = json_stem_ids(poscar.parent / "provenance_json")
    doi_ids = json_stem_ids(poscar.parent / "doi_json")
    if summary_ids != metadata_ids or provenance_ids != metadata_ids or not doi_ids <= metadata_ids:
        raise RuntimeError(
            "Archive evidence-set mismatch: "
            f"summary={len(summary_ids)} provenance={len(provenance_ids)} doi={len(doi_ids)} "
            f"summary_delta={len(summary_ids ^ metadata_ids)} "
            f"provenance_delta={len(provenance_ids ^ metadata_ids)} "
            f"doi_outside_metadata={len(doi_ids - metadata_ids)}"
        )
    if len(doi_ids) != 30119 or conventional_ids & (summary_ids | provenance_ids | doi_ids | metadata_ids):
        raise RuntimeError("Conventional structures unexpectedly overlap the metadata/summary/provenance/DOI evidence sets")

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
        raise RuntimeError(f"Structure parse audit changed: {parse_audit_path}")

    attach_summary_path = root / (
        "03_CLEANED_AND_MERGED_DATA/02_MERGED_WITH_STRUCTURES/"
        "merged_20260609_with_structures/structure_attach_summary.json"
    )
    attach = json.loads(attach_summary_path.read_text(encoding="utf-8"))
    attach_counts = attach["validation"]["counts"]
    expected_attach_counts = {
        "poscar_parsed": 13406,
        "cif_poscar_comp_match": 13405,
        "cif_poscar_comp_mismatch": 1,
        "mp_formula_matches_poscar": 17125,
        "parent_formula_matches_poscar": 17125,
    }
    mismatch_samples = attach["validation"]["samples"].get("cif_poscar_mismatches", [])
    if (
        attach["inputs"].get("unique_new_material_ids") != 13406
        or attach_counts != expected_attach_counts
        or len(mismatch_samples) != 1
        or mismatch_samples[0].get("material_id") != "mp-bcol"
    ):
        raise RuntimeError(f"Conventional structure attach audit changed: {attach_summary_path}")

    strict_root = root / "02_RAW_DATA/04_STRICT_FILTER_OUTPUTS/01_ALIGNMENT_SPLIT_REMOTE_FINAL"
    exact = load_jsonl_ids(strict_root / "strict_exact_only.jsonl")
    parent = load_jsonl_ids(strict_root / "strict_parent_aug.jsonl")
    strict_union = exact | parent

    stage2 = load_meta_ids(root / "06_TRAIN_READY_DATA/04_STAGE2_CANONICAL/stage2_full_cation_family_canonical_v1")
    stage3 = load_meta_ids(root / "06_TRAIN_READY_DATA/08_STAGE3_FAMILY_FULL/stage3_full_cation_family_v1")
    final_union = stage2 | stage3
    expected = {"exact": 3411, "parent": 383, "strict_union": 3451, "stage2": 2508, "stage3": 8229, "final_union": 8478}
    observed = {"exact": len(exact), "parent": len(parent), "strict_union": len(strict_union), "stage2": len(stage2), "stage3": len(stage3), "final_union": len(final_union)}
    if observed != expected:
        raise RuntimeError(f"Scope counts changed: {observed} != {expected}")
    if (len(metadata_backed_ids), len(conventional_ids)) != (49283, 13406):
        raise RuntimeError(
            f"Archive tier counts changed: metadata={len(metadata_backed_ids)}, "
            f"conventional={len(conventional_ids)}"
        )
    observed_tiers = {
        "strict_exact": tier_counts(exact, archive),
        "strict_parent": tier_counts(parent, archive),
        "strict_union": tier_counts(strict_union, archive),
        "stage2": tier_counts(stage2, archive),
        "stage3": tier_counts(stage3, archive),
        "final_union": tier_counts(final_union, archive),
    }
    expected_tiers = {
        "strict_exact": {"mp_metadata_backed": 3411, "conventional_supplement_no_upstream_metadata": 0},
        "strict_parent": {"mp_metadata_backed": 383, "conventional_supplement_no_upstream_metadata": 0},
        "strict_union": {"mp_metadata_backed": 3451, "conventional_supplement_no_upstream_metadata": 0},
        "stage2": {"mp_metadata_backed": 2508, "conventional_supplement_no_upstream_metadata": 0},
        "stage3": {"mp_metadata_backed": 2281, "conventional_supplement_no_upstream_metadata": 5948},
        "final_union": {"mp_metadata_backed": 2530, "conventional_supplement_no_upstream_metadata": 5948},
    }
    if observed_tiers != expected_tiers:
        raise RuntimeError(f"Scope tier counts changed: {observed_tiers} != {expected_tiers}")
    missing_strict = sorted(strict_union - set(archive))
    missing_final = sorted(final_union - set(archive))
    if missing_strict or missing_final:
        raise RuntimeError(f"Missing strict={missing_strict[:10]} final={missing_final[:10]}")

    (output / "strict_3451_material_ids.txt").write_text("\n".join(sorted(strict_union)) + "\n", encoding="utf-8")
    (output / "final_8478_material_ids.txt").write_text("\n".join(sorted(final_union)) + "\n", encoding="utf-8")
    write_mapping(
        output / "strict_3451_structure_mapping.tsv", strict_union, archive, root,
        {item: (item in exact, item in parent) for item in strict_union}, ("in_exact", "in_parent"),
    )
    write_mapping(
        output / "final_8478_structure_mapping.tsv", final_union, archive, root,
        {item: (item in stage2, item in stage3) for item in final_union}, ("in_stage2", "in_stage3"),
    )

    audit = {
        "schema": "synthmind_structure_scope_coverage_audit_v2",
        "release": root.name,
        "structure_archive": {
            "files": len(archive),
            "unique_material_ids": len(archive),
            "parseable": parse_audit["parseable_file_count"],
            "missing_or_invalid": 0,
            "source_evidence_tiers": {
                "mp_metadata_backed": {
                    "structures": len(metadata_backed_ids),
                    "metadata_rows": len(metadata_ids),
                    "metadata_identity_match": True,
                    "summary_identity_match": summary_ids == metadata_ids,
                    "provenance_identity_match": provenance_ids == metadata_ids,
                    "doi_snapshot_structures": len(doi_ids),
                    "package_evidence": "metadata + summary/provenance snapshots; DOI snapshot for 30,119 structures",
                },
                "conventional_supplement_no_upstream_metadata": {
                    "structures": len(conventional_ids),
                    "package_evidence": "local snapshot + parseability + material_id/formula alignment only",
                    "upstream_mp_provenance_independently_provable_from_package": False,
                },
            },
            "qualification": (
                "All 62,689 files are frozen local MP-labelled structures and parse successfully. "
                "Only the 49,283 metadata-backed files have package-contained upstream metadata; "
                "the 13,406 conventional supplements must not be described as independently MP-provenanced."
            ),
            "conventional_supplement_validation": {
                "unique_material_ids": attach["inputs"]["unique_new_material_ids"],
                **attach_counts,
                "single_source_cif_poscar_composition_mismatch": mismatch_samples[0],
                "interpretation": (
                    "All 13,406 POSCARs parse and all 17,125 merged rows match the POSCAR formula. "
                    "The source-CIF comparison has one mismatch (mp-bcol); this does not establish upstream MP/API provenance."
                ),
            },
        },
        "alignment_scope": {
            "strict_exact": {"unique_material_ids": len(exact), "covered": len(exact), "missing": 0, "source_evidence_tiers": observed_tiers["strict_exact"]},
            "strict_parent": {"unique_material_ids": len(parent), "covered": len(parent), "missing": 0, "source_evidence_tiers": observed_tiers["strict_parent"]},
            "exact_parent_union": {"unique_material_ids": len(strict_union), "covered": len(strict_union), "missing": 0, "source_evidence_tiers": observed_tiers["strict_union"]},
        },
        "final_training_scope": {
            "stage2": {"unique_material_ids": len(stage2), "covered": len(stage2), "missing": 0, "source_evidence_tiers": observed_tiers["stage2"]},
            "stage3": {"unique_material_ids": len(stage3), "covered": len(stage3), "missing": 0, "source_evidence_tiers": observed_tiers["stage3"]},
            "stage2_stage3_union": {"unique_material_ids": len(final_union), "covered": len(final_union), "missing": 0, "source_evidence_tiers": observed_tiers["final_union"]},
        },
        "sources": {
            "archive_manifest": "11_TESTS_AND_AUDITS/01_STRUCTURE_ARCHIVE_AUDIT/poscar_file_manifest_sha256.tsv",
            "archive_metadata": "02_RAW_DATA/02_MATERIALS_PROJECT_ARCHIVE/mp_full_archive_export/mp_full_archive_metadata.csv",
            "archive_parse_audit": "11_TESTS_AND_AUDITS/01_STRUCTURE_ARCHIVE_AUDIT/audit_summary.json",
            "conventional_attach_summary": "03_CLEANED_AND_MERGED_DATA/02_MERGED_WITH_STRUCTURES/merged_20260609_with_structures/structure_attach_summary.json",
            "alignment_exact": "02_RAW_DATA/04_STRICT_FILTER_OUTPUTS/01_ALIGNMENT_SPLIT_REMOTE_FINAL/strict_exact_only.jsonl",
            "alignment_parent": "02_RAW_DATA/04_STRICT_FILTER_OUTPUTS/01_ALIGNMENT_SPLIT_REMOTE_FINAL/strict_parent_aug.jsonl",
            "stage2_meta": "06_TRAIN_READY_DATA/04_STAGE2_CANONICAL/stage2_full_cation_family_canonical_v1/{train,val,test}_meta.csv",
            "stage3_meta": "06_TRAIN_READY_DATA/08_STAGE3_FAMILY_FULL/stage3_full_cation_family_v1/{train,val,test}_meta.csv",
        },
        "indexes": {
            "strict_ids": "11_TESTS_AND_AUDITS/03_STRUCTURE_SCOPE_INDEXES/strict_3451_material_ids.txt",
            "strict_mapping": "11_TESTS_AND_AUDITS/03_STRUCTURE_SCOPE_INDEXES/strict_3451_structure_mapping.tsv",
            "final_ids": "11_TESTS_AND_AUDITS/03_STRUCTURE_SCOPE_INDEXES/final_8478_material_ids.txt",
            "final_mapping": "11_TESTS_AND_AUDITS/03_STRUCTURE_SCOPE_INDEXES/final_8478_structure_mapping.tsv",
        },
        "status": "PASS",
    }
    audit_path = (args.audit_path or root / "11_TESTS_AND_AUDITS/STRUCTURE_SCOPE_COVERAGE_AUDIT.json").resolve()
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "audit": str(audit_path), **observed}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
