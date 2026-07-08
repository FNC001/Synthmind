#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/Users/wyc/SynPred}"

V43B_VERSION="benchmark_100_v43b_validate_recovered_poscar_structures"
OUT_ROOT="$PROJECT_ROOT/outputs/$V43B_VERSION"

POSCAR_DIR="$PROJECT_ROOT/outputs/benchmark_100_v43_reconstruct_poscar_from_mp_json_for_remaining_v33_targets/recovered_poscar_v43"
V43_AUDIT="$PROJECT_ROOT/outputs/benchmark_100_v43_reconstruct_poscar_from_mp_json_for_remaining_v33_targets/audit_v43/v43_recovered_poscar_audit.csv"

AUDIT_DIR="$OUT_ROOT/audit_v43b"
REPORT_DIR="$OUT_ROOT/FINAL_REPORT_V43B"

mkdir -p "$AUDIT_DIR" "$REPORT_DIR"

echo "============================================================"
echo "Benchmark-100 V43b Validate Recovered POSCAR Structures"
echo "PROJECT_ROOT = $PROJECT_ROOT"
echo "POSCAR_DIR   = $POSCAR_DIR"
echo "OUT_ROOT     = $OUT_ROOT"
echo "============================================================"

echo
echo "[STEP 1] Check inputs"

if [[ ! -d "$POSCAR_DIR" ]]; then
  echo "[ERROR] missing POSCAR_DIR: $POSCAR_DIR"
  exit 1
fi

if [[ ! -f "$V43_AUDIT" ]]; then
  echo "[ERROR] missing V43 audit: $V43_AUDIT"
  exit 1
fi

echo "[OK] $POSCAR_DIR"
echo "[OK] $V43_AUDIT"

echo
echo "[STEP 2] Validate recovered POSCAR files with pymatgen"

python - <<PY
import json
from pathlib import Path

import pandas as pd

poscar_dir = Path("$POSCAR_DIR")
v43_audit_path = Path("$V43_AUDIT")
audit_dir = Path("$AUDIT_DIR")
audit_dir.mkdir(parents=True, exist_ok=True)

try:
    from pymatgen.core import Structure
except Exception as e:
    summary = {
        "status": "blocked_pymatgen_not_available",
        "error": repr(e),
        "interpretation": "pymatgen is required to validate recovered POSCAR files."
    }
    (audit_dir / "v43b_poscar_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    raise SystemExit(1)

v43 = pd.read_csv(v43_audit_path)

expected = {}
for _, r in v43.iterrows():
    mp_id = str(r.get("mp_id", ""))
    expected[mp_id] = {
        "external_case_id": str(r.get("external_case_id", "")),
        "target_formula": str(r.get("target_formula", "")),
        "expected_n_sites": r.get("n_sites", ""),
        "expected_elements": str(r.get("elements", "")),
        "expected_element_counts": str(r.get("element_counts", "")),
    }

rows = []

for p in sorted(poscar_dir.glob("*.vasp")):
    mp_id = p.stem
    exp = expected.get(mp_id, {})
    status = "pass"
    error = ""

    try:
        s = Structure.from_file(str(p))
        reduced_formula = s.composition.reduced_formula
        n_sites = len(s)
        elements = ";".join(sorted([str(el) for el in s.composition.elements]))
        lattice_a, lattice_b, lattice_c = [float(x) for x in s.lattice.abc]
        alpha, beta, gamma = [float(x) for x in s.lattice.angles]

        expected_formula = str(exp.get("target_formula", ""))
        formula_match = (reduced_formula == expected_formula)

        if not formula_match:
            status = "review_formula_mismatch"

        rows.append({
            "mp_id": mp_id,
            "external_case_id": exp.get("external_case_id", ""),
            "poscar_path": str(p),
            "target_formula": expected_formula,
            "parsed_formula": reduced_formula,
            "formula_match": formula_match,
            "n_sites": int(n_sites),
            "parsed_elements": elements,
            "lattice_a": lattice_a,
            "lattice_b": lattice_b,
            "lattice_c": lattice_c,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "validation_status": status,
            "error": error,
        })

    except Exception as e:
        rows.append({
            "mp_id": mp_id,
            "external_case_id": exp.get("external_case_id", ""),
            "poscar_path": str(p),
            "target_formula": exp.get("target_formula", ""),
            "parsed_formula": "",
            "formula_match": False,
            "n_sites": "",
            "parsed_elements": "",
            "lattice_a": "",
            "lattice_b": "",
            "lattice_c": "",
            "alpha": "",
            "beta": "",
            "gamma": "",
            "validation_status": "failed_structure_read",
            "error": repr(e),
        })

audit = pd.DataFrame(rows)

out_csv = audit_dir / "v43b_poscar_validation_audit.csv"
out_md = audit_dir / "v43b_poscar_validation_audit.md"
summary_json = audit_dir / "v43b_poscar_validation_summary.json"

audit.to_csv(out_csv, index=False)
audit.to_markdown(out_md, index=False)

summary = {
    "status": "pass" if len(audit) and (audit["validation_status"] == "pass").all() else "pass_with_review_or_failure",
    "n_poscar_files": int(len(audit)),
    "n_valid_structures": int((audit["validation_status"] == "pass").sum()),
    "n_review_or_failed": int((audit["validation_status"] != "pass").sum()),
    "validated_mp_ids": sorted(audit.loc[audit["validation_status"] == "pass", "mp_id"].astype(str).tolist()),
    "review_or_failed_mp_ids": sorted(audit.loc[audit["validation_status"] != "pass", "mp_id"].astype(str).tolist()),
    "output_csv": str(out_csv),
    "interpretation": (
        "V43b validates that recovered V43 POSCAR files are readable by pymatgen and match the expected target formulas. "
        "Passing structures can be used as structure-backed inputs for V44 Stage3 feature/NPZ regeneration."
    ),
    "next_required_step": (
        "Proceed to V44 to regenerate Stage3-compatible input features and NPZ files from these validated structures."
    ),
}

summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(json.dumps(summary, indent=2))
print("[SAVE]", out_csv)
print("[SAVE]", out_md)
print("[SAVE]", summary_json)
PY

echo
echo "[STEP 3] Build V43b report"

cat > "$REPORT_DIR/FINAL_BENCHMARK_100_V43B_REPORT.md" <<MD
# Final Benchmark-100 V43b Report

## 1. Version

\`benchmark_100_v43b_validate_recovered_poscar_structures\`

## 2. Purpose

V43b validates the POSCAR-style structure files reconstructed in V43.

The goal is to confirm that each recovered structure can be parsed by pymatgen and that the parsed formula matches the expected target formula.

## 3. Key outputs

- Validation audit:
  \`audit_v43b/v43b_poscar_validation_audit.csv\`

- Markdown audit:
  \`audit_v43b/v43b_poscar_validation_audit.md\`

- Summary:
  \`audit_v43b/v43b_poscar_validation_summary.json\`

## 4. Interpretation

V43b is a structure validation layer.

A passing V43b result means the seven remaining V33 targets now have valid, locally recovered, structure-backed POSCAR inputs.

## 5. Next required step

Proceed to V44:

\`benchmark_100_v44_stage3_feature_npz_regeneration_from_v43_poscar\`

V44 should regenerate Stage3-compatible feature arrays and NPZ files from the validated POSCAR structures.
MD

cp "$AUDIT_DIR/v43b_poscar_validation_summary.json" "$REPORT_DIR/FINAL_V43B_METRIC_SUMMARY.json"

echo
echo "[STEP 4] Archive V43b checkpoint"

cd "$PROJECT_ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
BASE="checkpoint_${V43B_VERSION}_${TS}"

tar -czf "outputs/${BASE}.tar.gz" \
  "outputs/$V43B_VERSION" \
  "scripts/07_infer/structure_to_synthesis_route/benchmark/$V43B_VERSION"

cd outputs
shasum -a 256 "${BASE}.tar.gz" > "${BASE}.tar.gz.sha256"

echo
echo "============================================================"
echo "[DONE] V43b POSCAR validation completed."
echo "Report:"
echo "$REPORT_DIR/FINAL_BENCHMARK_100_V43B_REPORT.md"
echo
echo "Audit:"
echo "$AUDIT_DIR/v43b_poscar_validation_audit.md"
echo
echo "Summary:"
echo "$AUDIT_DIR/v43b_poscar_validation_summary.json"
echo
echo "Archive:"
echo "$PROJECT_ROOT/outputs/${BASE}.tar.gz"
echo "Checksum:"
echo "$PROJECT_ROOT/outputs/${BASE}.tar.gz.sha256"
echo "============================================================"
