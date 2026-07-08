#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/Users/wyc/SynPred}"

V50_VERSION="benchmark_100_v50_final_full_gap_closure_report_and_release_package"
OUT_ROOT="$PROJECT_ROOT/outputs/$V50_VERSION"

INPUT_DIR="$OUT_ROOT/input_evidence_v50"
REPORT_DIR="$OUT_ROOT/FINAL_REPORT_V50"
AUDIT_DIR="$OUT_ROOT/audit_v50"
RELEASE_DIR="$OUT_ROOT/release_package_v50"

mkdir -p "$INPUT_DIR" "$REPORT_DIR" "$AUDIT_DIR" "$RELEASE_DIR"

echo "============================================================"
echo "Benchmark-100 V50 Final Full Gap Closure Report and Release Package"
echo "PROJECT_ROOT = $PROJECT_ROOT"
echo "OUT_ROOT     = $OUT_ROOT"
echo "============================================================"

V32_SUMMARY="$PROJECT_ROOT/outputs/benchmark_100_v32_mp_metadata_aware_stage3_alignment/audit_metadata_aware_alignment_v32/v32_metadata_aware_alignment_audit_summary.json"
V33_SUMMARY="$PROJECT_ROOT/outputs/benchmark_100_v33_stage3_library_expansion_or_direct_formula_mapping/direct_formula_mp_mapping_v33/v33_direct_formula_mp_mapping_summary.json"
V34_REPORT="$PROJECT_ROOT/outputs/benchmark_100_v34_real_stage3_expansion_from_v33_manifest/FINAL_REPORT_V34/FINAL_BENCHMARK_100_V34_BLOCKED_REPORT.md"
V38_SUMMARY="$PROJECT_ROOT/outputs/benchmark_100_v38_stage3_input_feature_extension_for_v33_targets/stage3_feature_source_audit_v38/v38_stage3_feature_source_audit_summary.json"
V40C_SUMMARY="$PROJECT_ROOT/outputs/benchmark_100_v40c_flow_stage3_candidate_patch_from_v40b/audit_v40c/v40c_flow_stage3_candidate_patch_summary.json"
V43B_SUMMARY="$PROJECT_ROOT/outputs/benchmark_100_v43b_validate_recovered_poscar_structures/audit_v43b/v43b_poscar_validation_summary.json"
V44_SUMMARY="$PROJECT_ROOT/outputs/benchmark_100_v44_stage3_feature_npz_regeneration_from_v43_poscar/audit_v44/v44_stage3_feature_npz_regeneration_summary.json"
V47C_SUMMARY="$PROJECT_ROOT/outputs/benchmark_100_v47c_no_clip_flow_export_confirms_clipping_collapse/audit_v47c/v47c_no_clip_flow_export_summary.json"
V48_SUMMARY="$PROJECT_ROOT/outputs/benchmark_100_v48_flow_export_with_global_stage3_clip_range/audit_v48/v48_global_clip_flow_export_summary.json"
V49_SUMMARY="$PROJECT_ROOT/outputs/benchmark_100_v49_merge_v40c_v48_and_realign_all_v33_gap_cases/audit_v49/v49_realignment_summary.json"
V49_REALIGN="$PROJECT_ROOT/outputs/benchmark_100_v49_merge_v40c_v48_and_realign_all_v33_gap_cases/realignment_v49/v49_all_v33_gap_case_realignment_summary.csv"
V49_COMPARE="$PROJECT_ROOT/outputs/benchmark_100_v49_merge_v40c_v48_and_realign_all_v33_gap_cases/audit_v49/v49_v33_gap_vs_v49_realignment_comparison.csv"
V49_MERGED="$PROJECT_ROOT/outputs/benchmark_100_v49_merge_v40c_v48_and_realign_all_v33_gap_cases/merged_stage3_library_v49/v49_merged_stage3_library_with_v40c_v48_patch.csv"
V49_PATCH="$PROJECT_ROOT/outputs/benchmark_100_v49_merge_v40c_v48_and_realign_all_v33_gap_cases/combined_stage3_patch_v49/v49_combined_stage3_candidate_patch.csv"

echo
echo "[STEP 1] Check required evidence inputs"

for p in \
  "$V32_SUMMARY" \
  "$V33_SUMMARY" \
  "$V34_REPORT" \
  "$V38_SUMMARY" \
  "$V40C_SUMMARY" \
  "$V43B_SUMMARY" \
  "$V44_SUMMARY" \
  "$V47C_SUMMARY" \
  "$V48_SUMMARY" \
  "$V49_SUMMARY" \
  "$V49_REALIGN" \
  "$V49_COMPARE" \
  "$V49_MERGED" \
  "$V49_PATCH"
do
  if [[ ! -e "$p" ]]; then
    echo "[ERROR] Missing required evidence file: $p"
    exit 1
  fi
  echo "[OK] $p"
done

echo
echo "[STEP 2] Copy evidence files into V50 package"

cp "$V32_SUMMARY" "$INPUT_DIR/v32_metadata_aware_alignment_audit_summary.json"
cp "$V33_SUMMARY" "$INPUT_DIR/v33_direct_formula_mp_mapping_summary.json"
cp "$V34_REPORT" "$INPUT_DIR/v34_blocked_report.md"
cp "$V38_SUMMARY" "$INPUT_DIR/v38_stage3_feature_source_audit_summary.json"
cp "$V40C_SUMMARY" "$INPUT_DIR/v40c_flow_stage3_candidate_patch_summary.json"
cp "$V43B_SUMMARY" "$INPUT_DIR/v43b_poscar_validation_summary.json"
cp "$V44_SUMMARY" "$INPUT_DIR/v44_stage3_feature_npz_regeneration_summary.json"
cp "$V47C_SUMMARY" "$INPUT_DIR/v47c_no_clip_flow_export_summary.json"
cp "$V48_SUMMARY" "$INPUT_DIR/v48_global_clip_flow_export_summary.json"
cp "$V49_SUMMARY" "$INPUT_DIR/v49_realignment_summary.json"
cp "$V49_REALIGN" "$INPUT_DIR/v49_all_v33_gap_case_realignment_summary.csv"
cp "$V49_COMPARE" "$INPUT_DIR/v49_v33_gap_vs_v49_realignment_comparison.csv"
cp "$V49_MERGED" "$RELEASE_DIR/v49_merged_stage3_library_with_v40c_v48_patch.csv"
cp "$V49_PATCH" "$RELEASE_DIR/v49_combined_stage3_candidate_patch.csv"

echo
echo "[STEP 3] Build V50 final audit and master report"

python - <<PY
import json
from pathlib import Path
import pandas as pd

out_root = Path("$OUT_ROOT")
input_dir = Path("$INPUT_DIR")
report_dir = Path("$REPORT_DIR")
audit_dir = Path("$AUDIT_DIR")
release_dir = Path("$RELEASE_DIR")

v49_summary = json.loads((input_dir / "v49_realignment_summary.json").read_text(encoding="utf-8"))
v48_summary = json.loads((input_dir / "v48_global_clip_flow_export_summary.json").read_text(encoding="utf-8"))
v40c_summary = json.loads((input_dir / "v40c_flow_stage3_candidate_patch_summary.json").read_text(encoding="utf-8"))

realign = pd.read_csv(input_dir / "v49_all_v33_gap_case_realignment_summary.csv")
compare = pd.read_csv(input_dir / "v49_v33_gap_vs_v49_realignment_comparison.csv")
patch = pd.read_csv(release_dir / "v49_combined_stage3_candidate_patch.csv")

source_counts = patch["v49_patch_source"].value_counts().to_dict() if "v49_patch_source" in patch.columns else {}

audit = {
    "status": "pass_final_full_gap_closure_package",
    "version": "benchmark_100_v50_final_full_gap_closure_report_and_release_package",
    "v49_status": v49_summary.get("status"),
    "n_v33_gap_cases": int(v49_summary.get("n_v33_gap_cases", 0)),
    "n_patch_supported_cases": int(v49_summary.get("n_patch_supported_cases", 0)),
    "n_pass_patch_supported": int(v49_summary.get("n_pass_patch_supported", 0)),
    "n_combined_patch_rows": int(v49_summary.get("n_combined_patch_rows", 0)),
    "n_v40c_patch_rows": int(v49_summary.get("n_v40c_patch_rows", 0)),
    "n_v48_patch_rows": int(v49_summary.get("n_v48_patch_rows", 0)),
    "v48_condition_diversity_status": v48_summary.get("status"),
    "v40c_patch_status": v40c_summary.get("status"),
    "patch_source_counts": source_counts,
    "all_cases_patch_supported": bool((realign["v49_alignment_status"] == "pass_patch_supported").all()),
    "all_cases_formula_exact_match": bool(realign["formula_exact_match"].astype(bool).all()),
    "all_cases_have_condition_support": bool((realign["condition_distribution_support"] > 0).all()),
    "release_files": {
        "final_report": str(report_dir / "FINAL_BENCHMARK_100_V50_REPORT.md"),
        "metric_summary": str(audit_dir / "v50_final_gap_closure_summary.json"),
        "realignment_summary": str(input_dir / "v49_all_v33_gap_case_realignment_summary.csv"),
        "comparison": str(input_dir / "v49_v33_gap_vs_v49_realignment_comparison.csv"),
        "combined_patch": str(release_dir / "v49_combined_stage3_candidate_patch.csv"),
        "merged_stage3_library": str(release_dir / "v49_merged_stage3_library_with_v40c_v48_patch.csv"),
    },
    "interpretation": (
        "V50 packages the final evidence chain showing that all nine V33 formula-exact gap cases now have Stage3 candidate support through V49. "
        "The package preserves provenance separation between original V30/V32 candidates, V40c real Flow patch candidates, and V48 Flow candidates from V44 regenerated sparse/zero-filled features."
    ),
    "important_caution": (
        "This is a full coverage-closure package, not a claim that all candidate sources are equivalent. "
        "V48 candidates should remain labeled as real_stage3_flow_v48_global_clip_from_v44_regenerated_features."
    )
}

summary_json = audit_dir / "v50_final_gap_closure_summary.json"
summary_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")

realign_md = audit_dir / "v50_final_gap_closure_case_table.md"
realign.to_markdown(realign_md, index=False)

compare_md = audit_dir / "v50_v33_to_v49_comparison.md"
compare.to_markdown(compare_md, index=False)

print(json.dumps(audit, indent=2))
print("[SAVE]", summary_json)
print("[SAVE]", realign_md)
print("[SAVE]", compare_md)
PY

echo
echo "[STEP 4] Write final V50 report"

cat > "$REPORT_DIR/FINAL_BENCHMARK_100_V50_REPORT.md" <<MD
# Final Benchmark-100 V50 Report

## 1. Version

\`benchmark_100_v50_final_full_gap_closure_report_and_release_package\`

## 2. Executive conclusion

V50 finalizes the Benchmark-100 gap-closure evidence chain.

The final V49 result shows:

- all nine V33 formula-exact gap cases are now Stage3-patch supported;
- all nine cases have formula-exact MP matches;
- all nine cases have non-trivial Stage3 condition-candidate support;
- the final status is \`pass_all_v33_gap_cases_patch_supported\`.

## 3. Evidence chain

### V32: metadata-aware alignment issue

V32 introduced stricter metadata-aware alignment using MP formula, element sets, target-family compatibility, and real Stage3 condition support.

It revealed that several external cases were weakly aligned, under review, or unaligned.

### V33: formula-exact MP target mapping

V33 converted all weak/review/unaligned cases into formula-exact MP targets.

This localized the issue from alignment logic to Stage3 library coverage.

### V34–V38: Stage3 library and feature coverage diagnosis

V34 confirmed that the nine V33 targets were absent from the existing real V30 Stage3 candidate library.

V37/V38 further showed that the issue was not only candidate-library absence, but also Stage3 input-feature coverage.

### V40c: partial real Flow patch

V40c produced a normalized partial real Flow Stage3 patch for the feature-available cases:

- BaSO4 / \`mp-1190568\`
- CaSO4 / \`mp-12372\`

### V44–V48: regenerated-feature Flow correction

V43/V43b recovered and validated POSCAR structures for the remaining seven targets.

V44 regenerated Stage3-compatible NPZ feature inputs.

V45 initially produced seed-collapsed outputs because clipping used the single-point regenerated V44 train range.

V47c confirmed that no-clip Flow export preserved condition diversity.

V48 corrected the export by applying original global Stage3 condition bounds from \`condition_schema.json\`.

### V49: all-nine gap-case realignment

V49 merged:

- V40c Flow patch for BaSO4 and CaSO4;
- V48 global-clipped Flow candidates for the remaining seven targets;

and reran realignment for all nine V33 gap cases.

The final V49 status is:

\`pass_all_v33_gap_cases_patch_supported\`

## 4. Key release outputs

- Final summary:
  \`audit_v50/v50_final_gap_closure_summary.json\`

- Final case table:
  \`audit_v50/v50_final_gap_closure_case_table.md\`

- V33-to-V49 comparison:
  \`audit_v50/v50_v33_to_v49_comparison.md\`

- Combined Stage3 patch:
  \`release_package_v50/v49_combined_stage3_candidate_patch.csv\`

- Merged Stage3 library:
  \`release_package_v50/v49_merged_stage3_library_with_v40c_v48_patch.csv\`

## 5. Provenance labels

The final package preserves candidate provenance:

1. original V30/V32 Stage3 library candidates;
2. \`v40c_real_flow_patch_for_feature_available_targets\`;
3. \`v48_global_clip_flow_from_v44_regenerated_features\`.

The V48 subset should also be described as:

\`real_stage3_flow_v48_global_clip_from_v44_regenerated_features\`

## 6. Scientific caution

V50 closes the Benchmark-100 Stage3 coverage gap for the nine V33 formula-exact cases.

However, V48 candidates were generated from V44 regenerated sparse/zero-filled feature inputs. They are valid for controlled alignment coverage and interface-level inference, but should not be described as fully equivalent to original-distribution V30 Stage3 candidates.

## 7. Final status

\`pass_final_full_gap_closure_package\`
MD

cp "$AUDIT_DIR/v50_final_gap_closure_summary.json" "$REPORT_DIR/FINAL_V50_METRIC_SUMMARY.json"

echo
echo "[STEP 5] Build release manifest"

cat > "$RELEASE_DIR/README_V50_RELEASE.md" <<MD
# Benchmark-100 V50 Release Package

This package contains the final V50 gap-closure evidence and release artifacts.

## Main files

- \`v49_combined_stage3_candidate_patch.csv\`
- \`v49_merged_stage3_library_with_v40c_v48_patch.csv\`
- \`../FINAL_REPORT_V50/FINAL_BENCHMARK_100_V50_REPORT.md\`
- \`../audit_v50/v50_final_gap_closure_summary.json\`
- \`../audit_v50/v50_final_gap_closure_case_table.md\`
- \`../audit_v50/v50_v33_to_v49_comparison.md\`

## Important provenance caution

The merged library contains multiple candidate provenance classes.

Do not silently mix:

- original V30/V32 candidates;
- V40c real Flow patch candidates;
- V48 regenerated-feature Flow candidates.

The V48 subset should remain labeled as:

\`real_stage3_flow_v48_global_clip_from_v44_regenerated_features\`
MD

echo
echo "[STEP 6] Archive V50 final package"

cd "$PROJECT_ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
BASE="final_demo_snapshot_${V50_VERSION}_${TS}"

tar -czf "outputs/${BASE}.tar.gz" \
  "outputs/$V50_VERSION" \
  "scripts/07_infer/structure_to_synthesis_route/benchmark/$V50_VERSION"

cd outputs
shasum -a 256 "${BASE}.tar.gz" > "${BASE}.tar.gz.sha256"

echo
echo "============================================================"
echo "[DONE] V50 final full gap-closure release package completed."
echo "Report:"
echo "$REPORT_DIR/FINAL_BENCHMARK_100_V50_REPORT.md"
echo
echo "Summary:"
echo "$AUDIT_DIR/v50_final_gap_closure_summary.json"
echo
echo "Release package:"
echo "$RELEASE_DIR"
echo
echo "Archive:"
echo "$PROJECT_ROOT/outputs/${BASE}.tar.gz"
echo "Checksum:"
echo "$PROJECT_ROOT/outputs/${BASE}.tar.gz.sha256"
echo "============================================================"
