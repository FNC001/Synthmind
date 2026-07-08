#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/Users/wyc/SynPred}"
V33_NAME="benchmark_100_v33_stage3_library_expansion_or_direct_formula_mapping"
OUT_ROOT="$PROJECT_ROOT/outputs/$V33_NAME"
SCRIPT_ROOT="$PROJECT_ROOT/scripts/07_infer/structure_to_synthesis_route/benchmark/$V33_NAME"

echo "============================================================"
echo "Initialize Benchmark-100 V33"
echo "PROJECT_ROOT = $PROJECT_ROOT"
echo "SCRIPT_ROOT  = $SCRIPT_ROOT"
echo "OUT_ROOT     = $OUT_ROOT"
echo "============================================================"

cd "$PROJECT_ROOT"

mkdir -p "$OUT_ROOT"
mkdir -p "$OUT_ROOT/input_from_v32"
mkdir -p "$OUT_ROOT/direct_formula_mp_mapping_v33"
mkdir -p "$OUT_ROOT/stage3_library_gap_analysis_v33"
mkdir -p "$OUT_ROOT/expanded_stage3_targets_v33"
mkdir -p "$OUT_ROOT/FINAL_REPORT_V33"

echo
echo "[STEP 1] Copy frozen V32 outputs"

V32_OUT="$PROJECT_ROOT/outputs/benchmark_100_v32_mp_metadata_aware_stage3_alignment"

KEY_V32_FILES=(
  "FINAL_REPORT_V32/FINAL_BENCHMARK_100_V32_REPORT.md"
  "FINAL_REPORT_V32/FINAL_V32_METRIC_SUMMARY.md"
  "audit_metadata_aware_alignment_v32/v32_metadata_aware_alignment_audit_summary.json"
  "audit_metadata_aware_alignment_v32/v32_metadata_aware_alignment_unaligned_cases.csv"
  "audit_metadata_aware_alignment_v32/v32_metadata_aware_alignment_bad_or_review_cases.csv"
  "metadata_aware_alignment_v32/v32_metadata_aware_external_stage3_alignment_summary.csv"
  "metadata_aware_alignment_v32/v32_metadata_aware_external_to_stage3_alignment_topk.csv"
  "stage3_candidates_with_metadata_v32/v32_stage3_candidates_with_mp_metadata_summary.json"
  "mp_metadata_table_v32/v32_mp_metadata_table.csv"
)

for rel in "${KEY_V32_FILES[@]}"; do
  src="$V32_OUT/$rel"
  dst="$OUT_ROOT/input_from_v32/$(basename "$rel")"
  if [ -f "$src" ]; then
    cp "$src" "$dst"
    echo "[COPY] $src -> $dst"
  else
    echo "[WARN] missing V32 file skipped: $src"
  fi
done

echo
echo "[STEP 2] Write V33 master index"

cat > "$OUT_ROOT/FINAL_BENCHMARK_100_V33_MASTER_INDEX.md" <<INDEX
# Final Benchmark-100 V33 Master Index

## 1. Version

\`benchmark_100_v33_stage3_library_expansion_or_direct_formula_mapping\`

## 2. Purpose

V33 upgrades V32 by addressing chemistry-coverage gaps in the real Stage3 MDN/Flow condition-candidate library.

V32 already established a conservative MP-metadata-aware alignment. However, V32 also identified:

- one unaligned case: \`external_case_015 / NaCl\`
- eight weak or review aligned cases
- limited real Stage3 coverage: 249 unique MP-indexed materials

V33 focuses on resolving these gaps.

## 3. Main goals

1. Build a direct external-formula-to-MP mapping table.
2. Identify all unaligned and weak-review external cases from V32.
3. Check whether direct MP targets exist in the MP metadata table.
4. Prepare an expanded Stage3 target list for missing or weakly covered chemistries.
5. Preserve the conservative V32 audit logic.

## 4. Frozen V32 inputs

Copied into:

\`outputs/benchmark_100_v33_stage3_library_expansion_or_direct_formula_mapping/input_from_v32/\`

## 5. Planned components

1. \`01_build_v33_gap_case_table.py\`
2. \`02_build_direct_formula_mp_mapping_v33.py\`
3. \`03_prepare_expanded_stage3_target_list_v33.py\`
4. \`04_score_v33_expected_coverage_gain.py\`
5. \`05_build_v33_final_report.py\`
6. \`run_full_v33_stage3_library_expansion.sh\`

## 6. Current status

Initialized only.

V33 should not overwrite V32. V32 remains the frozen final demo release.
INDEX

echo "[SAVE] $OUT_ROOT/FINAL_BENCHMARK_100_V33_MASTER_INDEX.md"

echo
echo "[STEP 3] Write V33 TODO"

cat > "$OUT_ROOT/V33_TODO.md" <<TODO
# V33 TODO

## Priority 1: gap cases

Use V32 audit outputs to collect:

- unaligned cases
- bad/review aligned cases
- weak element-overlap cases
- family mismatch cases
- low condition-support cases

## Priority 2: direct formula-to-MP mapping

Start from these external formulas:

- NaCl
- Y2O3
- ZnO
- BaSO4
- CaSO4
- Li3N
- AlN
- CaCO3
- Bi2Se3

Try to locate exact formula matches in the MP metadata table.

## Priority 3: Stage3 expansion target list

For each direct MP match, create a Stage3 target expansion row containing:

- external_case_id
- external_formula
- external_elements
- external_family
- direct_mp_id
- mp_formula
- mp_elements
- mp_family
- reason_for_expansion

## Priority 4: next model action

After direct MP mapping is prepared, decide whether to:

1. export existing Stage3 candidates for these MP IDs if available;
2. generate new Stage3 MDN/Flow candidates;
3. add fallback retrieval-conditioned condition priors only when real Stage3 model output is unavailable.

## Priority 5: conservative audit

V33 should only promote a case from review to pass if:

- formula exact match exists, or
- element Jaccard is high and family compatibility is strong, and
- Stage3 condition support is non-trivial.
TODO

echo "[SAVE] $OUT_ROOT/V33_TODO.md"

echo
echo "[DONE] V33 initialized."
echo "OUT_ROOT:"
echo "$OUT_ROOT"
echo
echo "Next:"
echo "cat $OUT_ROOT/FINAL_BENCHMARK_100_V33_MASTER_INDEX.md"
echo "cat $OUT_ROOT/V33_TODO.md"
