#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/Users/wyc/SynPred}"

V33_NAME="benchmark_100_v33_stage3_library_expansion_or_direct_formula_mapping"
SCRIPT_ROOT="$PROJECT_ROOT/scripts/07_infer/structure_to_synthesis_route/benchmark/$V33_NAME"
OUT_ROOT="$PROJECT_ROOT/outputs/$V33_NAME"

cd "$PROJECT_ROOT"

echo "============================================================"
echo "Benchmark-100 V33 Stage3 Library Expansion / Direct Mapping"
echo "PROJECT_ROOT = $PROJECT_ROOT"
echo "SCRIPT_ROOT  = $SCRIPT_ROOT"
echo "OUT_ROOT     = $OUT_ROOT"
echo "============================================================"

echo
echo "[STEP 0] Check required scripts"
required_scripts=(
  "$SCRIPT_ROOT/init_v33_stage3_library_expansion.sh"
  "$SCRIPT_ROOT/01_build_v33_gap_case_table.py"
  "$SCRIPT_ROOT/02_build_direct_formula_mp_mapping_v33.py"
  "$SCRIPT_ROOT/03_prepare_expanded_stage3_target_list_v33.py"
  "$SCRIPT_ROOT/04_build_stage3_expansion_request_manifest_v33.py"
  "$SCRIPT_ROOT/05_build_v33_final_report.py"
)

for s in "${required_scripts[@]}"; do
  if [ ! -f "$s" ]; then
    echo "[ERROR] Missing required script: $s"
    exit 1
  fi
  echo "[OK] $s"
done

echo
echo "[STEP 1] Initialize V33 inputs if needed"
bash "$SCRIPT_ROOT/init_v33_stage3_library_expansion.sh" "$PROJECT_ROOT"

echo
echo "[STEP 2] Check required V33 frozen inputs"
required_inputs=(
  "$OUT_ROOT/input_from_v32/v32_metadata_aware_alignment_unaligned_cases.csv"
  "$OUT_ROOT/input_from_v32/v32_metadata_aware_alignment_bad_or_review_cases.csv"
  "$PROJECT_ROOT/outputs/benchmark_100_v32_mp_metadata_aware_stage3_alignment/mp_metadata_table_v32/v32_mp_metadata_table.csv"
  "$PROJECT_ROOT/outputs/benchmark_100_v32_mp_metadata_aware_stage3_alignment/stage3_candidates_with_metadata_v32/v32_stage3_candidates_with_mp_metadata.csv"
)

for f in "${required_inputs[@]}"; do
  if [ ! -f "$f" ]; then
    echo "[ERROR] Missing required input: $f"
    exit 1
  fi
  echo "[OK] $f"
done

echo
echo "[STEP 3] Build V33 gap case table"
python "$SCRIPT_ROOT/01_build_v33_gap_case_table.py" \
  --project_root "$PROJECT_ROOT" \
  --output_dir "$OUT_ROOT/stage3_library_gap_analysis_v33"

echo
echo "[STEP 4] Build direct formula-to-MP mapping"
python "$SCRIPT_ROOT/02_build_direct_formula_mp_mapping_v33.py" \
  --project_root "$PROJECT_ROOT" \
  --output_dir "$OUT_ROOT/direct_formula_mp_mapping_v33"

echo
echo "[STEP 5] Prepare expanded Stage3 target list"
python "$SCRIPT_ROOT/03_prepare_expanded_stage3_target_list_v33.py" \
  --project_root "$PROJECT_ROOT" \
  --output_dir "$OUT_ROOT/expanded_stage3_targets_v33"

echo
echo "[STEP 6] Build Stage3 expansion request manifest"
python "$SCRIPT_ROOT/04_build_stage3_expansion_request_manifest_v33.py" \
  --project_root "$PROJECT_ROOT" \
  --output_dir "$OUT_ROOT/stage3_expansion_request_manifest_v33"

echo
echo "[STEP 7] Build V33 final report"
python "$SCRIPT_ROOT/05_build_v33_final_report.py" \
  --project_root "$PROJECT_ROOT" \
  --output_dir "$OUT_ROOT"

echo
echo "[STEP 8] Final V33 output check"
find "$OUT_ROOT" -maxdepth 4 -type f | sort

echo
echo "[STEP 9] Show key summaries"
python - <<PY
from pathlib import Path
import json

root = Path("$OUT_ROOT")

summary_files = [
    root / "stage3_library_gap_analysis_v33/v33_gap_case_table_summary.json",
    root / "direct_formula_mp_mapping_v33/v33_direct_formula_mp_mapping_summary.json",
    root / "expanded_stage3_targets_v33/v33_expanded_stage3_target_list_summary.json",
    root / "stage3_expansion_request_manifest_v33/v33_stage3_expansion_request_manifest_summary.json",
]

for p in summary_files:
    print("\\n====", p.relative_to(root), "====")
    if not p.exists():
        print("[MISSING]", p)
        continue
    print(json.dumps(json.loads(p.read_text(encoding="utf-8")), indent=2, ensure_ascii=False))
PY

echo
echo "============================================================"
echo "[DONE] V33 planning layer completed."
echo "OUT_ROOT:"
echo "$OUT_ROOT"
echo
echo "Final report:"
echo "$OUT_ROOT/FINAL_REPORT_V33/FINAL_BENCHMARK_100_V33_REPORT.md"
echo
echo "Key files:"
echo "$OUT_ROOT/stage3_library_gap_analysis_v33/v33_gap_case_table.md"
echo "$OUT_ROOT/direct_formula_mp_mapping_v33/v33_direct_formula_mp_mapping.md"
echo "$OUT_ROOT/expanded_stage3_targets_v33/v33_expanded_stage3_target_list.md"
echo "$OUT_ROOT/stage3_expansion_request_manifest_v33/v33_stage3_expansion_request_manifest.md"
echo "============================================================"
