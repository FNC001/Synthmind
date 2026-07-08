#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/Users/wyc/SynPred}"
VERSION="benchmark_100_v32_mp_metadata_aware_stage3_alignment"

SCRIPT_ROOT="$PROJECT_ROOT/scripts/07_infer/structure_to_synthesis_route/benchmark/$VERSION"
OUT_ROOT="$PROJECT_ROOT/outputs/$VERSION"

echo "============================================================"
echo "Benchmark-100 V32 MP-Metadata-Aware Stage3 Alignment Pipeline"
echo "PROJECT_ROOT = $PROJECT_ROOT"
echo "SCRIPT_ROOT  = $SCRIPT_ROOT"
echo "OUT_ROOT     = $OUT_ROOT"
echo "============================================================"
echo

cd "$PROJECT_ROOT"

echo "[STEP 0] Check directories"
mkdir -p "$SCRIPT_ROOT"
mkdir -p "$OUT_ROOT"

echo "[OK] SCRIPT_ROOT = $SCRIPT_ROOT"
echo "[OK] OUT_ROOT    = $OUT_ROOT"
echo

echo "[STEP 1] Show current V32 outputs"
find "$OUT_ROOT" -type f 2>/dev/null | sort || true
echo

echo "[STEP 2] Discover executable V32 step scripts"
STEP_SCRIPTS=()
while IFS= read -r s; do
  STEP_SCRIPTS+=("$s")
done < <(
  find "$SCRIPT_ROOT" -maxdepth 1 -type f \
    \( -name "0*.py" -o -name "0*.sh" \) \
    ! -name "run_full_v32_mp_metadata_aware_stage3_alignment.sh" \
    | sort
)

if [ "${#STEP_SCRIPTS[@]}" -eq 0 ]; then
  echo "[BLOCKED] No numbered V32 step scripts were found in:"
  echo "          $SCRIPT_ROOT"
  echo
  echo "Current available files in SCRIPT_ROOT:"
  ls -lh "$SCRIPT_ROOT" || true
  echo
  echo "This means only V32 outputs were initialized, but the reproducible step scripts have not been written into scripts/ yet."
  echo "Please generate/copy the V32 step scripts first, then rerun this shell."
  exit 2
fi

printf '%s\n' "${STEP_SCRIPTS[@]}"
echo

echo "[STEP 3] Run discovered V32 steps in order"
for s in "${STEP_SCRIPTS[@]}"; do
  echo
  echo "------------------------------------------------------------"
  echo "[RUN] $s"
  echo "------------------------------------------------------------"

  case "$s" in
    *.py)
      base="$(basename "$s")"

      if [ "$base" = "01_inventory_mp_metadata_sources_v32.py" ]; then
        python "$s" \
          --project_root "$PROJECT_ROOT" \
          --output_dir "$OUT_ROOT/metadata_inventory_v32"

      elif [ "$base" = "02_build_mp_metadata_table_v32.py" ]; then
        python "$s" \
          --project_root "$PROJECT_ROOT" \
          --output_dir "$OUT_ROOT/mp_metadata_table_v32"

      elif [ "$base" = "03_attach_mp_metadata_to_stage3_candidates_v32.py" ]; then
        python "$s" \
          --project_root "$PROJECT_ROOT" \
          --output_dir "$OUT_ROOT/stage3_candidates_with_metadata_v32"

      elif [ "$base" = "04_build_metadata_aware_external_stage3_alignment_v32.py" ]; then
        python "$s" \
          --project_root "$PROJECT_ROOT" \
          --output_dir "$OUT_ROOT/metadata_aware_alignment_v32" \
          --top_k 10

      elif [ "$base" = "05_audit_metadata_aware_alignment_v32.py" ]; then
        python "$s" \
          --project_root "$PROJECT_ROOT" \
          --output_dir "$OUT_ROOT/audit_metadata_aware_alignment_v32"

      elif [ "$base" = "06_build_v32_final_report.py" ]; then
        python "$s" \
          --project_root "$PROJECT_ROOT" \
          --output_dir "$OUT_ROOT/FINAL_REPORT_V32"

      else
        python "$s" \
          --project_root "$PROJECT_ROOT"
      fi
      ;;
    *.sh)
      bash "$s" "$PROJECT_ROOT"
      ;;
    *)
      echo "[SKIP] Unknown script type: $s"
      ;;
  esac
done

echo
echo "[STEP 4] Final V32 output check"
find "$OUT_ROOT" -type f 2>/dev/null \
  | grep -E "metadata|alignment|confidence|audit|FINAL|summary|snapshot|archive|REPORT" \
  | sort || true

echo
echo "============================================================"
echo "[DONE] V32 pipeline shell finished."
echo "OUT_ROOT:"
echo "$OUT_ROOT"
echo "============================================================"
