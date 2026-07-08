#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/Users/wyc/SynPred"
DATA_ROOT="${PROJECT_ROOT}/data"
SCRIPTS_ROOT="${PROJECT_ROOT}/scripts"
DATA_SCRIPT_DIR="${SCRIPTS_ROOT}/03_data"

MODE_INPUT_ROOT="${DATA_ROOT}/interim/model_inputs/stage2_hybrid/stage2_hybrid"
GEN_ROOT="${DATA_ROOT}/interim/generative"

CVAE_ROOT="${GEN_ROOT}/stage2_cvae_dataset/hybrid"
AR_ROOT="${GEN_ROOT}/stage2_ar_dataset/hybrid"
SETPRED_ROOT="${GEN_ROOT}/stage2_setpred_dataset/hybrid"
GFLOWNET_ROOT="${GEN_ROOT}/stage2_gflownet_dataset/hybrid"
REFINE_ROOT="${GEN_ROOT}/stage2_refine_dataset/hybrid"

INPUT_MODE="hybrid"
MODES=("relaxed_only" "gold_only" "curriculum_phase1" "curriculum_phase2")

print_section() {
  echo
  echo "========== $1 =========="
}

check_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] Missing file: $path"
    exit 1
  fi
}

check_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "[ERROR] Missing directory: $path"
    exit 1
  fi
}

run_cmd() {
  echo
  echo "[RUN] $*"
  "$@"
}

print_section "STEP 0: Config"

echo "PROJECT_ROOT    = ${PROJECT_ROOT}"
echo "MODE_INPUT_ROOT = ${MODE_INPUT_ROOT}"
echo "CVAE_ROOT       = ${CVAE_ROOT}"
echo "AR_ROOT         = ${AR_ROOT}"
echo "SETPRED_ROOT    = ${SETPRED_ROOT}"
echo "GFLOWNET_ROOT   = ${GFLOWNET_ROOT}"
echo "REFINE_ROOT     = ${REFINE_ROOT}"

check_dir "${MODE_INPUT_ROOT}"
cd "${DATA_SCRIPT_DIR}"

print_section "STEP 1: Refresh standard mode inputs"

run_cmd python 00_prepare_mode_inputs_for_stage2.py \
  --source_root "${DATA_ROOT}/interim/training_modes/stage2_hybrid/stage2_hybrid" \
  --output_root "${DATA_ROOT}/interim/model_inputs/stage2_hybrid" \
  --dataset_name stage2_hybrid

print_section "STEP 2: Pre-check mode files"

check_file "${MODE_INPUT_ROOT}/relaxed_only/stage2_train_hybrid.csv"
check_file "${MODE_INPUT_ROOT}/relaxed_only/stage2_val_hybrid.csv"
check_file "${MODE_INPUT_ROOT}/relaxed_only/stage2_test_hybrid.csv"

check_file "${MODE_INPUT_ROOT}/gold_only/stage2_train_hybrid.csv"
check_file "${MODE_INPUT_ROOT}/gold_only/stage2_val_hybrid.csv"
check_file "${MODE_INPUT_ROOT}/gold_only/stage2_test_hybrid.csv"

check_file "${MODE_INPUT_ROOT}/curriculum/phase1/stage2_train_hybrid.csv"
check_file "${MODE_INPUT_ROOT}/curriculum/phase1/stage2_val_hybrid.csv"
check_file "${MODE_INPUT_ROOT}/curriculum/phase1/stage2_test_hybrid.csv"

check_file "${MODE_INPUT_ROOT}/curriculum/phase2/stage2_train_hybrid.csv"
check_file "${MODE_INPUT_ROOT}/curriculum/phase2/stage2_val_hybrid.csv"
check_file "${MODE_INPUT_ROOT}/curriculum/phase2/stage2_test_hybrid.csv"

echo "[OK] mode files found."

print_section "STEP 3: Build Stage2 CVAE datasets"

for mode in "${MODES[@]}"; do
  out="${CVAE_ROOT}/${mode}"

  echo
  echo "[INFO] CVAE mode=${mode}"
  echo "[INFO] mode_input_root=${MODE_INPUT_ROOT}"
  echo "[INFO] output=${out}"

  run_cmd python 20_build_stage2_cvae_dataset.py \
    --project_root "${PROJECT_ROOT}" \
    --input_mode "${INPUT_MODE}" \
    --mode_input_root "${MODE_INPUT_ROOT}" \
    --output_dir "${out}" \
    --train_mode "${mode}"
done

print_section "STEP 4: Build Stage2 AR-set datasets"

for mode in "${MODES[@]}"; do
  out="${AR_ROOT}/${mode}"

  echo
  echo "[INFO] AR mode=${mode}"
  echo "[INFO] mode_input_root=${MODE_INPUT_ROOT}"
  echo "[INFO] output=${out}"

  run_cmd python 25_build_stage2_ar_dataset.py \
    --project_root "${PROJECT_ROOT}" \
    --input_mode "${INPUT_MODE}" \
    --mode_input_root "${MODE_INPUT_ROOT}" \
    --output_dir "${out}" \
    --train_mode "${mode}"
done

print_section "STEP 5: Build Stage2 SetPred datasets"

for mode in "${MODES[@]}"; do
  out="${SETPRED_ROOT}/${mode}"

  echo
  echo "[INFO] SetPred mode=${mode}"
  echo "[INFO] mode_input_root=${MODE_INPUT_ROOT}"
  echo "[INFO] output=${out}"

  run_cmd python 26_build_stage2_setpred_dataset.py \
    --project_root "${PROJECT_ROOT}" \
    --input_mode "${INPUT_MODE}" \
    --mode_input_root "${MODE_INPUT_ROOT}" \
    --output_dir "${out}" \
    --train_mode "${mode}"
done

print_section "STEP 6: Build Stage2 GFlowNet datasets"

for mode in "${MODES[@]}"; do
  out="${GFLOWNET_ROOT}/${mode}"

  echo
  echo "[INFO] GFlowNet mode=${mode}"
  echo "[INFO] mode_input_root=${MODE_INPUT_ROOT}"
  echo "[INFO] output=${out}"

  run_cmd python 27_build_stage2_gflownet_dataset.py \
    --project_root "${PROJECT_ROOT}" \
    --input_mode "${INPUT_MODE}" \
    --mode_input_root "${MODE_INPUT_ROOT}" \
    --output_dir "${out}" \
    --train_mode "${mode}"
done

#print_section "STEP 7: Build Stage2 Refiner datasets"

#if [[ -f "${DATA_SCRIPT_DIR}/29_build_stage2_refine_dataset.py" ]]; then
#  for mode in "${MODES[@]}"; do
#    out="${REFINE_ROOT}/${mode}"

#    echo
#    echo "[INFO] Refiner mode=${mode}"
#    echo "[INFO] mode_input_root=${MODE_INPUT_ROOT}"
#    echo "[INFO] output=${out}"

#    python 29_build_stage2_refine_dataset.py \
#      --mode_input_root "${MODE_INPUT_ROOT}" \
#      --train_mode "${mode}" \
#      --output_dir "${out}" || {
#        echo "[WARN] Refiner failed for mode=${mode}; continue."
#      }
#  done
#else
#  echo "[SKIP] 29_build_stage2_refine_dataset.py not found."
#fi

print_section "STEP 7: Output summary"

echo
echo "[OK] CVAE datasets:"
find "${CVAE_ROOT}" -maxdepth 3 -type f 2>/dev/null | sort || true

echo
echo "[OK] AR datasets:"
find "${AR_ROOT}" -maxdepth 3 -type f 2>/dev/null | sort || true

echo
echo "[OK] SetPred datasets:"
find "${SETPRED_ROOT}" -maxdepth 3 -type f 2>/dev/null | sort || true

echo
echo "[OK] GFlowNet datasets:"
find "${GFLOWNET_ROOT}" -maxdepth 3 -type f 2>/dev/null | sort || true

#echo
#echo "[OK] Refiner datasets:"
#find "${REFINE_ROOT}" -maxdepth 3 -type f 2>/dev/null | sort || true

echo
echo "========== DONE =========="
