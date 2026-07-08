#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/Users/wyc/SynPred}"
SCRIPT_DIR="${PROJECT_ROOT}/scripts/03_data"

MODE_ROOT="${PROJECT_ROOT}/data/interim/training_modes/stage2_hybrid/stage2_hybrid"
OUT_ROOT="${PROJECT_ROOT}/data/interim/generative/stage2_gflownet_dataset/hybrid"

FORCE="${FORCE:-0}"

echo "============================================================"
echo "Build Stage2 GFlowNet datasets for pipeline_v3"
echo "PROJECT_ROOT = ${PROJECT_ROOT}"
echo "SCRIPT_DIR   = ${SCRIPT_DIR}"
echo "MODE_ROOT    = ${MODE_ROOT}"
echo "OUT_ROOT     = ${OUT_ROOT}"
echo "FORCE        = ${FORCE}"
echo "============================================================"

check_file() {
  local p="$1"
  if [[ ! -f "$p" ]]; then
    echo "[ERROR] Missing file: $p"
    exit 1
  fi
}

check_dir() {
  local p="$1"
  if [[ ! -d "$p" ]]; then
    echo "[ERROR] Missing directory: $p"
    exit 1
  fi
}

check_bundle() {
  local d="$1"

  check_file "${d}/train.npz"
  check_file "${d}/val.npz"
  check_file "${d}/test.npz"
  check_file "${d}/train_meta.csv"
  check_file "${d}/val_meta.csv"
  check_file "${d}/test_meta.csv"
  check_file "${d}/feature_cols.json"
  check_file "${d}/feature_mean.npy"
  check_file "${d}/feature_std.npy"
  check_file "${d}/action_to_id.json"
  check_file "${d}/action_vocab.json"
  check_file "${d}/precursor_names.json"
  check_file "${d}/summary.json"
}

run_mode() {
  local mode="$1"
  local out_dir="${OUT_ROOT}/${mode}"

  echo
  echo "------------------------------------------------------------"
  echo "[MODE] ${mode}"
  echo "OUT_DIR = ${out_dir}"
  echo "------------------------------------------------------------"

  if [[ "${FORCE}" != "1" && -f "${out_dir}/train.npz" && -f "${out_dir}/summary.json" ]]; then
    echo "[SKIP] Existing dataset bundle found: ${out_dir}"
    check_bundle "${out_dir}"
    return 0
  fi

  mkdir -p "${out_dir}"

  python "${SCRIPT_DIR}/27_build_stage2_gflownet_dataset.py" \
    --project_root "${PROJECT_ROOT}" \
    --input_mode hybrid \
    --mode_input_root "${MODE_ROOT}" \
    --train_mode "${mode}" \
    --output_dir "${out_dir}"

  check_bundle "${out_dir}"

  echo "[OK] Built dataset bundle:"
  echo "  ${out_dir}"
}

echo
echo "[STEP 1] Check required scripts"
check_file "${SCRIPT_DIR}/27_build_stage2_gflownet_dataset.py"
echo "[OK] 27_build_stage2_gflownet_dataset.py found"

echo
echo "[STEP 2] Check mode input root"
check_dir "${MODE_ROOT}"
check_dir "${MODE_ROOT}/gold_only"
check_dir "${MODE_ROOT}/relaxed_only"
check_dir "${MODE_ROOT}/curriculum"
echo "[OK] mode input root exists"

echo
echo "[STEP 3] Build mode-specific GFlowNet datasets"

run_mode gold_only
run_mode relaxed_only
run_mode curriculum_phase1
run_mode curriculum_phase2

echo
echo "============================================================"
echo "[DONE] Stage2 GFlowNet datasets built"
echo "============================================================"

echo
echo "Main pipeline_v3 required bundle:"
echo "  ${OUT_ROOT}/gold_only"

echo
echo "Generated files:"
find "${OUT_ROOT}" -maxdepth 2 -type f | sort

echo
echo "Next step:"
echo "  bash /Users/wyc/SynPred/scripts/04_train/stage2/run_retrain_stage2_gflownet_for_pipeline_v3.sh /Users/wyc/SynPred"
