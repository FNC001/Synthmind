#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# SynPred Benchmark Models Runner
#
# Purpose:
#   Benchmark candidate Stage2 / Stage3 algorithms and generate
#   recommended model configuration for final training / inference.
#
# This script is a clean wrapper around:
#   scripts/run_stage2_stage3_benchmark.sh
#
# Outputs:
#   outputs/stage2_stage3_benchmark_summary/benchmark_summary.csv
#   outputs/stage2_stage3_benchmark_summary/benchmark_summary.json
#   outputs/stage2_stage3_benchmark_summary/recommended_inference_config.sh
#   outputs/stage2_stage3_benchmark_summary/recommended_inference_config.json
#   outputs/stage2_stage3_benchmark_summary/benchmark_recommendation.md
#
# Usage:
#   bash scripts/run_benchmark_models.sh [PROJECT_ROOT] [DEVICE]
#
# Examples:
#   bash scripts/run_benchmark_models.sh /Users/wyc/SynPred cpu
#
#   FORCE=1 bash scripts/run_benchmark_models.sh /Users/wyc/SynPred cpu
#
#   FORCE=1 RUN_STAGE2_TREE=0 bash scripts/run_benchmark_models.sh /Users/wyc/SynPred cpu
#
#   FORCE=1 RUN_STAGE3_HCNAF=1 bash scripts/run_benchmark_models.sh /Users/wyc/SynPred cpu
#
# Notes:
#   - This script does not prepare datasets.
#   - Run run_prepare_training_data.sh before this script if datasets are missing.
#   - This script should be run before final model training if you want
#     data-driven algorithm selection.
# ============================================================

PROJECT_ROOT="${1:-/Users/wyc/SynPred}"
DEVICE="${2:-${DEVICE:-cpu}}"

PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
SCRIPTS="${PROJECT_ROOT}/scripts"

BENCHMARK_SCRIPT="${SCRIPTS}/run_stage2_stage3_benchmark.sh"
SUMMARY_DIR="${PROJECT_ROOT}/outputs/stage2_stage3_benchmark_summary"

print_section() {
  echo
  echo "============================================================"
  echo "  $1"
  echo "============================================================"
}

check_file() {
  local f="$1"
  if [[ ! -f "${f}" ]]; then
    echo "[ERROR] Missing file: ${f}"
    exit 1
  fi
  echo "[OK] ${f}"
}

check_dir() {
  local d="$1"
  if [[ ! -d "${d}" ]]; then
    echo "[ERROR] Missing dir: ${d}"
    exit 1
  fi
  echo "[OK] ${d}"
}

print_section "SynPred Benchmark Models"

echo "PROJECT_ROOT     = ${PROJECT_ROOT}"
echo "DEVICE           = ${DEVICE}"
echo "BENCHMARK_SCRIPT = ${BENCHMARK_SCRIPT}"
echo "SUMMARY_DIR      = ${SUMMARY_DIR}"
echo
echo "Benchmark switches inherited from environment:"
echo "  FORCE                  = ${FORCE:-0}"
echo "  RUN_STAGE2_BASELINE    = ${RUN_STAGE2_BASELINE:-1}"
echo "  RUN_STAGE2_GFLOWNET    = ${RUN_STAGE2_GFLOWNET:-1}"
echo "  RUN_STAGE2_TREE        = ${RUN_STAGE2_TREE:-1}"
echo "  RUN_STAGE2_CGCNN       = ${RUN_STAGE2_CGCNN:-0}"
echo "  RUN_STAGE3_LGBM        = ${RUN_STAGE3_LGBM:-1}"
echo "  RUN_STAGE3_MDN         = ${RUN_STAGE3_MDN:-1}"
echo "  RUN_STAGE3_FLOW        = ${RUN_STAGE3_FLOW:-1}"
echo "  RUN_STAGE3_HCNAF       = ${RUN_STAGE3_HCNAF:-0}"
echo "  RUN_STAGE3_CVAE        = ${RUN_STAGE3_CVAE:-0}"
echo "  RUN_STAGE3_DIFFUSION   = ${RUN_STAGE3_DIFFUSION:-0}"

print_section "Precheck"

check_dir "${PROJECT_ROOT}"
check_dir "${SCRIPTS}"
check_file "${BENCHMARK_SCRIPT}"

print_section "Run Stage2 / Stage3 Benchmark"

bash "${BENCHMARK_SCRIPT}" "${PROJECT_ROOT}" "${DEVICE}"

print_section "Check Benchmark Outputs"

check_file "${SUMMARY_DIR}/benchmark_summary.csv"
check_file "${SUMMARY_DIR}/benchmark_summary.json"
check_file "${SUMMARY_DIR}/recommended_inference_config.sh"
check_file "${SUMMARY_DIR}/recommended_inference_config.json"
check_file "${SUMMARY_DIR}/benchmark_recommendation.md"

print_section "BENCHMARK MODELS COMPLETE"

echo "Benchmark summary:"
echo "  ${SUMMARY_DIR}/benchmark_summary.csv"
echo "  ${SUMMARY_DIR}/benchmark_summary.json"
echo
echo "Recommended config:"
echo "  ${SUMMARY_DIR}/recommended_inference_config.sh"
echo "  ${SUMMARY_DIR}/recommended_inference_config.json"
echo
echo "Recommendation report:"
echo "  ${SUMMARY_DIR}/benchmark_recommendation.md"
echo
echo "Next step:"
echo "  bash scripts/run_train_models.sh ${PROJECT_ROOT} ${DEVICE}"
echo
echo "Or inspect recommendation:"
echo "  cat ${SUMMARY_DIR}/recommended_inference_config.sh"
echo
echo "============================================================"
echo "[DONE] SynPred benchmark models completed"
echo "============================================================"
