#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# SynPred Train Models Pipeline
#
# Purpose:
#   Train official SynPred production models only.
#
# This script does NOT:
#   - prepare raw training data
#   - run benchmark / ablation comparison
#   - run production inference
#
# Recommended workflow:
#   1) bash scripts/run_prepare_training_data.sh /Users/wyc/SynPred cpu
#   2) bash scripts/run_benchmark_models.sh /Users/wyc/SynPred cpu
#   3) bash scripts/run_train_models.sh /Users/wyc/SynPred cpu
#   4) bash scripts/run_inference_pipeline.sh /Users/wyc/SynPred cpu
#
# Usage:
#   bash scripts/run_train_models.sh [PROJECT_ROOT] [DEVICE]
#
# Examples:
#   bash scripts/run_train_models.sh /Users/wyc/SynPred cpu
#
#   FORCE=1 bash scripts/run_train_models.sh /Users/wyc/SynPred cpu
#
#   FORCE=1 START_FROM=train_stage3 STOP_AFTER=train_stage3 \
#     bash scripts/run_train_models.sh /Users/wyc/SynPred cpu
#
# Env options:
#   DEVICE=cpu|cuda|mps
#   FORCE=0|1
#   START_FROM=step_name
#   STOP_AFTER=step_name
#   CHECK_OUTPUTS=0|1
#   SKIP_RETRAIN=0|1
#
# Training switches:
#   RUN_TRAIN_STAGE2=1
#   RUN_TRAIN_STAGE2_EC=1
#   RUN_TRAIN_STAGE3=1
#   RUN_TRAIN_STAGE3_LGBM=1
#   RUN_TRAIN_STAGE35=1
#
# Steps:
#   train_stage2
#   train_stage2_ec
#   train_stage3
#   train_stage3_lgbm
#   train_stage35
# ============================================================

PROJECT_ROOT="${1:-/Users/wyc/SynPred}"
DEVICE="${2:-${DEVICE:-cpu}}"

FORCE="${FORCE:-0}"
START_FROM="${START_FROM:-}"
STOP_AFTER="${STOP_AFTER:-}"
CHECK_OUTPUTS="${CHECK_OUTPUTS:-1}"
SKIP_RETRAIN="${SKIP_RETRAIN:-0}"

RUN_TRAIN_STAGE2="${RUN_TRAIN_STAGE2:-1}"
RUN_TRAIN_STAGE2_EC="${RUN_TRAIN_STAGE2_EC:-1}"
RUN_TRAIN_STAGE3="${RUN_TRAIN_STAGE3:-1}"
RUN_TRAIN_STAGE3_LGBM="${RUN_TRAIN_STAGE3_LGBM:-1}"
RUN_TRAIN_STAGE35="${RUN_TRAIN_STAGE35:-1}"

PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"

SCRIPTS="${PROJECT_ROOT}/scripts"
DATA="${PROJECT_ROOT}/data"
INTERIM="${DATA}/interim"

RESUME_DIR="${PROJECT_ROOT}/outputs/resume/train_models"
LOG_DIR="${PROJECT_ROOT}/outputs/logs/train_models/$(date +%Y%m%d_%H%M%S)"

mkdir -p "${RESUME_DIR}" "${LOG_DIR}"

# ============================================================
# Expected data paths
# ============================================================

STAGE2_DATASET_DIR="${INTERIM}/generative/stage2_gflownet_dataset/hybrid/gold_only"
STAGE3_DATASET_DIR="${INTERIM}/generative/stage3_condition_dataset/hybrid_mixed_v1"
STAGE3_TASK_VIEWS_DIR="${INTERIM}/features/stage3_task_views"

# ============================================================
# Expected production run paths
# ============================================================

STAGE2_BASE_RUN_DIR="${PROJECT_ROOT}/runs/stage2/gflownet_joint_rerank_hybrid_gold_only_v1"
STAGE2_EC_RUN_DIR="${PROJECT_ROOT}/runs/stage2/gflownet_curriculum_ec_v1"

STAGE3_BASELINE_RUN_DIR="${PROJECT_ROOT}/runs/stage3/stage3_baseline_commonized_v1"
STAGE3_BASELINE_CKPT="${STAGE3_BASELINE_RUN_DIR}/best_model.pkl"

STAGE3_MDN_RUN_DIR="${PROJECT_ROOT}/runs/stage3/condition_residual_mdn_hybrid_mixed_v1_yset_conditioned_v2"
STAGE3_MDN_CKPT="${STAGE3_MDN_RUN_DIR}/best_stage3_condition_residual_mdn_mixed.pt"

STAGE3_FLOW_RUN_DIR="${PROJECT_ROOT}/runs/stage3/condition_mixture_flow_hybrid_mixed_v1_yset_conditioned_v2"
STAGE3_FLOW_CKPT="${STAGE3_FLOW_RUN_DIR}/best_stage3_condition_mixture_flow_mixed.pt"

STAGE3_LGBM_RUN_DIR="${PROJECT_ROOT}/runs/stage3/lgbm_quantile_ensemble_v2_fulldata"
STAGE35_RUN_DIR="${PROJECT_ROOT}/runs/stage35/route_ranker_v43_template_aware"

# ============================================================
# Step order
# ============================================================

STEPS=(
  train_stage2
  train_stage2_ec
  train_stage3
  train_stage3_lgbm
  train_stage35
)

# ============================================================
# Utility functions
# ============================================================

print_section() {
  echo
  echo "============================================================"
  echo "  $1"
  echo "============================================================"
}

mark_done() {
  echo "done $(date '+%Y-%m-%d %H:%M:%S')" > "${RESUME_DIR}/${1}.done"
}

is_done() {
  [[ "${FORCE}" != "1" && -f "${RESUME_DIR}/${1}.done" ]]
}

should_run() {
  local step="$1"
  local started="false"

  if [[ -z "${START_FROM}" ]]; then
    started="true"
  fi

  for s in "${STEPS[@]}"; do
    if [[ "${s}" == "${START_FROM}" ]]; then
      started="true"
    fi

    if [[ "${s}" == "${step}" && "${started}" == "true" ]]; then
      return 0
    fi

    if [[ "${s}" == "${STOP_AFTER}" && "${started}" == "true" ]]; then
      if [[ "${s}" == "${step}" ]]; then
        return 0
      fi
      return 1
    fi
  done

  if [[ "${started}" == "true" ]]; then
    return 0
  fi

  return 1
}

past_stop() {
  local step="$1"

  if [[ -z "${STOP_AFTER}" ]]; then
    return 1
  fi

  local past="false"

  for s in "${STEPS[@]}"; do
    if [[ "${past}" == "true" && "${s}" == "${step}" ]]; then
      return 0
    fi

    if [[ "${s}" == "${STOP_AFTER}" ]]; then
      past="true"
    fi
  done

  return 1
}

run_step() {
  local step_name="$1"
  local title="$2"
  shift 2

  if ! should_run "${step_name}"; then
    return 0
  fi

  if past_stop "${step_name}"; then
    return 0
  fi

  if is_done "${step_name}"; then
    echo "[SKIP] ${title} already done. Use FORCE=1 to rerun."
    return 0
  fi

  print_section "${title}"

  local log_file="${LOG_DIR}/${step_name}.log"

  if "$@" 2>&1 | tee "${log_file}"; then
    mark_done "${step_name}"
    echo "[OK] ${title}"
  else
    echo "[FAIL] ${title}"
    echo "Log file:"
    echo "  ${log_file}"
    exit 1
  fi
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

warn_missing_file() {
  local f="$1"
  if [[ ! -f "${f}" ]]; then
    echo "[WARN] Missing optional file: ${f}"
  else
    echo "[OK] ${f}"
  fi
}

warn_missing_dir() {
  local d="$1"
  if [[ ! -d "${d}" ]]; then
    echo "[WARN] Missing optional dir: ${d}"
  else
    echo "[OK] ${d}"
  fi
}

check_after_step() {
  local label="$1"
  shift

  if [[ "${CHECK_OUTPUTS}" != "1" ]]; then
    return 0
  fi

  echo
  echo "[CHECK_OUTPUTS] ${label}"

  for f in "$@"; do
    check_file "${f}"
  done
}

write_config_snapshot() {
  local snapshot="${LOG_DIR}/train_models_config.json"

  cat > "${snapshot}" <<EOF
{
  "project_root": "${PROJECT_ROOT}",
  "device": "${DEVICE}",
  "force": "${FORCE}",
  "start_from": "${START_FROM}",
  "stop_after": "${STOP_AFTER}",
  "check_outputs": "${CHECK_OUTPUTS}",
  "skip_retrain": "${SKIP_RETRAIN}",
  "run_train_stage2": "${RUN_TRAIN_STAGE2}",
  "run_train_stage2_ec": "${RUN_TRAIN_STAGE2_EC}",
  "run_train_stage3": "${RUN_TRAIN_STAGE3}",
  "run_train_stage3_lgbm": "${RUN_TRAIN_STAGE3_LGBM}",
  "run_train_stage35": "${RUN_TRAIN_STAGE35}",
  "stage2_dataset_dir": "${STAGE2_DATASET_DIR}",
  "stage3_dataset_dir": "${STAGE3_DATASET_DIR}",
  "stage2_base_run_dir": "${STAGE2_BASE_RUN_DIR}",
  "stage2_ec_run_dir": "${STAGE2_EC_RUN_DIR}",
  "stage3_baseline_run_dir": "${STAGE3_BASELINE_RUN_DIR}",
  "stage3_mdn_run_dir": "${STAGE3_MDN_RUN_DIR}",
  "stage3_flow_run_dir": "${STAGE3_FLOW_RUN_DIR}",
  "stage3_lgbm_run_dir": "${STAGE3_LGBM_RUN_DIR}",
  "stage35_run_dir": "${STAGE35_RUN_DIR}",
  "log_dir": "${LOG_DIR}",
  "resume_dir": "${RESUME_DIR}"
}
EOF

  echo "[SAVED] ${snapshot}"
}

precheck_training_data() {
  echo
  echo "[Precheck] Stage2 dataset"
  check_dir "${STAGE2_DATASET_DIR}"
  check_file "${STAGE2_DATASET_DIR}/train.npz"
  check_file "${STAGE2_DATASET_DIR}/val.npz"
  check_file "${STAGE2_DATASET_DIR}/test.npz"
  check_file "${STAGE2_DATASET_DIR}/action_vocab.json"
  check_file "${STAGE2_DATASET_DIR}/precursor_names.json"
  check_file "${STAGE2_DATASET_DIR}/summary.json"
  warn_missing_file "${STAGE2_DATASET_DIR}/train_meta.csv"
  warn_missing_file "${STAGE2_DATASET_DIR}/val_meta.csv"
  warn_missing_file "${STAGE2_DATASET_DIR}/test_meta.csv"
  warn_missing_file "${STAGE2_DATASET_DIR}/action_to_id.json"
  warn_missing_file "${STAGE2_DATASET_DIR}/label_cols.json"

  echo
  echo "[Precheck] Stage3 dataset"
  check_dir "${STAGE3_DATASET_DIR}"
  check_file "${STAGE3_DATASET_DIR}/train.npz"
  check_file "${STAGE3_DATASET_DIR}/val.npz"
  check_file "${STAGE3_DATASET_DIR}/test.npz"
  check_file "${STAGE3_DATASET_DIR}/schema.json"

  echo
  echo "[Precheck] Stage3 task views"
  warn_missing_dir "${STAGE3_TASK_VIEWS_DIR}"
  warn_missing_file "${STAGE3_TASK_VIEWS_DIR}/atmosphere_coarse_train.csv"
  warn_missing_file "${STAGE3_TASK_VIEWS_DIR}/time_bucket_train.csv"
}

# ============================================================
# Header
# ============================================================

print_section "SynPred Train Models"

echo "PROJECT_ROOT          = ${PROJECT_ROOT}"
echo "DEVICE                = ${DEVICE}"
echo "FORCE                 = ${FORCE}"
echo "START_FROM            = ${START_FROM:-<beginning>}"
echo "STOP_AFTER            = ${STOP_AFTER:-<end>}"
echo "CHECK_OUTPUTS         = ${CHECK_OUTPUTS}"
echo "SKIP_RETRAIN          = ${SKIP_RETRAIN}"
echo "RUN_TRAIN_STAGE2      = ${RUN_TRAIN_STAGE2}"
echo "RUN_TRAIN_STAGE2_EC   = ${RUN_TRAIN_STAGE2_EC}"
echo "RUN_TRAIN_STAGE3      = ${RUN_TRAIN_STAGE3}"
echo "RUN_TRAIN_STAGE3_LGBM = ${RUN_TRAIN_STAGE3_LGBM}"
echo "RUN_TRAIN_STAGE35     = ${RUN_TRAIN_STAGE35}"
echo "LOG_DIR               = ${LOG_DIR}"
echo "RESUME_DIR            = ${RESUME_DIR}"
echo
echo "Training data:"
echo "  STAGE2_DATASET_DIR  = ${STAGE2_DATASET_DIR}"
echo "  STAGE3_DATASET_DIR  = ${STAGE3_DATASET_DIR}"
echo
echo "Production model outputs:"
echo "  STAGE2_BASE_RUN_DIR = ${STAGE2_BASE_RUN_DIR}"
echo "  STAGE2_EC_RUN_DIR   = ${STAGE2_EC_RUN_DIR}"
echo "  STAGE3_BASELINE     = ${STAGE3_BASELINE_RUN_DIR}"
echo "  STAGE3_MDN_RUN_DIR  = ${STAGE3_MDN_RUN_DIR}"
echo "  STAGE3_FLOW_RUN_DIR = ${STAGE3_FLOW_RUN_DIR}"
echo "  STAGE3_LGBM_RUN_DIR = ${STAGE3_LGBM_RUN_DIR}"
echo "  STAGE35_RUN_DIR     = ${STAGE35_RUN_DIR}"

write_config_snapshot
precheck_training_data

# ============================================================
# STEP 7: Train Stage2 GFlowNet + reranker
# ============================================================

step_train_stage2() {
  if [[ "${RUN_TRAIN_STAGE2}" != "1" ]]; then
    echo "[SKIP] RUN_TRAIN_STAGE2=0"
    return 0
  fi

  if [[ "${SKIP_RETRAIN}" == "1" ]]; then
    echo "[SKIP] SKIP_RETRAIN=1"
    return 0
  fi

  local script="${SCRIPTS}/04_train/stage2/run_retrain_stage2_gflownet_for_pipeline.sh"
  check_file "${script}"

  DEVICE="${DEVICE}" bash "${script}" "${PROJECT_ROOT}"

  if [[ "${CHECK_OUTPUTS}" == "1" ]]; then
    echo
    echo "[CHECK_OUTPUTS] Stage2 GFlowNet outputs"
    check_file "${STAGE2_BASE_RUN_DIR}/best_model.pt"
    check_file "${STAGE2_BASE_RUN_DIR}/best_reranker.pt"
    check_file "${STAGE2_BASE_RUN_DIR}/metrics.json"
  fi
}

run_step train_stage2 "STEP 7: Train Stage2 GFlowNet + Reranker" step_train_stage2

# ============================================================
# STEP 7b: Train Stage2 Curriculum EC
# ============================================================

step_train_stage2_ec() {
  if [[ "${RUN_TRAIN_STAGE2_EC}" != "1" ]]; then
    echo "[SKIP] RUN_TRAIN_STAGE2_EC=0"
    return 0
  fi

  if [[ "${SKIP_RETRAIN}" == "1" ]]; then
    echo "[SKIP] SKIP_RETRAIN=1"
    return 0
  fi

  local script="${SCRIPTS}/04_train/stage2/run_curriculum_ec_training.sh"

  if [[ ! -f "${script}" ]]; then
    echo "[WARN] Missing optional Stage2 EC script:"
    echo "  ${script}"
    echo "[SKIP] Stage2 curriculum EC training"
    return 0
  fi

  bash "${script}"

  if [[ "${CHECK_OUTPUTS}" == "1" ]]; then
    echo
    echo "[CHECK_OUTPUTS] Stage2 EC outputs"
    warn_missing_file "${STAGE2_EC_RUN_DIR}/phase1_relaxed/best_model.pt"
    warn_missing_file "${STAGE2_EC_RUN_DIR}/phase2_gold/best_model.pt"
    warn_missing_file "${STAGE2_EC_RUN_DIR}/inference_ec/test_candidates.csv"
  fi
}

run_step train_stage2_ec "STEP 7b: Train Stage2 Curriculum EC + Reranker" step_train_stage2_ec

# ============================================================
# STEP 8: Train Stage3 baseline / MDN / flow
# ============================================================

step_train_stage3() {
  if [[ "${RUN_TRAIN_STAGE3}" != "1" ]]; then
    echo "[SKIP] RUN_TRAIN_STAGE3=0"
    return 0
  fi

  if [[ "${SKIP_RETRAIN}" == "1" ]]; then
    echo "[SKIP] SKIP_RETRAIN=1"
    return 0
  fi

  local script="${SCRIPTS}/04_train/stage3/run_retrain_stage3_all_for_pipeline.sh"
  check_file "${script}"

  bash "${script}" "${PROJECT_ROOT}" "${DEVICE}"

  if [[ "${CHECK_OUTPUTS}" == "1" ]]; then
    echo
    echo "[CHECK_OUTPUTS] Stage3 neural outputs"
    check_file "${STAGE3_BASELINE_CKPT}"
    warn_missing_file "${STAGE3_MDN_CKPT}"
    check_file "${STAGE3_FLOW_CKPT}"
    check_file "${STAGE3_FLOW_RUN_DIR}/metrics.json"
  fi
}

run_step train_stage3 "STEP 8: Train Stage3 Baseline / MDN / Flow" step_train_stage3

# ============================================================
# STEP 8b: Train Stage3 LightGBM
# ============================================================

step_train_stage3_lgbm() {
  if [[ "${RUN_TRAIN_STAGE3_LGBM}" != "1" ]]; then
    echo "[SKIP] RUN_TRAIN_STAGE3_LGBM=0"
    return 0
  fi

  if [[ "${SKIP_RETRAIN}" == "1" ]]; then
    echo "[SKIP] SKIP_RETRAIN=1"
    return 0
  fi

  local script="${SCRIPTS}/04_train/stage3/run_retrain_stage3_lgbm_for_pipeline.sh"
  check_file "${script}"

  bash "${script}" "${PROJECT_ROOT}"

  check_after_step "Stage3 LightGBM outputs" \
    "${STAGE3_LGBM_RUN_DIR}/metrics.json"
}

run_step train_stage3_lgbm "STEP 8b: Train Stage3 LightGBM Quantile Ensemble" step_train_stage3_lgbm

# ============================================================
# STEP 8c: Train Stage35 route ranker
# ============================================================

step_train_stage35() {
  if [[ "${RUN_TRAIN_STAGE35}" != "1" ]]; then
    echo "[SKIP] RUN_TRAIN_STAGE35=0"
    return 0
  fi

  if [[ "${SKIP_RETRAIN}" == "1" ]]; then
    echo "[SKIP] SKIP_RETRAIN=1"
    return 0
  fi

  local script="${SCRIPTS}/04_train/joint/run_retrain_stage35_ranker_for_pipeline.sh"

  if [[ ! -f "${script}" ]]; then
    echo "[WARN] Missing Stage35 ranker script:"
    echo "  ${script}"
    echo "[SKIP] Stage35 route ranker"
    return 0
  fi

  bash "${script}" "${PROJECT_ROOT}"

  if [[ "${CHECK_OUTPUTS}" == "1" ]]; then
    echo
    echo "[CHECK_OUTPUTS] Stage35 outputs"
    warn_missing_file "${STAGE35_RUN_DIR}/metrics.json"
  fi
}

run_step train_stage35 "STEP 8c: Train Stage35 Route Ranker" step_train_stage35

# ============================================================
# Final summary
# ============================================================

print_section "TRAIN MODELS COMPLETE"

echo "Logs:"
echo "  ${LOG_DIR}"
echo
echo "Resume markers:"
echo "  ${RESUME_DIR}"
echo
echo "Training data:"
echo "  Stage2 dataset:"
echo "    ${STAGE2_DATASET_DIR}/"
echo "  Stage3 dataset:"
echo "    ${STAGE3_DATASET_DIR}/"
echo
echo "Production model outputs:"
echo "  Stage2 base:"
echo "    ${STAGE2_BASE_RUN_DIR}/"
echo "  Stage2 EC:"
echo "    ${STAGE2_EC_RUN_DIR}/"
echo "  Stage3 baseline:"
echo "    ${STAGE3_BASELINE_RUN_DIR}/"
echo "  Stage3 MDN:"
echo "    ${STAGE3_MDN_RUN_DIR}/"
echo "  Stage3 neural flow:"
echo "    ${STAGE3_FLOW_RUN_DIR}/"
echo "  Stage3 LightGBM:"
echo "    ${STAGE3_LGBM_RUN_DIR}/"
echo "  Stage35 ranker:"
echo "    ${STAGE35_RUN_DIR}/"

echo
echo "Finished steps:"
find "${RESUME_DIR}" -type f -name "*.done" | sort || true

echo
echo "Useful commands:"
echo
echo "  Rerun all official model training:"
echo "    FORCE=1 bash scripts/run_train_models.sh ${PROJECT_ROOT} ${DEVICE}"
echo
echo "  Rerun Stage2 only:"
echo "    FORCE=1 START_FROM=train_stage2 STOP_AFTER=train_stage2 bash scripts/run_train_models.sh ${PROJECT_ROOT} ${DEVICE}"
echo
echo "  Rerun Stage3 only:"
echo "    FORCE=1 START_FROM=train_stage3 STOP_AFTER=train_stage3 bash scripts/run_train_models.sh ${PROJECT_ROOT} ${DEVICE}"
echo
echo "  Rerun Stage3 LightGBM only:"
echo "    FORCE=1 START_FROM=train_stage3_lgbm STOP_AFTER=train_stage3_lgbm bash scripts/run_train_models.sh ${PROJECT_ROOT} ${DEVICE}"
echo
echo "  Skip all retraining and only check existing outputs:"
echo "    FORCE=1 SKIP_RETRAIN=1 bash scripts/run_train_models.sh ${PROJECT_ROOT} ${DEVICE}"
echo
echo "  Run benchmark separately:"
echo "    FORCE=1 bash scripts/run_benchmark_models.sh ${PROJECT_ROOT} ${DEVICE}"
echo
echo "  Run inference after training:"
echo "    FORCE=1 bash scripts/run_inference_pipeline.sh ${PROJECT_ROOT} ${DEVICE}"
echo
echo "Available step names:"
printf "  %s\n" "${STEPS[@]}"

echo
echo "============================================================"
echo "[DONE] SynPred train models completed"
echo "============================================================"
