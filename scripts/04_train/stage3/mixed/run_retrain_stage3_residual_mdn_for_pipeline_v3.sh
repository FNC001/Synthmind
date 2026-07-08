#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/Users/wyc/SynPred}"
DEVICE="${2:-cpu}"

SCRIPT_DIR="${PROJECT_ROOT}/scripts/04_train/stage3/mixed"
DATASET_DIR="${PROJECT_ROOT}/data/interim/generative/stage3_condition_dataset/hybrid_mixed_v1"
RUN_DIR="${PROJECT_ROOT}/runs/stage3/condition_residual_mdn_hybrid_mixed_v1_yset_conditioned_v2"
EXPECTED="${RUN_DIR}/best_stage3_condition_residual_mdn_mixed.pt"
LOG_DIR="${PROJECT_ROOT}/outputs/logs/retrain_stage3_residual_mdn/$(date +%Y%m%d_%H%M%S)"

EPOCHS="${EPOCHS:-120}"
PATIENCE="${PATIENCE:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
METRIC_NAME="${METRIC_NAME:-val_loss}"

mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "Retrain Stage3 residual-MDN mixed baseline for pipeline_v3"
echo "PROJECT_ROOT = ${PROJECT_ROOT}"
echo "DEVICE       = ${DEVICE}"
echo "DATASET_DIR  = ${DATASET_DIR}"
echo "RUN_DIR      = ${RUN_DIR}"
echo "EXPECTED     = ${EXPECTED}"
echo "LOG_DIR      = ${LOG_DIR}"
echo "============================================================"

check_file() {
  local p="$1"
  if [[ ! -f "$p" ]]; then
    echo "[ERROR] Missing file: $p"
    exit 1
  fi
}

echo
echo "[STEP 1] Check required files"
check_file "${SCRIPT_DIR}/train_condition_residual_mdn_mixed.py"
check_file "${DATASET_DIR}/train.npz"
check_file "${DATASET_DIR}/val.npz"
check_file "${DATASET_DIR}/test.npz"
check_file "${DATASET_DIR}/schema.json"

echo
echo "[STEP 2] Backup old run dir if exists"
if [[ -d "${RUN_DIR}" ]]; then
  BACKUP_DIR="${PROJECT_ROOT}/runs/stage3/_backup/condition_residual_mdn_hybrid_mixed_v1_yset_conditioned_v2_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$(dirname "${BACKUP_DIR}")"
  mv "${RUN_DIR}" "${BACKUP_DIR}"
  echo "[BACKUP] ${RUN_DIR}"
  echo "      -> ${BACKUP_DIR}"
fi

echo
echo "[STEP 3] Start Stage3 residual-MDN training"
echo "------------------------------------------------------------"

python "${SCRIPT_DIR}/train_condition_residual_mdn_mixed.py" \
  --input_dir "${DATASET_DIR}" \
  --run_dir "${RUN_DIR}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --metric_name "${METRIC_NAME}" \
  --clip_to_train_range \
  --targets_are_standardized auto \
  2>&1 | tee "${LOG_DIR}/train.log"

echo
echo "[STEP 4] Check generated checkpoint"
find "${RUN_DIR}" -type f \( -name "*.pt" -o -name "*.pth" \) -print | sort || true

if [[ ! -f "${EXPECTED}" ]]; then
  echo
  echo "[WARNING] Expected checkpoint not found:"
  echo "  ${EXPECTED}"
  echo
  echo "Available checkpoint files:"
  find "${RUN_DIR}" -type f \( -name "*.pt" -o -name "*.pth" \) -print | sort || true
  echo
  echo "If the checkpoint name is different, use that file as BASELINE_CKPT for the flow model."
  exit 1
fi

echo
echo "============================================================"
echo "[DONE] Stage3 residual-MDN baseline training finished"
echo "BASELINE_CKPT = ${EXPECTED}"
echo "LOG           = ${LOG_DIR}/train.log"
echo "============================================================"
