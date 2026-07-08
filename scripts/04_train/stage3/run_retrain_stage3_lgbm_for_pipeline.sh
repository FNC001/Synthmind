#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Retrain Stage3 LightGBM models for pipeline_v3
#
# Trains:
#   1) Quantile ensemble (temperature + time, 9 quantiles each)
#      -> runs/stage3/lgbm_quantile_ensemble_v2_fulldata/
#   2) Atmosphere binary classifier
#      -> runs/stage3/lgbm_atmosphere_classifier_v1/
#   3) Time bucket multiclass classifier
#      -> runs/stage3/lgbm_time_bucket_classifier_v1/
# ============================================================

PROJECT_ROOT="${1:-/Users/wyc/SynPred}"
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"

FORCE="${FORCE:-0}"
SKIP_QUANTILE="${SKIP_QUANTILE:-0}"
SKIP_CLASSIFIERS="${SKIP_CLASSIFIERS:-0}"

SCRIPT_DIR="${PROJECT_ROOT}/scripts/04_train/stage3"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_lgbm_quantile_ensemble.py"

QE_RUN_DIR="${PROJECT_ROOT}/runs/stage3/lgbm_quantile_ensemble_v2_fulldata"
ATM_RUN_DIR="${PROJECT_ROOT}/runs/stage3/lgbm_atmosphere_classifier_v1"
TB_RUN_DIR="${PROJECT_ROOT}/runs/stage3/lgbm_time_bucket_classifier_v1"

BACKUP_ROOT="${PROJECT_ROOT}/runs/stage3/_backup"
LOG_DIR="${PROJECT_ROOT}/outputs/logs/stage3_lgbm_retrain"
LOG_FILE="${LOG_DIR}/retrain_stage3_lgbm_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${BACKUP_ROOT}" "${LOG_DIR}"

echo "============================================================"
echo "Retrain Stage3 LightGBM for pipeline_v3"
echo "PROJECT_ROOT     = ${PROJECT_ROOT}"
echo "QE_RUN_DIR       = ${QE_RUN_DIR}"
echo "ATM_RUN_DIR      = ${ATM_RUN_DIR}"
echo "TB_RUN_DIR       = ${TB_RUN_DIR}"
echo "FORCE            = ${FORCE}"
echo "SKIP_QUANTILE    = ${SKIP_QUANTILE}"
echo "SKIP_CLASSIFIERS = ${SKIP_CLASSIFIERS}"
echo "LOG_FILE         = ${LOG_FILE}"
echo "============================================================"
echo

# Check if already trained (unless FORCE)
QE_EXPECTED="${QE_RUN_DIR}/temp_q0.5.txt"
if [[ "${FORCE}" != "1" && -f "${QE_EXPECTED}" && "${SKIP_QUANTILE}" != "1" ]]; then
  echo "[SKIP] Quantile ensemble already exists: ${QE_EXPECTED}"
  echo "       Set FORCE=1 to retrain."
  SKIP_QUANTILE=1
fi

ATM_EXPECTED="${ATM_RUN_DIR}/model_atmosphere_binary_final.txt"
TB_EXPECTED="${TB_RUN_DIR}/model_time_bucket.txt"
if [[ "${FORCE}" != "1" && -f "${ATM_EXPECTED}" && -f "${TB_EXPECTED}" && "${SKIP_CLASSIFIERS}" != "1" ]]; then
  echo "[SKIP] Classifiers already exist."
  echo "       Set FORCE=1 to retrain."
  SKIP_CLASSIFIERS=1
fi

if [[ "${SKIP_QUANTILE}" == "1" && "${SKIP_CLASSIFIERS}" == "1" ]]; then
  echo "[DONE] All models already trained. Nothing to do."
  exit 0
fi

# Backup existing models if FORCE
if [[ "${FORCE}" == "1" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  if [[ -d "${QE_RUN_DIR}" && "${SKIP_QUANTILE}" != "1" ]]; then
    echo "[BACKUP] ${QE_RUN_DIR} -> ${BACKUP_ROOT}/lgbm_qe_${TS}/"
    cp -r "${QE_RUN_DIR}" "${BACKUP_ROOT}/lgbm_qe_${TS}/"
  fi
fi

# Build args
EXTRA_ARGS=""
if [[ "${SKIP_QUANTILE}" == "1" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --skip_quantile"
fi
if [[ "${SKIP_CLASSIFIERS}" == "1" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --skip_classifiers"
fi

echo "[RUN] python ${TRAIN_SCRIPT} --project_root ${PROJECT_ROOT} ${EXTRA_ARGS}"
python "${TRAIN_SCRIPT}" \
  --project_root "${PROJECT_ROOT}" \
  ${EXTRA_ARGS} \
  2>&1 | tee "${LOG_FILE}"

echo
echo "[OK] Stage3 LightGBM training complete."
echo "     Log: ${LOG_FILE}"
