#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Retrain Stage35 Route Ranker (v43 template-aware) for pipeline_v3
#
# Steps:
#   1) Build pairwise dataset from benchmark inference results
#      (skipped if dataset already exists)
#   2) Train ExtraTrees pairwise ranker (chem-only features)
#      -> runs/stage35/route_ranker_v43_template_aware/
#
# Prerequisites:
#   - Benchmark inference results with v43 template features
#     (data/interim/generative/stage35_route_ranker_dataset/v43_template_aware/)
# ============================================================

PROJECT_ROOT="${1:-/Users/wyc/SynPred}"
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"

FORCE="${FORCE:-0}"

RANKER_SCRIPT_DIR="${PROJECT_ROOT}/scripts/07_infer/structure_to_synthesis_route/route_ranker/v43_template_aware"
DATASET_DIR="${PROJECT_ROOT}/data/interim/generative/stage35_route_ranker_dataset/v43_template_aware"
RUN_DIR="${PROJECT_ROOT}/runs/stage35/route_ranker_v43_template_aware"

BACKUP_ROOT="${PROJECT_ROOT}/runs/stage35/_backup"
LOG_DIR="${PROJECT_ROOT}/outputs/logs/stage35_ranker_retrain"
LOG_FILE="${LOG_DIR}/retrain_stage35_ranker_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${BACKUP_ROOT}" "${LOG_DIR}" "${DATASET_DIR}" "${RUN_DIR}"

echo "============================================================"
echo "Retrain Stage35 Route Ranker (v43 template-aware)"
echo "PROJECT_ROOT = ${PROJECT_ROOT}"
echo "DATASET_DIR  = ${DATASET_DIR}"
echo "RUN_DIR      = ${RUN_DIR}"
echo "FORCE        = ${FORCE}"
echo "LOG_FILE     = ${LOG_FILE}"
echo "============================================================"
echo

EXPECTED="${RUN_DIR}/stage35_v43_template_pairwise_chemonly_extratrees.joblib"
if [[ "${FORCE}" != "1" && -f "${EXPECTED}" ]]; then
  echo "[SKIP] Ranker already exists: ${EXPECTED}"
  echo "       Set FORCE=1 to retrain."
  exit 0
fi

# --- Step 1: Build pairwise dataset (if not exists) ---
PAIRWISE_CSV="${DATASET_DIR}/stage35_v43_template_pairwise_all506_weakprefs.csv"
if [[ ! -f "${PAIRWISE_CSV}" ]]; then
  echo "[STEP 1] Building pairwise dataset..."
  python "${RANKER_SCRIPT_DIR}/02_build_v43_template_pairwise_dataset.py" \
    --project_root "${PROJECT_ROOT}" \
    --output_csv "${PAIRWISE_CSV}" \
    2>&1 | tee -a "${LOG_FILE}"
else
  echo "[SKIP] Pairwise dataset exists: ${PAIRWISE_CSV}"
fi

# --- Step 2: Backup existing model ---
if [[ "${FORCE}" == "1" && -f "${EXPECTED}" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  echo "[BACKUP] ${RUN_DIR} -> ${BACKUP_ROOT}/route_ranker_v43_${TS}/"
  cp -r "${RUN_DIR}" "${BACKUP_ROOT}/route_ranker_v43_${TS}/"
fi

# --- Step 3: Train ranker ---
echo "[STEP 2] Training ExtraTrees pairwise ranker..."
python "${RANKER_SCRIPT_DIR}/03_train_v43_template_pairwise_ranker_chemonly.py" \
  --input_csv "${PAIRWISE_CSV}" \
  --output_dir "${RUN_DIR}" \
  --n_estimators 600 \
  --max_depth 12 \
  --min_samples_leaf 2 \
  2>&1 | tee -a "${LOG_FILE}"

echo
echo "[OK] Stage35 ranker training complete."
echo "     Model: ${EXPECTED}"
echo "     Log:   ${LOG_FILE}"
