#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/Users/wyc/SynPred}"
DEVICE="${2:-cpu}"

SCRIPT_DIR="${PROJECT_ROOT}/scripts/04_train/stage3/mixed"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_condition_mixture_flow_mixed.py"

DATASET_DIR="${PROJECT_ROOT}/data/interim/generative/stage3_condition_dataset/hybrid_mixed_v1"
RUN_DIR="${PROJECT_ROOT}/runs/stage3/condition_mixture_flow_hybrid_mixed_v1_yset_conditioned_v2"
EXPECTED="${RUN_DIR}/best_stage3_condition_mixture_flow_mixed.pt"

BACKUP_ROOT="${PROJECT_ROOT}/runs/stage3/_backup"
LOG_DIR="${PROJECT_ROOT}/outputs/logs/retrain_stage3_flow/$(date +%Y%m%d_%H%M%S)"

mkdir -p "${BACKUP_ROOT}" "${LOG_DIR}"

echo "============================================================"
echo "Retrain Stage3 condition mixture-flow for pipeline_v3"
echo "PROJECT_ROOT = ${PROJECT_ROOT}"
echo "DEVICE       = ${DEVICE}"
echo "SCRIPT_DIR   = ${SCRIPT_DIR}"
echo "DATASET_DIR  = ${DATASET_DIR}"
echo "RUN_DIR      = ${RUN_DIR}"
echo "EXPECTED     = ${EXPECTED}"
echo "LOG_DIR      = ${LOG_DIR}"
echo "============================================================"
echo

echo "[STEP 1] Check required files"

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "[ERROR] Missing train script: ${TRAIN_SCRIPT}"
  exit 1
fi

for f in train.npz val.npz test.npz schema.json; do
  if [[ ! -f "${DATASET_DIR}/${f}" ]]; then
    echo "[ERROR] Missing Stage3 dataset file: ${DATASET_DIR}/${f}"
    exit 1
  fi
  echo "[OK] ${DATASET_DIR}/${f}"
done

echo
echo "[STEP 2] Show training script help"
python "${TRAIN_SCRIPT}" --help || true

echo
echo "[STEP 3] Backup old run dir if exists"

if [[ -d "${RUN_DIR}" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  BACKUP_DIR="${BACKUP_ROOT}/condition_mixture_flow_hybrid_mixed_v1_yset_conditioned_v2_${TS}"
  echo "[BACKUP] ${RUN_DIR}"
  echo "      -> ${BACKUP_DIR}"
  mv "${RUN_DIR}" "${BACKUP_DIR}"
fi

mkdir -p "${RUN_DIR}"

echo
echo "[STEP 4] Start Stage3 Flow training"
echo "------------------------------------------------------------"

python "${TRAIN_SCRIPT}" \
  --project_root "${PROJECT_ROOT}" \
  --input_dir "${DATASET_DIR}" \
  --run_dir "${RUN_DIR}" \
  --device "${DEVICE}" \
  --epochs 200 \
  --batch_size 256 \
  --lr 1e-3 \
  --weight_decay 1e-5 \
  2>&1 | tee "${LOG_DIR}/train_stage3_flow.log"

echo
echo "[STEP 5] Check expected checkpoint"

if [[ ! -f "${EXPECTED}" ]]; then
  echo "[ERROR] Training finished but expected checkpoint was not found:"
  echo "  ${EXPECTED}"
  echo
  echo "Available files in RUN_DIR:"
  find "${RUN_DIR}" -maxdepth 2 -type f | sort || true
  exit 1
fi

echo "[OK] Stage3 Flow checkpoint generated:"
ls -lh "${EXPECTED}"

echo
echo "============================================================"
echo "[DONE] Stage3 Flow retraining completed"
echo "============================================================"
echo
echo "Next command:"
echo "cd ${PROJECT_ROOT}/scripts/07_infer/structure_to_synthesis_route/pipeline_v3"
echo "python run_pipeline.py --config configs/full_route_stage3.yaml --infer_name benchmark_001 --start_from preflight"
