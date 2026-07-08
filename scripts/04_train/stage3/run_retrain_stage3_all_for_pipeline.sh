#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Retrain full Stage3 chain for pipeline_v3
#
# Stage3 chain:
#   1) baseline linear/logistic model
#      -> runs/stage3/stage3_baseline_commonized_v1/best_model.pkl
#
#   2) residual-MDN mixed model
#      -> runs/stage3/condition_residual_mdn_hybrid_mixed_v1_yset_conditioned_v2/
#         best_stage3_condition_residual_mdn_mixed.pt
#
#   3) mixture-flow mixed model
#      -> runs/stage3/condition_mixture_flow_hybrid_mixed_v1_yset_conditioned_v2/
#         best_stage3_condition_mixture_flow_mixed.pt
#
# This final checkpoint is required by:
#   scripts/07_infer/structure_to_synthesis_route/pipeline_v3/configs/full_route_stage3.yaml
# ============================================================

PROJECT_ROOT="${1:-/Users/wyc/SynPred}"
DEVICE="${2:-cpu}"

FORCE="${FORCE:-0}"
SKIP_BASELINE="${SKIP_BASELINE:-0}"
SKIP_MDN="${SKIP_MDN:-0}"
SKIP_FLOW="${SKIP_FLOW:-0}"

EPOCHS_MDN="${EPOCHS_MDN:-80}"
EPOCHS_FLOW="${EPOCHS_FLOW:-80}"
PATIENCE="${PATIENCE:-12}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-42}"

PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"

STAGE3_SCRIPT_DIR="${PROJECT_ROOT}/scripts/04_train/stage3"
MIXED_SCRIPT_DIR="${PROJECT_ROOT}/scripts/04_train/stage3/mixed"

DATASET_DIR="${PROJECT_ROOT}/data/interim/generative/stage3_condition_dataset/hybrid_mixed_v1"

BASELINE_RUN_DIR="${PROJECT_ROOT}/runs/stage3/stage3_baseline_commonized_v1"
BASELINE_CKPT="${BASELINE_RUN_DIR}/best_model.pkl"

MDN_RUN_DIR="${PROJECT_ROOT}/runs/stage3/condition_residual_mdn_hybrid_mixed_v1_yset_conditioned_v2"
MDN_CKPT="${MDN_RUN_DIR}/best_stage3_condition_residual_mdn_mixed.pt"

FLOW_RUN_DIR="${PROJECT_ROOT}/runs/stage3/condition_mixture_flow_hybrid_mixed_v1_yset_conditioned_v2"
FLOW_CKPT="${FLOW_RUN_DIR}/best_stage3_condition_mixture_flow_mixed.pt"

BACKUP_ROOT="${PROJECT_ROOT}/runs/stage3/_backup"
LOG_DIR="${PROJECT_ROOT}/outputs/logs/retrain_stage3_all_for_pipeline_v3/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${BACKUP_ROOT}" "${LOG_DIR}"

print_section() {
  echo
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

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

backup_dir_if_needed() {
  local d="$1"
  local name="$2"

  if [[ "${FORCE}" == "1" && -d "${d}" ]]; then
    local backup="${BACKUP_ROOT}/${name}_$(date +%Y%m%d_%H%M%S)"
    mv "${d}" "${backup}"
    echo "[BACKUP] ${d}"
    echo "      -> ${backup}"
  fi
}

run_and_log() {
  local name="$1"
  shift
  local log_file="${LOG_DIR}/${name}.log"

  echo "[RUN] $*"
  echo "[LOG] ${log_file}"

  set +e
  "$@" 2>&1 | tee "${log_file}"
  local status=${PIPESTATUS[0]}
  set -e

  if [[ "${status}" -ne 0 ]]; then
    echo
    echo "[ERROR] Command failed: ${name}"
    echo "[ERROR] Log file: ${log_file}"
    exit "${status}"
  fi
}

find_checkpoint_or_link() {
  local run_dir="$1"
  local expected="$2"
  local pattern="$3"

  if [[ -f "${expected}" ]]; then
    echo "[OK] Expected checkpoint exists:"
    echo "  ${expected}"
    return 0
  fi

  echo "[WARN] Expected checkpoint not found:"
  echo "  ${expected}"
  echo "[INFO] Searching possible checkpoints in:"
  echo "  ${run_dir}"

  local found
  found="$(find "${run_dir}" -maxdepth 3 -type f \( -name "${pattern}" -o -name "*.pt" -o -name "*.pkl" -o -name "*.joblib" \) | sort | head -n 1 || true)"

  if [[ -n "${found}" && -f "${found}" ]]; then
    echo "[LINK] Found possible checkpoint:"
    echo "  ${found}"
    echo "[LINK] Creating symlink/copy target:"
    echo "  ${expected}"
    mkdir -p "$(dirname "${expected}")"
    ln -sf "${found}" "${expected}"
    return 0
  fi

  echo "[ERROR] No usable checkpoint found in ${run_dir}"
  find "${run_dir}" -maxdepth 3 -type f | sort || true
  exit 1
}

print_section "CONFIG"

echo "PROJECT_ROOT     = ${PROJECT_ROOT}"
echo "DEVICE           = ${DEVICE}"
echo "DATASET_DIR      = ${DATASET_DIR}"
echo "BASELINE_RUN_DIR = ${BASELINE_RUN_DIR}"
echo "BASELINE_CKPT    = ${BASELINE_CKPT}"
echo "MDN_RUN_DIR      = ${MDN_RUN_DIR}"
echo "MDN_CKPT         = ${MDN_CKPT}"
echo "FLOW_RUN_DIR     = ${FLOW_RUN_DIR}"
echo "FLOW_CKPT        = ${FLOW_CKPT}"
echo "FORCE            = ${FORCE}"
echo "SKIP_BASELINE    = ${SKIP_BASELINE}"
echo "SKIP_MDN         = ${SKIP_MDN}"
echo "SKIP_FLOW        = ${SKIP_FLOW}"
echo "EPOCHS_MDN       = ${EPOCHS_MDN}"
echo "EPOCHS_FLOW      = ${EPOCHS_FLOW}"
echo "PATIENCE         = ${PATIENCE}"
echo "BATCH_SIZE       = ${BATCH_SIZE}"
echo "NUM_WORKERS      = ${NUM_WORKERS}"
echo "SEED             = ${SEED}"
echo "LOG_DIR          = ${LOG_DIR}"

print_section "STEP 0: Check required inputs"

check_dir "${PROJECT_ROOT}"
check_dir "${DATASET_DIR}"

check_file "${DATASET_DIR}/train.npz"
check_file "${DATASET_DIR}/val.npz"
check_file "${DATASET_DIR}/test.npz"
check_file "${DATASET_DIR}/schema.json"

check_file "${STAGE3_SCRIPT_DIR}/train_baseline_linear.py"
check_file "${MIXED_SCRIPT_DIR}/train_condition_residual_mdn_mixed.py"
check_file "${MIXED_SCRIPT_DIR}/train_condition_mixture_flow_mixed.py"

echo "[OK] Required data and scripts exist."

# ============================================================
# STEP 1: baseline
# ============================================================

print_section "STEP 1: Train Stage3 baseline"

if [[ "${SKIP_BASELINE}" == "1" && -f "${BASELINE_CKPT}" ]]; then
  echo "[SKIP] baseline already exists:"
  echo "  ${BASELINE_CKPT}"
elif [[ "${FORCE}" != "1" && -f "${BASELINE_CKPT}" ]]; then
  echo "[SKIP] baseline checkpoint already exists:"
  echo "  ${BASELINE_CKPT}"
else
  backup_dir_if_needed "${BASELINE_RUN_DIR}" "stage3_baseline_commonized_v1"
  mkdir -p "${BASELINE_RUN_DIR}"

  run_and_log "01_train_stage3_baseline" \
    python "${STAGE3_SCRIPT_DIR}/train_baseline_linear.py" \
      --project_root "${PROJECT_ROOT}" \
      --input_dir "${DATASET_DIR}" \
      --run_dir "${BASELINE_RUN_DIR}" \
      --use_y_set \
      --standardize \
      --ridge_alpha 1.0 \
      --logreg_C 1.0 \
      --max_iter 2000 \
      --seed "${SEED}"

  find_checkpoint_or_link "${BASELINE_RUN_DIR}" "${BASELINE_CKPT}" "best_model.pkl"
fi

check_file "${BASELINE_CKPT}"

# ============================================================
# STEP 2: residual-MDN mixed
# ============================================================

print_section "STEP 2: Train Stage3 residual-MDN mixed"

if [[ "${SKIP_MDN}" == "1" && -f "${MDN_CKPT}" ]]; then
  echo "[SKIP] residual-MDN already exists:"
  echo "  ${MDN_CKPT}"
elif [[ "${FORCE}" != "1" && -f "${MDN_CKPT}" ]]; then
  echo "[SKIP] residual-MDN checkpoint already exists:"
  echo "  ${MDN_CKPT}"
else
  backup_dir_if_needed "${MDN_RUN_DIR}" "condition_residual_mdn_hybrid_mixed_v1_yset_conditioned_v2"
  mkdir -p "${MDN_RUN_DIR}"

  run_and_log "02_train_stage3_residual_mdn_mixed" \
    python "${MIXED_SCRIPT_DIR}/train_condition_residual_mdn_mixed.py" \
      --project_root "${PROJECT_ROOT}" \
      --input_dir "${DATASET_DIR}" \
      --run_dir "${MDN_RUN_DIR}" \
      --baseline_ckpt "${BASELINE_CKPT}" \
      --use_layernorm \
      --clip_to_train_range \
      --targets_are_standardized false \
      --batch_size "${BATCH_SIZE}" \
      --epochs "${EPOCHS_MDN}" \
      --patience "${PATIENCE}" \
      --lr 1e-3 \
      --weight_decay 1e-5 \
      --n_gen_samples 32 \
      --grad_clip 5.0 \
      --num_workers "${NUM_WORKERS}" \
      --device "${DEVICE}" \
      --seed "${SEED}"

  find_checkpoint_or_link "${MDN_RUN_DIR}" "${MDN_CKPT}" "best_stage3_condition_residual_mdn_mixed.pt"
fi

check_file "${MDN_CKPT}"

# ============================================================
# STEP 3: mixture-flow mixed
# ============================================================

print_section "STEP 3: Train Stage3 mixture-flow mixed"

if [[ "${SKIP_FLOW}" == "1" && -f "${FLOW_CKPT}" ]]; then
  echo "[SKIP] mixture-flow already exists:"
  echo "  ${FLOW_CKPT}"
elif [[ "${FORCE}" != "1" && -f "${FLOW_CKPT}" ]]; then
  echo "[SKIP] mixture-flow checkpoint already exists:"
  echo "  ${FLOW_CKPT}"
else
  backup_dir_if_needed "${FLOW_RUN_DIR}" "condition_mixture_flow_hybrid_mixed_v1_yset_conditioned_v2"
  mkdir -p "${FLOW_RUN_DIR}"

  run_and_log "03_train_stage3_mixture_flow_mixed" \
    python "${MIXED_SCRIPT_DIR}/train_condition_mixture_flow_mixed.py" \
      --input_dir "${DATASET_DIR}" \
      --baseline_ckpt "${MDN_CKPT}" \
      --run_dir "${FLOW_RUN_DIR}" \
      --use_layernorm \
      --clip_to_train_range \
      --batch_size "${BATCH_SIZE}" \
      --epochs "${EPOCHS_FLOW}" \
      --patience "${PATIENCE}" \
      --lr 1e-3 \
      --weight_decay 1e-5 \
      --n_gen_samples 32 \
      --grad_clip 5.0 \
      --num_workers "${NUM_WORKERS}" \
      --device "${DEVICE}" \
      --seed "${SEED}"

  find_checkpoint_or_link "${FLOW_RUN_DIR}" "${FLOW_CKPT}" "best_stage3_condition_mixture_flow_mixed.pt"
fi

check_file "${FLOW_CKPT}"

# ============================================================
# STEP 4: pipeline config consistency check
# ============================================================

print_section "STEP 4: Check pipeline_v3 expected files"

PIPELINE_CONFIG="${PROJECT_ROOT}/scripts/07_infer/structure_to_synthesis_route/pipeline/configs/full_route_stage3.yaml"

check_file "${PIPELINE_CONFIG}"

echo "[OK] pipeline config:"
echo "  ${PIPELINE_CONFIG}"

echo
echo "[CHECK] Stage3 lines in config:"
grep -n "schema_json\|flow_ckpt\|flow_script" "${PIPELINE_CONFIG}" || true

echo
echo "[OK] Final Stage3 checkpoint for pipeline:"
echo "  ${FLOW_CKPT}"

# ============================================================
# DONE
# ============================================================

print_section "DONE"

echo "[OK] Stage3 full retraining chain finished."
echo
echo "Baseline checkpoint:"
echo "  ${BASELINE_CKPT}"
echo
echo "Residual-MDN checkpoint:"
echo "  ${MDN_CKPT}"
echo
echo "Mixture-flow checkpoint required by pipeline_v3:"
echo "  ${FLOW_CKPT}"
echo
echo "Logs:"
echo "  ${LOG_DIR}"
echo
echo "Next command:"
echo "  cd ${PROJECT_ROOT}/scripts/07_infer/structure_to_synthesis_route/pipeline"
echo "  python run_pipeline.py --config configs/full_route_stage3.yaml --infer_name benchmark_001 --start_from preflight"
