#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# SynPred Prepare Training Data Pipeline
#
# Purpose:
#   Prepare all training data required by Stage2 / Stage3 models.
#   This script does NOT train models and does NOT run inference.
#
# Usage:
#   bash scripts/run_prepare_training_data.sh [PROJECT_ROOT] [DEVICE]
#
# Examples:
#   bash scripts/run_prepare_training_data.sh /Users/wyc/SynPred cpu
#
#   FORCE=1 bash scripts/run_prepare_training_data.sh /Users/wyc/SynPred cpu
#
#   FORCE=1 START_FROM=stage2_dataset STOP_AFTER=stage2_dataset \
#     bash scripts/run_prepare_training_data.sh /Users/wyc/SynPred cpu
#
# Env options:
#   DEVICE=cpu|cuda|mps
#   FORCE=0|1
#   START_FROM=step_name
#   STOP_AFTER=step_name
#   CHECK_OUTPUTS=0|1
#
# Steps:
#   download
#   refine
#   split
#   features
#   graph_cache
#   stage2_modes
#   stage2_dataset
#   stage3_dataset
# ============================================================

PROJECT_ROOT="${1:-/Users/wyc/SynPred}"
DEVICE="${2:-${DEVICE:-cpu}}"

FORCE="${FORCE:-0}"
START_FROM="${START_FROM:-}"
STOP_AFTER="${STOP_AFTER:-}"
CHECK_OUTPUTS="${CHECK_OUTPUTS:-1}"

PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"

SCRIPTS="${PROJECT_ROOT}/scripts"
DATA="${PROJECT_ROOT}/data"
RAW="${DATA}/raw"
INTERIM="${DATA}/interim"

RESUME_DIR="${PROJECT_ROOT}/outputs/resume/prepare_training_data"
LOG_DIR="${PROJECT_ROOT}/outputs/logs/prepare_training_data/$(date +%Y%m%d_%H%M%S)"

mkdir -p "${RESUME_DIR}" "${LOG_DIR}"
mkdir -p "${RAW}" "${INTERIM}"

# ============================================================
# Expected paths
# ============================================================

MP_ARCHIVE_DIR="${RAW}/mp_full_archive_export"
MP_METADATA_CSV="${MP_ARCHIVE_DIR}/mp_full_archive_metadata.csv"

REFINED_DIR="${INTERIM}/refined/structdesc_refined"
SPLIT_DIR="${INTERIM}/splits/structdesc_splits"

STAGE2_DATASET_DIR="${INTERIM}/generative/stage2_gflownet_dataset/hybrid/gold_only"
STAGE3_DATASET_DIR="${INTERIM}/generative/stage3_condition_dataset/hybrid_mixed_v1"

STRUCTDESC_DIR="${INTERIM}/features/structdesc_features"
CHGNET_STAGE2_CACHE_DIR="${INTERIM}/graph_cache/chgnet_stage2"
CHGNET_STAGE3_CACHE_DIR="${INTERIM}/graph_cache/chgnet_stage3"
CHGNET_STAGE2_EMB_DIR="${INTERIM}/graph_embeddings/chgnet_stage2"
STAGE2_HYBRID_DIR="${INTERIM}/features/stage2_hybrid_features"
STAGE2_TRAINING_MODE_ROOT="${INTERIM}/training_modes/stage2_hybrid"

# ============================================================
# Step order
# ============================================================

STEPS=(
  download
  refine
  split
  features
  graph_cache
  stage2_modes
  stage2_dataset
  stage3_dataset
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

check_after_step_dir() {
  local label="$1"
  shift

  if [[ "${CHECK_OUTPUTS}" != "1" ]]; then
    return 0
  fi

  echo
  echo "[CHECK_OUTPUTS] ${label}"

  for d in "$@"; do
    check_dir "${d}"
  done
}

write_config_snapshot() {
  local snapshot="${LOG_DIR}/prepare_training_data_config.json"

  cat > "${snapshot}" <<EOF
{
  "project_root": "${PROJECT_ROOT}",
  "device": "${DEVICE}",
  "force": "${FORCE}",
  "start_from": "${START_FROM}",
  "stop_after": "${STOP_AFTER}",
  "check_outputs": "${CHECK_OUTPUTS}",
  "scripts": "${SCRIPTS}",
  "data": "${DATA}",
  "raw": "${RAW}",
  "interim": "${INTERIM}",
  "mp_metadata_csv": "${MP_METADATA_CSV}",
  "refined_dir": "${REFINED_DIR}",
  "split_dir": "${SPLIT_DIR}",
  "stage2_dataset_dir": "${STAGE2_DATASET_DIR}",
  "stage3_dataset_dir": "${STAGE3_DATASET_DIR}",
  "log_dir": "${LOG_DIR}",
  "resume_dir": "${RESUME_DIR}"
}
EOF

  echo "[SAVED] ${snapshot}"
}

# ============================================================
# Header
# ============================================================

print_section "SynPred Prepare Training Data"

echo "PROJECT_ROOT   = ${PROJECT_ROOT}"
echo "DEVICE         = ${DEVICE}"
echo "FORCE          = ${FORCE}"
echo "START_FROM     = ${START_FROM:-<beginning>}"
echo "STOP_AFTER     = ${STOP_AFTER:-<end>}"
echo "CHECK_OUTPUTS  = ${CHECK_OUTPUTS}"
echo "LOG_DIR        = ${LOG_DIR}"
echo "RESUME_DIR     = ${RESUME_DIR}"
echo
echo "Main outputs:"
echo "  REFINED_DIR         = ${REFINED_DIR}"
echo "  SPLIT_DIR           = ${SPLIT_DIR}"
echo "  STAGE2_DATASET_DIR  = ${STAGE2_DATASET_DIR}"
echo "  STAGE3_DATASET_DIR  = ${STAGE3_DATASET_DIR}"

write_config_snapshot

# ============================================================
# STEP 0: Download MP archive
# ============================================================

step_download() {
  if [[ -f "${MP_METADATA_CSV}" ]]; then
    local n_entries
    n_entries="$(wc -l < "${MP_METADATA_CSV}" || echo 0)"
    echo "[SKIP] MP archive already exists:"
    echo "  ${MP_METADATA_CSV}"
    echo "  lines=${n_entries}"
    return 0
  fi

  local script="${SCRIPTS}/00_refine/01_capture_experimental_structures.py"
  check_file "${script}"

  echo "[Run] Downloading / capturing MP archive"
  cd "${RAW}"

  python "${script}"

  check_after_step "MP archive" \
    "${MP_METADATA_CSV}"
}

run_step download "STEP 0: Download MP Archive" step_download

# ============================================================
# STEP 1: Refine raw data
# ============================================================

step_refine() {
  local prepare_script="${SCRIPTS}/00_refine/02_prepare_dataset.py"
  local refine_script="${SCRIPTS}/00_refine/04_refine_strict_exact_for_structdesc.py"
  local stat_script="${SCRIPTS}/00_refine/03_statistic.py"

  check_file "${prepare_script}"
  check_file "${refine_script}"

  cd "${RAW}"

  echo "[1a] Prepare dataset: MP matching + synthesis alignment"
  python "${prepare_script}"

  echo
  echo "[1b] Refine strict_exact for structdesc"
  python "${refine_script}" \
    --input_path "${RAW}/strict_exact_only.jsonl" \
    --output_dir "${REFINED_DIR}"

  echo
  echo "[1c] Statistics"
  if [[ -f "${stat_script}" ]]; then
    python "${stat_script}" || true
  else
    echo "[WARN] Optional statistic script not found:"
    echo "  ${stat_script}"
  fi

  check_after_step "refined data" \
    "${REFINED_DIR}/stage2_gold.jsonl" \
    "${REFINED_DIR}/stage2_train_relaxed.jsonl" \
    "${REFINED_DIR}/stage3_gold.jsonl" \
    "${REFINED_DIR}/stage3_train_relaxed.jsonl"
}

run_step refine "STEP 1: Data Refinement" step_refine

# ============================================================
# STEP 2: Train/val/test split
# ============================================================

step_split() {
  local script="${SCRIPTS}/01_split/01_make_group_split.py"
  check_file "${script}"

  cd "${SCRIPTS}/01_split"

  python "${script}" \
    --stage2_gold "${REFINED_DIR}/stage2_gold.jsonl" \
    --stage2_relaxed "${REFINED_DIR}/stage2_train_relaxed.jsonl" \
    --stage3_gold "${REFINED_DIR}/stage3_gold.jsonl" \
    --stage3_relaxed "${REFINED_DIR}/stage3_train_relaxed.jsonl" \
    --output_dir "${SPLIT_DIR}"

  check_after_step_dir "split directory" \
    "${SPLIT_DIR}"
}

run_step split "STEP 2: Train/Val/Test Split" step_split

# ============================================================
# STEP 3: Structural descriptors + hybrid features
# ============================================================

step_features() {
  local script="${SCRIPTS}/02_features/run_build_stage3_features_for_pipeline.sh"
  check_file "${script}"

  bash "${script}" "${PROJECT_ROOT}" "${DEVICE}"

  if [[ "${CHECK_OUTPUTS}" == "1" ]]; then
    echo
    echo "[CHECK_OUTPUTS] feature directories"
    check_dir "${INTERIM}/features"
    warn_missing_dir "${STRUCTDESC_DIR}"
  fi
}

run_step features "STEP 3: Structural Descriptors + Hybrid Features" step_features

# ============================================================
# STEP 4: Graph cache
# ============================================================

step_graph_cache() {
  local script_stage2="${SCRIPTS}/03_graph/03_build_chgnet_cache_stage2.py"
  local script_stage3="${SCRIPTS}/03_graph/03_build_chgnet_cache_stage3.py"

  cd "${SCRIPTS}/03_graph"

  echo "[4a] Build CHGNet cache for Stage2"
  if [[ -f "${script_stage2}" ]]; then
    python "${script_stage2}" \
      --base_dir "${DATA}" \
      --input_dir "${SPLIT_DIR}" || true
  else
    echo "[WARN] Missing optional Stage2 CHGNet cache script:"
    echo "  ${script_stage2}"
  fi

  echo
  echo "[4b] Build CHGNet cache for Stage3"
  if [[ -f "${script_stage3}" ]]; then
    python "${script_stage3}" \
      --base_dir "${DATA}" \
      --input_dir "${SPLIT_DIR}" || true
  else
    echo "[WARN] Missing optional Stage3 CHGNet cache script:"
    echo "  ${script_stage3}"
  fi

  if [[ "${CHECK_OUTPUTS}" == "1" ]]; then
    echo
    echo "[CHECK_OUTPUTS] graph cache outputs"
    warn_missing_file "${CHGNET_STAGE2_CACHE_DIR}/summary.json"
    warn_missing_file "${CHGNET_STAGE3_CACHE_DIR}/summary.json"
  fi
}

run_step graph_cache "STEP 4: Graph Embedding Caches" step_graph_cache

# ============================================================
# STEP 4b: Stage2 hybrid modes
# ============================================================

step_stage2_modes() {
  local export_script="${SCRIPTS}/03_graph/export_chgnet_stage2_embeddings.py"
  local build_hybrid_script="${SCRIPTS}/02_features/05_build_hybrid_features.py"
  local prepare_modes_script="${SCRIPTS}/02_features/06_prepare_training_modes.py"

  echo "[4b-1] Export CHGNet Stage2 embeddings"

  if [[ -f "${export_script}" ]]; then
    cd "${SCRIPTS}/03_graph"
    python "${export_script}" \
      --cache_dir "${CHGNET_STAGE2_CACHE_DIR}" \
      --output_dir "${CHGNET_STAGE2_EMB_DIR}"
  else
    echo "[WARN] Missing optional CHGNet embedding export script:"
    echo "  ${export_script}"
    echo "[WARN] Continue without exporting CHGNet Stage2 embeddings."
  fi

  echo
  echo "[4b-2] Build Stage2 hybrid features"

  check_file "${build_hybrid_script}"
  cd "${SCRIPTS}/02_features"

  if [[ -d "${CHGNET_STAGE2_EMB_DIR}" ]]; then
    python "${build_hybrid_script}" \
      --task stage2 \
      --descriptor_dir "${STRUCTDESC_DIR}" \
      --embedding_dirs "${CHGNET_STAGE2_EMB_DIR}" \
      --embedding_prefixes chgnet \
      --output_dir "${STAGE2_HYBRID_DIR}" \
      --descriptor_kind ml
  else
    echo "[WARN] CHGNet Stage2 embedding dir not found:"
    echo "  ${CHGNET_STAGE2_EMB_DIR}"
    echo "[WARN] Trying descriptor-only Stage2 hybrid feature build."

    python "${build_hybrid_script}" \
      --task stage2 \
      --descriptor_dir "${STRUCTDESC_DIR}" \
      --output_dir "${STAGE2_HYBRID_DIR}" \
      --descriptor_kind ml
  fi

  echo
  echo "[4b-3] Prepare Stage2 training modes"

  check_file "${prepare_modes_script}"

  python "${prepare_modes_script}" \
    --source_dir "${STAGE2_HYBRID_DIR}" \
    --output_root "${STAGE2_TRAINING_MODE_ROOT}" \
    --train_file stage2_train_hybrid.csv \
    --val_file stage2_val_hybrid.csv \
    --test_file stage2_test_hybrid.csv \
    --gold_train_holdout_file stage2_gold_train_holdout_hybrid.csv \
    --dataset_name stage2_hybrid

  if [[ "${CHECK_OUTPUTS}" == "1" ]]; then
    echo
    echo "[CHECK_OUTPUTS] Stage2 training modes"
    check_dir "${STAGE2_TRAINING_MODE_ROOT}"
  fi
}

run_step stage2_modes "STEP 4b: Stage2 Hybrid Features + Training Modes" step_stage2_modes

# ============================================================
# STEP 5: Stage2 GFlowNet dataset
# ============================================================

step_stage2_dataset() {
  local script="${SCRIPTS}/03_data/run_build_stage2_gflownet_dataset_for_pipeline.sh"
  check_file "${script}"

  bash "${script}" "${PROJECT_ROOT}"

  check_after_step "Stage2 GFlowNet dataset" \
    "${STAGE2_DATASET_DIR}/train.npz" \
    "${STAGE2_DATASET_DIR}/val.npz" \
    "${STAGE2_DATASET_DIR}/test.npz" \
    "${STAGE2_DATASET_DIR}/action_vocab.json" \
    "${STAGE2_DATASET_DIR}/precursor_names.json" \
    "${STAGE2_DATASET_DIR}/summary.json"
}

run_step stage2_dataset "STEP 5: Stage2 GFlowNet Dataset" step_stage2_dataset

# ============================================================
# STEP 6: Stage3 condition dataset
# ============================================================

step_stage3_dataset() {
  local script="${SCRIPTS}/03_data/27_build_stage3_condition_dataset_v5_mixed.py"
  check_file "${script}"

  cd "${SCRIPTS}/03_data"

  python "${script}" \
    --output_dir "${STAGE3_DATASET_DIR}"

  check_after_step "Stage3 condition dataset" \
    "${STAGE3_DATASET_DIR}/train.npz" \
    "${STAGE3_DATASET_DIR}/val.npz" \
    "${STAGE3_DATASET_DIR}/test.npz" \
    "${STAGE3_DATASET_DIR}/schema.json"
}

run_step stage3_dataset "STEP 6: Stage3 Condition Dataset" step_stage3_dataset

# ============================================================
# Final summary
# ============================================================

print_section "PREPARE TRAINING DATA COMPLETE"

echo "Logs:"
echo "  ${LOG_DIR}"
echo
echo "Resume markers:"
echo "  ${RESUME_DIR}"
echo
echo "Key outputs:"
echo "  MP archive:"
echo "    ${MP_ARCHIVE_DIR}/"
echo
echo "  Refined data:"
echo "    ${REFINED_DIR}/"
echo
echo "  Splits:"
echo "    ${SPLIT_DIR}/"
echo
echo "  Features:"
echo "    ${INTERIM}/features/"
echo
echo "  Graph cache:"
echo "    ${INTERIM}/graph_cache/"
echo
echo "  Stage2 training modes:"
echo "    ${STAGE2_TRAINING_MODE_ROOT}/"
echo
echo "  Stage2 dataset:"
echo "    ${STAGE2_DATASET_DIR}/"
echo
echo "  Stage3 dataset:"
echo "    ${STAGE3_DATASET_DIR}/"
echo
echo "Finished steps:"
find "${RESUME_DIR}" -type f -name "*.done" | sort || true

echo
echo "Useful commands:"
echo
echo "  Rerun all data preparation:"
echo "    FORCE=1 bash scripts/run_prepare_training_data.sh ${PROJECT_ROOT} ${DEVICE}"
echo
echo "  Rerun one step:"
echo "    FORCE=1 START_FROM=<step> STOP_AFTER=<step> bash scripts/run_prepare_training_data.sh ${PROJECT_ROOT} ${DEVICE}"
echo
echo "  Rerun Stage2 dataset only:"
echo "    FORCE=1 START_FROM=stage2_dataset STOP_AFTER=stage2_dataset bash scripts/run_prepare_training_data.sh ${PROJECT_ROOT} ${DEVICE}"
echo
echo "  Rerun Stage3 dataset only:"
echo "    FORCE=1 START_FROM=stage3_dataset STOP_AFTER=stage3_dataset bash scripts/run_prepare_training_data.sh ${PROJECT_ROOT} ${DEVICE}"
echo
echo "Available step names:"
printf "  %s\n" "${STEPS[@]}"

echo
echo "============================================================"
echo "[DONE] SynPred prepare training data completed"
echo "============================================================"
