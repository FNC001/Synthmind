#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/Users/wyc/SynPred}"
DEVICE="${2:-cpu}"

SCRIPT_DIR="${PROJECT_ROOT}/scripts/02_features"
LOG_DIR="${PROJECT_ROOT}/outputs/logs/stage3_feature_build"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG_DIR="${LOG_DIR}/${TS}"

mkdir -p "${RUN_LOG_DIR}"

echo "============================================================"
echo "Build Stage3 features/views for pipeline_v3"
echo "PROJECT_ROOT = ${PROJECT_ROOT}"
echo "SCRIPT_DIR   = ${SCRIPT_DIR}"
echo "DEVICE       = ${DEVICE}"
echo "LOG_DIR      = ${RUN_LOG_DIR}"
echo "============================================================"

cd "${PROJECT_ROOT}"

echo
echo "[STEP 0] Check scripts"
required_scripts=(
  "${SCRIPT_DIR}/01_build_structdesc_features.py"
  "${SCRIPT_DIR}/02_reprocess_stage3_targets.py"
  "${SCRIPT_DIR}/04_build_stage3_task_views.py"
  "${SCRIPT_DIR}/05_build_hybrid_features.py"
  "${SCRIPT_DIR}/06_prepare_training_modes.py"
)

for f in "${required_scripts[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] Missing script: $f"
    exit 1
  fi
  echo "[OK] $f"
done

echo
echo "[STEP 1] Build structural descriptor features"
python "${SCRIPT_DIR}/01_build_structdesc_features.py" \
  --base_dir "${PROJECT_ROOT}/data" \
  2>&1 | tee "${RUN_LOG_DIR}/01_build_structdesc_features.log"

echo
echo "[STEP 2] Reprocess Stage3 targets"
python "${SCRIPT_DIR}/02_reprocess_stage3_targets.py" \
  --input_dir "${PROJECT_ROOT}/data/interim/features/structdesc_features" \
  --output_dir "${PROJECT_ROOT}/data/interim/features/structdesc_features_stage3_v2" \
  2>&1 | tee "${RUN_LOG_DIR}/02_reprocess_stage3_targets.log"

echo
echo "[STEP 3] Build Stage3 task views"
python "${SCRIPT_DIR}/04_build_stage3_task_views.py" \
  --input_dir "${PROJECT_ROOT}/data/interim/features/structdesc_features_stage3_v2" \
  --output_dir "${PROJECT_ROOT}/data/interim/features/stage3_task_views" \
  2>&1 | tee "${RUN_LOG_DIR}/04_build_stage3_task_views.log"

STAGE3_EMBED_DIR="${PROJECT_ROOT}/data/interim/graph_embeddings"
STAGE3_HYBRID_DIR="${PROJECT_ROOT}/data/interim/features/stage3_hybrid_features"

echo
echo "[STEP 4] Build hybrid features"
if [[ -d "${STAGE3_EMBED_DIR}/chgnet_stage3" ]]; then
  python "${SCRIPT_DIR}/05_build_hybrid_features.py" \
    --task stage3 \
    --descriptor_dir "${PROJECT_ROOT}/data/interim/features/structdesc_features_stage3_v2" \
    --embedding_dirs "${STAGE3_EMBED_DIR}/chgnet_stage3" \
    --embedding_prefixes chgnet \
    --output_dir "${STAGE3_HYBRID_DIR}" \
    --descriptor_kind raw \
    2>&1 | tee "${RUN_LOG_DIR}/05_build_hybrid_features.log"
else
  echo "[SKIP] No Stage3 graph embeddings found at ${STAGE3_EMBED_DIR}/chgnet_stage3"
  echo "       Run graph embedding export for Stage3 first, or use FORCE=1 after generating them."
fi

TRAINING_MODE_ROOT="${PROJECT_ROOT}/data/interim/training_modes"

echo
echo "[STEP 5] Prepare training modes"
if [[ -d "${STAGE3_HYBRID_DIR}" ]] && [[ -n "$(find "${STAGE3_HYBRID_DIR}" -name '*.csv' -print -quit 2>/dev/null)" ]]; then
  python "${SCRIPT_DIR}/06_prepare_training_modes.py" \
    --source_dir "${STAGE3_HYBRID_DIR}" \
    --output_root "${TRAINING_MODE_ROOT}" \
    --train_file stage3_train_hybrid.csv \
    --val_file stage3_val_hybrid.csv \
    --test_file stage3_test_hybrid.csv \
    --gold_train_holdout_file stage3_gold_train_holdout_hybrid.csv \
    --dataset_name stage3_hybrid \
    2>&1 | tee "${RUN_LOG_DIR}/06_prepare_training_modes.log"
else
  echo "[SKIP] No Stage3 hybrid features found — skipping training mode preparation."
fi

echo
echo "[STEP 6] Check expected Stage3 dataset for pipeline_v3"

EXPECTED_DIR="${PROJECT_ROOT}/data/interim/generative/stage3_condition_dataset/hybrid_mixed_v1"

expected_files=(
  "${EXPECTED_DIR}/schema.json"
  "${EXPECTED_DIR}/train.npz"
  "${EXPECTED_DIR}/val.npz"
  "${EXPECTED_DIR}/test.npz"
)

for f in "${expected_files[@]}"; do
  if [[ -f "$f" ]]; then
    echo "[OK] $f"
  else
    echo "[WARN] Expected file not found: $f"
  fi
done

echo
echo "============================================================"
echo "[DONE] Stage3 feature construction finished."
echo "Logs:"
echo "  ${RUN_LOG_DIR}"
echo
echo "Expected Stage3 dataset dir:"
echo "  ${EXPECTED_DIR}"
echo "============================================================"
