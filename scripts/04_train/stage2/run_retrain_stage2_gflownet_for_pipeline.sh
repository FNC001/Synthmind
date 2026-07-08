#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/Users/wyc/SynPred}"
DEVICE="${DEVICE:-cpu}"
METRIC_NAME="${METRIC_NAME:-samples_f1}"

SCRIPT_DIR="${PROJECT_ROOT}/scripts/04_train/stage2"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_gflownet_rerank.py"

DATASET_DIR="${PROJECT_ROOT}/data/interim/generative/stage2_gflownet_dataset/hybrid/gold_only"

# Keep this canonical run_dir for pipeline_v3 compatibility.
RUN_DIR="${PROJECT_ROOT}/runs/stage2/gflownet_joint_rerank_hybrid_gold_only_v1"
EXPECTED="${RUN_DIR}/best_model.pt"
EXPECTED_RERANKER="${RUN_DIR}/best_reranker.pt"
EXPECTED_METRICS="${RUN_DIR}/metrics.json"

BACKUP_ROOT="${PROJECT_ROOT}/runs/stage2/_backup"
LOG_DIR="${PROJECT_ROOT}/outputs/logs/stage2_gflownet_retrain"
LOG_FILE="${LOG_DIR}/retrain_stage2_gflownet_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${BACKUP_ROOT}" "${LOG_DIR}"

echo "============================================================"
echo "Retrain Stage2 GFlowNet for pipeline_v3"
echo "PROJECT_ROOT = ${PROJECT_ROOT}"
echo "DEVICE       = ${DEVICE}"
echo "METRIC_NAME  = ${METRIC_NAME}"
echo "DATASET_DIR  = ${DATASET_DIR}"
echo "RUN_DIR      = ${RUN_DIR}"
echo "EXPECTED     = ${EXPECTED}"
echo "LOG_FILE     = ${LOG_FILE}"
echo "============================================================"
echo

echo "[STEP 1] Check required training script"
if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "[ERROR] Missing training script: ${TRAIN_SCRIPT}"
  exit 1
fi
echo "[OK] ${TRAIN_SCRIPT}"

echo
echo "[STEP 2] Check Stage2 GFlowNet dataset"
required_files=(
  "${DATASET_DIR}/train.npz"
  "${DATASET_DIR}/val.npz"
  "${DATASET_DIR}/test.npz"
  "${DATASET_DIR}/train_meta.csv"
  "${DATASET_DIR}/val_meta.csv"
  "${DATASET_DIR}/test_meta.csv"
  "${DATASET_DIR}/action_to_id.json"
  "${DATASET_DIR}/action_vocab.json"
  "${DATASET_DIR}/precursor_names.json"
  "${DATASET_DIR}/summary.json"
)

optional_files=(
  "${DATASET_DIR}/feature_cols.json"
  "${DATASET_DIR}/feature_mean.npy"
  "${DATASET_DIR}/feature_std.npy"
  "${DATASET_DIR}/label_cols.json"
  "${DATASET_DIR}/label_names.json"
  "${DATASET_DIR}/schema.json"
)

for f in "${required_files[@]}"; do
  if [[ ! -f "${f}" ]]; then
    echo "[ERROR] Missing required dataset file: ${f}"
    exit 1
  fi
  echo "[OK] ${f}"
done

for f in "${optional_files[@]}"; do
  if [[ -f "${f}" ]]; then
    echo "[OK-optional] ${f}"
  else
    echo "[WARN-optional-missing] ${f}"
  fi
done

echo
echo "[STEP 3] Backup old run dir if exists"
if [[ -d "${RUN_DIR}" ]]; then
  BACKUP_DIR="${BACKUP_ROOT}/gflownet_joint_rerank_hybrid_gold_only_v1_$(date +%Y%m%d_%H%M%S)"
  echo "[BACKUP] ${RUN_DIR}"
  echo "      -> ${BACKUP_DIR}"
  mv "${RUN_DIR}" "${BACKUP_DIR}"
fi

mkdir -p "${RUN_DIR}"

echo
echo "[STEP 4] Start training with mild element-bias + reranker"
echo "------------------------------------------------------------"

cd "${SCRIPT_DIR}"

python "${TRAIN_SCRIPT}" \
  --project_root "${PROJECT_ROOT}" \
  --input_mode hybrid \
  --input_dir "${DATASET_DIR}" \
  --run_dir "${RUN_DIR}" \
  --train_mode gold_only \
  --device "${DEVICE}" \
  --hidden_dim 256 \
  --x_mlp_hidden_dims 512 \
  --dropout 0.10 \
  --batch_size 128 \
  --epochs 100 \
  --patience 15 \
  --lr 1e-3 \
  --weight_decay 1e-5 \
  --metric_name "${METRIC_NAME}" \
  --seed 42 \
  --warmup_epochs 10 \
  --rl_weight 0.20 \
  --sample_temperature 1.0 \
  --exact_bonus 1.0 \
  --length_penalty 0.02 \
  --force_non_empty \
  --rerank_enabled \
  --rerank_num_samples_train 16 \
  --rerank_num_samples_eval 64 \
  --rerank_sample_temperatures 0.7,0.9,1.1,1.3,1.6 \
  --rerank_hidden_dims 512,256 \
  --rerank_dropout 0.10 \
  --rerank_lr 1e-3 \
  --rerank_weight_decay 1e-5 \
  --rerank_batch_size 512 \
  --rerank_epochs 30 \
  --rerank_patience 6 \
  --save_topn_candidates 20 \
  --topk_values 1,3,5,10,20 \
  --element_bias_enabled \
  --target_hit_bonus 3.0 \
  --extra_element_penalty 0.5 \
  --no_overlap_penalty 3.0 \
  --stop_bias -1.0 \
  --ignore_elements H,O \
  2>&1 | tee "${LOG_FILE}"

echo
echo "[STEP 5] Check output"

if [[ ! -f "${EXPECTED}" ]]; then
  echo "[ERROR] Training finished but missing expected checkpoint:"
  echo "  ${EXPECTED}"
  echo
  echo "Check log:"
  echo "  ${LOG_FILE}"
  exit 1
fi

if [[ ! -f "${EXPECTED_RERANKER}" ]]; then
  echo "[ERROR] Training finished but missing expected reranker checkpoint:"
  echo "  ${EXPECTED_RERANKER}"
  echo
  echo "Check log:"
  echo "  ${LOG_FILE}"
  exit 1
fi

if [[ ! -f "${EXPECTED_METRICS}" ]]; then
  echo "[ERROR] Training finished but missing metrics file:"
  echo "  ${EXPECTED_METRICS}"
  echo
  echo "Check log:"
  echo "  ${LOG_FILE}"
  exit 1
fi

echo "[OK] Stage2 GFlowNet checkpoint generated:"
echo "  ${EXPECTED}"

echo "[OK] Stage2 reranker checkpoint generated:"
echo "  ${EXPECTED_RERANKER}"

echo "[OK] Metrics generated:"
echo "  ${EXPECTED_METRICS}"

echo
echo "[STEP 6] Key metrics preview"
python - <<PY
import json
from pathlib import Path

p = Path("${EXPECTED_METRICS}")
m = json.loads(p.read_text())

print("best_epoch:", m.get("training", {}).get("best_epoch"))
print("best_val_metric:", m.get("training", {}).get("best_val_metric"))
print("greedy_test_samples_f1:", m.get("greedy_test_metrics", {}).get("samples_f1"))
print("greedy_test_subset_accuracy:", m.get("greedy_test_metrics", {}).get("subset_accuracy"))

rerank = m.get("rerank", {})
rt = rerank.get("rerank_test_metrics", {})
print("rerank_test_samples_f1:", rt.get("rerank_test_samples_f1"))
print("rerank_test_subset_accuracy:", rt.get("rerank_test_subset_accuracy"))
print("rerank_test_exact_hit@5:", rt.get("rerank_test_exact_hit@5"))
print("rerank_test_exact_hit@10:", rt.get("rerank_test_exact_hit@10"))

eb = rerank.get("element_bias", {})
print("element_bias_enabled:", eb.get("enabled"))
print("element_bias_config:", eb)
PY

echo
echo "Next step:"
echo "  cd ${PROJECT_ROOT}/scripts/07_infer/structure_to_synthesis_route/pipeline_v3"
echo "  python run_pipeline.py --config configs/full_route_stage3.yaml --infer_name benchmark_001 --start_from export_final_top_routes"
echo
echo "============================================================"
echo "[DONE] Stage2 GFlowNet retraining completed"
echo "============================================================"
