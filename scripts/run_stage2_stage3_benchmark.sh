#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# SynPred Stage2 + Stage3 Benchmark / Ablation Runner
#
# Function:
#   1. Run Stage2 benchmark models.
#   2. Run Stage3 benchmark models.
#   3. Collect all metrics.
#   4. Automatically recommend inference models.
#   5. Write recommended_inference_config.sh/json/md for run_full_pipeline.sh.
#
# Usage:
#   bash scripts/run_stage2_stage3_benchmark.sh [PROJECT_ROOT] [DEVICE]
#
# Example:
#   FORCE=1 bash scripts/run_stage2_stage3_benchmark.sh /Users/wyc/SynPred cpu
#
# Re-collect summary only:
#   FORCE=1 \
#   RUN_STAGE2_BASELINE=0 RUN_STAGE2_GFLOWNET=0 RUN_STAGE2_TREE=0 RUN_STAGE2_CGCNN=0 \
#   RUN_STAGE3_LGBM=0 RUN_STAGE3_MDN=0 RUN_STAGE3_FLOW=0 RUN_STAGE3_HCNAF=0 RUN_STAGE3_CVAE=0 RUN_STAGE3_DIFFUSION=0 \
#   bash scripts/run_stage2_stage3_benchmark.sh /Users/wyc/SynPred cpu
# ============================================================

PROJECT_ROOT="${1:-/Users/wyc/SynPred}"
DEVICE="${2:-${DEVICE:-cpu}}"
FORCE="${FORCE:-0}"

PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
SCRIPTS="${PROJECT_ROOT}/scripts"

LOG_DIR="${PROJECT_ROOT}/outputs/logs/stage2_stage3_benchmark/$(date +%Y%m%d_%H%M%S)"
RESUME_DIR="${PROJECT_ROOT}/outputs/resume/stage2_stage3_benchmark"
SUMMARY_DIR="${PROJECT_ROOT}/outputs/stage2_stage3_benchmark_summary"

mkdir -p "${LOG_DIR}" "${RESUME_DIR}" "${SUMMARY_DIR}"

# -----------------------------
# Switches
# -----------------------------
RUN_STAGE2_BASELINE="${RUN_STAGE2_BASELINE:-1}"
RUN_STAGE2_GFLOWNET="${RUN_STAGE2_GFLOWNET:-1}"
RUN_STAGE2_TREE="${RUN_STAGE2_TREE:-1}"
RUN_STAGE2_CGCNN="${RUN_STAGE2_CGCNN:-0}"

RUN_STAGE3_BASELINE_CHECK="${RUN_STAGE3_BASELINE_CHECK:-1}"
RUN_STAGE3_LGBM="${RUN_STAGE3_LGBM:-1}"
RUN_STAGE3_MDN="${RUN_STAGE3_MDN:-1}"
RUN_STAGE3_FLOW="${RUN_STAGE3_FLOW:-1}"
RUN_STAGE3_HCNAF="${RUN_STAGE3_HCNAF:-0}"
RUN_STAGE3_CVAE="${RUN_STAGE3_CVAE:-0}"
RUN_STAGE3_DIFFUSION="${RUN_STAGE3_DIFFUSION:-0}"

# -----------------------------
# Paths
# -----------------------------
STAGE2_DATASET="${PROJECT_ROOT}/data/interim/generative/stage2_gflownet_dataset/hybrid/gold_only"
STAGE3_DATASET="${PROJECT_ROOT}/data/interim/generative/stage3_condition_dataset/hybrid_mixed_v1"
STAGE3_TASK_VIEWS="${PROJECT_ROOT}/data/interim/features/stage3_task_views"

STAGE3_BASELINE_RUN_DIR="${PROJECT_ROOT}/runs/stage3/stage3_baseline_commonized_v1"
STAGE3_BASELINE="${STAGE3_BASELINE_RUN_DIR}/best_model.pkl"

# ============================================================
# Utilities
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

run_step() {
  local step="$1"
  local title="$2"
  shift 2

  if is_done "${step}"; then
    echo "[SKIP] ${title} already done. Use FORCE=1 to rerun."
    return 0
  fi

  print_section "${title}"
  local log_file="${LOG_DIR}/${step}.log"

  if "$@" 2>&1 | tee "${log_file}"; then
    mark_done "${step}"
    echo "[OK] ${title}"
  else
    echo "[FAIL] ${title}"
    echo "Check log: ${log_file}"
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

check_optional_file() {
  local f="$1"
  if [[ ! -f "${f}" ]]; then
    echo "[WARN-optional-missing] ${f}"
    return 1
  fi
  echo "[OK-optional] ${f}"
  return 0
}

find_first_script() {
  for s in "$@"; do
    if [[ -f "${s}" ]]; then
      echo "${s}"
      return 0
    fi
  done
  return 1
}

# ============================================================
# Header
# ============================================================

print_section "SynPred Stage2 + Stage3 Benchmark"

echo "PROJECT_ROOT = ${PROJECT_ROOT}"
echo "DEVICE       = ${DEVICE}"
echo "FORCE        = ${FORCE}"
echo "LOG_DIR      = ${LOG_DIR}"
echo "RESUME_DIR   = ${RESUME_DIR}"
echo "SUMMARY_DIR  = ${SUMMARY_DIR}"
echo
echo "Stage2 switches:"
echo "  RUN_STAGE2_BASELINE = ${RUN_STAGE2_BASELINE}"
echo "  RUN_STAGE2_GFLOWNET = ${RUN_STAGE2_GFLOWNET}"
echo "  RUN_STAGE2_TREE     = ${RUN_STAGE2_TREE}"
echo "  RUN_STAGE2_CGCNN    = ${RUN_STAGE2_CGCNN}"
echo
echo "Stage3 switches:"
echo "  RUN_STAGE3_BASELINE_CHECK = ${RUN_STAGE3_BASELINE_CHECK}"
echo "  RUN_STAGE3_LGBM           = ${RUN_STAGE3_LGBM}"
echo "  RUN_STAGE3_MDN            = ${RUN_STAGE3_MDN}"
echo "  RUN_STAGE3_FLOW           = ${RUN_STAGE3_FLOW}"
echo "  RUN_STAGE3_HCNAF          = ${RUN_STAGE3_HCNAF}"
echo "  RUN_STAGE3_CVAE           = ${RUN_STAGE3_CVAE}"
echo "  RUN_STAGE3_DIFFUSION      = ${RUN_STAGE3_DIFFUSION}"

# ============================================================
# Precheck
# ============================================================

step_precheck() {
  echo "[Check] Stage2 dataset"
  check_dir "${STAGE2_DATASET}"
  check_file "${STAGE2_DATASET}/train.npz"
  check_file "${STAGE2_DATASET}/val.npz"
  check_file "${STAGE2_DATASET}/test.npz"
  check_file "${STAGE2_DATASET}/action_vocab.json"
  check_file "${STAGE2_DATASET}/precursor_names.json"
  check_file "${STAGE2_DATASET}/summary.json"
  check_optional_file "${STAGE2_DATASET}/action_to_id.json" || true
  check_optional_file "${STAGE2_DATASET}/label_cols.json" || true
  check_optional_file "${STAGE2_DATASET}/train_meta.csv" || true
  check_optional_file "${STAGE2_DATASET}/val_meta.csv" || true
  check_optional_file "${STAGE2_DATASET}/test_meta.csv" || true

  echo
  echo "[Check] Stage3 dataset"
  check_dir "${STAGE3_DATASET}"
  check_file "${STAGE3_DATASET}/train.npz"
  check_file "${STAGE3_DATASET}/val.npz"
  check_file "${STAGE3_DATASET}/test.npz"
  check_file "${STAGE3_DATASET}/schema.json"
  check_optional_file "${STAGE3_DATASET}/condition_schema.json" || true
  check_optional_file "${STAGE3_DATASET}/train_meta.csv" || true
  check_optional_file "${STAGE3_DATASET}/val_meta.csv" || true
  check_optional_file "${STAGE3_DATASET}/test_meta.csv" || true

  echo
  echo "[Check] Stage3 task views for LightGBM classifiers"
  if [[ -d "${STAGE3_TASK_VIEWS}" ]]; then
    echo "[OK] ${STAGE3_TASK_VIEWS}"
    check_optional_file "${STAGE3_TASK_VIEWS}/atmosphere_coarse_train.csv" || true
    check_optional_file "${STAGE3_TASK_VIEWS}/time_bucket_train.csv" || true
  else
    echo "[WARN-optional-missing] ${STAGE3_TASK_VIEWS}"
  fi

  echo
  echo "[Check] Stage3 baseline checkpoint"
  if [[ "${RUN_STAGE3_BASELINE_CHECK}" == "1" || "${RUN_STAGE3_MDN}" == "1" || "${RUN_STAGE3_FLOW}" == "1" || "${RUN_STAGE3_HCNAF}" == "1" || "${RUN_STAGE3_CVAE}" == "1" || "${RUN_STAGE3_DIFFUSION}" == "1" ]]; then
    check_file "${STAGE3_BASELINE}"
  else
    check_optional_file "${STAGE3_BASELINE}" || true
  fi
}

run_step precheck "Precheck datasets and checkpoints" step_precheck

# ============================================================
# Stage2 MLP baseline
# ============================================================

step_stage2_multilabel_mlp() {
  local script="${SCRIPTS}/04_train/stage2/train_stage2_multilabel_mlp_baseline.py"
  local run_dir="${PROJECT_ROOT}/runs/stage2/baseline_multilabel_mlp_v1"

  check_file "${script}"

  python "${script}" \
    --input_dir "${STAGE2_DATASET}" \
    --run_dir "${run_dir}" \
    --device "${DEVICE}" \
    --hidden_dims 512,256 \
    --dropout 0.10 \
    --batch_size 128 \
    --epochs 80 \
    --patience 12 \
    --lr 1e-3 \
    --weight_decay 1e-5 \
    --threshold 0.5 \
    --topk_from_true_count \
    --seed 42
}

if [[ "${RUN_STAGE2_BASELINE}" == "1" ]]; then
  run_step stage2_multilabel_mlp "Stage2 baseline: multilabel MLP" step_stage2_multilabel_mlp
fi

# ============================================================
# Stage2 GFlowNet + reranker
# ============================================================

step_stage2_gflownet_mild_element_bias() {
  local script="${SCRIPTS}/04_train/stage2/train_gflownet_rerank.py"
  local run_dir="${PROJECT_ROOT}/runs/stage2/gflownet_benchmark_element_bias_mild_v1"

  check_file "${script}"

  python "${script}" \
    --project_root "${PROJECT_ROOT}" \
    --input_mode hybrid \
    --input_dir "${STAGE2_DATASET}" \
    --run_dir "${run_dir}" \
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
    --metric_name samples_f1 \
    --seed 42 \
    --warmup_epochs 10 \
    --rl_weight 0.20 \
    --sample_temperature 1.0 \
    --exact_bonus 1.0 \
    --length_penalty 0.02 \
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
    --ignore_elements H,O
}

if [[ "${RUN_STAGE2_GFLOWNET}" == "1" ]]; then
  run_step stage2_gflownet_mild_element_bias "Stage2 GFlowNet + mild element bias + reranker" step_stage2_gflownet_mild_element_bias
fi

# ============================================================
# Stage2 ExtraTrees / tree ensemble
# ============================================================

step_stage2_tree_ensemble() {
  local script=""

  script="$(find_first_script \
    "${SCRIPTS}/04_train/stage2/train_extratrees.py" \
    "${SCRIPTS}/04_train/stage2/train_tree_ensemble_triple_hybrid.py" \
    "${SCRIPTS}/04_train/stage2/train_tree_ensemble_triple_hybrid" \
  )" || {
    echo "[ERROR] Missing Stage2 tree/ExtraTrees script."
    exit 1
  }

  local run_dir="${PROJECT_ROOT}/runs/stage2/benchmark_tree_ensemble_v1"

  echo "[Info] Using Stage2 tree script: ${script}"

  python "${script}" \
    --input_dir "${STAGE2_DATASET}" \
    --run_dir "${run_dir}" \
    --seed 42
}

if [[ "${RUN_STAGE2_TREE}" == "1" ]]; then
  run_step stage2_tree_ensemble "Stage2 tree / ExtraTrees baseline" step_stage2_tree_ensemble
fi

# ============================================================
# Stage2 CGCNN optional
# ============================================================

step_stage2_cgcnn() {
  local script=""

  script="$(find_first_script \
    "${SCRIPTS}/04_train/stage2/train_cgcnn_stage2_synpred_compare.py" \
    "${SCRIPTS}/04_train/stage2/train_cgcnn_stage2_training_script.py" \
    "${SCRIPTS}/04_train/stage2/train_cgcnn_encoder.py" \
  )" || {
    echo "[ERROR] Missing Stage2 CGCNN script."
    exit 1
  }

  local run_dir="${PROJECT_ROOT}/runs/stage2/benchmark_cgcnn_v1"

  echo "[Info] Using Stage2 CGCNN script: ${script}"

  python "${script}" \
    --project_root "${PROJECT_ROOT}" \
    --input_dir "${STAGE2_DATASET}" \
    --run_dir "${run_dir}" \
    --device "${DEVICE}" \
    --seed 42
}

if [[ "${RUN_STAGE2_CGCNN}" == "1" ]]; then
  run_step stage2_cgcnn "Stage2 CGCNN / graph baseline optional" step_stage2_cgcnn
fi

# ============================================================
# Stage3 LightGBM
# ============================================================

step_stage3_lgbm() {
  local script="${SCRIPTS}/04_train/stage3/train_lgbm_quantile_ensemble.py"

  check_file "${script}"

  python "${script}" \
    --project_root "${PROJECT_ROOT}" \
    --input_dir "${STAGE3_DATASET}" \
    --task_views_dir "${STAGE3_TASK_VIEWS}" \
    --output_dir "${PROJECT_ROOT}/runs/stage3/lgbm_quantile_ensemble_benchmark_v1" \
    --atm_output_dir "${PROJECT_ROOT}/runs/stage3/lgbm_atmosphere_classifier_benchmark_v1" \
    --time_bucket_output_dir "${PROJECT_ROOT}/runs/stage3/lgbm_time_bucket_classifier_benchmark_v1" \
    --num_boost_round 300 \
    --early_stopping_rounds 30
}

if [[ "${RUN_STAGE3_LGBM}" == "1" ]]; then
  run_step stage3_lgbm "Stage3 LightGBM quantile ensemble" step_stage3_lgbm
fi

# ============================================================
# Stage3 residual MDN
# ============================================================

step_stage3_residual_mdn() {
  local script="${SCRIPTS}/04_train/stage3/mixed/train_condition_residual_mdn_mixed.py"
  local run_dir="${PROJECT_ROOT}/runs/stage3/benchmark_residual_mdn_mixed_v1"

  check_file "${script}"

  python "${script}" \
    --project_root "${PROJECT_ROOT}" \
    --input_dir "${STAGE3_DATASET}" \
    --run_dir "${run_dir}" \
    --baseline_ckpt "${STAGE3_BASELINE}" \
    --device "${DEVICE}" \
    --hidden_dims 512,256 \
    --set_proj_dim 256 \
    --fuse_mode concat \
    --dropout 0.10 \
    --n_mixtures 5 \
    --batch_size 128 \
    --epochs 60 \
    --patience 10 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --metric_name top1_continuous_mean_mae_raw \
    --n_gen_samples 16 \
    --grad_clip 5.0 \
    --seed 42 \
    --use_class_weights \
    --clip_to_train_range
}

if [[ "${RUN_STAGE3_MDN}" == "1" ]]; then
  run_step stage3_residual_mdn "Stage3 residual MDN mixed" step_stage3_residual_mdn
fi

# ============================================================
# Stage3 residual normalizing flow
# ============================================================

step_stage3_residual_flow_mixed() {
  local script=""

  script="$(find_first_script \
    "${SCRIPTS}/04_train/stage3/mixed/train_condition_residual_flow_mixed.py" \
    "${SCRIPTS}/04_train/stage3/train_condition_residual_flow_mixed.py" \
    "${SCRIPTS}/04_train/stage3/train_condition_residual_flow.py" \
  )" || {
    echo "[ERROR] Missing Stage3 residual normalizing flow script."
    exit 1
  }

  local run_dir="${PROJECT_ROOT}/runs/stage3/benchmark_residual_flow_mixed_v1"

  echo "[Info] Using Stage3 flow script: ${script}"

  python "${script}" \
    --project_root "${PROJECT_ROOT}" \
    --input_dir "${STAGE3_DATASET}" \
    --run_dir "${run_dir}" \
    --baseline_ckpt "${STAGE3_BASELINE}" \
    --device "${DEVICE}" \
    --hidden_dims 512,256 \
    --set_proj_dim 256 \
    --fuse_mode concat \
    --dropout 0.10 \
    --flow_hidden_dim 256 \
    --n_flow_layers 4 \
    --batch_size 128 \
    --epochs 60 \
    --patience 10 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --metric_name top1_continuous_mean_mae_raw \
    --n_gen_samples 16 \
    --grad_clip 5.0 \
    --seed 42 \
    --use_class_weights \
    --clip_to_train_range
}

if [[ "${RUN_STAGE3_FLOW}" == "1" ]]; then
  run_step stage3_residual_flow_mixed "Stage3 residual normalizing flow mixed" step_stage3_residual_flow_mixed
fi

# ============================================================
# Stage3 HCNAF optional
# ============================================================

step_stage3_hcnaf() {
  local script="${SCRIPTS}/04_train/stage3/train_condition_hcnaf.py"
  local run_dir="${PROJECT_ROOT}/runs/stage3/benchmark_hcnaf_v1"

  check_file "${script}"

  python "${script}" \
    --project_root "${PROJECT_ROOT}" \
    --input_dir "${STAGE3_DATASET}" \
    --run_dir "${run_dir}" \
    --baseline_ckpt "${STAGE3_BASELINE}" \
    --device "${DEVICE}" \
    --hidden_dims 512,256 \
    --set_proj_dim 256 \
    --fuse_mode concat \
    --dropout 0.10 \
    --flow_hidden_dim 256 \
    --maf_hidden_dims 256,256 \
    --batch_size 128 \
    --epochs 60 \
    --patience 10 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --metric_name top1_continuous_mean_mae_raw \
    --n_gen_samples 16 \
    --grad_clip 5.0 \
    --seed 42 \
    --use_class_weights \
    --clip_to_train_range
}

if [[ "${RUN_STAGE3_HCNAF}" == "1" ]]; then
  run_step stage3_hcnaf "Stage3 HCNAF normalizing flow optional" step_stage3_hcnaf
fi

# ============================================================
# Stage3 CVAE optional
# ============================================================

step_stage3_cvae() {
  local script="${SCRIPTS}/04_train/stage3/train_condition_residual_cvae.py"
  local run_dir="${PROJECT_ROOT}/runs/stage3/benchmark_residual_cvae_v1"

  check_file "${script}"

  python "${script}" \
    --project_root "${PROJECT_ROOT}" \
    --input_dir "${STAGE3_DATASET}" \
    --run_dir "${run_dir}" \
    --baseline_ckpt "${STAGE3_BASELINE}" \
    --device "${DEVICE}" \
    --hidden_dims 512,256 \
    --batch_size 128 \
    --epochs 60 \
    --patience 10 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --seed 42
}

if [[ "${RUN_STAGE3_CVAE}" == "1" ]]; then
  run_step stage3_cvae "Stage3 residual CVAE optional" step_stage3_cvae
fi

# ============================================================
# Stage3 latent diffusion optional
# ============================================================

step_stage3_latent_diffusion() {
  local script="${SCRIPTS}/04_train/stage3/train_condition_latent_diffusion.py"
  local run_dir="${PROJECT_ROOT}/runs/stage3/benchmark_latent_diffusion_v1"

  check_file "${script}"

  python "${script}" \
    --project_root "${PROJECT_ROOT}" \
    --input_dir "${STAGE3_DATASET}" \
    --run_dir "${run_dir}" \
    --baseline_ckpt "${STAGE3_BASELINE}" \
    --device "${DEVICE}" \
    --hidden_dims 512,256 \
    --batch_size 128 \
    --epochs 60 \
    --patience 10 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --seed 42
}

if [[ "${RUN_STAGE3_DIFFUSION}" == "1" ]]; then
  run_step stage3_latent_diffusion "Stage3 latent diffusion optional" step_stage3_latent_diffusion
fi

# ============================================================
# Collect metrics + recommend inference config
# ============================================================

step_collect_summary_and_recommend() {
  mkdir -p "${SUMMARY_DIR}"

  PROJECT_ROOT_ENV="${PROJECT_ROOT}" SUMMARY_DIR_ENV="${SUMMARY_DIR}" python - <<'PY'
import json
import math
import os
from pathlib import Path

import pandas as pd

root = Path(os.environ["PROJECT_ROOT_ENV"])
out = Path(os.environ["SUMMARY_DIR_ENV"])
out.mkdir(parents=True, exist_ok=True)

candidate_metrics = [
    ("stage2_multilabel_mlp", root / "runs/stage2/baseline_multilabel_mlp_v1/metrics.json"),
    ("stage2_gflownet_element_bias", root / "runs/stage2/gflownet_benchmark_element_bias_mild_v1/metrics.json"),
    ("stage2_tree_ensemble", root / "runs/stage2/benchmark_tree_ensemble_v1/metrics.json"),
    ("stage2_cgcnn", root / "runs/stage2/benchmark_cgcnn_v1/metrics.json"),

    ("stage3_lgbm_quantile", root / "runs/stage3/lgbm_quantile_ensemble_benchmark_v1/metrics.json"),
    ("stage3_residual_mdn", root / "runs/stage3/benchmark_residual_mdn_mixed_v1/metrics.json"),
    ("stage3_residual_flow_mixed", root / "runs/stage3/benchmark_residual_flow_mixed_v1/metrics.json"),
    ("stage3_hcnaf", root / "runs/stage3/benchmark_hcnaf_v1/metrics.json"),
    ("stage3_cvae", root / "runs/stage3/benchmark_residual_cvae_v1/metrics.json"),
    ("stage3_latent_diffusion", root / "runs/stage3/benchmark_latent_diffusion_v1/metrics.json"),
]

run_dirs = {name: str(path.parent) for name, path in candidate_metrics}


def get_nested(d, keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def first_existing(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def is_finite(x):
    try:
        return x is not None and not pd.isna(x) and math.isfinite(float(x))
    except Exception:
        return False


items = []

for name, path in candidate_metrics:
    row = {
        "model": name,
        "metrics_path": str(path),
        "run_dir": run_dirs.get(name, ""),
        "exists": path.exists(),
    }

    if not path.exists():
        items.append(row)
        continue

    try:
        m = json.loads(path.read_text())
    except Exception as e:
        row["error"] = str(e)
        items.append(row)
        continue

    # Stage2 normal style
    row["stage2_test_samples_f1"] = first_existing(
        get_nested(m, ["test_metrics", "samples_f1"]),
        get_nested(m, ["test_topk_from_true_count_metrics", "samples_f1"]),
    )
    row["stage2_test_subset_accuracy"] = first_existing(
        get_nested(m, ["test_metrics", "subset_accuracy"]),
        get_nested(m, ["test_topk_from_true_count_metrics", "subset_accuracy"]),
    )
    row["stage2_test_micro_f1"] = first_existing(
        get_nested(m, ["test_metrics", "micro_f1"]),
        get_nested(m, ["test_topk_from_true_count_metrics", "micro_f1"]),
    )

    # Stage2 GFlowNet/rerank style
    row["stage2_greedy_test_samples_f1"] = get_nested(m, ["greedy_test_metrics", "samples_f1"])
    row["stage2_greedy_test_subset_accuracy"] = get_nested(m, ["greedy_test_metrics", "subset_accuracy"])
    row["stage2_rerank_test_samples_f1"] = get_nested(m, ["rerank", "rerank_test_metrics", "rerank_test_samples_f1"])
    row["stage2_rerank_test_subset_accuracy"] = get_nested(m, ["rerank", "rerank_test_metrics", "rerank_test_subset_accuracy"])
    row["stage2_rerank_test_exact_hit@5"] = get_nested(m, ["rerank", "rerank_test_metrics", "rerank_test_exact_hit@5"])
    row["stage2_rerank_test_exact_hit@10"] = get_nested(m, ["rerank", "rerank_test_metrics", "rerank_test_exact_hit@10"])
    row["stage2_rerank_test_exact_hit@20"] = get_nested(m, ["rerank", "rerank_test_metrics", "rerank_test_exact_hit@20"])

    # Stage3 neural styles
    row["stage3_test_top1_mae_raw"] = first_existing(
        get_nested(m, ["test_metrics", "top1_continuous_mean_mae_raw"]),
        get_nested(m, ["test_metrics", "top1_mae_mean"]),
        get_nested(m, ["test", "top1_continuous_mean_mae_raw"]),
        get_nested(m, ["test", "top1_mae_mean"]),
    )
    row["stage3_test_oracle_mae_raw"] = first_existing(
        get_nested(m, ["test_metrics", "oracle_best_of_k_continuous_mean_mae_raw"]),
        get_nested(m, ["test_metrics", "oracle_best_of_k_mae_mean"]),
        get_nested(m, ["test", "oracle_best_of_k_continuous_mean_mae_raw"]),
        get_nested(m, ["test", "oracle_best_of_k_mae_mean"]),
    )
    row["stage3_test_disc_macro_f1"] = first_existing(
        get_nested(m, ["test_metrics", "top1_discrete_mean_macro_f1"]),
        get_nested(m, ["test_metrics", "top1_disc_macro_f1_mean"]),
        get_nested(m, ["test", "top1_discrete_mean_macro_f1"]),
        get_nested(m, ["test", "top1_disc_macro_f1_mean"]),
    )
    row["stage3_test_disc_accuracy"] = first_existing(
        get_nested(m, ["test_metrics", "top1_discrete_mean_accuracy"]),
        get_nested(m, ["test_metrics", "top1_disc_accuracy_mean"]),
        get_nested(m, ["test", "top1_discrete_mean_accuracy"]),
        get_nested(m, ["test", "top1_disc_accuracy_mean"]),
    )

    # Stage3 LightGBM style
    row["lgbm_temp_top1_mae"] = first_existing(
        get_nested(m, ["temperature", "top1_mae"]),
        get_nested(m, ["temperature_c", "top1_mae"]),
        get_nested(m, ["target_temperature_c_clean", "top1_mae"]),
        get_nested(m, ["temp", "top1_mae"]),
    )
    row["lgbm_time_top1_mae"] = first_existing(
        get_nested(m, ["time", "top1_mae"]),
        get_nested(m, ["time_h", "top1_mae"]),
        get_nested(m, ["target_time_h_clean", "top1_mae"]),
    )

    items.append(row)

df = pd.DataFrame(items)

summary_csv = out / "benchmark_summary.csv"
summary_json = out / "benchmark_summary.json"

df.to_csv(summary_csv, index=False)
summary_json.write_text(json.dumps(items, ensure_ascii=False, indent=2))


def row_of(model):
    hit = df[df["model"] == model]
    if hit.empty:
        return None
    return hit.iloc[0].to_dict()


def row_exists(row):
    return bool(row and row.get("exists", False))


def val(row, key):
    if row is None:
        return None
    x = row.get(key)
    if is_finite(x):
        return float(x)
    return None


# -----------------------------
# Recommend Stage2
# Higher F1 is better.
# Use rerank F1 for GFlowNet if available.
# -----------------------------
stage2_candidates = []

for model in [
    "stage2_multilabel_mlp",
    "stage2_tree_ensemble",
    "stage2_gflownet_element_bias",
    "stage2_cgcnn",
]:
    row = row_of(model)
    if not row_exists(row):
        continue

    if model == "stage2_gflownet_element_bias":
        score = val(row, "stage2_rerank_test_samples_f1")
        metric = "rerank_test_samples_f1"
        if score is None:
            score = val(row, "stage2_greedy_test_samples_f1")
            metric = "greedy_test_samples_f1"
    else:
        score = val(row, "stage2_test_samples_f1")
        metric = "test_samples_f1"

    if score is not None:
        stage2_candidates.append((score, model, metric))

stage2_candidates.sort(reverse=True, key=lambda x: x[0])

if stage2_candidates:
    best_stage2_score, best_stage2_model, best_stage2_metric = stage2_candidates[0]
else:
    best_stage2_score, best_stage2_model, best_stage2_metric = None, "", ""


# -----------------------------
# Recommend Stage3 top1
# Lower MAE is better.
# -----------------------------
stage3_top1_candidates = []

for model in [
    "stage3_residual_mdn",
    "stage3_residual_flow_mixed",
    "stage3_hcnaf",
    "stage3_cvae",
    "stage3_latent_diffusion",
    "stage3_lgbm_quantile",
]:
    row = row_of(model)
    if not row_exists(row):
        continue
    score = val(row, "stage3_test_top1_mae_raw")
    if score is not None:
        stage3_top1_candidates.append((score, model, "test_top1_continuous_mean_mae_raw"))

stage3_top1_candidates.sort(key=lambda x: x[0])

if stage3_top1_candidates:
    best_stage3_top1_score, best_stage3_top1_model, best_stage3_top1_metric = stage3_top1_candidates[0]
else:
    best_stage3_top1_score, best_stage3_top1_model, best_stage3_top1_metric = None, "", ""


# -----------------------------
# Recommend Stage3 generative
# Lower oracle MAE is better.
# Usually MDN/flow-style models can produce candidate distributions.
# -----------------------------
stage3_gen_candidates = []

for model in [
    "stage3_residual_mdn",
    "stage3_residual_flow_mixed",
    "stage3_hcnaf",
    "stage3_cvae",
    "stage3_latent_diffusion",
]:
    row = row_of(model)
    if not row_exists(row):
        continue
    score = val(row, "stage3_test_oracle_mae_raw")
    if score is not None:
        stage3_gen_candidates.append((score, model, "test_oracle_best_of_k_continuous_mean_mae_raw"))

stage3_gen_candidates.sort(key=lambda x: x[0])

if stage3_gen_candidates:
    best_stage3_gen_score, best_stage3_gen_model, best_stage3_gen_metric = stage3_gen_candidates[0]
else:
    best_stage3_gen_score, best_stage3_gen_model, best_stage3_gen_metric = None, "", ""


recommendation = {
    "project_root": str(root),
    "summary_csv": str(summary_csv),
    "summary_json": str(summary_json),
    "recommended_stage2": {
        "model": best_stage2_model,
        "score": best_stage2_score,
        "selection_metric": best_stage2_metric,
        "run_dir": run_dirs.get(best_stage2_model, ""),
    },
    "recommended_stage3_top1": {
        "model": best_stage3_top1_model,
        "score": best_stage3_top1_score,
        "selection_metric": best_stage3_top1_metric,
        "run_dir": run_dirs.get(best_stage3_top1_model, ""),
    },
    "recommended_stage3_generative": {
        "model": best_stage3_gen_model,
        "score": best_stage3_gen_score,
        "selection_metric": best_stage3_gen_metric,
        "run_dir": run_dirs.get(best_stage3_gen_model, ""),
    },
    "deployment_policy": {
        "stage2": "Use recommended_stage2 for precursor-set generation.",
        "stage3_top1": "Use recommended_stage3_top1 for default single condition prediction.",
        "stage3_generative": "Use recommended_stage3_generative for multi-candidate or uncertainty-aware condition generation.",
        "note": "Benchmark should not be rerun inside production inference. Source recommended_inference_config.sh in run_full_pipeline.sh.",
    },
}

rec_json = out / "recommended_inference_config.json"
rec_sh = out / "recommended_inference_config.sh"
rec_md = out / "benchmark_recommendation.md"

rec_json.write_text(json.dumps(recommendation, ensure_ascii=False, indent=2))

rec_sh.write_text(
    "#!/usr/bin/env bash\n"
    "# Auto-generated by run_stage2_stage3_benchmark.sh\n"
    "\n"
    f"export SYNPRED_STAGE2_MODEL=\"{best_stage2_model}\"\n"
    f"export SYNPRED_STAGE2_RUN_DIR=\"{run_dirs.get(best_stage2_model, '')}\"\n"
    f"export SYNPRED_STAGE2_SELECTION_METRIC=\"{best_stage2_metric}\"\n"
    f"export SYNPRED_STAGE2_SELECTION_SCORE=\"{'' if best_stage2_score is None else best_stage2_score}\"\n"
    "\n"
    f"export SYNPRED_STAGE3_TOP1_MODEL=\"{best_stage3_top1_model}\"\n"
    f"export SYNPRED_STAGE3_TOP1_RUN_DIR=\"{run_dirs.get(best_stage3_top1_model, '')}\"\n"
    f"export SYNPRED_STAGE3_TOP1_SELECTION_METRIC=\"{best_stage3_top1_metric}\"\n"
    f"export SYNPRED_STAGE3_TOP1_SELECTION_SCORE=\"{'' if best_stage3_top1_score is None else best_stage3_top1_score}\"\n"
    "\n"
    f"export SYNPRED_STAGE3_GENERATIVE_MODEL=\"{best_stage3_gen_model}\"\n"
    f"export SYNPRED_STAGE3_GENERATIVE_RUN_DIR=\"{run_dirs.get(best_stage3_gen_model, '')}\"\n"
    f"export SYNPRED_STAGE3_GENERATIVE_SELECTION_METRIC=\"{best_stage3_gen_metric}\"\n"
    f"export SYNPRED_STAGE3_GENERATIVE_SELECTION_SCORE=\"{'' if best_stage3_gen_score is None else best_stage3_gen_score}\"\n"
    "\n"
    f"export SYNPRED_BENCHMARK_SUMMARY=\"{summary_csv}\"\n"
    f"export SYNPRED_RECOMMENDED_CONFIG=\"{rec_json}\"\n"
)

rec_md.write_text(
    "# SynPred Stage2/Stage3 Benchmark Recommendation\n\n"
    "## Stage2 recommendation\n\n"
    f"Recommended model: {best_stage2_model}\n\n"
    f"Selection metric: {best_stage2_metric}\n\n"
    f"Score: {best_stage2_score}\n\n"
    "Run directory:\n\n"
    "```text\n"
    f"{run_dirs.get(best_stage2_model, '')}\n"
    "```\n\n"
    "## Stage3 top1 recommendation\n\n"
    f"Recommended model: {best_stage3_top1_model}\n\n"
    f"Selection metric: {best_stage3_top1_metric}\n\n"
    f"Score: {best_stage3_top1_score}\n\n"
    "Run directory:\n\n"
    "```text\n"
    f"{run_dirs.get(best_stage3_top1_model, '')}\n"
    "```\n\n"
    "## Stage3 generative recommendation\n\n"
    f"Recommended model: {best_stage3_gen_model}\n\n"
    f"Selection metric: {best_stage3_gen_metric}\n\n"
    f"Score: {best_stage3_gen_score}\n\n"
    "Run directory:\n\n"
    "```text\n"
    f"{run_dirs.get(best_stage3_gen_model, '')}\n"
    "```\n\n"
    "## Production usage\n\n"
    "Do not rerun benchmark inside production inference. Source this file in run_full_pipeline.sh:\n\n"
    "```bash\n"
    f"source {rec_sh}\n"
    "```\n"
)

print(df.to_string(index=False))
print()
print(json.dumps(recommendation, ensure_ascii=False, indent=2))
print()
print(f"[SAVED] {summary_csv}")
print(f"[SAVED] {summary_json}")
print(f"[SAVED] {rec_json}")
print(f"[SAVED] {rec_sh}")
print(f"[SAVED] {rec_md}")
PY

  chmod +x "${SUMMARY_DIR}/recommended_inference_config.sh"
}

run_step collect_summary_and_recommend "Collect metrics and recommend inference configuration" step_collect_summary_and_recommend

# ============================================================
# Final summary
# ============================================================

print_section "BENCHMARK COMPLETE"

echo "Logs:"
echo "  ${LOG_DIR}"
echo
echo "Summary:"
echo "  ${SUMMARY_DIR}/benchmark_summary.csv"
echo "  ${SUMMARY_DIR}/benchmark_summary.json"
echo
echo "Recommended inference config:"
echo "  ${SUMMARY_DIR}/recommended_inference_config.json"
echo "  ${SUMMARY_DIR}/recommended_inference_config.sh"
echo "  ${SUMMARY_DIR}/benchmark_recommendation.md"
echo
echo "Use recommendation in run_full_pipeline.sh:"
echo "  source ${SUMMARY_DIR}/recommended_inference_config.sh"
echo
echo "Quick commands:"
echo
echo "  Rerun all default benchmark models:"
echo "    FORCE=1 bash scripts/run_stage2_stage3_benchmark.sh ${PROJECT_ROOT} ${DEVICE}"
echo
echo "  Re-collect summary and recommendation only:"
echo "    FORCE=1 RUN_STAGE2_BASELINE=0 RUN_STAGE2_GFLOWNET=0 RUN_STAGE2_TREE=0 RUN_STAGE2_CGCNN=0 RUN_STAGE3_LGBM=0 RUN_STAGE3_MDN=0 RUN_STAGE3_FLOW=0 RUN_STAGE3_HCNAF=0 RUN_STAGE3_CVAE=0 RUN_STAGE3_DIFFUSION=0 bash scripts/run_stage2_stage3_benchmark.sh ${PROJECT_ROOT} ${DEVICE}"
echo
echo "  Run Stage2 only:"
echo "    FORCE=1 RUN_STAGE3_LGBM=0 RUN_STAGE3_MDN=0 RUN_STAGE3_FLOW=0 bash scripts/run_stage2_stage3_benchmark.sh ${PROJECT_ROOT} ${DEVICE}"
echo
echo "  Run Stage3 only:"
echo "    FORCE=1 RUN_STAGE2_BASELINE=0 RUN_STAGE2_GFLOWNET=0 RUN_STAGE2_TREE=0 bash scripts/run_stage2_stage3_benchmark.sh ${PROJECT_ROOT} ${DEVICE}"
echo
echo "============================================================"
echo "[DONE] SynPred Stage2 + Stage3 benchmark completed"
echo "============================================================"
