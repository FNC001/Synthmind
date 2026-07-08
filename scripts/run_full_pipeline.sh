#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/Users/wyc/SynPred}"
DEVICE="${2:-${DEVICE:-cpu}}"

RUN_PREPARE="${RUN_PREPARE:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_INFER="${RUN_INFER:-1}"

if [[ "${RUN_PREPARE}" == "1" ]]; then
  bash "${PROJECT_ROOT}/scripts/run_prepare_training_data.sh" "${PROJECT_ROOT}" "${DEVICE}"
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
  bash "${PROJECT_ROOT}/scripts/run_train_models.sh" "${PROJECT_ROOT}" "${DEVICE}"
fi

if [[ "${RUN_INFER}" == "1" ]]; then
  bash "${PROJECT_ROOT}/scripts/run_inference_pipeline.sh" "${PROJECT_ROOT}" "${DEVICE}"
fi
