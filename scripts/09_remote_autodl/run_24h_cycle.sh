#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="${AUTODL_REMOTE_ROOT:-/root/SynPred_autorun_20260613}"
PROJECT_DIR="${AUTODL_PROJECT_DIR:-${REMOTE_ROOT}/SynPred}"
OUTPUT_ROOT="${AUTODL_OUTPUT_ROOT:-outputs/autorun/24h_optimization_20260613}"
SESSION_NAME="${AUTODL_TMUX_SESSION:-synpred_24h_20260613}"
mkdir -p "${REMOTE_ROOT}/logs"
LOG_FILE="${REMOTE_ROOT}/logs/24h_cycle_$(date +%Y%m%d_%H%M%S).log"

detect_python() {
  if command -v conda >/dev/null 2>&1; then
    local base
    base="$(conda info --base 2>/dev/null || true)"
    for p in "${base}/envs/py311/bin/python" "${base}/envs/synpred/bin/python" "${base}/bin/python"; do
      [[ -x "${p}" ]] && { echo "${p}"; return; }
    done
  fi
  command -v python3 2>/dev/null || command -v python
}
PYTHON_BIN="${PYTHON_BIN:-$(detect_python)}"

CMD="cd '${PROJECT_DIR}' && env KMP_DUPLICATE_LIB_OK=TRUE '${PYTHON_BIN}' scripts/10_autorun/run_synpred_24h_optimization.py --project_root '${PROJECT_DIR}' --output_root '${OUTPUT_ROOT}' --max_train_hours 12 --max_total_hours 24 --scope all --run_stage2 1 --run_stage3 1 --run_stage35 1 --run_expensive_stage2_oof 0 --kfold 3 --run_test_only_if_val_pass 1 --allow_update_selector 0 --make_paper_figures 1 --make_article_draft 1 --seed 42"

if command -v tmux >/dev/null 2>&1; then
  tmux new-session -d -s "${SESSION_NAME}" "bash -lc \"${CMD} 2>&1 | tee '${LOG_FILE}'\""
  echo "Started tmux session: ${SESSION_NAME}"
else
  nohup bash -lc "${CMD}" > "${LOG_FILE}" 2>&1 &
  echo "Started nohup PID: $!"
fi
echo "Log: ${LOG_FILE}"

