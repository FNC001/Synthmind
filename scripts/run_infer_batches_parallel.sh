#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# SynPred Parallel Batch Inference Runner
#
# Purpose:
#   Run inference_pipeline on many batch_* conditioned_x_csv files
#   in parallel.
#
# This script does NOT:
#   - prepare training data
#   - train models
#   - run benchmark
#
# It only calls:
#   scripts/run_inference_pipeline.sh
#
# Usage:
#   bash scripts/run_infer_batches_parallel.sh [PROJECT_ROOT] [DEVICE]
#
# Examples:
#   bash scripts/run_infer_batches_parallel.sh /Users/wyc/SynPred cpu
#
#   N_JOBS=4 bash scripts/run_infer_batches_parallel.sh /Users/wyc/SynPred cpu
#
#   FORCE=1 N_JOBS=4 bash scripts/run_infer_batches_parallel.sh /Users/wyc/SynPred cpu
#
# Env options:
#   FORCE=0|1
#   N_JOBS=2
#   BATCH_ROOT=/path/to/batch_root
#   PIPELINE_OUT_ROOT=/path/to/output_root
#   TOP_K_CONDITIONS=5
#   N_FLOW_SAMPLES=32
#   SEED=42
# ============================================================

PROJECT_ROOT="${1:-/Users/wyc/SynPred}"
DEVICE="${2:-${DEVICE:-cpu}}"

FORCE="${FORCE:-1}"
N_JOBS="${N_JOBS:-2}"

PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
SCRIPTS="${PROJECT_ROOT}/scripts"

BATCH_ROOT="${BATCH_ROOT:-${PROJECT_ROOT}/data/interim/infer/GNoME_selected_Wyckoff_CLscore_vasp_batches}"
PIPELINE_OUT_ROOT="${PIPELINE_OUT_ROOT:-${PROJECT_ROOT}/outputs/inference_batches_parallel}"

TOP_K_CONDITIONS="${TOP_K_CONDITIONS:-5}"
N_FLOW_SAMPLES="${N_FLOW_SAMPLES:-32}"
SEED="${SEED:-42}"

INFER_SCRIPT="${SCRIPTS}/run_inference_pipeline.sh"

LOG_DIR="${PROJECT_ROOT}/outputs/logs/batch_infer/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}" "${PIPELINE_OUT_ROOT}"

print_section() {
  echo
  echo "============================================================"
  echo "  $1"
  echo "============================================================"
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

print_section "SynPred Parallel Batch Inference"

echo "PROJECT_ROOT      = ${PROJECT_ROOT}"
echo "DEVICE            = ${DEVICE}"
echo "FORCE             = ${FORCE}"
echo "N_JOBS            = ${N_JOBS}"
echo "BATCH_ROOT        = ${BATCH_ROOT}"
echo "PIPELINE_OUT_ROOT = ${PIPELINE_OUT_ROOT}"
echo "INFER_SCRIPT      = ${INFER_SCRIPT}"
echo "LOG_DIR           = ${LOG_DIR}"
echo "TOP_K_CONDITIONS  = ${TOP_K_CONDITIONS}"
echo "N_FLOW_SAMPLES    = ${N_FLOW_SAMPLES}"
echo "SEED              = ${SEED}"

check_dir "${PROJECT_ROOT}"
check_dir "${SCRIPTS}"
check_file "${INFER_SCRIPT}"
check_dir "${BATCH_ROOT}"

LIST_FILE="${LOG_DIR}/conditioned_x_files.txt"

find "${BATCH_ROOT}" \
  -type f \
  -name "stage3_conditioned_x_fallback_retrieval_baseline_element_reranked.csv" \
  | sort > "${LIST_FILE}"

if [[ ! -s "${LIST_FILE}" ]]; then
  echo "[ERROR] No conditioned_x csv files found under:"
  echo "  ${BATCH_ROOT}"
  echo
  echo "Expected file name:"
  echo "  stage3_conditioned_x_fallback_retrieval_baseline_element_reranked.csv"
  exit 1
fi

N_FILES="$(wc -l < "${LIST_FILE}" | tr -d ' ')"

echo
echo "[Info] Found conditioned_x files: ${N_FILES}"
echo "[SAVED] ${LIST_FILE}"

if ! command -v parallel >/dev/null 2>&1; then
  echo
  echo "[ERROR] GNU parallel is not installed or not in PATH."
  echo "Install it first, for example:"
  echo "  brew install parallel"
  echo
  echo "Or run sequentially with:"
  echo "  while read -r f; do CONDITIONED_X_CSV=\"\$f\" bash ${INFER_SCRIPT} ${PROJECT_ROOT} ${DEVICE}; done < ${LIST_FILE}"
  exit 1
fi

export PROJECT_ROOT DEVICE FORCE INFER_SCRIPT PIPELINE_OUT_ROOT LOG_DIR
export TOP_K_CONDITIONS N_FLOW_SAMPLES SEED

print_section "Run parallel inference"

parallel -j "${N_JOBS}" --bar --halt soon,fail=1 '
csv="{}"
batch_name="$(basename "$(dirname "$csv")")"

out_dir="${PIPELINE_OUT_ROOT}/${batch_name}"
log_file="${LOG_DIR}/infer_${batch_name}.log"

mkdir -p "${out_dir}"

echo "[START] ${batch_name}"
echo "  csv     = ${csv}"
echo "  out_dir = ${out_dir}"
echo "  log     = ${log_file}"

FORCE="${FORCE}" \
RUN_PREFLIGHT=1 \
RUN_COMBINED_INFER=1 \
RUN_ORIGINAL_PIPELINE=0 \
RUN_EXPORT=1 \
CONDITIONED_X_CSV="${csv}" \
PIPELINE_OUT_DIR="${out_dir}" \
COMBINED_OUT_DIR="${out_dir}/combined" \
EXPORT_OUT_DIR="${out_dir}/export" \
TOP_K_CONDITIONS="${TOP_K_CONDITIONS}" \
N_FLOW_SAMPLES="${N_FLOW_SAMPLES}" \
SEED="${SEED}" \
bash "${INFER_SCRIPT}" "${PROJECT_ROOT}" "${DEVICE}" \
> "${log_file}" 2>&1

echo "[DONE] ${batch_name}"
' :::: "${LIST_FILE}"

print_section "Collect batch outputs"

SUMMARY_JSON="${PIPELINE_OUT_ROOT}/batch_parallel_summary.json"
SUMMARY_CSV="${PIPELINE_OUT_ROOT}/batch_parallel_summary.csv"

python - <<PY
import json
from pathlib import Path
import pandas as pd

root = Path("${PIPELINE_OUT_ROOT}")
items = []

for batch_dir in sorted(root.glob("batch_*")):
    export_csv = batch_dir / "export" / "all_test_candidates_flat.csv"
    combined_summary = batch_dir / "combined" / "combined_summary.json"
    manifest = batch_dir / "export" / "manifest.json"

    item = {
        "batch_name": batch_dir.name,
        "batch_dir": str(batch_dir),
        "export_csv": str(export_csv),
        "export_csv_exists": export_csv.exists(),
        "combined_summary": str(combined_summary),
        "combined_summary_exists": combined_summary.exists(),
        "manifest": str(manifest),
        "manifest_exists": manifest.exists(),
        "n_rows": 0,
    }

    if export_csv.exists():
        try:
            df = pd.read_csv(export_csv)
            item["n_rows"] = int(len(df))
        except Exception as e:
            item["read_error"] = str(e)

    items.append(item)

summary_json = Path("${SUMMARY_JSON}")
summary_csv = Path("${SUMMARY_CSV}")

summary_json.write_text(json.dumps({
    "pipeline_out_root": str(root),
    "n_batches": len(items),
    "items": items,
}, ensure_ascii=False, indent=2))

pd.DataFrame(items).to_csv(summary_csv, index=False)

print(f"[SAVED] {summary_json}")
print(f"[SAVED] {summary_csv}")
print(f"[Info] batches = {len(items)}")
PY

print_section "PARALLEL BATCH INFERENCE COMPLETE"

echo "Logs:"
echo "  ${LOG_DIR}"
echo
echo "Input list:"
echo "  ${LIST_FILE}"
echo
echo "Outputs:"
echo "  ${PIPELINE_OUT_ROOT}"
echo
echo "Summary:"
echo "  ${SUMMARY_JSON}"
echo "  ${SUMMARY_CSV}"
echo
echo "Useful commands:"
echo
echo "  Rerun with 4 parallel jobs:"
echo "    FORCE=1 N_JOBS=4 bash scripts/run_infer_batches_parallel.sh ${PROJECT_ROOT} ${DEVICE}"
echo
echo "  Use another batch root:"
echo "    FORCE=1 N_JOBS=4 BATCH_ROOT=/path/to/batches bash scripts/run_infer_batches_parallel.sh ${PROJECT_ROOT} ${DEVICE}"
echo
echo "============================================================"
echo "[DONE] SynPred parallel batch inference completed"
echo "============================================================"
