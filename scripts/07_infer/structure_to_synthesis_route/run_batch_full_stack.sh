#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/Users/wyc/SynPred}"
BATCH_NAME="${2:-batch_500}"
TOP_N_PER_CASE="${3:-5}"
RELIABILITY_MODE="${4:-dry_run}"   # dry_run or execute

BASE_DIR="${PROJECT_ROOT}/scripts/07_infer/structure_to_synthesis_route"
ADAPTIVE_DIR="${BASE_DIR}/batch_adaptive"
RELIABILITY_DIR="${BASE_DIR}/batch_reliability"

echo "============================================================"
echo "Full-stack batch route inference"
echo "PROJECT_ROOT     = ${PROJECT_ROOT}"
echo "BATCH_NAME       = ${BATCH_NAME}"
echo "TOP_N_PER_CASE   = ${TOP_N_PER_CASE}"
echo "RELIABILITY_MODE = ${RELIABILITY_MODE}"
echo "============================================================"

echo
echo "[STEP 1] Export adaptive results, dispatch recovery, and build compact overview"
cd "${ADAPTIVE_DIR}"
bash run_export_and_dispatch.sh "${PROJECT_ROOT}" "${BATCH_NAME}" "${TOP_N_PER_CASE}"

echo
echo "[STEP 2] Run batch reliability using adaptive compact overview"
cd "${RELIABILITY_DIR}"

if [[ "${RELIABILITY_MODE}" == "execute" ]]; then
  python run_batch_reliability_pipeline.py \
    --config configs/batch_reliability.yaml \
    --execute \
    --start_from audit_batch
else
  python run_batch_reliability_pipeline.py \
    --config configs/batch_reliability.yaml \
    --dry_run \
    --start_from audit_batch
fi

echo
echo "============================================================"
echo "[DONE] Full-stack batch postprocessing finished"
echo "Main outputs:"
echo "  ${PROJECT_ROOT}/outputs/batch_adaptive/${BATCH_NAME}/batch_compact_overview/compact_overview_report.md"
echo "  ${PROJECT_ROOT}/outputs/batch_adaptive/${BATCH_NAME}/batch_compact_overview/compact_watchlist.csv"
echo "  ${PROJECT_ROOT}/outputs/batch_adaptive/${BATCH_NAME}/route_splits/route_recommended_high_medium.csv"
echo "  ${PROJECT_ROOT}/outputs/batch_reliability/${BATCH_NAME}/gap_closure_report/gap_closure_report.md"
echo "  ${PROJECT_ROOT}/outputs/batch_adaptive/${BATCH_NAME}/failed_case_inspection/failed_cases_inspection.md"
echo "  ${PROJECT_ROOT}/outputs/batch_adaptive/${BATCH_NAME}/manual_review/manual_review_cases.csv"
echo "  ${PROJECT_ROOT}/outputs/batch_adaptive/${BATCH_NAME}/manual_review/manual_review_routes.csv"
echo "============================================================"
