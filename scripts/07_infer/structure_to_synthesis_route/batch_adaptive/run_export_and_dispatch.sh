#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/Users/wyc/SynPred}"
BATCH_NAME="${2:-batch_001}"
TOP_N_PER_CASE="${3:-5}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BATCH_OUT_DIR="${PROJECT_ROOT}/outputs/batch_adaptive/${BATCH_NAME}"
DISPATCH_DIR="${BATCH_OUT_DIR}/recovery_dispatch"

echo "============================================================"
echo "Export and dispatch adaptive batch results"
echo "PROJECT_ROOT    = ${PROJECT_ROOT}"
echo "BATCH_NAME      = ${BATCH_NAME}"
echo "TOP_N_PER_CASE  = ${TOP_N_PER_CASE}"
echo "SCRIPT_DIR      = ${SCRIPT_DIR}"
echo "BATCH_OUT_DIR   = ${BATCH_OUT_DIR}"
echo "============================================================"

echo
echo "[STEP 0] Check required inputs"
MASTER_CSV="${BATCH_OUT_DIR}/master_status.csv"

if [[ ! -f "${MASTER_CSV}" ]]; then
  echo "[ERROR] Missing master_status.csv:"
  echo "  ${MASTER_CSV}"
  echo
  echo "Please run adaptive batch first, for example:"
  echo "  python ${SCRIPT_DIR}/run_adaptive_batch_pipeline.py \\"
  echo "    --config ${SCRIPT_DIR}/configs/batch_adaptive.yaml \\"
  echo "    --mode run_and_audit"
  exit 1
fi

echo "[OK] ${MASTER_CSV}"

echo
echo "[STEP 1] Export batch recommendations with refined status"
python "${SCRIPT_DIR}/scripts/export_batch_recommendations.py" \
  --project_root "${PROJECT_ROOT}" \
  --batch_name "${BATCH_NAME}" \
  --top_n_per_case "${TOP_N_PER_CASE}"

echo
echo "[STEP 2] Dispatch recovery/review/pass cases"
python "${SCRIPT_DIR}/scripts/dispatch_recovery_cases.py" \
  --batch_name "${BATCH_NAME}"

echo
echo "[STEP 3] Refine review-required route-level severity"
python "${SCRIPT_DIR}/scripts/refine_review_required_routes.py" \
  --batch_name "${BATCH_NAME}"

echo
echo "[STEP 4] Split routes by route-level refined status"
python "${SCRIPT_DIR}/scripts/split_routes_by_refined_status.py" \
  --batch_name "${BATCH_NAME}"

echo
echo "[STEP 5] Build recovery plan for abnormal cases"
python "${SCRIPT_DIR}/scripts/build_recovery_plan.py" \
  --project_root "${PROJECT_ROOT}" \
  --batch_name "${BATCH_NAME}"

echo
echo "[STEP 6] Prepare recovery actions by condition"
python "${SCRIPT_DIR}/scripts/run_recovery_by_condition.py" \
  --project_root "${PROJECT_ROOT}" \
  --batch_name "${BATCH_NAME}" \
  --dry_run

echo
echo "[STEP 7] Build compact batch overview"
python "${SCRIPT_DIR}/scripts/build_compact_batch_overview.py" \
  --project_root "${PROJECT_ROOT}" \
  --batch_name "${BATCH_NAME}" \
  --top_n_routes 100

echo
echo "[STEP 8] Inspect failed cases"
python "${SCRIPT_DIR}/scripts/inspect_failed_cases.py" \
  --project_root "${PROJECT_ROOT}" \
  --batch_name "${BATCH_NAME}" \
  --tail_lines 120

echo
echo "[STEP 9] Export manual review cases and routes"
python "${SCRIPT_DIR}/scripts/export_manual_review_cases.py" \
  --project_root "${PROJECT_ROOT}" \
  --batch_name "${BATCH_NAME}" \
  --top_n_per_case "${TOP_N_PER_CASE}"

echo
echo "[STEP 10] Build final structure-route evidence table"
python "${SCRIPT_DIR}/scripts/build_final_structure_route_evidence_table.py" \
  --project_root "${PROJECT_ROOT}" \
  --batch_name "${BATCH_NAME}" \
  --top_n_per_case 3 \
  --use_high_medium_only

echo
echo "============================================================"
echo "[DONE] Export and dispatch finished"
echo "============================================================"

echo
echo "Main outputs:"
echo "  ${BATCH_OUT_DIR}/batch_recommended_routes_topn.csv"
echo "  ${BATCH_OUT_DIR}/batch_recommended_routes_topn.md"
echo "  ${BATCH_OUT_DIR}/batch_recommended_routes_topn_review_refined.csv"
echo "  ${BATCH_OUT_DIR}/batch_recommended_routes_topn_review_refined.md"
echo "  ${BATCH_OUT_DIR}/batch_route_refined_summary.csv"
echo "  ${BATCH_OUT_DIR}/batch_route_refined_summary.md"
echo "  ${BATCH_OUT_DIR}/route_splits/route_split_summary.md"
echo "  ${BATCH_OUT_DIR}/route_splits/route_recommended_high_medium.csv"
echo "  ${BATCH_OUT_DIR}/route_splits/route_recommended_high_medium.md"
echo "  ${BATCH_OUT_DIR}/recovery_plan/recovery_plan_summary.md"
echo "  ${BATCH_OUT_DIR}/recovery_plan/recovery_plan.csv"
echo "  ${BATCH_OUT_DIR}/recovery_plan/recovery_commands.sh"
echo "  ${BATCH_OUT_DIR}/recovery_runs/recovery_run_results.csv"
echo "  ${BATCH_OUT_DIR}/recovery_runs/recovery_run_results.md"
echo "  ${DISPATCH_DIR}/recovery_dispatch_summary.csv"
echo "  ${DISPATCH_DIR}/recovery_dispatch_summary.md"

echo
echo "Preview: recovery dispatch summary"
echo "------------------------------------------------------------"
if [[ -f "${DISPATCH_DIR}/recovery_dispatch_summary.md" ]]; then
  cat "${DISPATCH_DIR}/recovery_dispatch_summary.md"
else
  echo "[WARN] Missing ${DISPATCH_DIR}/recovery_dispatch_summary.md"
fi

echo
echo "Preview: refined route-level review summary"
echo "------------------------------------------------------------"
if [[ -f "${BATCH_OUT_DIR}/batch_review_refined_summary.md" ]]; then
  cat "${BATCH_OUT_DIR}/batch_review_refined_summary.md"
else
  echo "[WARN] Missing ${BATCH_OUT_DIR}/batch_review_refined_summary.md"
fi

echo
echo "Preview: route split summary"
echo "------------------------------------------------------------"
if [[ -f "${BATCH_OUT_DIR}/route_splits/route_split_summary.md" ]]; then
  cat "${BATCH_OUT_DIR}/route_splits/route_split_summary.md"
else
  echo "[WARN] Missing ${BATCH_OUT_DIR}/route_splits/route_split_summary.md"
fi

echo
echo "Preview: recovery plan summary"
echo "------------------------------------------------------------"
if [[ -f "${BATCH_OUT_DIR}/recovery_plan/recovery_plan_summary.md" ]]; then
  cat "${BATCH_OUT_DIR}/recovery_plan/recovery_plan_summary.md"
else
  echo "[WARN] Missing ${BATCH_OUT_DIR}/recovery_plan/recovery_plan_summary.md"
fi

echo
echo "Preview: recovery run results"
echo "------------------------------------------------------------"
if [[ -f "${BATCH_OUT_DIR}/recovery_runs/recovery_run_results.md" ]]; then
  cat "${BATCH_OUT_DIR}/recovery_runs/recovery_run_results.md"
else
  echo "[WARN] Missing ${BATCH_OUT_DIR}/recovery_runs/recovery_run_results.md"
fi

echo
echo "============================================================"
