#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# SynPred Inference Pipeline
#
# Purpose:
#   Inference-only pipeline.
#   It does NOT prepare training data.
#   It does NOT train models.
#
# Main usage:
#   1. Use existing conditioned_x_csv and run Stage3 flow inference.
#   2. Optionally run the original integrated pipeline run_pipeline.py.
#
# Usage:
#   bash scripts/run_inference_pipeline.sh [PROJECT_ROOT] [DEVICE]
#
# Examples:
#   bash scripts/run_inference_pipeline.sh /Users/wyc/SynPred cpu
#
#   FORCE=1 \
#   CONDITIONED_X_CSV=/Users/wyc/SynPred/data/interim/infer/case_000001_batch_001_poscars_POSCAR/stage3_conditioned_x_fallback_retrieval_baseline_element_reranked.csv \
#   bash scripts/run_inference_pipeline.sh /Users/wyc/SynPred cpu
#
#   FORCE=1 CONDITIONED_X_ROOT=/Users/wyc/SynPred/data/interim/infer \
#   bash scripts/run_inference_pipeline.sh /Users/wyc/SynPred cpu
#
# Optional env vars:
#   FORCE=0|1
#   STRICT_INFER=0|1
#   RUN_PREFLIGHT=1
#   RUN_COMBINED_INFER=1
#   RUN_ORIGINAL_PIPELINE=0
#   RUN_EXPORT=1
#
# Input selection:
#   CONDITIONED_X_CSV=/path/to/one/stage3_conditioned_x_*.csv
#   CONDITIONED_X_ROOT=/path/to/root/search/dir
#
# Flow settings:
#   TOP_K_CONDITIONS=5
#   N_FLOW_SAMPLES=32
#   SEED=42
#
# Original pipeline settings:
#   INFER_NAME=demo_poscar_test
#   ORIGINAL_PIPELINE_START_FROM=preflight
# ============================================================

PROJECT_ROOT="${1:-/Users/wyc/SynPred}"
DEVICE="${2:-${DEVICE:-cpu}}"
FORCE="${FORCE:-0}"
STRICT_INFER="${STRICT_INFER:-0}"

PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
SCRIPTS="${PROJECT_ROOT}/scripts"

RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
RUN_COMBINED_INFER="${RUN_COMBINED_INFER:-1}"
RUN_ORIGINAL_PIPELINE="${RUN_ORIGINAL_PIPELINE:-0}"
RUN_EXPORT="${RUN_EXPORT:-1}"

INFER_NAME="${INFER_NAME:-demo_poscar_test}"
ORIGINAL_PIPELINE_START_FROM="${ORIGINAL_PIPELINE_START_FROM:-preflight}"

PIPELINE_OUT_DIR="${PIPELINE_OUT_DIR:-${PROJECT_ROOT}/outputs/inference_pipeline}"
COMBINED_OUT_DIR="${COMBINED_OUT_DIR:-${PIPELINE_OUT_DIR}/combined}"
EXPORT_OUT_DIR="${EXPORT_OUT_DIR:-${PIPELINE_OUT_DIR}/export}"

LOG_DIR="${PROJECT_ROOT}/outputs/logs/inference_pipeline/$(date +%Y%m%d_%H%M%S)"
RESUME_DIR="${PROJECT_ROOT}/outputs/resume/inference_pipeline"

mkdir -p "${PIPELINE_OUT_DIR}" "${COMBINED_OUT_DIR}" "${EXPORT_OUT_DIR}" "${LOG_DIR}" "${RESUME_DIR}"

SUMMARY_DIR="${PROJECT_ROOT}/outputs/stage2_stage3_benchmark_summary"
RECOMMENDED_CONFIG_SH="${SUMMARY_DIR}/recommended_inference_config.sh"
RECOMMENDED_CONFIG_JSON="${SUMMARY_DIR}/recommended_inference_config.json"

STAGE3_SCHEMA_JSON="${STAGE3_SCHEMA_JSON:-${PROJECT_ROOT}/data/interim/generative/stage3_condition_dataset/hybrid_mixed_v1/schema.json}"

COMBINED_INFER_SCRIPT="${COMBINED_INFER_SCRIPT:-${SCRIPTS}/07_infer/structure_to_synthesis_route/pipeline/src/13_run_stage3_infer_mixture_flow_conditioned.py}"

# Important:
# The working script you verified is train_condition_mixture_flow_mixed.py,
# not train_condition_residual_flow_mixed.py.
FLOW_SCRIPT="${FLOW_SCRIPT:-${SCRIPTS}/04_train/stage3/mixed/train_condition_mixture_flow_mixed.py}"

CONDITIONED_X_ROOT="${CONDITIONED_X_ROOT:-${PROJECT_ROOT}/data/interim/infer}"
CONDITIONED_X_CSV="${CONDITIONED_X_CSV:-}"

TOP_K_CONDITIONS="${TOP_K_CONDITIONS:-5}"
N_FLOW_SAMPLES="${N_FLOW_SAMPLES:-32}"
SEED="${SEED:-42}"

# ============================================================
# Load recommended model dirs
# ============================================================

if [[ -f "${RECOMMENDED_CONFIG_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${RECOMMENDED_CONFIG_SH}"
  echo "[Info] Loaded recommended inference config:"
  echo "  ${RECOMMENDED_CONFIG_SH}"
else
  echo "[WARN] recommended_inference_config.sh not found:"
  echo "  ${RECOMMENDED_CONFIG_SH}"
  echo "[WARN] Falling back to default benchmark dirs."

  export SYNPRED_STAGE2_MODEL="${SYNPRED_STAGE2_MODEL:-stage2_gflownet_element_bias}"
  export SYNPRED_STAGE2_RUN_DIR="${SYNPRED_STAGE2_RUN_DIR:-${PROJECT_ROOT}/runs/stage2/gflownet_benchmark_element_bias_mild_v1}"

  export SYNPRED_STAGE3_TOP1_MODEL="${SYNPRED_STAGE3_TOP1_MODEL:-stage3_residual_mdn}"
  export SYNPRED_STAGE3_TOP1_RUN_DIR="${SYNPRED_STAGE3_TOP1_RUN_DIR:-${PROJECT_ROOT}/runs/stage3/benchmark_residual_mdn_mixed_v1}"

  export SYNPRED_STAGE3_GENERATIVE_MODEL="${SYNPRED_STAGE3_GENERATIVE_MODEL:-stage3_residual_flow_mixed}"
  export SYNPRED_STAGE3_GENERATIVE_RUN_DIR="${SYNPRED_STAGE3_GENERATIVE_RUN_DIR:-${PROJECT_ROOT}/runs/stage3/benchmark_residual_flow_mixed_v1}"
fi

SYNPRED_STAGE2_MODEL="${SYNPRED_STAGE2_MODEL:-stage2_gflownet_element_bias}"
SYNPRED_STAGE2_RUN_DIR="${SYNPRED_STAGE2_RUN_DIR:-${PROJECT_ROOT}/runs/stage2/gflownet_benchmark_element_bias_mild_v1}"

SYNPRED_STAGE3_TOP1_MODEL="${SYNPRED_STAGE3_TOP1_MODEL:-stage3_residual_mdn}"
SYNPRED_STAGE3_TOP1_RUN_DIR="${SYNPRED_STAGE3_TOP1_RUN_DIR:-${PROJECT_ROOT}/runs/stage3/benchmark_residual_mdn_mixed_v1}"

SYNPRED_STAGE3_GENERATIVE_MODEL="${SYNPRED_STAGE3_GENERATIVE_MODEL:-stage3_residual_flow_mixed}"
SYNPRED_STAGE3_GENERATIVE_RUN_DIR="${SYNPRED_STAGE3_GENERATIVE_RUN_DIR:-${PROJECT_ROOT}/runs/stage3/benchmark_residual_flow_mixed_v1}"

FLOW_CKPT="${FLOW_CKPT:-auto}"

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
    echo "Check log:"
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

check_optional_file() {
  local f="$1"
  if [[ ! -f "${f}" ]]; then
    echo "[WARN-optional-missing] ${f}"
    return 1
  fi
  echo "[OK-optional] ${f}"
  return 0
}

resolve_flow_ckpt() {
  if [[ "${FLOW_CKPT}" != "auto" && -n "${FLOW_CKPT}" ]]; then
    if [[ ! -f "${FLOW_CKPT}" ]]; then
      echo "[ERROR] FLOW_CKPT was specified but does not exist:"
      echo "  ${FLOW_CKPT}"
      exit 1
    fi
    echo "${FLOW_CKPT}"
    return 0
  fi

  local candidates=(
    "${SYNPRED_STAGE3_GENERATIVE_RUN_DIR}/best_stage3_residual_flow_mixed.pt"
    "${SYNPRED_STAGE3_GENERATIVE_RUN_DIR}/best_stage3_condition_mixture_flow_mixed.pt"
    "${SYNPRED_STAGE3_GENERATIVE_RUN_DIR}/best_stage3_condition_residual_flow_mixed.pt"
    "${PROJECT_ROOT}/runs/stage3/benchmark_residual_flow_mixed_v1/best_stage3_residual_flow_mixed.pt"
    "${PROJECT_ROOT}/runs/stage3/condition_mixture_flow_hybrid_mixed_v1_yset_conditioned_v2/best_stage3_condition_mixture_flow_mixed.pt"
  )

  local c
  for c in "${candidates[@]}"; do
    if [[ -f "${c}" ]]; then
      echo "${c}"
      return 0
    fi
  done

  echo "[ERROR] Could not auto-resolve FLOW_CKPT." >&2
  echo "[ERROR] Checked:" >&2
  for c in "${candidates[@]}"; do
    echo "  ${c}" >&2
  done
  exit 1
}

write_config_snapshot() {
  local resolved_flow_ckpt="$1"

  python - <<PY
import json
from pathlib import Path

payload = {
    "project_root": "${PROJECT_ROOT}",
    "device": "${DEVICE}",
    "force": "${FORCE}",
    "strict_infer": "${STRICT_INFER}",
    "pipeline_out_dir": "${PIPELINE_OUT_DIR}",
    "combined_out_dir": "${COMBINED_OUT_DIR}",
    "export_out_dir": "${EXPORT_OUT_DIR}",
    "recommended_config_sh": "${RECOMMENDED_CONFIG_SH}",
    "recommended_config_json": "${RECOMMENDED_CONFIG_JSON}",
    "stage2_model": "${SYNPRED_STAGE2_MODEL}",
    "stage2_run_dir": "${SYNPRED_STAGE2_RUN_DIR}",
    "stage3_top1_model": "${SYNPRED_STAGE3_TOP1_MODEL}",
    "stage3_top1_run_dir": "${SYNPRED_STAGE3_TOP1_RUN_DIR}",
    "stage3_generative_model": "${SYNPRED_STAGE3_GENERATIVE_MODEL}",
    "stage3_generative_run_dir": "${SYNPRED_STAGE3_GENERATIVE_RUN_DIR}",
    "combined_infer_script": "${COMBINED_INFER_SCRIPT}",
    "flow_script": "${FLOW_SCRIPT}",
    "flow_ckpt": "${resolved_flow_ckpt}",
    "stage3_schema_json": "${STAGE3_SCHEMA_JSON}",
    "conditioned_x_csv": "${CONDITIONED_X_CSV}",
    "conditioned_x_root": "${CONDITIONED_X_ROOT}",
    "top_k_conditions": "${TOP_K_CONDITIONS}",
    "n_flow_samples": "${N_FLOW_SAMPLES}",
    "seed": "${SEED}",
    "run_combined_infer": "${RUN_COMBINED_INFER}",
    "run_original_pipeline": "${RUN_ORIGINAL_PIPELINE}",
    "infer_name": "${INFER_NAME}",
    "original_pipeline_start_from": "${ORIGINAL_PIPELINE_START_FROM}",
}
out = Path("${PIPELINE_OUT_DIR}") / "inference_config_snapshot.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
print(f"[SAVED] {out}")
PY
}

discover_conditioned_x_files() {
  local list_file="$1"

  mkdir -p "$(dirname "${list_file}")"
  : > "${list_file}"

  if [[ -n "${CONDITIONED_X_CSV}" ]]; then
    if [[ ! -f "${CONDITIONED_X_CSV}" ]]; then
      echo "[ERROR] CONDITIONED_X_CSV does not exist:"
      echo "  ${CONDITIONED_X_CSV}"
      exit 1
    fi
    echo "${CONDITIONED_X_CSV}" > "${list_file}"
    return 0
  fi

  if [[ ! -d "${CONDITIONED_X_ROOT}" ]]; then
    echo "[ERROR] CONDITIONED_X_ROOT does not exist:"
    echo "  ${CONDITIONED_X_ROOT}"
    exit 1
  fi

  find "${CONDITIONED_X_ROOT}" \
    -type f \
    \( -name "stage3_conditioned_x_fallback_retrieval_baseline_element_reranked.csv" \
       -o -name "*conditioned*x*.csv" \
       -o -name "*stage3*conditioned*.csv" \) \
    | sort > "${list_file}"

  if [[ ! -s "${list_file}" ]]; then
    echo "[ERROR] No conditioned_x_csv files found under:"
    echo "  ${CONDITIONED_X_ROOT}"
    echo
    echo "Set one explicitly, for example:"
    echo "  CONDITIONED_X_CSV=/path/to/stage3_conditioned_x_fallback_retrieval_baseline_element_reranked.csv"
    exit 1
  fi
}

safe_case_name_from_csv() {
  local csv="$1"
  local parent
  parent="$(basename "$(dirname "${csv}")")"

  if [[ "${parent}" == "." || -z "${parent}" ]]; then
    parent="$(basename "${csv}" .csv)"
  fi

  echo "${parent}" | tr ' /' '__'
}

write_minimal_export() {
  python - <<PY
import json
from pathlib import Path
import pandas as pd

combined = Path("${COMBINED_OUT_DIR}")
export = Path("${EXPORT_OUT_DIR}")
export.mkdir(parents=True, exist_ok=True)

rows = []
for csv in sorted(combined.glob("*/test_candidates_flat.csv")):
    try:
        df = pd.read_csv(csv)
        df.insert(0, "case_name", csv.parent.name)
        df.insert(1, "source_file", str(csv))
        rows.append(df)
    except Exception as e:
        print(f"[WARN] failed to read {csv}: {e}")

manifest = {
    "combined_out_dir": str(combined),
    "export_out_dir": str(export),
    "n_case_dirs": len([p for p in combined.iterdir() if p.is_dir()]) if combined.exists() else 0,
    "note": "Minimal export generated by run_inference_pipeline.sh."
}

if rows:
    out_csv = export / "all_test_candidates_flat.csv"
    all_df = pd.concat(rows, ignore_index=True)
    all_df.to_csv(out_csv, index=False)
    manifest["all_candidates_csv"] = str(out_csv)
    manifest["n_rows"] = int(len(all_df))
    print(f"[SAVED] {out_csv}")
else:
    manifest["all_candidates_csv"] = ""
    manifest["n_rows"] = 0
    print("[WARN] No test_candidates_flat.csv files found for export.")

manifest_path = export / "manifest.json"
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
print(f"[SAVED] {manifest_path}")
PY
}

# ============================================================
# Header
# ============================================================

RESOLVED_FLOW_CKPT="$(resolve_flow_ckpt)"

print_section "SynPred Inference Pipeline"

echo "PROJECT_ROOT          = ${PROJECT_ROOT}"
echo "DEVICE                = ${DEVICE}"
echo "FORCE                 = ${FORCE}"
echo "STRICT_INFER          = ${STRICT_INFER}"
echo "LOG_DIR               = ${LOG_DIR}"
echo "RESUME_DIR            = ${RESUME_DIR}"
echo
echo "Switches:"
echo "  RUN_PREFLIGHT        = ${RUN_PREFLIGHT}"
echo "  RUN_COMBINED_INFER   = ${RUN_COMBINED_INFER}"
echo "  RUN_ORIGINAL_PIPELINE= ${RUN_ORIGINAL_PIPELINE}"
echo "  RUN_EXPORT           = ${RUN_EXPORT}"
echo
echo "Recommended models:"
echo "  Stage2 model         = ${SYNPRED_STAGE2_MODEL}"
echo "  Stage2 run dir       = ${SYNPRED_STAGE2_RUN_DIR}"
echo "  Stage3 top1 model    = ${SYNPRED_STAGE3_TOP1_MODEL}"
echo "  Stage3 top1 run dir  = ${SYNPRED_STAGE3_TOP1_RUN_DIR}"
echo "  Stage3 gen model     = ${SYNPRED_STAGE3_GENERATIVE_MODEL}"
echo "  Stage3 gen run dir   = ${SYNPRED_STAGE3_GENERATIVE_RUN_DIR}"
echo
echo "Combined inference:"
echo "  COMBINED_INFER_SCRIPT = ${COMBINED_INFER_SCRIPT}"
echo "  FLOW_SCRIPT           = ${FLOW_SCRIPT}"
echo "  FLOW_CKPT             = ${RESOLVED_FLOW_CKPT}"
echo "  STAGE3_SCHEMA_JSON    = ${STAGE3_SCHEMA_JSON}"
echo "  CONDITIONED_X_CSV     = ${CONDITIONED_X_CSV:-<auto-discover>}"
echo "  CONDITIONED_X_ROOT    = ${CONDITIONED_X_ROOT}"
echo "  TOP_K_CONDITIONS      = ${TOP_K_CONDITIONS}"
echo "  N_FLOW_SAMPLES        = ${N_FLOW_SAMPLES}"
echo "  SEED                  = ${SEED}"
echo
echo "Outputs:"
echo "  PIPELINE_OUT_DIR      = ${PIPELINE_OUT_DIR}"
echo "  COMBINED_OUT_DIR      = ${COMBINED_OUT_DIR}"
echo "  EXPORT_OUT_DIR        = ${EXPORT_OUT_DIR}"
echo
echo "Original pipeline:"
echo "  INFER_NAME            = ${INFER_NAME}"
echo "  START_FROM            = ${ORIGINAL_PIPELINE_START_FROM}"

# ============================================================
# Preflight
# ============================================================

step_preflight() {
  check_dir "${PROJECT_ROOT}"
  check_dir "${SCRIPTS}"

  check_dir "${SYNPRED_STAGE2_RUN_DIR}"
  check_dir "${SYNPRED_STAGE3_TOP1_RUN_DIR}"
  check_dir "${SYNPRED_STAGE3_GENERATIVE_RUN_DIR}"

  check_optional_file "${RECOMMENDED_CONFIG_JSON}" || true
  check_optional_file "${RECOMMENDED_CONFIG_SH}" || true

  if [[ "${RUN_COMBINED_INFER}" == "1" ]]; then
    check_file "${COMBINED_INFER_SCRIPT}"
    check_file "${FLOW_SCRIPT}"
    check_file "${RESOLVED_FLOW_CKPT}"
    check_file "${STAGE3_SCHEMA_JSON}"
  fi

  write_config_snapshot "${RESOLVED_FLOW_CKPT}"
}

if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
  run_step preflight "Preflight inference inputs and model files" step_preflight
fi

# ============================================================
# Combined inference
# ============================================================

step_combined_infer() {
  check_file "${COMBINED_INFER_SCRIPT}"
  check_file "${FLOW_SCRIPT}"
  check_file "${RESOLVED_FLOW_CKPT}"
  check_file "${STAGE3_SCHEMA_JSON}"

  local list_file="${COMBINED_OUT_DIR}/conditioned_x_files.txt"
  discover_conditioned_x_files "${list_file}"

  local n_files
  n_files="$(wc -l < "${list_file}" | tr -d ' ')"
  echo "[Info] Conditioned_x files to process: ${n_files}"
  echo "[SAVED] ${list_file}"

  local summary_jsonl="${COMBINED_OUT_DIR}/combined_summary.jsonl"
  : > "${summary_jsonl}"

  local csv
  while IFS= read -r csv; do
    [[ -z "${csv}" ]] && continue

    local case_name
    case_name="$(safe_case_name_from_csv "${csv}")"

    local out_dir="${COMBINED_OUT_DIR}/${case_name}"
    mkdir -p "${out_dir}"

    echo
    echo "------------------------------------------------------------"
    echo "[Combined] case_name = ${case_name}"
    echo "[Combined] csv       = ${csv}"
    echo "[Combined] out_dir   = ${out_dir}"
    echo "------------------------------------------------------------"

    python "${COMBINED_INFER_SCRIPT}" \
      --conditioned_x_csv "${csv}" \
      --schema_json "${STAGE3_SCHEMA_JSON}" \
      --flow_ckpt "${RESOLVED_FLOW_CKPT}" \
      --flow_script "${FLOW_SCRIPT}" \
      --output_dir "${out_dir}" \
      --top_k_conditions "${TOP_K_CONDITIONS}" \
      --n_flow_samples "${N_FLOW_SAMPLES}" \
      --device "${DEVICE}" \
      --seed "${SEED}"

    python - <<PY >> "${summary_jsonl}"
import json
from pathlib import Path

case_name = "${case_name}"
out_dir = Path("${out_dir}")
payload = {
    "case_name": case_name,
    "conditioned_x_csv": "${csv}",
    "out_dir": str(out_dir),
    "candidate_summary": str(out_dir / "candidate_summary.json"),
    "jsonl": str(out_dir / "test_candidates.jsonl"),
    "flat_csv": str(out_dir / "test_candidates_flat.csv"),
    "debug_parent_candidates": str(out_dir / "debug_parent_candidates.csv"),
}
print(json.dumps(payload, ensure_ascii=False))
PY

  done < "${list_file}"

  python - <<PY
import json
from pathlib import Path

jsonl = Path("${summary_jsonl}")
out = Path("${COMBINED_OUT_DIR}") / "combined_summary.json"

items = []
if jsonl.exists():
    for line in jsonl.read_text().splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))

out.write_text(json.dumps({
    "n_cases": len(items),
    "items": items,
}, ensure_ascii=False, indent=2))

print(f"[SAVED] {out}")
PY
}

if [[ "${RUN_COMBINED_INFER}" == "1" ]]; then
  run_step combined_infer "Combined inference: conditioned precursor candidates + Stage3 Flow conditions" step_combined_infer
fi

# ============================================================
# Optional original pipeline
# ============================================================

step_original_pipeline() {
  local pipeline_dir="${SCRIPTS}/07_infer/structure_to_synthesis_route/pipeline"
  local config="${pipeline_dir}/configs/full_route_stage3.yaml"
  local runner="${pipeline_dir}/run_pipeline.py"

  check_file "${runner}"
  check_file "${config}"

  cd "${pipeline_dir}"

  python "${runner}" \
    --config "${config}" \
    --infer_name "${INFER_NAME}" \
    --project_root "${PROJECT_ROOT}" \
    --start_from "${ORIGINAL_PIPELINE_START_FROM}"
}

if [[ "${RUN_ORIGINAL_PIPELINE}" == "1" ]]; then
  run_step original_pipeline "Original integrated route inference pipeline" step_original_pipeline
fi

# ============================================================
# Export
# ============================================================

step_export() {
  write_minimal_export
}

if [[ "${RUN_EXPORT}" == "1" ]]; then
  run_step export "Export inference results" step_export
fi

# ============================================================
# Final summary
# ============================================================

print_section "INFERENCE PIPELINE COMPLETE"

echo "Logs:"
echo "  ${LOG_DIR}"
echo
echo "Config snapshot:"
echo "  ${PIPELINE_OUT_DIR}/inference_config_snapshot.json"
echo
echo "Combined output:"
echo "  ${COMBINED_OUT_DIR}"
echo "  ${COMBINED_OUT_DIR}/combined_summary.json"
echo
echo "Export output:"
echo "  ${EXPORT_OUT_DIR}"
echo
echo "Useful commands:"
echo
echo "  Run one conditioned_x_csv:"
echo "    FORCE=1 CONDITIONED_X_CSV=/path/to/stage3_conditioned_x_fallback_retrieval_baseline_element_reranked.csv bash scripts/run_inference_pipeline.sh ${PROJECT_ROOT} ${DEVICE}"
echo
echo "  Run all discovered conditioned_x_csv files:"
echo "    FORCE=1 CONDITIONED_X_ROOT=${CONDITIONED_X_ROOT} bash scripts/run_inference_pipeline.sh ${PROJECT_ROOT} ${DEVICE}"
echo
echo "  Check only:"
echo "    FORCE=1 RUN_COMBINED_INFER=0 RUN_ORIGINAL_PIPELINE=0 RUN_EXPORT=0 bash scripts/run_inference_pipeline.sh ${PROJECT_ROOT} ${DEVICE}"
echo
echo "  Run original integrated pipeline instead:"
echo "    FORCE=1 RUN_COMBINED_INFER=0 RUN_ORIGINAL_PIPELINE=1 bash scripts/run_inference_pipeline.sh ${PROJECT_ROOT} ${DEVICE}"
echo
echo "============================================================"
echo "[DONE] SynPred inference pipeline completed"
echo "============================================================"
