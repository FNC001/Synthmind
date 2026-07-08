#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
INFER_NAME="${1:-demo_poscar_test}"
PYTHON_BIN="${PYTHON:-python3}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/pipeline/run_pipeline.py" \
  --config "${PROJECT_ROOT}/configs/full_route.yaml" \
  --project_root "${PROJECT_ROOT}" \
  --infer_name "${INFER_NAME}"
