#!/usr/bin/env bash
set -euo pipefail
STEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$STEP_DIR/../common.sh"
require_data_root 00_OVERVIEW_AND_MANIFEST 02_RAW_DATA 11_TESTS_AND_AUDITS
announce_context
OUT="$WORK_ROOT/01_validate_raw_data/release_audit.json"
mkdir -p "$(dirname "$OUT")"
announce "Running package and raw-data audit"
if test "${MODE:-validate}" = full_audit; then
  "$PYTHON" "$STEP_DIR/build_structure_scope_indexes.py" "$RELEASE_ROOT" \
    --output-dir "$WORK_ROOT/01_validate_raw_data/structure_scope_indexes" \
    --audit-path "$WORK_ROOT/01_validate_raw_data/STRUCTURE_SCOPE_COVERAGE_AUDIT.json"
fi
"$PYTHON" "$DATA_ROOT/11_TESTS_AND_AUDITS/verify_release.py" \
  --release-root "$DATA_ROOT" --output "$OUT"
announce "Audit report: $OUT"
