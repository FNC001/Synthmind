#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/Users/wyc/SynPred}"
VERSION="benchmark_100_v34_real_stage3_expansion_from_v33_manifest"
TS="$(date +%Y%m%d_%H%M%S)"

OUT_DIR="$PROJECT_ROOT/outputs/$VERSION"
ARCHIVE_BASE="final_demo_snapshot_${VERSION}_${TS}"
ARCHIVE_PATH="$PROJECT_ROOT/outputs/${ARCHIVE_BASE}.tar.gz"

echo "============================================================"
echo "Archive V34 blocked release"
echo "PROJECT_ROOT = $PROJECT_ROOT"
echo "OUT_DIR      = $OUT_DIR"
echo "ARCHIVE      = $ARCHIVE_PATH"
echo "============================================================"

cd "$PROJECT_ROOT"

if [ ! -d "$OUT_DIR" ]; then
  echo "[ERROR] Missing OUT_DIR: $OUT_DIR"
  exit 1
fi

echo "[STEP 1] Check required V34 files"

REQ_FILES=(
  "$OUT_DIR/expanded_stage3_candidates_v34/v34_existing_stage3_target_check.md"
  "$OUT_DIR/expanded_stage3_candidates_v34/v34_existing_stage3_target_check.csv"
  "$OUT_DIR/audit_v34/v34_audit_summary.json"
  "$OUT_DIR/FINAL_REPORT_V34/FINAL_V34_METRIC_SUMMARY.md"
  "$OUT_DIR/FINAL_REPORT_V34/FINAL_BENCHMARK_100_V34_BLOCKED_REPORT.md"
)

for f in "${REQ_FILES[@]}"; do
  if [ ! -f "$f" ]; then
    echo "[ERROR] Missing required file: $f"
    exit 1
  fi
  echo "[OK] $f"
done

echo
echo "[STEP 2] Write ARCHIVE_INFO.txt"

cat > "$OUT_DIR/ARCHIVE_INFO.txt" <<EOF
Archive name: ${ARCHIVE_BASE}
Version: ${VERSION}
Created at: ${TS}
Project root: ${PROJECT_ROOT}

Status:
blocked_pending_real_stage3_generation_or_export

Interpretation:
V34 confirms that none of the V33 formula-exact MP expansion targets currently exist in the real V30/V32 Stage3 MDN/Flow candidate library.
This is a Stage3 library coverage problem, not an alignment-code failure.

Required next action:
Generate or export real Stage3 MDN/Flow condition candidates for the missing MP IDs, then rerun V34 merge and realignment.

Key files:
- FINAL_REPORT_V34/FINAL_BENCHMARK_100_V34_BLOCKED_REPORT.md
- FINAL_REPORT_V34/FINAL_V34_METRIC_SUMMARY.md
- audit_v34/v34_audit_summary.json
- expanded_stage3_candidates_v34/v34_existing_stage3_target_check.md
EOF

echo
echo "[STEP 3] Create tar.gz"

cd "$PROJECT_ROOT/outputs"
tar -czf "$ARCHIVE_PATH" "$VERSION"

echo
echo "[STEP 4] Create checksum"

cd "$PROJECT_ROOT/outputs"
shasum -a 256 "$(basename "$ARCHIVE_PATH")" > "$(basename "$ARCHIVE_PATH").sha256"

echo
echo "[STEP 5] Verify archive"

shasum -a 256 -c "$(basename "$ARCHIVE_PATH").sha256"

echo
echo "[STEP 6] List important archived files"

tar -tzf "$(basename "$ARCHIVE_PATH")" | grep -E "ARCHIVE_INFO.txt|FINAL_BENCHMARK_100_V34_BLOCKED_REPORT.md|FINAL_V34_METRIC_SUMMARY.md|v34_audit_summary.json|v34_existing_stage3_target_check.md" || true

echo
echo "[STEP 7] Archive size"

ls -lh "$ARCHIVE_PATH" "$ARCHIVE_PATH.sha256"

echo
echo "============================================================"
echo "[DONE] V34 blocked release archived."
echo "Archive:"
echo "$ARCHIVE_PATH"
echo "Checksum:"
echo "$ARCHIVE_PATH.sha256"
echo "============================================================"
