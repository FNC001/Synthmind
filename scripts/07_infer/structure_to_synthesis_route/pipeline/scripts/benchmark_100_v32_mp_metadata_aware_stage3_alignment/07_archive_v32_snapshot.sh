#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/Users/wyc/SynPred}"
VERSION="benchmark_100_v32_mp_metadata_aware_stage3_alignment"
OUT_ROOT="$PROJECT_ROOT/outputs/$VERSION"

TS="$(date +%Y%m%d_%H%M%S)"
SNAPSHOT_DIR="$PROJECT_ROOT/outputs/final_demo_snapshot_${VERSION}_${TS}"
ARCHIVE="$SNAPSHOT_DIR.tar.gz"

echo "============================================================"
echo "Archive Benchmark-100 V32 snapshot"
echo "PROJECT_ROOT = $PROJECT_ROOT"
echo "OUT_ROOT     = $OUT_ROOT"
echo "SNAPSHOT_DIR = $SNAPSHOT_DIR"
echo "ARCHIVE      = $ARCHIVE"
echo "============================================================"

if [ ! -d "$OUT_ROOT" ]; then
  echo "[ERROR] OUT_ROOT not found: $OUT_ROOT"
  exit 1
fi

mkdir -p "$SNAPSHOT_DIR"

echo "[STEP 1] Copy V32 outputs"
cp -R "$OUT_ROOT" "$SNAPSHOT_DIR/$VERSION"

echo "[STEP 2] Copy final report files to snapshot root"
mkdir -p "$SNAPSHOT_DIR/FINAL_REPORT_V32"

cp "$OUT_ROOT/FINAL_REPORT_V32/FINAL_BENCHMARK_100_V32_REPORT.md" \
   "$SNAPSHOT_DIR/FINAL_REPORT_V32/" 2>/dev/null || true

cp "$OUT_ROOT/FINAL_REPORT_V32/FINAL_V32_METRIC_SUMMARY.md" \
   "$SNAPSHOT_DIR/FINAL_REPORT_V32/" 2>/dev/null || true

cp "$OUT_ROOT/FINAL_REPORT_V32/FINAL_V32_METRIC_SUMMARY.csv" \
   "$SNAPSHOT_DIR/FINAL_REPORT_V32/" 2>/dev/null || true

cp "$OUT_ROOT/FINAL_BENCHMARK_100_V32_MASTER_INDEX.md" \
   "$SNAPSHOT_DIR/" 2>/dev/null || true

echo "[STEP 3] Write archive info"

cat > "$SNAPSHOT_DIR/ARCHIVE_INFO.txt" <<EOF
SynPred Structure-to-Synthesis Route Inference Final Demo Snapshot

Version:
$VERSION

Timestamp:
$TS

Main purpose:
V32 upgrades V31 by attaching Materials Project formula/elements/family metadata to real Stage3 MDN/Flow condition candidates, enabling metadata-aware external-case Stage3 alignment.

Key interpretation:
V32 is intentionally conservative. It validates chemistry-aware matches, flags weak matches for review, and leaves unmatched cases unaligned rather than overclaiming condition support.

Key metrics:
- MP metadata rows: 49283
- Stage3 candidate rows: 158145
- Stage3 unique MP ids: 249
- Metadata coverage ratio: 1.0
- External cases: 20
- Aligned cases: 19
- Unaligned cases: 1
- Unaligned case: external_case_015 / NaCl
- Bad or review aligned cases: 8
- Formula exact matches: 7
- Alignment mode: metadata_aware_formula_elements_family_condition
- Audit status: pass_with_review

Main files:
- FINAL_REPORT_V32/FINAL_BENCHMARK_100_V32_REPORT.md
- FINAL_REPORT_V32/FINAL_V32_METRIC_SUMMARY.md
- $VERSION/metadata_aware_alignment_v32/v32_metadata_aware_external_stage3_alignment_summary.md
- $VERSION/audit_metadata_aware_alignment_v32/v32_metadata_aware_alignment_audit_summary.md
- $VERSION/audit_metadata_aware_alignment_v32/v32_metadata_aware_alignment_bad_or_review_cases.md
- $VERSION/audit_metadata_aware_alignment_v32/v32_metadata_aware_alignment_unaligned_cases.md
EOF

echo "[STEP 4] Create tar.gz"
cd "$PROJECT_ROOT/outputs"
tar -czf "$(basename "$ARCHIVE")" "$(basename "$SNAPSHOT_DIR")"

echo "[STEP 5] Create sha256"
shasum -a 256 "$(basename "$ARCHIVE")" > "$(basename "$ARCHIVE").sha256"

echo
echo "============================================================"
echo "[DONE] V32 snapshot archived"
echo "Snapshot dir:"
echo "$SNAPSHOT_DIR"
echo "Archive:"
echo "$ARCHIVE"
echo "SHA256:"
echo "$ARCHIVE.sha256"
echo "============================================================"
