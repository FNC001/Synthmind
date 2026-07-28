#!/usr/bin/env bash
set -euo pipefail
STEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$STEP_DIR/../common.sh"
ALIGN="$RELEASE_ROOT/02_RAW_DATA/03_STRUCTURE_SYNTHESIS_ALIGNMENT/mp_synth_direct_aligned"
require_file "$ALIGN/direct_aligned_dataset.jsonl"
require_file "$ALIGN/unmatched_records.jsonl"
require_file "$ALIGN/all_candidates.jsonl"
announce "Frozen alignment checkpoints are present"
wc -l "$ALIGN/direct_aligned_dataset.jsonl" "$ALIGN/unmatched_records.jsonl" "$ALIGN/all_candidates.jsonl"

