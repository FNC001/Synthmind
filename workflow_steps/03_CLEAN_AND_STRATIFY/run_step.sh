#!/usr/bin/env bash
set -euo pipefail
STEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$STEP_DIR/../common.sh"
require_dir "$RELEASE_ROOT/02_RAW_DATA/04_STRICT_FILTER_OUTPUTS/01_ALIGNMENT_SPLIT_REMOTE_FINAL"
require_dir "$RELEASE_ROOT/02_RAW_DATA/04_STRICT_FILTER_OUTPUTS/02_LEGACY_RAW_MERGE_BASE"
require_dir "$RELEASE_ROOT/03_CLEANED_AND_MERGED_DATA/02_MERGED_WITH_STRUCTURES/merged_20260609_with_structures"
require_dir "$RELEASE_ROOT/03_CLEANED_AND_MERGED_DATA/05_UNITS_NORMALIZED/structdesc_refined_route_unified_20260609_units_normalized"
announce "Clean/merge checkpoints and both strict branches are present"

