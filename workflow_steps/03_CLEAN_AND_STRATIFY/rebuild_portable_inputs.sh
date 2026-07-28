#!/usr/bin/env bash
set -euo pipefail
STEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$STEP_DIR/../common.sh"
STANDARDIZER="$STEP_DIR/standardize_jsonl_paths.py"
PORTABLE_OUTPUT_ROOT="${PORTABLE_OUTPUT_ROOT:-$WORK_ROOT/03_clean_and_stratify/portable}"

"$PYTHON" "$STANDARDIZER" \
  --release-root "$RELEASE_ROOT" \
  --output-dir "$PORTABLE_OUTPUT_ROOT/03_CLEANED_AND_MERGED_DATA/00_PORTABLE_STANDARDIZED_INPUTS" \
  --input direct_aligned=02_RAW_DATA/03_STRUCTURE_SYNTHESIS_ALIGNMENT/mp_synth_direct_aligned/direct_aligned_dataset.jsonl \
  --input alignment_exact=02_RAW_DATA/04_STRICT_FILTER_OUTPUTS/01_ALIGNMENT_SPLIT_REMOTE_FINAL/strict_exact_only.jsonl \
  --input alignment_parent=02_RAW_DATA/04_STRICT_FILTER_OUTPUTS/01_ALIGNMENT_SPLIT_REMOTE_FINAL/strict_parent_aug.jsonl \
  --input legacy_exact=02_RAW_DATA/04_STRICT_FILTER_OUTPUTS/02_LEGACY_RAW_MERGE_BASE/strict_exact_only.jsonl \
  --input legacy_parent=02_RAW_DATA/04_STRICT_FILTER_OUTPUTS/02_LEGACY_RAW_MERGE_BASE/strict_parent_aug.jsonl \
  --input new_20260608_aligned=02_RAW_DATA/06_NEW_20260608_DIRECT_ALIGNMENT/direct_aligned_json_20260608/codex_final_database_aligned_full.jsonl \
  --input merged_train_ready=03_CLEANED_AND_MERGED_DATA/02_MERGED_WITH_STRUCTURES/merged_20260609_with_structures/strict_matched_train_ready.jsonl

"$PYTHON" "$STANDARDIZER" \
  --release-root "$RELEASE_ROOT" \
  --output-dir "$PORTABLE_OUTPUT_ROOT/04_SPLITS/00_PORTABLE_SPLITS/01_BASE_GROUP_SPLIT" \
  --input stage2_train=04_SPLITS/01_BASE_GROUP_SPLIT/structdesc_splits/stage2_train.jsonl \
  --input stage2_val=04_SPLITS/01_BASE_GROUP_SPLIT/structdesc_splits/stage2_val.jsonl \
  --input stage2_test=04_SPLITS/01_BASE_GROUP_SPLIT/structdesc_splits/stage2_test.jsonl \
  --input stage2_gold_train_holdout=04_SPLITS/01_BASE_GROUP_SPLIT/structdesc_splits/stage2_gold_train_holdout.jsonl \
  --input stage3_train=04_SPLITS/01_BASE_GROUP_SPLIT/structdesc_splits/stage3_train.jsonl \
  --input stage3_val=04_SPLITS/01_BASE_GROUP_SPLIT/structdesc_splits/stage3_val.jsonl \
  --input stage3_test=04_SPLITS/01_BASE_GROUP_SPLIT/structdesc_splits/stage3_test.jsonl \
  --input stage3_gold_train_holdout=04_SPLITS/01_BASE_GROUP_SPLIT/structdesc_splits/stage3_gold_train_holdout.jsonl

"$PYTHON" "$STANDARDIZER" \
  --release-root "$RELEASE_ROOT" \
  --output-dir "$PORTABLE_OUTPUT_ROOT/04_SPLITS/00_PORTABLE_SPLITS/02_ROUTE_GROUP_SPLIT" \
  --input stage2_train=04_SPLITS/02_ROUTE_GROUP_SPLIT/structdesc_splits_route_unified_20260609_units_normalized/stage2_train.jsonl \
  --input stage2_val=04_SPLITS/02_ROUTE_GROUP_SPLIT/structdesc_splits_route_unified_20260609_units_normalized/stage2_val.jsonl \
  --input stage2_test=04_SPLITS/02_ROUTE_GROUP_SPLIT/structdesc_splits_route_unified_20260609_units_normalized/stage2_test.jsonl \
  --input stage2_gold_train_holdout=04_SPLITS/02_ROUTE_GROUP_SPLIT/structdesc_splits_route_unified_20260609_units_normalized/stage2_gold_train_holdout.jsonl \
  --input stage3_train=04_SPLITS/02_ROUTE_GROUP_SPLIT/structdesc_splits_route_unified_20260609_units_normalized/stage3_train.jsonl \
  --input stage3_val=04_SPLITS/02_ROUTE_GROUP_SPLIT/structdesc_splits_route_unified_20260609_units_normalized/stage3_val.jsonl \
  --input stage3_test=04_SPLITS/02_ROUTE_GROUP_SPLIT/structdesc_splits_route_unified_20260609_units_normalized/stage3_test.jsonl \
  --input stage3_gold_train_holdout=04_SPLITS/02_ROUTE_GROUP_SPLIT/structdesc_splits_route_unified_20260609_units_normalized/stage3_gold_train_holdout.jsonl

announce "Portable JSONL and split manifests rebuilt"
