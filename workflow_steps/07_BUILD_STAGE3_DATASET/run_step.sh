#!/usr/bin/env bash
set -euo pipefail
STEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$STEP_DIR/../common.sh"
SOURCE="${STAGE3_SOURCE_DIR:-$DATA_ROOT/06_TRAIN_READY_DATA/07_STAGE3_CHEM_CHECKED_CORE/core_methods_v5_20260610}"
SCHEMA="${STAGE3_SOURCE_SCHEMA:-$DATA_ROOT/06_TRAIN_READY_DATA/07_STAGE3_CHEM_CHECKED_CORE/source_schema_hybrid_mixed_v1.json}"
FINAL="${STAGE3_DATA_DIR:-$DATA_ROOT/06_TRAIN_READY_DATA/08_STAGE3_FAMILY_FULL/stage3_full_cation_family_v1}"
if test "${MODE:-validate}" = rebuild; then
  OUT="${STAGE3_REBUILD_DIR:-$WORK_ROOT/07_build_stage3_dataset/stage3_full_cation_family_v1}"; mkdir -p "$(dirname "$OUT")"
  "$PYTHON" "$CORE/training/family/build_stage3_full_family_dataset.py" \
    --source_dir "$SOURCE" --source_schema "$SCHEMA" --output_dir "$OUT" \
    --seed 20260713 --n_folds 10 --val_fold 5 --test_fold 8 \
    --precursor_column predicted_precursor_set_chem_checked
else
  require_file "$FINAL/train.npz"; require_file "$FINAL/val.npz"; require_file "$FINAL/test.npz"
  require_file "$FINAL/split_manifest.json"
  announce "Frozen final Stage3 dataset is present"
fi
