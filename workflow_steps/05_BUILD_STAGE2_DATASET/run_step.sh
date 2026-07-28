#!/usr/bin/env bash
set -euo pipefail
STEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$STEP_DIR/../common.sh"
SOURCE="${STAGE2_RELAXED_DIR:-$DATA_ROOT/06_TRAIN_READY_DATA/01_STAGE2_GFLOWNET_RELAXED/relaxed_only}"
GOLD="${STAGE2_GOLD_META:-$DATA_ROOT/06_TRAIN_READY_DATA/02_STAGE2_GFLOWNET_GOLD/gold_only/train_meta.csv}"
FINAL="${STAGE2_DATA_DIR:-$DATA_ROOT/06_TRAIN_READY_DATA/04_STAGE2_CANONICAL/stage2_full_cation_family_canonical_v1}"
if test "${MODE:-validate}" = rebuild; then
  OUT="${STAGE2_REBUILD_ROOT:-$WORK_ROOT/05_build_stage2_dataset}"
  mkdir -p "$OUT"
  "$PYTHON" "$CORE/training/family/build_full_database_split.py" \
    --source_dir "$SOURCE" --gold_meta "$GOLD" --output_dir "$OUT/stage2_full_cation_family_v1" \
    --seed 20260713 --n_folds 10 --val_fold 5 --test_fold 8 --relaxed_weight 0.5
  "$PYTHON" "$CORE/training/family/build_stage2_canonical_label_dataset.py" \
    --input_dir "$OUT/stage2_full_cation_family_v1" \
    --output_dir "$OUT/stage2_full_cation_family_canonical_v1"
  announce "Rebuilt Stage2 data: $OUT/stage2_full_cation_family_canonical_v1"
else
  require_file "$FINAL/train.npz"; require_file "$FINAL/val.npz"; require_file "$FINAL/test.npz"
  require_file "$FINAL/split_manifest.json"; require_file "$FINAL/precursor_canonicalization.json"
  announce "Frozen final Stage2 dataset is present"
fi
