#!/usr/bin/env bash
set -euo pipefail
STEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$STEP_DIR/../common.sh"
OUT="${1:-$WORK_ROOT/08_train_stage3/stage3_ensemble_rebuilt.npz}"
mkdir -p "$(dirname "$OUT")"
CUSTOM_SAMPLES=0
if test -n "${NF_SAMPLES:-}${CVAE_SAMPLES:-}${DIFFUSION_SAMPLES:-}"; then CUSTOM_SAMPLES=1; fi
NF_SAMPLES="${NF_SAMPLES:-$DATA_ROOT/08_GENERATED_OUTPUTS/03_STAGE3_COMPONENT_SAMPLES/NF_s8060/val_samples.npz}"
CVAE_SAMPLES="${CVAE_SAMPLES:-$DATA_ROOT/08_GENERATED_OUTPUTS/03_STAGE3_COMPONENT_SAMPLES/CVAE_s8040/val_samples.npz}"
DIFFUSION_SAMPLES="${DIFFUSION_SAMPLES:-$DATA_ROOT/08_GENERATED_OUTPUTS/03_STAGE3_COMPONENT_SAMPLES/Diffusion_s8320/val_samples.npz}"
"$PYTHON" "$CORE/training/family/ensemble_stage3_samples.py" \
  --input_npz "$NF_SAMPLES" \
  --input_npz "$CVAE_SAMPLES" \
  --input_npz "$DIFFUSION_SAMPLES" \
  --sample_limit 64 --sample_limit 64 --sample_limit 64 --output_npz "$OUT"
if test "$CUSTOM_SAMPLES" = 0; then
  EXPECTED_SHA=14f8306db675cfc30bf5d87224af64e049f667fcabbd2f0f99d112eb79b7e41a
  ACTUAL_SHA="$(sha256sum "$OUT" | awk '{print $1}')"
  test "$ACTUAL_SHA" = "$EXPECTED_SHA" || { echo "Stage3 frozen ensemble hash mismatch: $ACTUAL_SHA" >&2; exit 3; }
fi
announce "Rebuilt ensemble: $OUT"
