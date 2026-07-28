#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT=""
WORK_ROOT="$REPO_ROOT/work/v1"
PYTHON_BIN="${PYTHON:-python3}"
PROFILE="validate-full"
DEVICE_VALUE="${DEVICE:-cuda}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Synthmind V1.0 workflow

Usage:
  scripts/run_v1_workflow.sh --data-root PATH [options]

Required:
  --data-root PATH       Authorized artifact root containing 00_*, 02_* ... 11_*.

Options:
  --profile NAME         validate-full (default), validate-fast, train, rebuild-train.
  --work-root PATH       Output root. Default: ./work/v1
  --python PATH          Python interpreter. Default: $PYTHON or python3
  --device DEVICE        Training device. Default: cuda
  --dry-run              Print the ordered commands without executing them.
  -h, --help             Show this help.

Profiles:
  validate-fast          Validate Steps 01-08 and V1 inference dependencies.
  validate-full          validate-fast plus exact frozen metric reproduction (Step 09).
  train                  Train Stage2 and Stage3 from prepared datasets in 06_*.
  rebuild-train          Rebuild final Stage2/Stage3 datasets, then train both stages.
EOF
}

while (($#)); do
  case "$1" in
    --data-root)
      DATA_ROOT="${2:?missing value for --data-root}"
      shift 2
      ;;
    --work-root)
      WORK_ROOT="${2:?missing value for --work-root}"
      shift 2
      ;;
    --python)
      PYTHON_BIN="${2:?missing value for --python}"
      shift 2
      ;;
    --profile)
      PROFILE="${2:?missing value for --profile}"
      shift 2
      ;;
    --device)
      DEVICE_VALUE="${2:?missing value for --device}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

test -n "$DATA_ROOT" || { echo "--data-root is required" >&2; exit 2; }
DATA_ROOT="$(cd "$DATA_ROOT" && pwd)"
mkdir -p "$WORK_ROOT"
WORK_ROOT="$(cd "$WORK_ROOT" && pwd)"

export SYNTHMIND_REPO_ROOT="$REPO_ROOT"
export SYNTHMIND_DATA_ROOT="$DATA_ROOT"
export WORK_ROOT
export PYTHON="$PYTHON_BIN"
export DEVICE="$DEVICE_VALUE"

run_step() {
  local step="$1"
  local mode="${2:-validate}"
  local script="$REPO_ROOT/workflow_steps/$step/run_step.sh"
  test -f "$script" || { echo "Missing workflow step: $script" >&2; exit 2; }
  echo
  echo "=== $step (MODE=$mode) ==="
  if test "$DRY_RUN" = 1; then
    printf 'MODE=%q SYNTHMIND_DATA_ROOT=%q WORK_ROOT=%q PYTHON=%q DEVICE=%q bash %q\n' \
      "$mode" "$DATA_ROOT" "$WORK_ROOT" "$PYTHON_BIN" "$DEVICE_VALUE" "$script"
    return
  fi
  MODE="$mode" bash "$script"
}

validate_inputs() {
  run_step 01_VALIDATE_RAW_DATA validate
  run_step 02_ALIGN_SYNTHESIS_STRUCTURES validate
  run_step 03_CLEAN_AND_STRATIFY validate
  run_step 04_BUILD_FEATURES validate
}

case "$PROFILE" in
  validate-fast)
    validate_inputs
    run_step 05_BUILD_STAGE2_DATASET validate
    run_step 06_TRAIN_STAGE2 validate
    run_step 07_BUILD_STAGE3_DATASET validate
    run_step 08_TRAIN_STAGE3 validate
    run_step 10_INFER_END_TO_END validate
    ;;
  validate-full)
    validate_inputs
    run_step 05_BUILD_STAGE2_DATASET validate
    run_step 06_TRAIN_STAGE2 validate
    run_step 07_BUILD_STAGE3_DATASET validate
    run_step 08_TRAIN_STAGE3 validate
    run_step 09_EVALUATE_THREE_METRICS reproduce
    run_step 10_INFER_END_TO_END validate
    ;;
  train)
    validate_inputs
    run_step 05_BUILD_STAGE2_DATASET validate
    run_step 06_TRAIN_STAGE2 train_full
    run_step 07_BUILD_STAGE3_DATASET validate
    run_step 08_TRAIN_STAGE3 train_all
    ;;
  rebuild-train)
    validate_inputs
    run_step 05_BUILD_STAGE2_DATASET rebuild
    export STAGE2_DATA_DIR="$WORK_ROOT/05_build_stage2_dataset/stage2_full_cation_family_canonical_v1"
    run_step 06_TRAIN_STAGE2 train_full
    run_step 07_BUILD_STAGE3_DATASET rebuild
    export STAGE3_DATA_DIR="$WORK_ROOT/07_build_stage3_dataset/stage3_full_cation_family_v1"
    run_step 08_TRAIN_STAGE3 train_all
    ;;
  *)
    echo "Unsupported profile: $PROFILE" >&2
    usage >&2
    exit 2
    ;;
esac

echo
echo "Synthmind V1.0 profile '$PROFILE' completed."
echo "Outputs: $WORK_ROOT"
