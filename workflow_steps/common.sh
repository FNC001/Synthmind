#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SYNTHMIND_REPO_ROOT:-$(cd "$SOURCE_DIR/.." && pwd)}"
DATA_ROOT="${SYNTHMIND_DATA_ROOT:-${DATA_ROOT:-$REPO_ROOT}}"
# Compatibility alias for the historical wrappers. V1.0 code lives in
# REPO_ROOT; large data/model/evidence assets live in DATA_ROOT.
RELEASE_ROOT="$DATA_ROOT"
WORKFLOW_ROOT="$REPO_ROOT/workflow_steps"
CORE="$REPO_ROOT"
PYTHON="${PYTHON:-python3}"
WORK_ROOT="${WORK_ROOT:-$REPO_ROOT/work/v1}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$CORE${PYTHONPATH:+:$PYTHONPATH}"
export SYNTHMIND_REPO_ROOT="$REPO_ROOT"
export SYNTHMIND_DATA_ROOT="$DATA_ROOT"

require_file() {
  test -f "$1" || { echo "Missing required file: $1" >&2; exit 2; }
}

require_dir() {
  test -d "$1" || { echo "Missing required directory: $1" >&2; exit 2; }
}

announce() {
  echo "[$(basename "$0")] $*"
}

require_data_root() {
  require_dir "$DATA_ROOT"
  for directory in "$@"; do
    require_dir "$DATA_ROOT/$directory"
  done
}

announce_context() {
  announce "repo_root=$REPO_ROOT"
  announce "data_root=$DATA_ROOT"
  announce "work_root=$WORK_ROOT"
  announce "python=$PYTHON"
}
