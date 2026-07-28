#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <rankaware_pid>" >&2
  exit 2
fi

ROOT="${ROOT:-/root/autodl-tmp/synthmind_family_routed_v1}"
RANKAWARE_PID="$1"

while kill -0 "$RANKAWARE_PID" 2>/dev/null; do
  sleep 30
done

cd "$ROOT/code"
bash scripts/remote/run_stage2_post_rankaware_pipeline_s8514_s8515.sh
