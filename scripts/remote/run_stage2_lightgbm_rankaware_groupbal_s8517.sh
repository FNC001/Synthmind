#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/synthmind_family_routed_v1}"
RUN="$ROOT/runs/canonical_v1/stage2_lightgbm_rankaware_groupbal_s8517" \
SEED=8517 \
bash "$ROOT/code/scripts/remote/run_stage2_lightgbm_rankaware_s8516.sh" \
  --group_balance_power 1.0 \
  --family_balance_power 0.5
