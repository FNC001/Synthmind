#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/synthmind_family_routed_v1}"
RUN="$ROOT/runs/canonical_v1/stage2_validation_meta_hardmiss_s8519" \
SEED=8519 \
bash "$ROOT/code/scripts/remote/run_stage2_validation_meta_lambdarank_s8518.sh" \
  --num_boost_round 300 \
  --num_leaves 31 \
  --min_data_in_leaf 60 \
  --group_balance_power 0.5 \
  --family_balance_power 0.0 \
  --base_miss_weight 8.0 \
  --exclude_base_rank_features
