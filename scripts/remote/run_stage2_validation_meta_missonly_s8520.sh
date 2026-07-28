#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/synthmind_family_routed_v1}"
RUN="$ROOT/runs/canonical_v1/stage2_validation_meta_missonly_s8520" \
SEED=8520 \
bash "$ROOT/code/scripts/remote/run_stage2_validation_meta_lambdarank_s8518.sh" \
  --num_boost_round 120 \
  --learning_rate 0.04 \
  --num_leaves 15 \
  --min_data_in_leaf 20 \
  --group_balance_power 0.25 \
  --family_balance_power 0.0 \
  --base_miss_weight 1.0 \
  --exclude_base_rank_features \
  --train_accessible_base_misses_only
