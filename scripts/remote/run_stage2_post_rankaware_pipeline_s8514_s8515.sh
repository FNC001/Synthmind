#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/synthmind_family_routed_v1}"
cd "$ROOT/code"

bash scripts/remote/run_stage2_matscibert_aligned_rescore_s8514.sh
bash scripts/remote/run_stage2_score_ensemble_grid_s8515.sh
