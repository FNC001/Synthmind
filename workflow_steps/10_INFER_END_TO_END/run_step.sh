#!/usr/bin/env bash
set -euo pipefail
STEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$STEP_DIR/../common.sh"
require_file "$CORE/scripts/07_infer/structure_to_synthesis_route/pipeline/run_pipeline.py"
require_file "$CORE/scripts/infer/synthmind_gnome_frozen_adapter.py"
require_file "$DATA_ROOT/07_BEST_MODELS/02_STAGE2_FACTORIZED_EXPERTS/stage2_factorized_h2048_b6_top20_g1_s9140/best_factorized_generator.pt"
require_file "$DATA_ROOT/07_BEST_MODELS/02_STAGE2_FACTORIZED_EXPERTS/stage2_factorized_h1536_b4_top20_g2_s9151/best_factorized_generator.pt"
require_file "$DATA_ROOT/07_BEST_MODELS/02_STAGE2_FACTORIZED_EXPERTS/stage2_factorized_h2048_b6_top20_g1_s9152/best_factorized_generator.pt"
require_file "$DATA_ROOT/07_BEST_MODELS/04_STAGE3_NF/stage3_conditional_flow_h1024_s8060/best_model.pt"
require_file "$DATA_ROOT/07_BEST_MODELS/05_STAGE3_CVAE/stage3_hybrid_cvae_h1024_s8040/best_model.pt"
require_file "$DATA_ROOT/07_BEST_MODELS/06_STAGE3_DIFFUSION/stage3_conditional_diffusion_h1536_s8320/best_model.pt"
announce "V1.0 inference source and six frozen expert/generative weights are present."
