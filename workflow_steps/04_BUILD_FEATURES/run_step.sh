#!/usr/bin/env bash
set -euo pipefail
STEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$STEP_DIR/../common.sh"
require_dir "$RELEASE_ROOT/04_SPLITS/01_BASE_GROUP_SPLIT/structdesc_splits"
require_file "$RELEASE_ROOT/04_SPLITS/00_PORTABLE_SPLITS/01_BASE_GROUP_SPLIT/manifest.json"
require_file "$RELEASE_ROOT/04_SPLITS/00_PORTABLE_SPLITS/02_ROUTE_GROUP_SPLIT/manifest.json"
require_dir "$RELEASE_ROOT/05_FEATURES_AND_EMBEDDINGS/01_STRUCTURAL_DESCRIPTORS/structdesc_features"
require_dir "$RELEASE_ROOT/05_FEATURES_AND_EMBEDDINGS/02_CHGNET_GRAPH_CACHE/chgnet_stage2"
require_dir "$RELEASE_ROOT/05_FEATURES_AND_EMBEDDINGS/03_CHGNET_EMBEDDINGS/chgnet_stage2"
require_dir "$RELEASE_ROOT/05_FEATURES_AND_EMBEDDINGS/04_HYBRID_FEATURES/stage2_hybrid_features"
announce "Split, descriptor, CHGNet and hybrid checkpoints are present"
