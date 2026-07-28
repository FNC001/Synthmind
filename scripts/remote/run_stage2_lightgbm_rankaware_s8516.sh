#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/synthmind_family_routed_v1}"
DATA="$ROOT/data/stage2_full_cation_family_canonical_v1"
SRC="$ROOT/metrics/canonical_v1/sources"
RUN="${RUN:-$ROOT/runs/canonical_v1/stage2_lightgbm_rankaware_s8516}"

mkdir -p "$RUN"
cd "$ROOT/code"
export PYTHONPATH=.

/root/miniconda3/bin/python3 training/family/train_stage2_lightgbm_ranker.py \
  --input_dir "$DATA" \
  --matsci_embeddings "$DATA/matscibert_embeddings_canonical_v1.npz" \
  --matsci_components 64 \
  --base_val_candidates "$ROOT/runs/canonical_v1/stage2_s8502_top10_s8494_depth_s8504/val_candidates.jsonl" \
  --train_candidate_source "$SRC/stage2_oof5_fixedrrf10_train_candidates.jsonl" \
  --train_candidate_source "$SRC/stage2_oof_familypool_train1000_candidates.jsonl" \
  --train_candidate_source "$SRC/stage2_autoregressive_oof_fixed38_train_candidates.jsonl" \
  --train_candidate_source "$SRC/stage2_gflownet_film_oof_fixed31_beam100_train_candidates.jsonl" \
  --train_candidate_source "$SRC/stage2_substitution_oof_allgroups_exact1p0_train_candidates.jsonl" \
  --train_candidate_source "$SRC/stage2_substitution_oof_periodic_train_candidates.jsonl" \
  --train_candidate_source "$SRC/stage2_fulltrain_oof6_aligned_matsci32_g0p25_s8005_train_candidates.jsonl" \
  --train_candidate_source "$SRC/stage2_fulltrain_oof6_aligned_fast_g0p25_s8007_train_candidates.jsonl" \
  --train_candidate_source "$SRC/stage2_oof_element_cartesian_zeroshot_train2000_candidates.jsonl" \
  --train_aux_rank_source "$ROOT/runs/canonical_v1/stage2_family_template_oof5_s8511/train_candidates.jsonl" \
  --val_aux_rank_source "$ROOT/runs/canonical_v1/stage2_family_template_canonical_s8508/val_candidates.jsonl" \
  --train_aux_rank_source "$SRC/stage2_oof_familypool_train1000_candidates.jsonl" \
  --val_aux_rank_source "$SRC/stage2_union_familyfirst_h1024_s6262_val_candidates.jsonl" \
  --train_aux_rank_source "$SRC/stage2_autoregressive_oof_fixed38_train_candidates.jsonl" \
  --val_aux_rank_source "$SRC/stage2_autoregressive_cardinality_beam_val_candidates.jsonl" \
  --train_aux_rank_source "$SRC/stage2_gflownet_film_oof_fixed31_beam100_train_candidates.jsonl" \
  --val_aux_rank_source "$SRC/stage2_gflow_relativechem_s8006_val_candidates.jsonl" \
  --train_aux_rank_source "$SRC/stage2_substitution_oof_allgroups_exact1p0_train_candidates.jsonl" \
  --val_aux_rank_source "$SRC/stage2_family_substitution_allgroups_exact0p1_val_candidates.jsonl" \
  --train_aux_rank_source "$SRC/stage2_substitution_oof_periodic_train_candidates.jsonl" \
  --val_aux_rank_source "$SRC/stage2_family_substitution_periodic_knn1000_val_candidates.jsonl" \
  --train_aux_rank_source "$SRC/stage2_oof_element_cartesian_zeroshot_train2000_candidates.jsonl" \
  --val_aux_rank_source "$SRC/stage2_element_cartesian_deep_p120_z200_a50_val_candidates.jsonl" \
  --candidate_limit 800 \
  --source_union_limit 128 \
  --train_pool_limit 128 \
  --cross_family_negatives 32 \
  --aux_rank_limit 100 \
  --num_boost_round 1600 \
  --early_stopping_rounds 120 \
  --learning_rate 0.025 \
  --num_leaves 127 \
  --min_data_in_leaf 100 \
  --feature_fraction 0.85 \
  --bagging_fraction 0.85 \
  --lambda_l1 0.01 \
  --lambda_l2 0.10 \
  --num_threads 64 \
  --seed "${SEED:-8516}" \
  --output_json "$RUN/metrics.json" \
  --output_candidates_jsonl "$RUN/val_candidates.jsonl" \
  --output_model "$RUN/model.txt" \
  --output_scores_npz "$RUN/val_scores.npz" \
  "$@" \
  2>&1 | tee "$RUN/run.log"
