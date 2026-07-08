# Stage2 Precision Report

Date: 2026-06-10

Scope: Stage2 only, structure/descriptor to precursor set prediction. Stage3/process-condition optimization is paused.

## 1. Data And Label Canonicalization

| Dataset | Label count | Merged labels | Notes |
|---|---:|---:|---|
| Raw method-stratified | 4520 | 0 | Original precursor strings |
| canonical v1 | 4452 | 68 | Mostly hydrate dot/H20 variants |
| canonical v2 | 4440 | 80 | Adds `*`, `solution`, `powder`, `nH2O`-style cleanup, phrase aliases, and alias report |

Important guardrail: I first tested a more aggressive `pymatgen.Composition` canonical label, but it merged chemically distinct same-composition precursors. I reverted that behavior. In v2, `pymatgen` is used for formula-key reporting/validation, not for blindly replacing the precursor label.

Artifacts:
- `data/interim/generative/stage2_setpred_dataset/descriptor/route_method_stratified_canonical_v2_20260610_relaxed_only`
- `data/interim/generative/stage2_setpred_dataset/descriptor/route_method_stratified_canonical_v2_20260610_relaxed_only/precursor_alias_report.csv`

## 2. MLP Results

All metrics below are on the method-stratified fair test set.

| Model/data | Threshold exact | Samples-F1 | Jaccard | Diagnostic exact using true set size |
|---|---:|---:|---:|---:|
| Raw labels | 7.45% | 40.40% | 30.03% | 30.21% |
| canonical v1 | 7.34% | 40.48% | 30.29% | 30.05% |
| canonical v2 | 9.12% | 41.25% | 31.37% | 31.11% |

canonical v2 is a real but modest improvement. It does not by itself solve the Stage2 problem.

## 3. Hybrid Candidate Pool v2

Candidate sources:
- MLP probability/beam candidates
- Historical templates by formula/material/method
- Descriptor + composition nearest-neighbor retrieval
- Chemistry-constrained beam scoring with element coverage, extra/missing element penalties, precursor family priors, set-size probability, route frequency, and co-occurrence features

Test-set candidate-pool metrics:

| Metric | Result |
|---|---:|
| top1 exact from candidate score | 33.76% |
| top1 F1 | 50.24% |
| top1 Jaccard | 45.67% |
| top10 exact recall | 60.45% |
| top20 exact recall | 64.00% |
| top50 exact recall | 66.40% |
| top100 exact recall | 68.29% |
| top200 exact recall | 70.33% |
| top500 exact recall | 73.06% |
| top200 best F1 | 84.09% |
| top200 best Jaccard | 80.69% |
| top500 best F1 | 86.00% |
| top500 best Jaccard | 82.93% |

This improves over the previous method-stratified MLP+retrieval top200 upper bound of about 62.2%, but it does not yet reach the short-term 75% top200 exact target.

Artifacts:
- `outputs/evaluation/stage2_candidate_pool_v2_20260610/test_stage2_candidate_pool_v2_summary.json`
- `outputs/evaluation/stage2_candidate_pool_v2_20260610/test_stage2_candidate_pool_v2.csv`

## 4. Set-Level Reranker

Trained:
- LightGBM LambdaRank
- LightGBM dense-target regressor
- Candidate features include probability stats, size features, element coverage, extra/missing element counts, family counts, retrieval similarity, route frequency, co-occurrence, method prior, and source flags.

Result: the pure reranker overfit/distributed poorly and underperformed the original candidate score. A small blend helped only slightly.

| Scoring | top1 exact | top3 exact | top5 exact | top10 exact | top200 exact |
|---|---:|---:|---:|---:|---:|
| Original candidate score | 33.76% | 49.13% | 54.72% | 60.45% | 70.33% |
| Ranker only | 31.03% | 45.55% | 50.63% | 57.64% | 66.68% |
| Best blend: `0.9*ranker + 0.1*score` | 33.95% | 49.70% | 54.86% | 60.81% | 70.41% |

Conclusion: reranker v2 is not yet strong enough. The candidate-pool score is currently the safer default.

Artifacts:
- `runs/stage2/ranker_stage2_candidate_set_lgbm_v2_20260610/metrics.json`
- `runs/stage2/ranker_stage2_candidate_set_lgbm_v2_20260610/blend_sweep_test.json`
- `runs/stage2/ranker_stage2_candidate_set_lgbm_v2_20260610/stage2_candidate_set_lgbm_reranker.joblib`

## 5. OOV And Train-Coverage Analysis

On canonical v2:

| Metric | Result |
|---|---:|
| True precursor label occurrences that are OOV | 4.27% |
| Test rows containing at least one OOV true precursor | 8.52% |
| Exact true precursor set seen in train | 55.27% |
| top1 exact for rows with OOV | 0.00% |
| top10 exact for rows with OOV | 0.00% |
| top1 exact for rows without OOV | 36.90% |
| top10 exact for rows without OOV | 66.08% |

OOV by reaction method is concentrated in:
- `other`: 105 OOV label occurrences
- `solution`: 95
- `solid_state`: 60
- `precipitation`: 22
- `hydro_solvothermal`: 18

OOV by family:
- `other`: 136
- `halide`: 88
- `oxide`: 32
- `hydroxide`: 18
- `nitrate`: 14
- `acetate`: 13

This confirms that a closed-label classifier cannot reach high exact-set accuracy on these OOV rows.

Artifacts:
- `outputs/evaluation/stage2_oov_canonical_v2_20260610/test_oov_summary.json`
- `outputs/evaluation/stage2_oov_canonical_v2_20260610/test_oov_rows.csv`

## 6. Current Best Numbers

Fair method-stratified test:

| Category | Best current result |
|---|---:|
| Stage2 deployable top1 exact | 33.95% |
| Stage2 top10 exact recall | 60.81% |
| Stage2 top50 exact recall | 66.59% |
| Stage2 top100 exact recall | 68.45% |
| Stage2 top200 exact recall | 70.41% |
| Stage2 top500 exact recall | 73.06% |
| Stage2 top500 best Jaccard | 82.93% |

Older solid-state-biased test remains much easier:
- top1 exact was about 52.4%
- top10 exact recall was about 66.7%
- MLP + retrieval top200 exact recall could exceed 80%

That old split should not be used as the main target for the fair benchmark.

## 7. Can We Reach 80%?

Short answer: not with the current closed-vocabulary MLP + candidate reranking stack alone.

What is now realistic:
- Short term: top200 exact recall 75% is close but not yet reached; current is 70.4%.
- Short term top1 exact 45% was not reached; current best is 33.95%.
- Medium term: top50 exact recall 80% likely requires a stronger candidate generator, not just reranking.
- Long term top1 exact 80% needs a different formulation:
  - precursor family prediction,
  - element-source prediction,
  - open-formula generation for OOV precursors,
  - and a set-level model trained directly on exact-set quality.

Recommended next technical step:
1. Build an element-source/family model: for each target element, predict likely precursor family and concrete formula candidates.
2. Add open-vocabulary formula generation for OOV-like labels.
3. Rebuild reranker training without train-template leakage and with hard negatives sampled from val-like retrieval distributions.
4. Train a set decoder or autoregressive precursor model that optimizes set-level exact/F1 directly instead of independent BCE labels.
