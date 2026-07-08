# STAGE2 CORE METHOD REPORT

## Scope

Stage3 was not changed. This report evaluates a Stage2 core-method benchmark restricted to `solid_state`, `solution`, and `melt_arc`, while preserving the original train/val/test split.

## Why Core Methods

- `solid_state`: large sample count, clear oxide/carbonate/hydroxide precursor logic, stable Stage2 behavior.
- `solution`: large sample count, interpretable nitrate/acetate/halide/organic salt chemistry, useful for precursor prediction despite lower OOV robustness.
- `melt_arc`: smaller than the first two but chemically clean, dominated by elemental/alloy/direct precursors, and already high precision.

## Why Exclude Other Methods

- `other`: sample count is large but mechanism is mixed and not a coherent synthesis method.
- `hydro_solvothermal`, `precipitation`, `flux_molten_salt`: smaller sample counts and low v5 top500, better handled by few-shot/template-specific work.
- `thermal_decomposition`, `mechanochemical`, `sol_gel`, `combustion`: too small for a stable core benchmark right now.

## Core Dataset

- dataset: `/Users/lihonglin/Desktop/Syn_DP/SynPred/data/interim/generative/stage2_setpred_dataset/descriptor/core_methods_ss_solution_meltarc_20260610_relaxed_only`
- all-method label count: 4414
- core label count: 2940
- dropped labels: 1474
- test OOV labels vs core train: 133
- test OOV rows vs core train: 172

| split | n | method distribution | avg set size | max set size |
|---|---:|---|---:|---:|
| train | 19722 | {'solution': 8940, 'solid_state': 8899, 'melt_arc': 1883} | 2.042 | 6 |
| val | 2441 | {'solution': 1124, 'solid_state': 1061, 'melt_arc': 256} | 2.050 | 5 |
| test | 2468 | {'solid_state': 1142, 'solution': 1102, 'melt_arc': 224} | 2.060 | 5 |

Family distribution by label occurrence:

- oxide: 15516
- elemental: 10116
- halide: 6931
- carbonate: 4980
- nitrate: 4211
- organic: 1969
- other_salt: 1701
- phosphate: 1691
- sulfate: 1334
- hydroxide: 1161
- acetate: 747
- unknown: 3

## Core MLP Baseline

- run: `/Users/lihonglin/Desktop/Syn_DP/SynPred/runs/stage2/mlp_core_methods_ss_solution_meltarc_20260610_descriptor`
- best epoch: 26
- threshold subset accuracy/top1 exact: 9.97%
- threshold samples-F1: 44.16%
- threshold Jaccard: 33.93%
- diagnostic true-count subset accuracy: 35.21%

Simple MLP prefix candidate baseline:

| metric | value |
|---|---:|
| prefix_top1_exact | 8.10% |
| prefix_top3_exact | 34.97% |
| prefix_top5_exact | 35.21% |
| prefix_top10_exact | 35.21% |
| prefix_top10_best_f1 | 64.94% |
| prefix_top10_best_jaccard | 57.16% |

## Core Candidate Pool

- candidate dir: `/Users/lihonglin/Desktop/Syn_DP/SynPred/outputs/evaluation/stage2_candidate_pool_core_methods_20260610_top1000`
- generated top1000 per sample so calibration could pull candidates into top500.

| split | top1 | top10 | top200 | top500 | top1000 | best Jaccard@500 |
|---|---:|---:|---:|---:|---:|---:|
| val raw | 31.50% | 57.89% | 82.92% | 86.28% | 86.28% | 93.03% |
| test raw | 28.28% | 57.33% | 80.79% | 85.33% | 85.33% | 92.30% |

Raw test by method:

| method | n | top1 | top10 | top200 | top500 |
|---|---:|---:|---:|---:|---:|
| solid_state | 1142 | 23.12% | 50.00% | 79.68% | 85.11% |
| solution | 1102 | 27.40% | 61.25% | 79.49% | 83.94% |
| melt_arc | 224 | 58.93% | 75.45% | 92.86% | 93.30% |

## Core Calibration

- global calibration dir: `/Users/lihonglin/Desktop/Syn_DP/SynPred/outputs/evaluation/stage2_score_calibration_core_methods_20260610`
- per-method calibration dir: `/Users/lihonglin/Desktop/Syn_DP/SynPred/outputs/evaluation/stage2_score_calibration_core_methods_20260610_per_method`
- final reported result uses per-method calibration because it gives the best top1 objective without changing candidate coverage.

| calibration | top1 | top10 | top200 | top500 | top1000 | OOV top500 | non-OOV top500 |
|---|---:|---:|---:|---:|---:|---:|---:|
| global | 44.65% | 69.45% | 81.85% | 85.33% | 85.37% | 18.99% | 89.87% |
| per-method | 46.15% | 69.17% | 81.69% | 85.33% | 85.37% | 18.99% | 89.87% |

Per-method calibrated test:

| method | n | top1 | top10 | top200 | top500 | top1000 |
|---|---:|---:|---:|---:|---:|---:|
| solid_state | 1142 | 50.44% | 67.16% | 80.39% | 85.11% | 85.20% |
| solution | 1102 | 35.48% | 67.97% | 80.76% | 83.94% | 83.94% |
| melt_arc | 224 | 76.79% | 85.27% | 92.86% | 93.30% | 93.30% |

## Comparison To All-Method V5

| benchmark | top1 | top10 | top200 | top500 |
|---|---:|---:|---:|---:|
| all-method v5 | 39.47% | 63.35% | 77.02% | 80.24% |
| estimated v5 core subset | ~44.8% | ~70.1% | ~82.1% | ~85.3% |
| core-method per-method calibrated | 46.15% | 69.17% | 81.69% | 85.33% |

## Target Check

| target | result | status |
|---|---:|---|
| core top1 >= 48% | 46.15% | FAIL |
| core top10 >= 72% | 69.17% | FAIL |
| core top200 >= 84% | 81.69% | FAIL |
| core top500 >= 87% | 85.33% | FAIL |
| solid_state top500 >= 87% | 85.11% | FAIL |
| solution top500 >= 86% | 83.94% | FAIL |
| melt_arc top500 >= 94% | 93.30% | FAIL, very close |

## Ranker Decision

A hard-negative core ranker was not trained because core candidate-pool top500 is only 85.33%, below the required 87% threshold. Training a ranker now would mainly reshuffle an insufficient candidate pool.

## Interpretation

Restricting to core methods improves stability and interpretability, but it does not by itself improve the candidate ceiling beyond the v5 core subset. The final per-method calibration reaches top1 46.15%, close to but below the 48% target, while top500 remains fixed at 85.33%.

The bottleneck is now candidate coverage for solid_state and solution, not ranking. Melt_arc is already strong: top1 76.79% and top500 93.30%, just below the 94% target. Solid_state and solution need better method-specific candidate generation, especially for OOV and exact hydrated/organic/complex precursor variants.

Recommended next steps: keep core-method benchmark as a clean paper benchmark, add targeted core candidate expansion for solid_state oxide/carbonate/hydroxide variants and solution nitrate/acetate/halide hydrates, then train a core hard-negative ranker only after top500 exceeds 87%. All-method benchmark should remain the broad-coverage supplement, and `other` should be reclassified rather than mixed into core.
