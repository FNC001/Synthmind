# STAGE2 V5 OOV TEMPLATE REPORT

## Scope

Stage3 was not changed. Stage2 v5 added no-leakage failure decomposition, train/val-only OOV clinic, train-derived method templates, v5 candidate generation, conservative assembly repair, and val-only score calibration.

## Artifacts

- failure decomposition: `/Users/lihonglin/Desktop/Syn_DP/SynPred/outputs/evaluation/stage2_v5_failure_decomposition_20260610/test_failure_decomposition.csv`
- OOV clinic: `/Users/lihonglin/Desktop/Syn_DP/SynPred/outputs/evaluation/stage2_v5_oov_clinic_20260610/oov_clinic_train_val.csv`
- alias patch v5: `/Users/lihonglin/Desktop/Syn_DP/SynPred/data/interim/ontology/precursor_alias_patch_v5_20260610/precursor_alias_patch_v5.csv`
- method templates: `/Users/lihonglin/Desktop/Syn_DP/SynPred/data/interim/templates/stage2_method_precursor_templates_v5_20260610/method_precursor_templates.csv`
- v5 candidate pool: `/Users/lihonglin/Desktop/Syn_DP/SynPred/outputs/evaluation/stage2_candidate_pool_v5_20260610`
- v5 calibration: `/Users/lihonglin/Desktop/Syn_DP/SynPred/outputs/evaluation/stage2_score_calibration_v5_20260610`

## V4 Baseline

| metric | value |
|---|---:|
| top1 exact | 34.99% |
| top10 exact | 58.57% |
| top200 exact | 73.36% |
| top500 exact | 76.83% |
| OOV top500 exact | 24.68% |
| non-OOV top500 exact | 81.68% |

## Failure Decomposition From V4 Pool

- top500 misses: 849 / 3664
- missing label failure among top500 misses: 62.07%
- assembly failure among top500 misses: 37.93%
- OOV failure among top500 misses: 26.38%
- parse failure among top500 misses: 2.36%
- weak-method/template failure among top500 misses: 71.38%

Missing-label family distribution in misses:

- halide: 164
- oxide: 146
- other_salt: 66
- nitrate: 53
- hydroxide: 47
- carbonate: 44
- acetate: 35
- organic: 35
- sulfate: 28
- elemental: 21
- phosphate: 9

## No-Leakage OOV Clinic

- clinic rows from train/val labels: 4211
- allowed v5 patches: 526
- forbidden test-derived patch count: 0
- patch type counts: {'hydrate_to_base': 503, 'solvent_adduct_suffix_remove': 21, 'concentration_suffix_remove': 1, 'phase_prefix_remove': 1}
- rule source counts: {'general_chemistry_rule': 503, 'manual_general_rule': 21, 'train_pattern': 2}
- train/val parse-failed labels in clinic: 170
- v5 remaining OOV label count after patch: 215
- v5 parse-failed label count after patch: 181

## Method-Specific Templates

- template count: 15809
- train rows used: 29208

By method:

- solid_state: 4544
- other: 4211
- solution: 3186
- melt_arc: 1357
- hydro_solvothermal: 870
- flux_molten_salt: 642
- precipitation: 552
- thermal_decomposition: 219
- mechanochemical: 181
- sol_gel: 31
- combustion: 18

## Candidate Pool V5

| split/pool | top1 | top10 | top50 | top100 | top200 | top500 | best F1@500 | best Jaccard@500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| val v5 | 39.50% | 63.83% | 73.70% | 76.05% | 79.47% | 82.29% | 93.27% | 90.85% |
| test v5 | 35.89% | 59.74% | 70.31% | 73.61% | 76.86% | 80.05% | 92.41% | 89.58% |

Candidate source contribution before calibration:

- method_template: candidates=296763, exact samples=1138
- train_label_assembly: candidates=122787, exact samples=52
- v4_base: candidates=1243981, exact samples=1743

## Assembly Repair

| pool | top10 | top200 | top500 | best Jaccard@500 |
|---|---:|---:|---:|---:|
| test before repair | 59.74% | 76.86% | 80.05% | 89.58% |
| test after conservative repair | 59.74% | 76.86% | 80.24% | 89.77% |

Aggressive repair was tested first and hurt ranking severely; the final saved repair is conservative, preserving the original top470 and using repair only for tail coverage.

## Calibration V5

- best val objective: 0.6043
- val top1/top10/top500: 43.30% / 67.52% / 82.42%
- test top1/top10/top200/top500: 39.47% / 63.35% / 77.02% / 80.24%
- test best F1@500 / Jaccard@500: 92.56% / 89.77%
- test OOV top500: 19.71%
- test non-OOV top500: 85.13%

Best weights emphasized v3_reranker_score, total_score_v5, method_prior_score, family_score, set_size_score, and a negative repair_source_flag. This matches the observed behavior: templates help coverage, but repair should stay in the tail.

## By Reaction Method After Calibration

| method | n | top1 | top10 | top200 | top500 |
|---|---:|---:|---:|---:|---:|
| solid_state | 1142 | 49.12% | 69.09% | 80.39% | 85.03% |
| solution | 1102 | 34.03% | 68.06% | 81.94% | 83.85% |
| other | 737 | 30.53% | 51.56% | 70.01% | 73.54% |
| melt_arc | 224 | 75.89% | 85.71% | 91.96% | 93.30% |
| hydro_solvothermal | 191 | 20.94% | 45.55% | 59.16% | 62.83% |
| precipitation | 128 | 25.78% | 50.78% | 61.72% | 64.84% |
| flux_molten_salt | 68 | 20.59% | 33.82% | 48.53% | 51.47% |
| thermal_decomposition | 32 | 34.38% | 40.62% | 78.12% | 84.38% |
| mechanochemical | 28 | 42.86% | 50.00% | 64.29% | 64.29% |
| sol_gel | 7 | 42.86% | 71.43% | 85.71% | 85.71% |
| combustion | 5 | 40.00% | 60.00% | 100.00% | 100.00% |

## By Original Failure Type After Calibration

| failure type | n | top1 | top10 | top200 | top500 |
|---|---:|---:|---:|---:|---:|
| missing_label_failure | 527 | 0.19% | 3.98% | 7.59% | 12.90% |
| assembly_failure | 322 | 0.31% | 2.48% | 11.80% | 18.63% |
| oov_failure | 276 | 0.00% | 1.81% | 13.77% | 19.93% |
| parse_failure | 39 | 0.00% | 41.03% | 51.28% | 53.85% |
| method_template_failure | 2226 | 30.86% | 58.63% | 73.85% | 76.55% |

## By Candidate Source After Calibration

| source | n samples | top1 | top10 | top200 | top500 |
|---|---:|---:|---:|---:|---:|
| v4_base | 3664 | 22.98% | 34.66% | 45.61% | 47.54% |
| assembly_repair_add | 3578 | 0.00% | 0.00% | 0.03% | 0.11% |
| train_label_assembly | 3419 | 0.18% | 0.32% | 0.76% | 1.52% |
| method_template | 3385 | 17.67% | 30.72% | 33.21% | 33.62% |
| assembly_repair_swap | 1874 | 0.00% | 0.00% | 0.00% | 0.16% |
| assembly_repair_drop | 884 | 0.00% | 0.00% | 0.00% | 0.11% |

## Target Check

| target | result | status |
|---|---:|---|
| full top500 >= 80% | 80.24% | PASS |
| OOV top500 >= 50% | 19.71% | FAIL |
| top10 >= 61% | 63.35% | PASS |
| top1 >= 40% | 39.47% | NEAR MISS |

## Interpretation

Stage2 v5 achieved the full-test top500 target and restored top10 above 61%. The main lift came from method-specific templates and conservative canonicalization, not from generic open-vocab expansion. Top1 is just below 40%, so a ranker may help, but OOV coverage remains the larger structural bottleneck.

OOV did not improve to the requested 50%+. The remaining OOV labels are mostly true unseen or messy complex precursors, including mixed salts, hydrated/organic adducts, flux salts, and target-like intermediates. Since these labels are absent from train, a ranker cannot recover them unless the candidate generator creates them first.

I did not train a new hard-negative ranker in this pass. Although full top500 is now above 80%, OOV top500 remains low; the next highest-return step is method-specific OOV template generation and controlled chemical formula generation for the remaining OOV families, then a no-leakage ranker once OOV coverage is materially higher.

Recommended next steps: build method-specific OOV generators for oxide/halide/other_salt, add mixed-salt and flux-salt templates, manually review the top remaining OOV/failed-parse labels, and only then train the hard-negative ranker or set decoder.
