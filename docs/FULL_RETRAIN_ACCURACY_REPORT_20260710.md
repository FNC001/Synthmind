# Full Retrain and Accuracy Report

Date: 2026-07-10

Code baseline: `dd73520` (`main`)

Scope: retrain every model family exposed by the current public training entry points, using the preserved fixed train/validation/test artifacts and the previous evaluation protocols. Historical datasets and model files were not overwritten.

## 1. Environment

| Item | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB |
| Python | 3.12 |
| PyTorch | 2.8.0+cu128 |
| LightGBM | 4.6.0 |
| scikit-learn | 1.9.0 |
| NumPy | 2.3.2 |
| Random seed | 42 for Stage2/Stage3; 20260612 for Stage35 |
| Remote run root | `runs/retrain_20260710` |

The following public entry points were run to completion:

- `training/precursor/train_gflownet.py`
- `training/precursor/train_mlp_baseline.py`
- `training/conditions/train_lgbm_quantile_ensemble.py`
- `training/conditions/train_lgbm_method_experts.py`
- `training/ranking/train_route_reranker.py`

## 2. Data and Format Compatibility

The original split files were reused without reshuffling.

### Stage2 precursor data

Directory: `data/interim/generative/stage2_gflownet_dataset/hybrid/gold_only`

| Split | Rows | Feature dimension | Label dimension | Maximum trajectory length |
|---|---:|---:|---:|---:|
| train | 2,978 | 195 | 1,968 | 5 |
| validation | 635 | 195 | 1,968 | 5 |
| test | 629 | 195 | 1,968 | 5 |

Each split remains an NPZ file with the established keys:

```text
x_raw, x, y_multi_hot, traj_actions, traj_mask, set_len
```

Metadata remains in `<split>_meta.csv`; action and precursor mappings remain in `action_vocab.json`, `action_to_id.json`, and `precursor_names.json`.

### Stage3 condition data

Directory: `data/interim/generative/stage3_condition_dataset/hybrid_mixed_v1`

| Split | Rows | Structure features | Precursor-set features |
|---|---:|---:|---:|
| train | 21,165 | 131 | 1,971 |
| validation | 533 | 131 | 1,971 |
| test | 533 | 131 | 1,971 |

Each split remains an NPZ file with the established keys:

```text
x, y_set, y_cond_continuous, y_cond_continuous_mask,
y_cond_discrete, y_cond_discrete_mask, sample_id
```

The current schema uses `target_temperature_c_clean` and `target_time_h_clean`. Training now accepts these names as well as the legacy names without changing the stored data format.

### Stage35 route data

Directory: `outputs/evaluation/stage35_route_candidates_v3_final_20260612`

| Split | Samples | Candidate rows |
|---|---:|---:|
| train | 29,208 | 540,982 |
| validation | 3,658 | 1,447,340 |
| test | 3,664 | 1,449,580 |

The existing CSV candidate schema and the previous train/validation/test separation were preserved.

## 3. Stage2 Results

### GFlowNet plus candidate reranker

Best epoch: 38. Best validation samples-F1: 75.03%.

| Test metric | Greedy | Candidate reranker |
|---|---:|---:|
| Exact set / subset accuracy | 55.33% | 56.76% |
| Samples-F1 | 72.02% | 73.29% |
| Samples Jaccard | 66.96% | 68.27% |
| Exact hit@3 | n/a | 65.02% |
| Exact hit@5 | n/a | 65.98% |
| Exact hit@10 | n/a | 67.09% |

Comparison with the previous `gflownet_joint_rerank_hybrid_gold_only_v1` run on the same split:

| Metric | Previous | Retrain | Change |
|---|---:|---:|---:|
| Reranked exact@1 | 58.03% | 56.76% | -1.27 pp |
| Reranked samples-F1 | 74.70% | 73.29% | -1.41 pp |
| Reranked Jaccard | 69.66% | 68.27% | -1.38 pp |
| Reranked exact@10 | 72.97% | 67.09% | -5.88 pp |
| Mean unique candidates | 7.66 | 3.69 | -3.98 |

Decision: do not replace the previous production GFlowNet checkpoint. The new run is valid and reproducible, but candidate diversity and top-k coverage regressed.

### MLP baseline

Best epoch: 17. Best validation samples-F1: 67.00%.

| Test metric | Deployable threshold result | True-count diagnostic |
|---|---:|---:|
| Exact set / subset accuracy | 27.66% | 53.10% |
| Samples-F1 | 66.76% | 70.83% |
| Samples Jaccard | 57.13% | 65.21% |

The true-count result is diagnostic only because the true precursor-set size is not available during real inference.

Compared with the previous baseline, deployable exact accuracy improved by 0.32 pp and samples-F1 improved by 0.82 pp. The GFlowNet remains substantially stronger.

## 4. Stage3 Results

### Quantile ensemble in physical units

The historical quantile report computed errors in normalized space. This run fixes the report by applying the `schema.json` mean and standard deviation before calculating °C/h metrics.

| Test metric | Temperature | Time |
|---|---:|---:|
| Comparable rows | 507 | 460 |
| Top1 MAE | 196.85 °C | 21.58 h |
| Median absolute error | 123.62 °C | 10.98 h |
| Oracle MAE across 9 quantiles | 57.04 °C | 7.96 h |
| Within 100 °C / 10 h | 43.00% | 45.22% |
| Within 200 °C | 65.09% | n/a |

The old normalized-space numbers are invalid for physical-unit accuracy and are intentionally not used as a baseline.

### Discrete condition heads

| Head | Test size | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Atmosphere binary | 294 | 56.80% | 52.23% |
| Time bucket | 460 | 47.17% | 40.45% |

### Reaction-method experts

| Test metric | Result |
|---|---:|
| Temperature MAE | 180.06 °C |
| Temperature RMSE | 271.08 °C |
| Temperature R² | 0.3473 |
| Time MAE | 21.48 h |
| Time RMSE | 44.49 h |
| Atmosphere coarse accuracy | 78.91% |
| Atmosphere coarse macro-F1 | 45.18% |

The synthesis-type head reports 0% on this preserved schema and should not be used as a quality claim; its label space is degenerate for this test artifact.

The previous method-expert report used a different 35,454/538/538 dataset and different discrete label schema, so direct percentage deltas would not be methodologically valid.

## 5. Stage35 Route Results

Validation selected a blend weight of 0.70. The retrain exactly reproduces the previous Stage35 metrics because the same candidate pool, features, split and seed were used.

### Missing-aware protocol

| Rank cutoff | Strict route | Relaxed route | Usable relaxed route |
|---|---:|---:|---:|
| top1 | 17.06% | 20.72% | 28.33% |
| top3 | 24.18% | 28.58% | 37.55% |
| top5 | 26.34% | 31.09% | 40.28% |
| top10 | 29.59% | 34.55% | 44.54% |
| top200 | 46.75% | 48.85% | 60.04% |

### Strict-comparable protocol

Strict condition means temperature error ≤100 °C, time error ≤24 h, and exact lower-case atmosphere match when the reference is known. Relaxed condition uses ≤200 °C and ≤48 h with the same atmosphere rule.

| Rank cutoff | Strict route | Relaxed route | Usable relaxed route |
|---|---:|---:|---:|
| top1 | 8.71% | 10.45% | 13.51% |
| top3 | 12.77% | 14.96% | 18.42% |
| top5 | 13.95% | 16.32% | 19.87% |
| top10 | 15.53% | 18.04% | 21.89% |
| top200 | 25.46% | 26.97% | 31.66% |

Decision: the Stage35 retrain passes reproducibility. Its model checksum is identical in behavior to the previous fixed-seed result and is safe to retain.

## 6. Artifact Integrity

Large models remain outside Git according to repository policy. Remote artifacts occupy approximately 340 MB.

| Artifact | SHA-256 |
|---|---|
| Stage2 GFlowNet | `6212459de830a5625a197b3265208284f325ae7c3870cb3bb25e0f7271055f0c` |
| Stage2 candidate reranker | `5c3599539cb7cf98a5e147bba607e19bd92d1e3d215c2ccd94ee5e916e23da51` |
| Stage2 MLP | `1d97fc2a755a447ee9690604b8c5fe0eaa85ae5a01515cf0b4703894ca762f78` |
| Stage3 method experts | `3c34711e0f584609cb780f0683dbd700570011afd92ee840bd23efdf40fe6ed2` |
| Stage35 route reranker | `50cdfa4b23e26a473ba83f5502c6dca73f9374a2d536b9004d1044c6785a1c49` |

## 7. Final Recommendation

1. Keep the previous production GFlowNet checkpoint because the retrain has lower candidate diversity and top-k exact coverage.
2. The new MLP is a modestly improved baseline, but it does not replace the GFlowNet.
3. Use the corrected physical-unit quantile metrics for all future Stage3 reports; discard the historical normalized-space accuracy values.
4. The method-expert temperature and atmosphere results are useful, but time prediction and discrete macro-F1 remain bottlenecks.
5. The Stage35 reranker is fully reproducible on the fixed candidate set and remains the strongest evaluated final-route ordering.

