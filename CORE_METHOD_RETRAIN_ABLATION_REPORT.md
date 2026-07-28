# Core-Method Retrain Ablation Report

## 1. 数据设置

本轮只把 `solid_state`、`solution`、`melt_arc` 定义为 core methods。`other`、`hydro_solvothermal`、`precipitation`、`flux_molten_salt`、`thermal_decomposition`、`mechanochemical`、`sol_gel`、`combustion` 暂不进入核心主模型，因为它们要么是混杂类别，要么样本数较少且当前预测稳定性不足。

All-method method-stratified 数据规模：

| split | n |
|---|---:|
| train | 29,208 |
| val | 3,658 |
| test | 3,664 |

Core-method 数据规模：

| split | n | solid_state | solution | melt_arc |
|---|---:|---:|---:|---:|
| train | 19,722 | 8,899 | 8,940 | 1,883 |
| val | 2,441 | 1,061 | 1,124 | 256 |
| test | 2,468 | 1,142 | 1,102 | 224 |

Core dataset artifacts:

- Stage2 core dataset: `data/interim/generative/stage2_setpred_dataset/descriptor/core_methods_ss_solution_meltarc_20260610_relaxed_only`
- Stage3 core dataset: `data/interim/generative/stage3_condition_dataset_mixed/core_methods_units_normalized_20260610_poscar_geom1024`
- Core templates: `data/interim/templates/stage2_core_method_templates_20260610`

## 2. Stage2 对照结果

三组主对照：

| Setting | train methods | val/calibration methods | test methods | top1 | top10 | top200 | top500 |
|---|---|---|---|---:|---:|---:|---:|
| A all-train core-eval | all | all | core | 44.81% | 70.14% | 82.13% | 85.25% |
| B core-only best | core | core | core | 46.15% | 69.17% | 81.69% | 85.33% |
| B core-retrain no external-rerank | core | core | core | 39.71% | 63.94% | 81.28% | 85.33% |
| C all-train core-val | all | core | core | 41.61% | 68.52% | 81.77% | 85.25% |

说明：`B core-only best` 使用此前已验证的 core-method per-method calibrated 结果，包含 core candidate pool top1000 和 external rerank features；本轮新生成的 `core_retrain` 目录作为无 external-rerank 消融，top500 保持但 top1 明显下降，说明排序特征对 top1 很关键。

By-method 对比：

| Setting | method | n | top1 | top10 | top200 | top500 |
|---|---|---:|---:|---:|---:|---:|
| A | solid_state | 1,142 | 49.12% | 69.09% | 80.39% | 85.03% |
| A | solution | 1,102 | 34.03% | 68.06% | 81.94% | 83.85% |
| A | melt_arc | 224 | 75.89% | 85.71% | 91.96% | 93.30% |
| B best | solid_state | 1,142 | 50.44% | 67.16% | 80.39% | 85.11% |
| B best | solution | 1,102 | 35.48% | 67.97% | 80.76% | 83.94% |
| B best | melt_arc | 224 | 76.79% | 85.27% | 92.86% | 93.30% |
| C | solid_state | 1,142 | 47.02% | 66.46% | 79.77% | 85.03% |
| C | solution | 1,102 | 29.22% | 67.51% | 81.76% | 83.85% |
| C | melt_arc | 224 | 75.00% | 83.93% | 91.96% | 93.30% |

Stage2 结论：

- 如果目标是 top1 推荐，`core-only best` 最优，top1 达到 46.15%。
- 如果目标是 top10 覆盖，A 略高，为 70.14%，但 top1 低于 B。
- 三组 top500 基本都在 85.25%-85.33%，说明当前核心瓶颈不是候选池总覆盖，而是 top1/top10 排序和 OOV/alias。
- 本轮目标 `top1 ≥48%、top10 ≥72%、top200 ≥84%、top500 ≥87%` 尚未达到；最接近的是 B best 的 top1 46.15% 和 top500 85.33%。

## 3. Stage3 对照结果

Stage3 使用相同 core test 集评估。`strict condition` 表示温度误差 ≤100 °C、时间误差 ≤24 h、气氛正确；`relaxed condition` 表示温度误差 ≤200 °C、时间误差 ≤48 h、气氛正确。下表 success 使用 all core rows 作为分母，缺失条件视为未命中。

| Setting | train methods | val/calibration methods | test methods | temp MAE | time MAE | atm acc | strict cond | relaxed cond |
|---|---|---|---|---:|---:|---:|---:|---:|
| A all-train core-eval | all | all | core | 201.06 °C | 29.61 h | 56.44% | 10.58% | 16.45% |
| B core-only retrain | core | core | core | 200.50 °C | 29.60 h | 55.35% | 11.02% | 16.61% |
| C all-train core-val calibration | all | core | core | 201.01 °C | 29.57 h | 56.44% | 10.62% | 16.45% |

By-method Stage3 core-only retrain:

| method | n | temp MAE | time MAE | atm acc | strict cond | relaxed cond |
|---|---:|---:|---:|---:|---:|---:|
| solid_state | 1,142 | 204.94 °C | 33.30 h | 60.25% | 10.25% | 17.43% |
| solution | 1,102 | 184.39 °C | 19.52 h | 47.09% | 13.61% | 17.79% |
| melt_arc | 224 | 263.60 °C | 77.74 h | 73.72% | 2.23% | 6.70% |

Stage3 结论：

- Core-only retrain 对 Stage3 是小幅正向：strict condition 从 10.58% 到 11.02%，relaxed 从 16.45% 到 16.61%，time MAE 维持在约 29.6 h。
- A 的 atmosphere accuracy 略高，但 B 的 condition success 更好，因此核心路线推荐中 Stage3 建议使用 B。
- `melt_arc` 的温度/时间字段噪声大，strict/relaxed condition 较低；后续应降低其工艺条件置信度，更多依赖 process-type / atmosphere / direct elemental route。

## 4. 完整路线结果

已联通的完整路线候选目前来自 all-method Stage2/Stage3/Stage35，再过滤 core-like route rows；新的 `Stage2 core-only + Stage3 core-only` 组合尚未重新接入 Stage35 route candidate generator，因此下面是可用端到端参考，不作为 B 组合的最终上限。

| Route setting | n | top1 strict route | top1 relaxed route | oracle strict route | oracle relaxed route |
|---|---:|---:|---:|---:|---:|
| method-stratified route top10, core-filtered available rows | 935 | 5.13% | 12.09% | 6.63% | 14.44% |
| Stage35 rerank, core-filtered available rows | 935 | 4.92% | 11.66% | 6.63% | 14.44% |
| previous old/core Stage35 reference | 538 | 8.18% | 16.73% | 10.97% | 21.75% |
| full method-stratified reference | 3,664 | 3.49% | 6.85% | 5.90% | 11.35% |

Core-filtered complete-route success is already higher than full method-stratified reference, but the new best core Stage2/Stage3 artifacts still need a dedicated Stage35 regeneration step before claiming final end-to-end route accuracy.

## 5. 最终推荐

1. 是否应该 core-only retrain？

Stage2 推荐使用 core-only best，而不是 all-train core-val。原因是 B 的 top1 最高，且核心方法更可解释。A 的 top10 略高，但 top1 不如 B；C 在 top1 上明显下降。

2. all-train + core-val 是否更好？

不更好。C 的 top1 为 41.61%，低于 A 的 44.81% 和 B 的 46.15%。保留 all-method 训练确实保留了部分标签覆盖，但 core-val calibration 没有解决排序干扰。

3. Stage2 和 Stage3 是否选择同一方案？

建议 Stage2 与 Stage3 都选择 core-only best/retrain。Stage2 的收益更明显；Stage3 收益较小，但 strict/relaxed condition success 最好。

4. 当前最佳核心推理 artifacts：

- Stage2 dataset: `data/interim/generative/stage2_setpred_dataset/descriptor/core_methods_ss_solution_meltarc_20260610_relaxed_only`
- Stage2 MLP: `runs/stage2/mlp_core_methods_ss_solution_meltarc_20260610_descriptor`
- Stage2 set-size: `runs/stage2/set_size_core_methods_ss_solution_meltarc_20260610_descriptor`
- Stage2 best calibrated candidates: `outputs/evaluation/stage2_score_calibration_core_methods_20260610_per_method`
- Stage2 formal core_retrain ablation: `outputs/evaluation/stage2_score_calibration_core_retrain_20260610`
- Stage3 core model: `runs/stage3/lgbm_method_experts_core_methods_20260610`
- Inference router: `scripts/07_infer/best_current_route_predictor/predict_best_current_route.py`

5. 非核心方法如何 fallback？

如果用户指定 `solid_state`、`solution`、`melt_arc`，默认使用 core-method artifacts。若用户指定非核心方法，则回退到 all-method v5 Stage2 和 all-method Stage3，并降低 confidence。若用户未指定反应方式，则默认尝试三类 core methods 并提示可靠性下降。

## 6. 产物检查

新增或更新的主要产物：

- `outputs/evaluation/stage2_core_ablation_20260610/expA_alltrain_coreeval`
- `outputs/evaluation/stage2_core_ablation_20260610/expC_alltrain_coreval`
- `outputs/evaluation/stage2_candidate_pool_core_retrain_20260610`
- `outputs/evaluation/stage2_score_calibration_core_retrain_20260610`
- `outputs/evaluation/stage3_core_ablation_20260610/expA_alltrain_coreeval`
- `outputs/evaluation/stage3_core_ablation_20260610/expB_core_retrain`
- `outputs/evaluation/stage3_core_ablation_20260610/expC_alltrain_coreval`
- `outputs/evaluation/core_route_available_20260610`

下一步最高收益工作：将 `Stage2 core-only best + Stage3 core-only retrain` 接入 Stage35 route candidate generator，重新训练/校准核心路线 reranker；同时针对 solid_state 和 solution 的 OOV/alias 做高频人工清洗，以冲击 top1 48% 和 top500 87%。
