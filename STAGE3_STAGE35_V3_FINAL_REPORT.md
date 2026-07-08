# SynPred Stage3 / Stage35 v3 Final Report

生成日期：2026-06-12

本轮目标是把上一轮 Stage3 v3 / Stage35 v3 从诊断实验整理成严格无泄漏、可正式汇报的 final version。最终完成了 train/val/test route candidates v3 final、dual-protocol 评价、正式 Stage35 reranker v3 final、score calibration final，并更新了 best-current inference selector。

## 1. 数据一致性检查

审计脚本：`scripts/06_eval/audit_stage3_v3_data_consistency.py`

输出目录：`outputs/evaluation/stage3_stage35_v3_final_audit_20260612`

审计结论：未发现 blocking consistency issues。

| split | predicted-precursor rows | target rows | sample_id 对齐 | chemistry ok rows | mean precursor F1 | mean precursor Jaccard | open-generated rows | repair rows |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| train | 29,208 | 29,208 | yes | 29,208 | 0.4532 | 0.4092 | 9,340 | 9,340 |
| val | 3,658 | 3,658 | yes | 3,658 | 0.4363 | 0.3952 | 0 | 0 |
| test | 3,664 | 3,664 | yes | 3,664 | 0.4606 | 0.4172 | 0 | 0 |

候选表检查：

| table | split | rows | samples | NaN score | empty precursor | empty condition | source flags |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Stage3 condition candidates | train | 547,852 | 29,208 | 0 | n/a | 0 | open/repair penalty retained |
| Stage3 condition candidates | val | 73,106 | 3,658 | 0 | n/a | 0 | retained |
| Stage3 condition candidates | test | 73,237 | 3,664 | 0 | n/a | 0 | retained |
| Stage35 route candidates | train | 540,982 | 29,208 | 0 | 0 | 0 | retained |
| Stage35 route candidates | val | 1,447,340 | 3,658 | 0 | 0 | 0 | retained |
| Stage35 route candidates | test | 1,449,580 | 3,664 | 0 | 0 | 0 | retained |

重要说明：由于当前没有 Stage2 v5 train top20 OOF precursor candidate pool，train route candidates 使用 `stage3_condition_dataset_predprec_oof_v3_20260610/train.csv` 中的一条 no-leakage pseudo precursor candidate 与 top20 condition candidates 组合。该设置避免了 test 泄漏，但 train candidate 分布与 val/test 的 Stage2 top20 candidate 分布不完全一致。正式报告中可以使用 final reranker test 指标，但需要保留这一限制说明。

## 2. Route Candidates v3 Final

构建脚本：`scripts/06_eval/build_stage35_route_candidates_v3.py`

输出目录：`outputs/evaluation/stage35_route_candidates_v3_final_20260612`

| split | samples | route candidates | avg candidates/sample | top1 relaxed route | top10 relaxed route | top200 relaxed route | top400 relaxed route | top400 usable relaxed | precursor exact upper bound | condition relaxed upper bound |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 29,208 | 540,982 | 18.52 | 15.23% | 23.94% | 24.63% | 24.63% | 32.14% | 30.31% | 78.95% |
| val | 3,658 | 1,447,340 | 395.67 | 15.55% | 24.44% | 51.09% | 52.10% | 62.99% | 68.15% | 76.57% |
| test | 3,664 | 1,449,580 | 395.63 | 15.26% | 26.12% | 50.46% | 51.94% | 63.21% | 67.52% | 76.26% |

Route candidate 上限说明：test top400 relaxed route 上限为 51.94%，usable relaxed route 上限为 63.21%。这说明 Stage35 排序器的 top1 提升空间存在，但 route 级 exact 上限仍主要受 Stage2 precursor exact coverage 和 Stage3 condition coverage 共同限制。

## 3. Dual Protocol Metrics

评价脚本：`scripts/06_eval/evaluate_stage3_stage35_metrics_dual_protocol.py`

输出目录：`outputs/evaluation/stage3_stage35_v3_dual_protocol_20260612`

两套口径：

- missing-aware：atmosphere/solvent 缺失样本不因为缺失字段被额外惩罚，是 v3 的主要优化口径。
- strict-comparable-to-v2：missing/unknown atmosphere 按更严格方式处理，用于和旧版结果横向比较。

### Stage3 Condition

| protocol | top1 strict | top1 relaxed | top10 strict | top10 relaxed | top20 relaxed | oracle relaxed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| missing-aware | 16.54% | 36.49% | 66.40% | 69.27% | 76.26% | 76.83% |
| strict-comparable | 8.65% | 17.96% | 31.80% | 33.19% | 39.85% | 40.42% |

### Stage35 Route Raw

| protocol | top1 strict | top1 relaxed | top10 strict | top10 relaxed | top200 relaxed | top400 relaxed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| missing-aware | 7.10% | 15.26% | 12.66% | 26.12% | 50.46% | 51.94% |
| strict-comparable | 3.98% | 8.13% | 7.12% | 14.33% | 28.52% | 29.89% |

解读：missing-aware 的 Stage3 condition top10 relaxed 达到 69.27%，说明 v3 对温度/时间区间推荐有明显价值；strict-comparable 下显著下降，主要来自 atmosphere missing/unknown 被严格惩罚，因此不能直接把 missing-aware 指标与旧 v2 strict 指标混写。

## 4. Reranker v3 Final

训练脚本：`scripts/04_train/stage35/train_stage35_route_reranker_v3_final.py`

输出目录：`runs/stage35/route_reranker_v3_final_20260612`

训练设置：

- train route candidates 用于训练。
- val route candidates 用于 early stopping、blend alpha 选择。
- test route candidates 只用于最终评估。
- 禁止将 `precursor_exact_if_eval`、`precursor_jaccard_if_eval`、condition/route hit 等 evaluation 字段作为特征。
- 模型包含 LightGBM LambdaRank、LightGBM binary classifier、raw score + LambdaRank blend。
- validation 选择的 blend alpha = 0.70。

Test 对比：

| model | protocol | top1 strict | top1 relaxed | top10 strict | top10 relaxed | top200 relaxed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| raw route score | missing-aware | 7.10% | 15.26% | 12.66% | 26.12% | 50.46% |
| diagnostic reranker v3 | missing-aware | 19.41% | 21.32% | n/a | 33.08% | 50.22% |
| final LambdaRank | missing-aware | 14.14% | 15.72% | 26.15% | 29.28% | 47.84% |
| final binary | missing-aware | 9.53% | 13.35% | 21.18% | 28.08% | 49.13% |
| final blend | missing-aware | 17.06% | 20.72% | 29.59% | 34.55% | 48.85% |
| raw route score | strict-comparable | 3.98% | 8.13% | 7.12% | 14.33% | 28.52% |
| final LambdaRank | strict-comparable | 6.33% | 7.01% | 13.05% | 14.57% | 26.20% |
| final binary | strict-comparable | 5.21% | 7.10% | 12.06% | 15.53% | 27.48% |
| final blend | strict-comparable | 8.71% | 10.45% | 15.53% | 18.04% | 26.97% |

结论：final blend 在 missing-aware 和 strict-comparable 两套口径的 top1/top10 上均超过 raw route score，因此可以作为当前默认 Stage35 ranking。需要注意，final blend 的 top200 relaxed 低于 raw upper-bound 排序，这说明 reranker 更偏向把高置信路线提前，而不是最大化 top200 oracle coverage。

## 5. Stage35 Score Calibration Final

校准脚本：`scripts/06_eval/calibrate_stage35_route_scores_v3_final.py`

输出目录：`outputs/evaluation/stage35_route_score_calibration_v3_final_20260612`

最佳权重在两套协议下均为 search_id=4：

```json
{
  "route_total_score_raw": 0.8,
  "precursor_rank_score": 0.5,
  "condition_rank_score": 0.5,
  "rank_product_score": 0.7,
  "contains_open_generated_precursor": -0.2,
  "contains_repair_precursor": -0.1
}
```

Test 指标：

| protocol | top1 strict | top1 relaxed | top10 strict | top10 relaxed | top200 relaxed | top400 relaxed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| missing-aware | 7.10% | 15.26% | 15.61% | 30.46% | 50.60% | 51.94% |
| strict-comparable | 3.98% | 8.13% | 8.60% | 16.35% | 28.66% | 29.89% |

校准权重对 top10 有提升，但 top1 没有超过 final blend。因此最终部署选择 reranker final blend，calibrated raw score 作为 fallback。

## 6. 与上一轮对比

上一轮 v3 diagnostic：

| item | metric |
| --- | ---: |
| Stage3 top1 relaxed condition | 36.49% |
| Stage3 top10 relaxed condition | 69.27% |
| Stage35 raw top1 relaxed route | 15.26% |
| Stage35 raw top10 relaxed route | 26.12% |
| Stage35 diagnostic reranker top1 relaxed | 21.32% |
| Stage35 diagnostic reranker top10 relaxed | 33.08% |

本轮 final：

| item | missing-aware | strict-comparable |
| --- | ---: | ---: |
| Stage3 top1 relaxed condition | 36.49% | 17.96% |
| Stage3 top10 relaxed condition | 69.27% | 33.19% |
| Stage35 raw top1 relaxed route | 15.26% | 8.13% |
| Stage35 raw top10 relaxed route | 26.12% | 14.33% |
| Stage35 final blend top1 relaxed route | 20.72% | 10.45% |
| Stage35 final blend top10 relaxed route | 34.55% | 18.04% |

最终 reranker top1 relaxed 比 diagnostic 的 21.32% 略低，为 20.72%，但它是正式 train/val/test 隔离流程下得到的结果，可信度更高。top10 relaxed 从 diagnostic 的 33.08% 提升到 final 的 34.55%。

## 7. 最终部署选择

当前 best-current pipeline：

1. Stage2：`stage2_v5` chemistry-checked precursor candidates。
2. Stage3：`distributional_condition_v3_20260610`，输出 temperature/time 区间、atmosphere/solvent 置信度。
3. Stage35：`route_reranker_v3_final_20260612` final blend，validation-selected alpha=0.70。
4. Fallback：`route_calibrated_score_v3_final`。

是否默认使用 reranker v3 final：是。

理由：final blend 在 test missing-aware 与 strict-comparable 两套协议中，top1 relaxed 和 top10 relaxed 均优于 raw route score。

可写入正式报告的指标：

- Stage3 v3 missing-aware condition 指标。
- Stage3 v3 strict-comparable condition 指标。
- Stage35 raw / calibrated / final reranker 的 dual-protocol 指标。
- final reranker 的 test 指标。

只能作为诊断说明的指标：

- 上一轮 diagnostic reranker v3 指标，因为它使用 val route candidates 训练，不是严格 final split 流程。

推理输出注意事项：

- 温度和时间推荐应输出为区间，不应解释为唯一精确值。
- atmosphere/solvent 如果 missing 或 low confidence，需要标注 low confidence。
- open-generated / repair precursor 会降低 route confidence。
- route score 是模型排序分数，不是实验成功概率。

## 8. 下一步建议

当前 final route top1 的主要限制仍来自 Stage2 precursor coverage 和 Stage3 atmosphere/solvent 缺失处理。建议下一轮优先做：

1. 生成真正的 Stage2 train OOF top20 precursor candidates，使 reranker train 分布和 val/test 完全一致。
2. 继续提高 Stage3 condition candidate quality，特别是 atmosphere/solvent 的 missing-aware modeling。
3. 引入 synthesis paragraph / route text embedding，用文本路线语义帮助区分相近条件。
4. 对 melt_arc、hydro_solvothermal、flux_molten_salt、non-core methods 做 method-specific condition expert。
5. 对 high-frequency open-generated / repair precursor 进行人工清洗，减少 Stage35 对修复候选的惩罚不确定性。

## 9. 关键输出文件

- `outputs/evaluation/stage3_stage35_v3_final_audit_20260612/data_consistency_audit.md`
- `outputs/evaluation/stage35_route_candidates_v3_final_20260612/route_candidate_build_report.md`
- `outputs/evaluation/stage3_stage35_v3_dual_protocol_20260612/dual_protocol_metrics_report.md`
- `runs/stage35/route_reranker_v3_final_20260612/reranker_v3_final_training_report.md`
- `outputs/evaluation/stage35_route_score_calibration_v3_final_20260612/route_score_calibration_v3_final_report.md`
- `scripts/07_infer/best_current_route_predictor/predict_best_current_route.py`
