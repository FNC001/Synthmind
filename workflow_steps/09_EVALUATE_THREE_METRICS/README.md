# Step 09：独立复算三项精度

该入口会重建 Stage2 最终 gate 候选、重建 Stage3 192-sample 集成、重新生成条件候选，并用独立评估器逐整数命中数对比冻结 JSON。

```bash
SYNTHMIND_DATA_ROOT=/path/to/data \
  bash workflow_steps/09_EVALUATE_THREE_METRICS/run_step.sh /tmp/synthmind_three_metrics
```

全部使用验证集；本步骤不计算最终模型 test 指标。注意当前 test 分割曾被历史非最终模型访问，不能称为从未开启的 lockbox；详见 `09_ACCURACY_EVALUATION/02_TEST_LOCKBOX/TEST_SPLIT_DISCLOSURE.json`。
