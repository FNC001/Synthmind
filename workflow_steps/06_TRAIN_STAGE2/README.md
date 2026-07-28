# Step 06：训练最终 Stage2

默认验证冻结权重。`MODE=train_experts` 只训练三个因子化专家；`MODE=train_full` 依次运行 A/B/C → s9141 meta → s9144 gate → s9156 meta → s9161 final gate，并把全部结果写到包外。最终冻结候选的逐字节复现由 Step 09 完成，因为它使用保存的权重/策略并同时验证三项指标。

```bash
MODE=train_full OUTPUT_DIR=/path/out \
  SYNTHMIND_DATA_ROOT=/path/to/data \
  bash workflow_steps/06_TRAIN_STAGE2/run_step.sh
```

三专家配置与冻结 metrics.json 一致：A=`2048/6/seed9140`，B=`1536/4/seed9151`，C=`2048/6/seed9152`。所有模型使用 canonical 数据和 GPU。

早期 `s8720` 是导入的不可变候选 checkpoint；完整串联从它开始，不能伪称本步骤从原始数据重新训练了 s8720。GPU、PyTorch、LightGBM 版本差异可能使从头复训结果有小幅数值波动；“逐字节复现”仅指冻结权重的 Step 09 回放。
