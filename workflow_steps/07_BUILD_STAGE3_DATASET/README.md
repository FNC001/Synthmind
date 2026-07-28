# Step 07：构建最终 Stage3 数据

默认验证冻结数据；`MODE=rebuild` 从 chem-checked core-method checkpoint 和冻结 schema 重新进行 family 分组切分。

```bash
MODE=rebuild SYNTHMIND_DATA_ROOT=/path/to/data WORK_ROOT=/path/to/work \
  bash workflow_steps/07_BUILD_STAGE3_DATASET/run_step.sh
```

期望：19,788/2,422/2,421 行，155 特征，formula/material/family 分组零交叉。
