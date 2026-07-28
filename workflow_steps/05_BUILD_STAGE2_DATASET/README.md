# Step 05：构建最终 Stage2 数据

默认 `MODE=validate`；设置 `MODE=rebuild` 会在包外重新生成 full-family 和 canonical 数据。MatSciBERT embedding 属于随后可选的 GPU/Transformer 步骤，命令记录在本目录 README，冻结 embedding 已包含在最终 canonical 数据中。

```bash
MODE=rebuild SYNTHMIND_DATA_ROOT=/path/to/data WORK_ROOT=/path/to/work \
  bash workflow_steps/05_BUILD_STAGE2_DATASET/run_step.sh
```

期望：18,127/2,170/2,300 行，219 特征，1,780 个规范标签。
