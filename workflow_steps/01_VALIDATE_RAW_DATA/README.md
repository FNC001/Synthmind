# Step 01：校验原始数据

`run_step.sh` 校验完整发布目录、两套 strict 分支、最终数据行数、三项指标和最终候选哈希。结构归档的 62,689 文件深度审计证据位于 `11_TESTS_AND_AUDITS/01_STRUCTURE_ARCHIVE_AUDIT/`。

```bash
SYNTHMIND_DATA_ROOT=/path/to/data bash workflow_steps/01_VALIDATE_RAW_DATA/run_step.sh
```

该步骤只读；报告写入包外 `WORK_ROOT`。

`MODE=full_audit` 还会从 strict 和最终 Stage2/Stage3 metadata 重建 3,451/8,478 材料结构范围索引与覆盖审计，同样写入包外工作目录：

```bash
MODE=full_audit SYNTHMIND_DATA_ROOT=/path/to/data \
  bash workflow_steps/01_VALIDATE_RAW_DATA/run_step.sh
```
