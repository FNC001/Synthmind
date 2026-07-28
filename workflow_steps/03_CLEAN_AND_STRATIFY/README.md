# Step 03：清洗、合并与分层

本步骤保存五类不可互换的产物：legacy merge、补结构 merge、base refined、route unified、units normalized。默认入口验证关键文件和两套 strict 目录均存在。

历史实现位于共享源码：

- `scripts/00_refine/03_statistic.py`
- `scripts/00_refine/04_refine_strict_exact_for_structdesc.py`
- `scripts/00_refine/05_merge_new_training_data.py`
- `scripts/00_refine/06_attach_new_structures_to_merged_data.py`
- `scripts/00_refine/07_audit_and_normalize_condition_units.py`

重建时依次执行，且输出必须写到包外 `WORK_ROOT/03_clean_and_stratify/`。旧 merge 使用 `02_LEGACY_RAW_MERGE_BASE`，基础 Stage2 使用 `01_ALIGNMENT_SPLIT_REMOTE_FINAL`，禁止交换。

发布版另提供标准化入口：

```bash
python workflow_steps/03_CLEAN_AND_STRATIFY/standardize_jsonl_paths.py \
  --release-root /path/to/authorized/data
```

它不修改 `02_RAW_DATA/`，而是在 `03_CLEANED_AND_MERGED_DATA/00_PORTABLE_STANDARDIZED_INPUTS/` 生成标准 JSONL：历史绝对路径保存到 `source_*` 字段，正式路径改为包根相对路径，非标准 `NaN/Infinity` 改为 JSON `null`。验收以该目录 `manifest.json` 中 `status=PASS` 且 `total_missing_relative_targets=0` 为准。
