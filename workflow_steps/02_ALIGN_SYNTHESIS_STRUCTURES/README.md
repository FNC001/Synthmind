# Step 02：结构—合成对齐

默认模式只验证冻结的 direct-aligned checkpoint。需要从两个最初合成 JSON 和本地 MP-labelled 结构归档重建时，必须显式给出新的输出目录。归档包含 `mp_metadata_backed` 与 `conventional_supplement_no_upstream_metadata` 两种来源证据层级，输出与报告不能把后者改写成已有官方 MP/API provenance：

```bash
python workflow_steps/02_ALIGN_SYNTHESIS_STRUCTURES/run_alignment.py \
  --data-root /path/to/authorized/data \
  --source-root "$PWD" \
  --output-dir /path/to/work/02_alignment_rebuild
```

包装器加载历史匹配算法后覆盖机器绝对路径；不会修改冻结原始目录。输出的 `poscar_path` 仍可能是运行机器绝对路径，进入下一步前由清洗器转换为发布包相对路径并保留 `source_poscar_path`。
