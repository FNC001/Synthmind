# Step 04：分组切分、描述符与结构嵌入

执行顺序：group split → 131 维描述符 → CHGNet graph cache → 64 维 CHGNet embedding → 195 维 hybrid features → Stage2/Stage3 task views。

共享源码中的对应入口：

1. `scripts/01_split/01_make_group_split.py`
2. `scripts/02_features/01_build_structdesc_features.py`
3. `scripts/03_graph/03_build_chgnet_cache_stage2.py`
4. `scripts/03_graph/export_chgnet_stage2_embeddings.py`
5. `scripts/02_features/05_build_hybrid_features.py`

这些历史脚本参数较多，运行前先执行各脚本 `--help`，输出统一写入 `WORK_ROOT/04_build_features/`。

特征重建必须以 `04_SPLITS/00_PORTABLE_SPLITS/` 中对应的 base 或 route 子目录为输入；`01_BASE_GROUP_SPLIT` 和 `02_ROUTE_GROUP_SPLIT` 保留的是历史 checkpoint，路径基准不统一。便携目录保留 `source_*_path` 追溯字段，工作路径统一指向发布包内结构，16 个文件的目标缺失数为 0。
