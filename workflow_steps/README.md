# Synthmind V1.0 编号化工作流

功能：共享源码保持原导入结构；01–10 包装入口定义执行顺序。

- 上游：`发布包数据`
- 下游：`训练、评估和推理工作目录`
- 状态：V1.0 正式代码入口；数据通过 `SYNTHMIND_DATA_ROOT` 外挂。
- 执行入口：`scripts/run_v1_workflow.sh` 或 `synthmind workflow`。

代码位于仓库根目录，输出写入 `WORK_ROOT`；不会覆盖外部数据根目录。
