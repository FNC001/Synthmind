# Step 10：新结构推理

共享源码保留历史 structure-to-route pipeline，但它最初绑定旧 production 模型，不等同于本发布版最终 family ensemble。为避免误用，当前入口只做依赖检查，不自动把旧 pipeline 标成最终模型部署。

正式部署适配必须同时满足：

1. 新结构使用与训练完全相同的 219/155 维特征构建；
2. Stage2 加载 s9161 所需候选专家和门控器；
3. Stage3 加载 NF/CVAE/Diffusion 三权重并按 64+64+64 集成；
4. 输出记录模型哈希、family 路由、候选和条件分桶参数。

冻结验证和训练复现不依赖本部署适配器。

