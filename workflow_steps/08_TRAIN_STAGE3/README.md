# Step 08：训练 Stage3 与重建集成

`MODE=validate` 检查三个 checkpoint 和冻结集成。`MODE=train_all` 使用冻结配置在 GPU 上重训 NF、CVAE、Diffusion，然后从新生成的三个 `val_samples.npz` 自动重建 64+64+64 集成，并导出条件指标和候选；输出全部写到包外。只重建冻结的 192-sample 集成时运行 `run_rebuild_ensemble.sh`，也可通过 `NF_SAMPLES/CVAE_SAMPLES/DIFFUSION_SAMPLES` 指定新的样本。

三个模型训练约使用 29.2M、43.2M、128.6M 参数。冻结运行分别采用 seed 8060、8040、8320。

注意：最终 Stage3 数据中 train/val/test 分别有约 80.43%/78.28%/78.93% 的行使用 `true_precursor_fallback`。因此模型是 mixed-input conditional generator，不是纯预测前驱体条件模型；范围限制见 `09_ACCURACY_EVALUATION/03_THREE_METRICS/E2E_SCOPE_AUDIT.json`。
