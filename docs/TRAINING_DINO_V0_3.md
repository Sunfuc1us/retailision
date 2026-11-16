# 六 SKU DINO 识别映射层训练

2026-11-11（Asia/Shanghai）。实验 `retail_6sku_dino_v003`，模型版本 `dinov2_b14_6sku_v003`。

使用现有 `datasets/retail_6sku_v002/recognition` 的 90 个商品 ROI，六款各 15 个。原数据来自同一采集场景，仍保留 `no_split=true`，没有伪造采集组、拆分历史数据或把训练数据当作验证数据。

训练显式使用 `--task recognition --no-validation`：冻结本地 DINOv2 ViT-B/14 主干，训练 Linear 768→256，TripletMarginLoss 与 hard triplet mining；batch=12，每批六款各两个样本，AdamW、lr=0.0001、seed=42，共 20 轮。全部模型计算位于 RTX 4090 CUDA，训练期间禁止网络连接。实际耗时约 23.6 秒。

第一轮训练 loss 为 0.159453，最后一轮为 0.037502；映射层参数相对初始化的 L2 变化为 0.879694，确认实际发生更新。这些是训练指标，不是商品识别准确率。没有验证分数，也没有选择 best checkpoint，保存的是最后一轮权重。

[训练 loss 曲线](../work_dirs/retail_6sku_dino_v003/loss_curve.png)

产物：

- `models/recognition/dinov2_b14_6sku_v003/model.pth`：纯张量映射层，推理时作为 `projection`，不能当作 DINO 主干加载。
- `work_dirs/retail_6sku_dino_v003/last.pth`：含优化器和随机状态的训练恢复文件。
- `datasets/gallery_6sku_dino_v003/`：与新映射层匹配的 90 个 256 维参考向量。
- `work_dirs/retail_6sku_dino_v003/training_summary.json`、`loss_history.json`、`reload_verification.json`：训练及独立进程重新加载记录。

独立进程从保存的映射层重新加载 CUDA 模型，确认 head 张量与训练输出一致、输出为 256 维，并与保存图库的对应参考向量一致。此项验证保存/重载与建库链路，不构成独立商品精度评估。

旧 DINO 主干、历史分割权重和原六 SKU 数据的哈希保持不变。新模型 manifest 标记 `validation_performed=false`、`calibration_required=true`、`production_eligible=false`。旧的 SKU 相似度阈值不能直接沿用到新特征空间，当前 Workbench 推理配置保持原模型，等待独立验证和拒识校准后再切换。

后续训练默认使用 [联合训练入口](TRAINING.md)：分割和 DINO 共用独立的训练/验证划分，依次训练并分别验证，再仅用训练参考样本建库。合成 Fixture 的联合 CUDA 测试只验证这条流程，不代表零售识别精度。
