# RetailVision v0.4 · 29 SKU 联合训练

2026-11-13（Asia/Shanghai）。模型与实验版本为 `retail_29sku_v004`。

已完成 RTX 4090 CUDA 分割与识别训练、最佳权重保存、训练图库重建、版本发布和独立进程的离线 CUDA 重载。训练及建库时间为 12:03:03–12:19:22，约 16 分 19 秒，不含此前的数据准备与此后的发布验证。

## 数据与划分

| 来源 | 图片 | 实例 | SKU |
|---|---:|---:|---:|
| v0.3 已保存的合并数据 | 51 | 306 | 19 |
| `4722943 -1113_1-10.zip`，原图来自 `20261112_1-10` | 39 | 220 | 10 |
| 合计 | 90 | 526 | 29 |

分割和识别共享整图划分：训练 81 张、475 个实例，验证 9 张、51 个实例。两边均覆盖 29 个 SKU，每款 SKU 至少有两个训练参考实例。保留 v0.3 的全部五张验证图，新数据使用固定 seed=42 选择四张且覆盖其全部十个 SKU。没有完全相同的重复图片，训练／验证图片 ID 交集为零；图库仅包含训练参考实例。

这是同场景图片留出实验，记录 `split_unit=image`、`validation_scope=same_capture_image_holdout`、`independent_validation=false`。没有为新图伪造采集组，也没有将不同拍摄角度当成独立采集组。

数据版本：`814e69cdb4701c18d98a79609da6e7187dd0f9409e160dbc0f7ebb4f3d2e01bc`。

原始标注 ZIP、SKU 清单、逐图来源及 SHA256、合并脚本和验证名单保存在 `datasets/incoming/retail_29sku_v004/`。沿用 v0.3 已审核的派生标注；本次新标注不改轮廓、SKU 或 `occluded`。SKU 清单的 `specificationl` 拼字仅在派生主表中规范为 `specification`。原标注 125／127 的轻微边界交叠及未填写的遮挡属性保留并记录，当前训练不使用 `occluded`。合并数据通过正式导入器的图像解码、尺寸、SKU 属性及几何校验。

## 训练

两套模型都从本地通用预训练权重开始：RTMDet-Ins-s 使用 COCO 权重，DINOv2 使用通用 ViT-B/14 并重新初始化映射层，避免将已见过验证图片的商品模型作为训练起点。

| 设置 | 分割 | 识别 |
|---|---|---|
| 模型 | RTMDet-Ins-s，单类商品实例 | 冻结 DINOv2 ViT-B/14 + Linear 768→256 |
| 轮数／Batch | 50／4，共 1050 次迭代 | 20／12 |
| 输入 | 960×960 范围、保持比例 | 224×224 商品 ROI |
| 优化 | AdamW，lr=0.001，AMP、warmup 和余弦衰减 | AdamW，lr=0.0001，TripletMarginLoss、hard triplet mining |
| 最佳选择 | 验证 Mask mAP，第 40 轮 | 验证 Top-1，平分时比较平均 SKU 间隔，第 19 轮 |

沿用 v0.3 训练配置，未启用新增的 ROI 扰动或分类辅助损失。两阶段依次在 `retailvision-seg` 和 `retailvision-reid` 环境运行，两个神经网络均要求 CUDA。

## 最佳权重结果

| 指标 | 结果 |
|---|---:|
| Mask mAP@0.50:0.95 | 0.976 |
| Mask AP@0.50 | 1.000 |
| BBox mAP@0.50:0.95 | 0.889 |
| DINO 标注 ROI Top-1 | 51/51（100%） |
| DINO 平均正确 SKU 与其他 SKU 相似度间隔 | 0.221257 |

分割表格引用最佳第 40 轮的指标，而非最后一轮。DINO 映射层参数实际更新，最后一轮相对初始化的参数 L2 变化为 1.263817。

识别数字基于人工标注 Mask 裁切的 ROI，不是完整分割加识别的端到端准确率。当前数据为同场景小样本，不能据此得出独立场景泛化、未知商品拒识或生产精度结论；本轮扩展了类别和验证图片，与 v0.3 的汇总指标不可直接作等条件比较。

## 产物与验证

- 分割：`models/segmentation/retail_29sku_v004/model.pth` 及展开后的 `inference_config.py`。
- 识别：`models/recognition/retail_29sku_v004/model.pth` 为最佳映射层；通用主干仍为 `models/recognition/dinov2_b14_v001/model.pth`。
- 图库：`datasets/gallery_retail_29sku_v004/`，475 个训练参考实例，29 SKU，256 维；实验原始图库仍保留在 `work_dirs/retail_29sku_v004/gallery/`。
- 推理配置：`configs/inference/retail_29sku_v004.json`；拒识配置为同名 `_policy.json`。
- 实验记录：`work_dirs/retail_29sku_v004/experiment.json`、`training_summary.json`、`split_summary.json`、`publication.json` 和 `console.log`。
- 离线验证：`work_dirs/retail_29sku_v004/reload_verification.json`，以及可复现脚本 `verify_saved_models.py`。

通过现有 `tools/publish.py` 核对最佳权重、数据版本与图库来源后发布。独立进程禁止 Python 网络连接，实际重新加载两模型至 CUDA：参数与检查点一致，分割样例输出非空 Mask，51 个验证 ROI 复核仍为 51/51。参考向量重算与图库的最大绝对误差为 1.49e-08；图库和验证图片交集为零。检查的 24 份历史权重及数据文件 SHA256 全部未变化，原始新 ZIP 的归档哈希匹配。

拒识沿用显式临时默认值：余弦阈值 0.8、不同 SKU Top-2 间隔 0.01，标记 `calibrated=false`、`production_eligible=false`。这不是独立已知／未知 SKU 的校准结果。

## 使用与复现

工作站模型选项已加入 **RetailVision v0.4 · 29 SKU**，可在运行设置中选择。启动指定版本：

```bash
./start-web --model retail_29sku_v004
```

用新的输出目录重建相同数据划分：

```bash
PYTHONNOUSERSITE=1 conda run -n retailvision-seg python tools/prepare_dataset.py \
  --coco datasets/incoming/retail_29sku_v004/merged.json \
  --images datasets/incoming/retail_29sku_v004/images \
  --catalog datasets/incoming/retail_29sku_v004/skus.json \
  --validation-images datasets/incoming/retail_29sku_v004/validation_images.json \
  --output datasets/retail_29sku_v004_reproduction

PYTHONNOUSERSITE=1 conda run --no-capture-output -n retailvision-seg python tools/train.py \
  --dataset datasets/retail_29sku_v004_reproduction \
  --config configs/segmentation/retail_29sku_v004.py \
  --recognition-config configs/recognition/retail_29sku_v004.py \
  --experiment-id retail_29sku_v004_reproduction
```
