# RetailVision v0.3 · 19 SKU 监督微调

2026-11-12（Asia/Shanghai）。实验和分割模型版本为 `retail_19sku_v003`，DINO 映射层版本为 `dinov2_b14_19sku_v003`。

已在 RTX 4090 上完成 RTMDet-Ins-s 与 DINOv2 映射层的监督训练、共用验证划分、最佳权重保存、训练集图库重建，以及独立进程中的离线 CUDA 重新加载。训练时间为 13:54:04–13:59:08，约 5 分 4 秒。

## 数据与划分

| 来源 | 图片 | 标注实例 | SKU |
|---|---:|---:|---:|
| v0.2 六 SKU 数据集 | 15 | 90 | 6 |
| `4718408-1112-1-16.zip` + `202061111_1-7.zip` | 16 | 112 | 7 |
| 修正版 `4718417-1112-1-20(1).zip` + `20261111_8-13.zip` | 20 | 104 | 6 |
| 合计 | 51 | 306 | 19 |

按来源分层、seed=42 选取验证图片。训练集 46 张、274 个实例；验证集 5 张、32 个实例，占图片数的 9.8%。验证图来自旧数据 1 张、新七 SKU 数据 2 张、新六 SKU 数据 2 张。训练和验证两边均覆盖 19 个 SKU。

两模型共用整图划分，商品 ROI 随原图划分。图库仅包含训练集的 274 个实例，验证图片 ID 与图库参考图片 ID 交集为零。没有完全相同的重复图片。

原始归档、合并脚本、来源哈希及修复记录保存在 `datasets/incoming/retail_19sku_v003/`。原文件保留，派生标注中修复了六个微小自交轮廓，每个改变 0–13 个 Mask 像素；并按来源 `sku.txt` 修正三处规格单位笔误（`500m` → `500ml` 两处、`80` → `80g` 一处）。文件、图像解码、尺寸、SKU 属性和多边形通过现有严格导入校验。

数据版本：`dda5e82528e46fe7b6a13df3b99e07f6875cf599e1d3e989a6ed0074d09e662a`。数据目录为 `datasets/retail_19sku_v003/`，验证图片名单保存在其 `manifest.json` 及来源目录的 `validation_images.json`。

## 训练与验证

本次从本地通用预训练权重进行监督微调。RTM 初始权重为 COCO RTMDet-Ins-s，DINO 主干为通用 DINOv2 ViT-B/14，识别映射层重新初始化。v0.2 模型及之前六 SKU 的映射层均保留，不作为本次初始化，以确保留出的旧图片也没有被训练起点见过。

| 设置 | RTMDet-Ins-s | DINOv2 识别 |
|---|---|---|
| 监督信号 | `retail_product` 单类框和 Polygon Mask | 19 个 SKU 标签 |
| 训练部分 | 分割模型参数 | 冻结 ViT-B/14 主干，训练 Linear 768→256 |
| 训练长度 / Batch | 50 epoch，600 次迭代 / 4 | 20 epoch / 12 |
| 优化 | AdamW，lr=0.001，AMP，warmup 后余弦衰减 | AdamW，lr=0.0001，TripletMarginLoss 和 hard triplet mining |
| 输入 | 960×960 范围、保持比例 | 224×224 标注商品 ROI |
| 最佳权重选择 | 验证 Mask mAP，选第 50 轮 | 验证 Top-1，选第 8 轮 |

增强保留包装文字方向，没有水平镜像。分割与识别依次使用独立 Conda 环境运行。

| 最佳权重的验证指标 | 结果 |
|---|---:|
| RTM Mask mAP@0.50:0.95 | 0.974 |
| RTM Mask AP@0.50 | 0.999 |
| RTM BBox mAP@0.50:0.95 | 0.960 |
| DINO 标注 ROI Top-1 | 32/32（100%） |

DINO 第 20 轮的 Top-1 为 31/32，实际交付与图库使用第 8 轮的最佳映射层。重新加载该映射层后复核仍为 32/32。最后一轮映射层相对初始化的参数 L2 变化为 1.011346，训练过程实际更新了参数。

**适用范围：这是同场景、同批商品的图片留出验证。** 保留了真实场景分组，并显式记录 `split_unit=image`、`validation_scope=same_capture_image_holdout`、`independent_validation=false`。DINO 数字基于标注商品 ROI 检索训练参考图；端到端分割加身份识别的准确率未在本次单独评估。验证集规模较小，结果用于本轮实验比较；独立场景、不同实物、未知 SKU 拒识能力还需另外采集数据验证和校准。

## 产物与验证证据

- `models/segmentation/retail_19sku_v003/model.pth`：最佳分割权重；配套 `inference_config.py` 为展开后的模型配置。
- `models/recognition/dinov2_b14_19sku_v003/model.pth`：最佳映射层，推理时作为 `projection`；主干仍使用 `models/recognition/dinov2_b14_v001/model.pth`。
- `work_dirs/retail_19sku_v003/gallery/`：274 个训练参考向量、19 个 SKU、256 维及来源记录。
- `work_dirs/retail_19sku_v003/training_summary.json`、`split_summary.json`：训练、验证、来源和最佳检查点汇总。
- `work_dirs/retail_19sku_v003/reload_verification.json`：独立进程的离线 CUDA 重新加载结果、32 个验证实例的逐项识别结果及图库交集检查。
- `work_dirs/retail_19sku_v003/console.log`：两阶段训练日志；子实验目录分别以 `_segmentation`、`_recognition` 结尾，包含恢复检查点、标量和 loss 记录。

离线检查沿用项目禁止 Python 网络连接的方法，两个神经网络实际运行于 CUDA。重载后的模型张量与保存检查点一致；分割主干张量相对通用初始权重发生变化；重载映射层提取的参考向量与图库一致。四份历史模型权重及旧数据 manifest、标注的哈希均未改变。受影响的数据准备与联合训练测试共 49 项通过。

本次产物标记 `calibration_required=true`、`production_eligible=false`。模型保存到独立的新版本目录。

## 工作站

已加入工作站选项 **RetailVision v0.3 · 19 SKU**。已有工作站刷新后，在“运行设置 → 分割权重”选择该项；也可执行 `./start-web --model retail_19sku_v003` 启动。

推理配置为 `configs/inference/retail_19sku_v003.json`，对应最佳分割权重、已注册的最佳 DINO 映射层及 `datasets/gallery_19sku_dino_v003/` 中的 274 个训练参考向量。部署图库从训练图库复制，并核对映射层内容一致；训练实验的原始图库保留。

`retail_19sku_v003_policy.json` 为单独的未校准实验配置，余弦阈值 0.8、Top-2 不同 SKU 间隔 0.01，均为临时默认值，没有未知 SKU 的误接受率保证。旧模型的反馈阈值没有作为新特征空间的校准结论。

工作站配置相关 14 项测试通过；`start-web` 实际构建和启动成功。在原工作站 8018 端口验证了三组数据各一张验证图的目标推理和 Mask 下载、19 SKU 主表、256 维识别特征，以及 v0.2/v0.3 切换。此项检查验证接入流程，不替代上表的训练验证指标。桌面及 390px 手机截图与结果保存在 `work_dirs/retail_19sku_v003/workbench/`。

## 复现入口

在新输出目录准备相同划分，避免覆盖已保存的数据版本：

```bash
conda run --no-capture-output -n retailvision-seg python tools/prepare_dataset.py \
  --coco datasets/incoming/retail_19sku_v003/merged.json \
  --images datasets/incoming/retail_19sku_v003/images \
  --catalog datasets/incoming/retail_19sku_v003/skus.json \
  --validation-images datasets/incoming/retail_19sku_v003/validation_images.json \
  --output datasets/retail_19sku_v003_reproduction

conda run --no-capture-output -n retailvision-seg python tools/train.py \
  --dataset datasets/retail_19sku_v003_reproduction \
  --config configs/segmentation/retail_19sku_v003.py \
  --recognition-config configs/recognition/retail_19sku_v003.py \
  --experiment-id retail_19sku_v003_reproduction
```
