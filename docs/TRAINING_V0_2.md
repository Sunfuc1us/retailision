# RetailVision v0.2 六 SKU 训练

2026-11-11（Asia/Shanghai）。模型版本 `retail_6sku_v002`。

已完成真实 CUDA RTMDet-Ins-s 训练、权重持久化和独立进程重新加载，以及六 SKU 的 DINOv2/FAISS 图库重建。所有模型从本地加载，训练和离线推理检查禁止网络连接；训练进程仅允许数据加载所需的本机 Unix socket 通信。

## 数据与结果的适用范围

输入 `/home/gzh/下载/6sku_test.zip`：15 张图、90 个 Polygon 实例，六款 SKU 各 15 个实例。SKU 主表直接来自标注属性，90 个实例通过现有文件、图像尺寸、几何和 SKU 一致性校验，没有修改原始标注。

图片来自同一场景、同一批商品的不同视角，标注没有提供独立 `capture_group`。沿用项目 `--no-split` 模式，全部数据用于训练和建库，没有随机拆图冒充独立验证集，没有选择 best checkpoint。下述数字是**训练集回代检查**，图库也来自这些图片，不能作为独立商品精度、泛化能力或未知 SKU 拒识能力。

- 原始压缩包、解压标注和来源记录：`datasets/incoming/6sku_v002/`。
- 转换后的数据、90 个实例 ROI 和图库：`datasets/retail_6sku_v002/`。
- 数据版本：`dbcca6c5452d6fe8c639fdc0aeac9c1f4fb0d635f5d1bee9bfdb03b283c7f610`。

## 训练

| 设置 | 实际值 |
|---|---|
| GPU / 环境 | RTX 4090 / retailvision-seg，PyTorch 2.1.2+cu121 |
| 初始权重 | 本地 COCO RTMDet-Ins-s，单类 retail_product 头重新初始化 |
| 配置 | `configs/segmentation/retail_6sku_v002.py` |
| 输入 / Batch | 960×960 范围，保持比例；batch=3 |
| 增强 | 0.8–1.2 随机缩放、裁剪、HSV；没有文字镜像翻转 |
| 优化 | AdamW，lr=0.001，25 步 warmup 后余弦衰减，AMP 初始 scale=128，梯度裁剪=10 |
| 训练长度 | 500 次迭代，实际优化器更新 500 次，约 100 次数据遍历 |
| 耗时 | 58.98 秒，含 Runner 初始化、保存和训练后记录 |
| 首条记录 loss（第 5 步） / 最终 loss | 3.371741 / 0.123327 |
| 最终 cls / bbox / mask loss | 0.030668 / 0.076313 / 0.016346 |

Checkpoint 保存为 `iter_250.pth` 和 `iter_500.pth`，最终模型注册到 `models/segmentation/retail_6sku_v002/model.pth`，没有覆盖历史模型或数据。

最终权重 SHA256：`0979d1c231237fddc45055ddae3aae4eb8f50e1bece0f264dadda0eb2553e900`。

DINOv2 ViT-B/14 保持已有预训练权重，没有训练识别主干或映射层。90 个标注 ROI 在 CUDA 上提取 768 维特征，保存六款商品的 FAISS 图库，每款 15 个参考向量。

## 实际检查

新进程通过现有 `load_pipeline()` 加载保存的 v0.2。分类头张量与 Checkpoint 完全相同，backbone 张量与初始权重不同；两个神经网络均运行于 `cuda:0`。旧 COCO、DINOv2 和三图训练权重的 SHA256 在训练前后保持一致。

训练集回代检查使用检测阈值 0.3，先按 Mask IoU 做一对一匹配，再检查 SKU：

| 训练集检查 | 结果 |
|---|---:|
| 图像 / 标注实例 / 预测实例 | 15 / 90 / 90 |
| 与标注匹配的 Mask（IoU ≥ 0.5） | 90 / 90 |
| 匹配且 SKU 正确 | 90 / 90 |
| 匹配 Mask 的平均 IoU | 0.965685 |

没有独立阈值校准数据。训练交付时使用实验性正余弦 Top-1 策略（similarity=0、margin=0），`calibrated=false`、`production_eligible=false`。后续已根据混合货架反馈修复拒识门槛与 ROI 性能，当前行为和数据范围见 [v0.2 推理修复](INFERENCE_FIX_V0_2.md)；本报告保留训练当时的记录。配置中的 `fixture=true` 表示工程验证范围，输入数据本身是真实商品图片。

数据与模型选择相关的现有测试：14 项通过。Workbench 实际页面完成模型选择、六 SKU 主表加载、CUDA 目标推理和 Mask 下载检查；桌面和 390px 移动端截图已保存并检查。

实验记录均位于 `work_dirs/retail_6sku_v002/`：

- `training_summary.json`、`resolved_config.py`、`train-console.log`、`training/`：配置、训练日志和 Checkpoint。
- `loss_history.json`、[loss_curve.png](../work_dirs/retail_6sku_v002/loss_curve.png)：训练损失。
- `validation_summary.json`、`training_set_matches.json`、`inference_results.json`：训练集逐实例真实结果。
- [visualizations/summary.jpg](../work_dirs/retail_6sku_v002/visualizations/summary.jpg)：15 图分割与 SKU 匹配概览，另有逐图可视化。
- `web-verification.json`、`web-result.json`、`web-desktop.png`、`web-mobile.png`：Workbench 实际验证。
- `protected_hashes.json`：历史权重哈希。初始化阶段的数据尚未就绪、Runner 测试配置和本机 IPC 拦截失败日志分别保留为 `data-not-ready.log`、`runner-config-attempt.log`、`local-ipc-attempt.log`；这些尝试未执行参数更新，最终成功训练日志独立保存。

## 使用

刷新现有 Workbench，在分割模型选择 **RetailVision v0.2 · 6 SKU**；也可显式启动：

```bash
./start-web --model retail_6sku_v002
```

推理配置为 `configs/inference/retail_6sku_v002.json`。当前支持 PyTorch CUDA，沿用已有 Backend，不新增导出产物。默认启动模型保持原有三图模型。

命令行示例：

```bash
conda run --no-capture-output -n retailvision-seg python tools/predict.py \
  --config configs/inference/retail_6sku_v002.json \
  --image datasets/retail_6sku_v002/segmentation/images/20261111-091730.170-12.jpg \
  --target COKE_ZERO_330ML
```
