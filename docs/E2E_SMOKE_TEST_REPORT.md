# RetailVision 第一次端到端训练与推理实验

实验名称：`retailvision_e2e_smoke_v001`。执行日期：2026-11-10（Asia/Shanghai）。

**验收通过：真实 CUDA RTMDet-Ins 训练 → Checkpoint 保存与新进程重新加载 → DINOv2 特征提取 → FAISS SKU Gallery → 统一目标商品推理全部执行成功。未使用 Mock。**

本实验是 **3 张图片的小样本工程验证**。训练图同时作为推理图，图库来自相同图片的已标注 ROI，允许对训练集过拟合；结果不能作为真实商品识别精度、泛化能力或独立测试集指标。DINOv2 未进行 Metric Learning，正式模型版本与权重未修改。

## 数据、环境与范围

- 输入：`datasets/retailtest_smoke_v001/`，dataset_version：`d21d4e9b27a4ed12d7a1f067825266c765d508776ad773f3d8c3f7743b0fef62`。
- 使用现有 `segmentation/annotations/all.json` 的全部 3 图、9 实例，类别仅 retail_product/id=1；识别侧使用现有 9 个 mask-aware ROI。
- 没有修改数据集、补造采集组或创建 Train/Val/Test 划分；没有独立验证、COCO mAP 评估或选 best 模型。
- 环境：retailvision-seg，Python 3.10.22，torch 2.1.2+cu121，CUDA runtime 12.1，NVIDIA GeForce RTX 4090。
- 两个神经网络均在 cuda:0；FAISS 使用现有 faiss-cpu。Python socket.connect 在训练与推理中被禁止，全部模型与权重由本地加载。
- 除本报告外，所有脚本、配置、日志、Checkpoint、Gallery、JSON 结果和图片均保存在 `work_dirs/experiments/retailvision_e2e_smoke_v001/`。本次没有修改正式训练/推理模块，没有升级依赖。

## RTMDet-Ins 真实训练

直接使用现有 MMDetection/MMEngine Runner，在实验脚本中配置单类 RTMDet-Ins-s，避免现有常规训练入口要求 train/val/test 文件；未更改常规入口。

| 项目 | 实际设置或结果 |
|---|---|
| 初始化权重 | models/segmentation/rtmdet_ins_s_coco_v001/model.pth（只读） |
| 单类头 | num_classes=1，retail_product |
| 训练长度 | 120 个训练迭代，batch size=3，全部图片重复用于小样本过拟合 |
| 实际优化器更新 | 117 次；AMP 在初始3次非有限梯度时跳过更新 |
| 图像处理 | 保持比例缩放至640范围，Pad到640×640；没有 Mosaic、MixUp或随机翻转 |
| 优化 | AdamW，初始基础 lr=.001、weight_decay=.0001、梯度裁剪max_norm=10、AMP动态loss scaling、10迭代warmup及短程降学习率 |
| 训练耗时 | **14.937 秒**，包含 Runner 初始化、训练和保存，不是纯 CUDA kernel 时间 |
| 初始总 Loss | 3.260772 |
| 最终总 Loss | **0.018725** |
| 最终 cls / bbox / mask Loss | 0.004266 / 0.014273 / 0.000186 |
| 迭代日志中的最大 GPU memory | 1062 MiB（MMEngine迭代分配显存计数） |
| 训练阶段峰值 reserved | 1176.0 MiB |

最终 Checkpoint：

```text
work_dirs/experiments/retailvision_e2e_smoke_v001/training/iter_120.pth
```

SHA256：`3e8acc5f229b4a47b4425082ccd27b2868595599b1e6c0d3193cd6bfe46793a6`。另保存 iter_60.pth，均为新的实验文件，没有覆盖原权重。没有独立验证，因此没有声称 best checkpoint。

训练记录：`train-console.log`、`training/20261110_151411/`、`train_config.py`、`training_summary.json`、`loss_history.json`、TensorBoard event 文件。Loss 曲线：[loss_curve.png](../work_dirs/experiments/retailvision_e2e_smoke_v001/loss_curve.png)。

训练的 memory 日志会重置 PyTorch peak allocated 计数，故报告使用120条迭代日志中最大值；training_summary 中 post_training_counter_allocated_bytes 只是训练结束后的计数，不冒充全程峰值。显存数字均不等于整张GPU或桌面进程占用。

## Checkpoint 重新加载证明

训练进程结束后，另起 `infer_experiment.py` 进程，通过现有 `load_pipeline()` 和 RTMDet Adapter 从 iter_120.pth 加载模型并在 cuda:0 推理，3张图各输出3个非空预测实例 Mask。

- 新模型的一份分类头权重张量与保存的 Checkpoint **逐元素完全相等**。
- Checkpoint 的一份 backbone 权重与原始 COCO 预训练权重不同，证明加载的是实际更新后的模型。
- Checkpoint 内 AdamW state 的 step 为117，与 AMP 初始跳过3次更新的记录一致。
- 加载使用独立的 `inference_rtmdet_config.py`；训练配置保留原样，失败过程见后文。

证据：`inference_summary.json` 的 checkpoint_reload_tensor_exact 和 trained_backbone_differs_from_pretrained 均为 true；`checkpoint-inspection.log` 保存优化器step检查结果。

## DINOv2 与临时 FAISS Gallery

使用现有 DINOv2 Adapter 加载本地预训练 ViT-B/14，pretrained=False建结构后读取既有本地权重，未训练backbone或projection。

- GPU提取9个 ROI 的 Embedding：shape=9×768，全部有限且L2归一化，提取耗时 **0.189秒**（不含模型初始化）。
- DINOv2初次加载耗时 1.363秒。
- FAISS IndexFlatIP 实际构建并保存9个向量；从保存的 index.faiss 读取检查 ntotal=9、d=768。
- 对9个 ROI 再次提取特征查询，9项自参考检索的Top-1都与对应SKU一致。只证明检索链路，不是独立识别准确率。

| Annotation | 查询 SKU | 返回 Top-1 SKU | 余弦匹配分数 |
|---|---|---|---:|
| 1 | `VITA_YASHIXIANG_LEMON_TEA_250ML` | `VITA_YASHIXIANG_LEMON_TEA_250ML` | 1.000000 |
| 2 | `VITA_BLACKCURRANT_DRINK_250ML` | `VITA_BLACKCURRANT_DRINK_250ML` | 1.000000 |
| 3 | `VITA_APPLE_JUICE_250ML` | `VITA_APPLE_JUICE_250ML` | 1.000000 |
| 4 | `FENTA_ORANGE_330ML` | `FENTA_ORANGE_330ML` | 1.000000 |
| 5 | `COKE_ZERO_330ML` | `COKE_ZERO_330ML` | 1.000000 |
| 6 | `SPRITE_ORIGINAL_330ML` | `SPRITE_ORIGINAL_330ML` | 1.000000 |
| 7 | `XX_KBD_KFY_18` | `XX_KBD_KFY_18` | 1.000000 |
| 8 | `999_GML_KL_9` | `999_GML_KL_9` | 1.000000 |
| 9 | `BYS_XCH_KL_10` | `BYS_XCH_KL_10` | 1.000000 |

Gallery目录：`gallery/`，包含 `index.faiss`、`embeddings.npy`、`skus.json`、`catalog.json` 和含真实Top-5结果的 `queries.json`。

## Unified Inference 实际结果

使用新进程加载的单类RTMDet模型生成Mask，现有 mask_crop 处理预测Mask区域，DINOv2提取预测实例特征，FAISS检索SKU，再由现有 VisionPipeline 输出目标实例Mask、SKU和匹配分数。

运行前指定目标分别为苹果汁、无糖可乐和999感冒灵，记录在 requested_targets.json；三个目标都实际找到。3张图总计输出9个非空预测Mask。

| 图片 | 指定目标 SKU | 预测实例数 | 目标余弦分数 | 分割 ms | 识别与输出 ms | 总计 ms | 目标Mask已保存 |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | `VITA_APPLE_JUICE_250ML` | 3 | 0.971947 | 24.95 | 123.08 | 148.04 | 是 |
| 2 | `COKE_ZERO_330ML` | 3 | 0.975408 | 24.54 | 120.40 | 144.94 | 是 |
| 3 | `999_GML_KL_9` | 3 | 0.979396 | 26.13 | 118.72 | 144.86 | 是 |

时间为本机单次驻留模型请求；每张图此前已执行真实预测用于生成全部实例可视化，以上不是冷启动或稳定性能基准。recognition_ms沿用管线定义，包含特征提取、检索及结果输出开销。统一管线模型重新加载为2.208秒；最终成功推理脚本共8.237秒，包含模型加载、Gallery操作、校验及可视化保存，不包含首次失败尝试。

推理脚本峰值 allocated **545.8 MiB**，reserved **608.0 MiB**；两模型参数设备均明确检查为cuda:0。

完整结果：`inference_results.json`。明确的文件系统输出映射：`target_outputs.json`，含每个目标的SKU ID、分数、Box及Mask实际路径。目标Mask文件：

- `work_dirs/experiments/retailvision_e2e_smoke_v001/unified/f28effd49eba4478a6eff026b1ccff8f/mask_002.png`
- `work_dirs/experiments/retailvision_e2e_smoke_v001/unified/5dc2a3c60f974a35b8311158f816dc67/mask_002.png`
- `work_dirs/experiments/retailvision_e2e_smoke_v001/unified/95393cae54f547b795f73195a3ceefa7/mask_002.png`

现有管线的HTTP样式mask_path字段沿用原接口格式；本次直接调用Python管线，实际文件位置以上述target_outputs.json为准，没有声称执行Workbench/API联调。

为使无独立校准数据的小样本链路可执行，实验使用 **similarity=0、margin=0 的临时正余弦Top-1策略**；`smoke_policy.json` 明确 calibrated=false、production_eligible=false。配置仅用于本实验，未生成或替换生产校准，**不验证未知SKU拒识能力**。匹配分数不是概率。inference_config 的fixture=true用于标记工程验证，输入仍是真实商品图片，未替换为合成或Mock响应。

## 可视化

已检查原始图、预测实例Mask与统一推理结果汇总，以及SKU匹配图：预测覆盖真实商品，指定目标Mask与返回SKU相符；Loss曲线显示小样本过拟合过程。

- [三图完整结果汇总](../work_dirs/experiments/retailvision_e2e_smoke_v001/visualizations/summary.jpg)
- [训练 Loss 曲线](../work_dirs/experiments/retailvision_e2e_smoke_v001/loss_curve.png)
- 原始图：`visualizations/image_1_original.png` 至 `image_3_original.png`。
- RTMDet Mask Overlay：`visualizations/image_*_rtmdet_masks.png`。
- 全部实例的SKU匹配：`visualizations/image_*_sku_matches.png`。
- 指定目标完整推理结果：`visualizations/image_*_unified.png`。
- 9个真实预测实例的黑白Mask：`predicted_masks/`；三个目标Mask另保存在 `unified/<result_id>/`。

## 所有异常与失败

| 情况 | 实际记录与处理 |
|---|---|
| COCO80类权重与单类头形状不同 | 初始化时6个分类头weight/bias形状不匹配，保留新单类头并训练；backbone等兼容权重加载。属于80→1迁移的预期日志，没有修改原权重。 |
| 初始AMP非有限梯度 | 迭代1、2的grad_norm为NaN，迭代3为Inf。动态loss scaling跳过3次更新，后续117次更新正常；120条记录中的各项Loss均有限。优化器step=117，最终权重与推理有效。未隐藏这些记录，也未把120迭代声称为120次成功参数更新。 |
| 第一次推理进程失败 | 训练未配置测试集，train_config.py中test_dataloader=None；MMDetection init_detector读取dataset元信息时报AttributeError。 |
| 对首次推理失败的修复 | 只在实验内增加inference_rtmdet_config.py，提供数据集元信息和无标注图片推理pipeline；未改Adapter或正式配置。保留首次尝试全部日志、配置、Gallery及状态于 attempts/inference_01/，第二次真实推理通过，未重训。 |
| xFormers未安装 | DINOv2使用原生PyTorch CUDA attention，两模型仍为cuda:0，没有CPU回退。 |
| 上游弃用提醒 | pkg_resources、FileClient/HardDiskBackend、TypedStorage、torch.meshgrid相关提醒保留于原始日志；未升级依赖绕过。 |

原始完整日志含训练最初的NaN/Inf；原始loss_history.raw.json、training_summary.raw.json保留Python导出的原始非有限数值。面向标准JSON的loss_history.json、training_summary.json将这些grad_norm值记为null，并在anomalies.json中记录原始值与迭代号；没有改写原始训练日志或Loss值。

## 产物与验收证据

所有相对产物路径基于：`work_dirs/experiments/retailvision_e2e_smoke_v001/`。

| 验收 | 证据 | 结果 |
|---|---|---|
| 真实RTMDet训练 | 120迭代日志、117次optimizer step、有限Loss和变化后的backbone张量 | 通过 |
| Checkpoint保存 | training/iter_60.pth、training/iter_120.pth及SHA256 | 通过 |
| Checkpoint新进程重载 | inference_summary.json张量相等断言、真实9实例预测 | 通过 |
| DINOv2 CUDA特征提取 | 9×768单位向量、设备cuda:0、真实二次查询 | 通过 |
| FAISS检索 | 9向量持久化索引、queries.json的9次自参考检索 | 通过 |
| 统一目标推理 | 三个实际目标结果、真实预测Mask及target_outputs.json | 通过 |
| 原始权重和数据保护 | input_hashes.json与preservation.json，28个输入文件哈希全部不变 | 通过 |

experiment.json保存完成状态，environment.json保存环境与nvidia-smi记录。没有注册正式模型版本、修改正式权重、覆盖历史Checkpoint或追加大规模训练。沿用上次数据管线Smoke Test的工作区修改，本实验没有再修改这些业务代码；本次新增Git可见文件仅为本报告，实验目录沿用既有.gitignore保留在本地。

## 复现入口

训练和推理采用本实验目录中的脚本：train_experiment.py、infer_experiment.py；plot_losses.py可仅从持久化Loss数据重画曲线。实际启动命令：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 conda run --no-capture-output -n retailvision-seg \
  python work_dirs/experiments/retailvision_e2e_smoke_v001/train_experiment.py
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 conda run --no-capture-output -n retailvision-seg \
  python work_dirs/experiments/retailvision_e2e_smoke_v001/infer_experiment.py
```

以上记录本次启动方式。复现实验应复制脚本到新的experiment目录名后运行，保留本次目录不变；Gallery保存禁止覆盖。各阶段重载、特征、检索、目标Mask和原始资产哈希均由真实执行结果检查支持，无Mock替代。
