# 训练

`tools/train.py` 统一任务路由，每次在 work_dirs/<experiment_id> 保存实验 JSON、完整输入配置、起止时间和检查点路径。分割 resolved_config.py 还保存实际 Runner 配置。失败保存 traceback，API stdout 在 logs/<id>.log。

默认一次训练两个阶段：

```bash
conda run --no-capture-output -n retailvision-seg python tools/train.py --dataset datasets/retail_v001
```

`--dataset` 为准备后的数据集根目录。默认 `--task both`，依次在 `retailvision-seg` 和 `retailvision-reid` 环境运行分割与识别，分别验证，避免同时争用 GPU。两阶段共用导入时约 90%/10% 的采集组划分及同一数据版本；缺少独立划分、识别训练样本不足或两阶段数据版本不一致时，在启动训练前拒绝。可用 `--config` 指定分割配置，`--recognition-config` 指定 DINO 配置。

父实验记录两个子实验的权重、状态和各自验证指标；任一阶段失败会停止后续步骤。两阶段均验证完成后，用识别模型选出的映射层和训练集 ROI 重建父实验目录下的 `gallery/`，不把验证图加入图库。新映射层需要单独校准 SKU 拒识阈值，训练完成不会自动替换运行中的推理配置。分割验证没有任何检测结果时记零指标，不把验证耗时当作准确率指标。

当用户明确要求同批次图片留出验证时，准备数据可显式传入 `--validation-images filenames.json`（图片相对文件名的 JSON 列表）。此模式按整张图片划分，两模型共享划分，图库仅使用训练图片；保留真实 `capture_group`，记录 `split_unit=image`、`validation_scope=same_capture_image_holdout` 和 `independent_validation=false`。它用于同批次实验，不代表独立场景或新物理样本的泛化能力。验证图还应排除训练起始权重已经见过的图片。

API `POST /api/training/start` 默认也为联合任务，最少传入 `{"dataset":"datasets/retail_v001"}`。既有 `--task segmentation`、`--task recognition` 单阶段入口仍可用于单独实验或恢复；联合任务不接收一个含义不明的 `--resume`，恢复时明确指定阶段和对应 checkpoint。

RTMDet-Ins-s 初始迁移 COCO 权重，分类 head 改为一个类别，MMDetection 原生数据增强、AdamW、调度器、AMP、EMA、验证和 CheckpointHook。默认 30 epoch、batch 4、lr .001 是可编辑实验起点，不代表经过优化。Resume 委托 Runner。

DINOv2-B/14 主干固定 eval 且冻结，只训练 Linear 768→256，TripletMargin 或 Contrastive loss，pytorch-metric-learning hard triplet miner、每类至少两样本采样。验证 Top-1 使用训练参考图对验证查询检索，不将验证图作为其自身参考。保存 head、optimizer、epoch、随机状态、数据版本；TensorBoard 保存 loss 和验证结果。映射层改变后必须重建图库并重新校准。

真实 CVAT 数据已用于三图实验和 [v0.2 六 SKU 训练](TRAINING_V0_2.md)。v0.2 使用全部 15 图、90 实例训练，未提供独立采集组，不能报告独立验证精度或泛化结论。分割中型对照与部分 backbone 解冻为后续实验，当前不自动执行。不要将 synthetic fixture 训练当作零售训练效果。

现有六 SKU 的 [DINO 映射层训练](TRAINING_DINO_V0_3.md) 已使用显式 `--task recognition --no-validation` 完成。这个选项只允许 `no_split=true`、全部条目标记 `all` 的数据：不执行验证、不选 best，只保存 `last.pth`、`last_head.pth` 和训练 loss。默认训练和联合训练仍要求验证集。

映射层推理使用 `best_head.pth`（纯张量），`best.pth`/`last.pth` 同时含优化器与随机状态，用于继续训练。不要将含 NumPy 状态的恢复检查点当作 weights_only 推理文件。

## 发布到工作站

完成联合训练和验证后，通过正式发布入口接入工作站：

```bash
conda run --no-capture-output -n retailvision-seg python tools/publish.py \
  --experiment work_dirs/my_joint_run \
  --version retail_next \
  --name 'RetailVision · Next' \
  --policy configs/inference/retail_next_input_policy.json
```

策略文件须明确提供 `similarity`、`margin`、`dataset_version`、`model_version=retail_next`、`gallery_version=gallery_retail_next`。未校准的实验策略须标记 `calibrated=false`、`production_eligible=false`，正式校准仍需独立已知/未知 SKU 数据。发布入口核对两个阶段与图库来源，并选择分割最佳检查点对应的验证指标；仅复制训练参考实例。输出为新的两套权重版本、图库、推理配置和工作站选项，已有版本不会被覆盖。

新实验自动记录数据路径；旧实验缺少该字段时，发布命令需显式加 `--dataset datasets/the_original_dataset`，并核对数据版本。历史实验目录中的单次 `finalize.py` 等脚本不再作为当前发布入口。
