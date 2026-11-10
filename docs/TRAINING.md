# 训练

`tools/train.py` 统一任务路由，每次在 work_dirs/<experiment_id> 保存实验 JSON、完整输入配置、起止时间和检查点路径。分割 resolved_config.py 还保存实际 Runner 配置。失败保存 traceback，API stdout 在 logs/<id>.log。

RTMDet-Ins-s 初始迁移 COCO 权重，分类 head 改为一个类别，MMDetection 原生数据增强、AdamW、调度器、AMP、EMA、验证和 CheckpointHook。默认 30 epoch、batch 4、lr .001 是可编辑实验起点，不代表经过优化。Resume 委托 Runner。

DINOv2-B/14 主干固定 eval 且冻结，只训练 Linear 768→256，TripletMargin 或 Contrastive loss，pytorch-metric-learning hard triplet miner、每类至少两样本采样。验证 Top-1 使用训练参考图对验证查询检索，不将验证图作为其自身参考。保存 head、optimizer、epoch、随机状态、数据版本；TensorBoard 保存 loss 和验证结果。映射层改变后必须重建图库并重新校准。

真实零售数据未提供，不能执行有意义的零售训练。分割中型对照与部分 backbone 解冻为后续实验，当前不自动执行。不要将 synthetic fixture 训练当作零售训练效果。

映射层推理使用 `best_head.pth`（纯张量），`best.pth`/`last.pth` 同时含优化器与随机状态，用于继续训练。不要将含 NumPy 状态的恢复检查点当作 weights_only 推理文件。
