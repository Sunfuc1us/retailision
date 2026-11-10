# 推理

RTMDet 输出 image-sized bool Mask、xyxy bbox、检测 score。mask_crop 负责同训练一致的前景提取；DINO 输出单位 L2 embedding；FAISS IndexFlatIP 在 CPU 做余弦检索。相同 SKU 多参考图取最高分，身份 margin 使用不同 SKU。

目标名称先查 catalog；无结果返回 unknown_target，多结果返回 ambiguous_target 与候选，均不运行视觉模型。只有独立验证集阈值及模型/图库版本一致才能组装生产管线。不接受单 SKU 图库自动匹配。

校准工具输入数组：`{"known":true,"correct":true,"score":0.9,"margin":0.2}`；unknown 行 known=false。通过已知样本召回最大化并限制未知误接受率选择 similarity/margin。必须由调用者确保独立验证组；有限验证集上的 FAR 不是未来数据保证。校准后用独立 test 报告性能。

results/<uuid> 保存 JSON 和每个目标 PNG Mask。API 路径供前端叠加、切换与导出。无目标返回 no_confident_match。生产配置通过 RETAILVISION_CONFIG 设置，首次请求加载；后续请求复用 GPU 模型，锁保证单请求执行。版本切换重启服务。

离线：DINO 从本地 source 直接导入官方 dinov2_vitb14 工厂并 pretrained=False，显式本地 state_dict；RTM 清空 backbone 初始化下载项且显式 checkpoint。verify_models.py 禁止 socket connect 进行验证。图库与校准文件必须事先准备。
