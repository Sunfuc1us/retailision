# 第三方许可核对

2026-11-10 核对代码仓库、模型卡及本地安装 metadata。实际版本见 environments/*freeze.txt；自动生成的安装包许可清单见 licenses-installed.json。网络下载仅用于初始化，运行时只用本地模型。

| 名称 | 版本/来源 | 代码许可 | 权重/使用说明 |
|---|---|---|---|
| MMDetection | v3.3.0 / 44ebd17b145c2372c4b700bfb9cb20dbd28ab64a | Apache-2.0 | [仓库 LICENSE](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/LICENSE) |
| RTMDet-Ins-s COCO checkpoint | 官方 20221121_212604-fdc5d7ec | 代码 Apache-2.0 | [官方模型表](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/configs/rtmdet/README.md)确认来源；未找到 checkpoint 独立权重授权声明。**商业权重许可待确认**，不可将代码许可直接推断为权重许可。当前用于本机技术验证 |
| DINOv2 | 7764ea0f912e53c92e82eb78a2a1631e92725fc8 | Apache-2.0 | [MODEL_CARD](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)明确模型许可 Apache-2.0；官方 ViT-B/14 无 register 权重 |
| PyTorch / TorchVision | 2.1.2+cu121 / .16.2+cu121 | BSD-style | 保留 LICENSE/NOTICE，CUDA runtime 另有 NVIDIA 许可 |
| MMEngine / MMCV | 实际锁定 | Apache-2.0 | MMCV wheel 对应 cu121/torch2.1 ABI |
| FAISS CPU | 实际锁定 | MIT | [官方 LICENSE](https://github.com/facebookresearch/faiss/blob/main/LICENSE)，本地保存索引，无云依赖 |
| pycocotools | 实际锁定 | BSD-style | 工具代码许可与 COCO 数据分别核对 |
| OpenCV | 实际锁定 | Apache-2.0 | 使用 headless，避免无关 GUI 依赖 |
| pytorch-metric-learning | 实际锁定 | MIT | 本地映射层训练 |
| scikit-learn | 实际锁定 | BSD-3-Clause | 本地评估 |
| NumPy / Pillow / Shapely | 实际锁定 | BSD-3 / MIT-CMU / BSD-3 | Pillow 的 bundled libraries 另保留 notices |
| FastAPI / Uvicorn / Pydantic | 实际锁定 | MIT / BSD-3 / MIT | 本地 HTTP，不传图至云 |
| React / Vite / Tailwind / Lucide | package-lock.json | MIT / MIT / MIT / ISC | Lucide 图形保留 ISC，测试工具 Playwright Apache-2.0 |
| NVIDIA CUDA runtime / cuDNN | torch wheel 内附 | NVIDIA 专有再分发条款 | 不是 MIT/Apache；仅按随附条款使用，不替换系统驱动 |
| COCO 原始图片 / 标注 | 未下载训练集 | 独立数据条款 | [官方 Terms](https://cocodataset.org/#termsofuse)，图片版权不因框架 Apache 自动转移；仓库 demo.jpg 仅 smoke test |
| 用户商品照片 / CVAT 标注 / SKU master | 未提供 | 用户确认授权 | 商标、包装图片和药品数据权利需由数据方确认 |

官方权重位置与 SHA256 由 models/*/*/manifest.json 单独记录。未安装 OCR、YOLOE、SAM、云视觉 SDK 或 GPU FAISS。未对未确认的 RTMDet checkpoint 宣称商业授权完成。

## DINOv2 仓库许可边界

当前官方仓库含 CELL-DINO / XRay 等额外模块，其中 CELL-DINO 代码为 CC-BY-NC。RetailVision 不使用这些模块，也不执行会自动导入它们的顶层 hubconf.py；只从本地 `dinov2.hub.backbones` 导入 Apache-2.0 的 ViT-B/14 工厂。分发时只包含所需代码及其 notices，不能笼统声称整个当前仓库均可商用。
