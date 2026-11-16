# RetailVision

本地零售与药房商品视觉研发平台。模块化单体；二维 RGB 输入；RTMDet-Ins 分割与 DINOv2 商品身份识别分离。

## 当前状态

已完成 **v0.4 二十九 SKU 联合训练**：合并 v0.3 与新十 SKU 数据，共 90 张图、526 个实例；训练 81 张、验证 9 张，两边覆盖全部 SKU，图库仅含 475 个训练实例。最佳同场景图片留出 Mask mAP@0.50:0.95=0.976，DINO 标注 ROI Top-1=51/51；离线 CUDA 重载通过，历史权重与数据保留。工作站可选择 `retail_29sku_v004`，独立场景泛化与未知 SKU 拒识仍待验证、校准。详见 [v0.4 训练报告](docs/TRAINING_V0_4.md)。

已完成 **v0.3 十九 SKU 监督微调**：合并 v0.2 旧数据与新两套数据，从通用预训练权重训练 RTMDet-Ins-s 和 DINOv2 映射层。46 张训练图、5 张验证图（9.8%）共用划分，训练图库含 274 个参考实例。最佳同场景留出结果为 Mask mAP@0.50:0.95=0.974、DINO 标注 ROI Top-1=32/32；独立场景泛化及拒识能力仍需另行验证、校准。新模型与历史版本分别保存，详见 [v0.3 训练报告](docs/TRAINING_V0_3.md)。

RetailVision 已保存 **v0.2 六 SKU 模型**：使用新标注的 15 图、90 实例完成 CUDA RTMDet-Ins 训练，并重建 DINOv2 商品图库。Workbench 支持切换权重及配套 SKU 图库，同时保留三图模型、COCO 预训练模型和既有 ONNX/TensorRT 部署产物。当前训练模型均没有独立验证集商品精度或生产校准结论。配置、结果与使用说明见 [v0.2 训练报告](docs/TRAINING_V0_2.md)。

v0.2 已修复大图 ROI 处理耗时，并启用基于反馈样本的临时 SKU 拒识门槛；混合货架实测与限制见 [推理修复记录](docs/INFERENCE_FIX_V0_2.md)。

六 SKU 的 DINO 识别映射层已另存新版本，使用全部 90 个样本完成 20 轮 CUDA 训练并重建配套图库，尚无独立验证或拒识校准，当前推理配置未切换。详见 [DINO 训练记录](docs/TRAINING_DINO_V0_3.md)。后续默认联合训练分割和识别，共用约 90%/10% 的训练/验证划分。

唯一正式开发根目录为 `/home/gzh/rocco/retailvision`；旧 DemoRoom 副本只作历史保留。历史冻结基线为 **RetailVision V1.0**（Git 标签 `v1.0.0`），报告见 [V1.0 基线验收](docs/BASELINE_V1.0.md)。后续实验与部署见 [三图训练实验](docs/E2E_SMOKE_TEST_REPORT.md) 和 [部署说明](docs/DEPLOYMENT.md)。

## 启动

项目根目录的唯一 Web 启动入口为 `./start-web`。与 Hexapose 的 `./run --model …` 一样，在启动入口明确选择模型；页面也可直接切换。

```bash
cd /home/gzh/rocco/retailvision
./start-web
```

打开 **http://127.0.0.1:8018**。默认使用独立的 `retailvision-deploy` 环境，先构建当前前端，再启动 API 和 Workbench。默认选择六 SKU 权重 `retail_6sku_v002`，检测阈值为 0.80，可按 0.01 步长选择 0.00–1.00。首次点击“分析图片”时实际加载模型。若已设置 `RETAILVISION_CONFIG`，使用该配置；显式 `--model` / `--config` 优先。

其他启动方式（每次选择一种，Ctrl+C 停止当前服务）：

```bash
./start-web --model retailtest_smoke_v001    # 三图训练模型，PyTorch
./start-web --model retail_6sku_v002         # v0.2，六 SKU / 15 图训练，PyTorch
./start-web --model retail_19sku_v003        # v0.3，19 SKU，配套 DINO 映射层与训练图库，PyTorch
./start-web --model retail_29sku_v004        # v0.4，29 SKU，配套最佳 DINO 映射层与 475 个训练参考实例
./start-web --model coco_pretrained         # COCO 预训练模型，PyTorch / ORT / TensorRT
./start-web --config configs/inference/local.json
./start-web --port 8020
./start-web --help
```

- Linux 不会默认在当前目录搜索裸命令 `start-web`，应输入 `./start-web`；也可从其他目录使用 `/home/gzh/rocco/retailvision/start-web`。
- 画布默认 Mask 透明度为 0.80；滚轮以鼠标位置缩放，按住拖动平移，“适应窗口”复位。聚焦画布后也可用 `+` / `-` 缩放、方向键平移、`Home` 复位。
- 页面“分割权重”选择模型，下方显示实际 Checkpoint 文件名，悬停可查看完整路径。切换时同步更换模型配置、SKU 列表、图库和阈值，清空旧结果与目标，并过滤 Backend。
- 目标商品可输入完整 SKU、中文名称（如“维他苹果汁”“苹果汁”）或空格分隔的关键词（如“维他 苹果汁 250ml”）。输入框候选使用中文商品名称；关键词匹配多款商品时需选择具体商品，不会自动猜测 SKU。
- 三图模型和 v0.2 当前只提供 PyTorch；其权重尚未导出为 ONNX/TensorRT。COCO 模型可选择三种已验证 Backend。DINOv2 当前使用已有预训练权重。
- 三图验证可上传 `datasets/retailtest_smoke_v001/segmentation/images/20261110-141818.474-1.jpg`，目标填写 `VITA_APPLE_JUICE_250ML`。
- 联合训练完成后使用 `tools/publish.py` 一次发布最佳权重、配套图库和明确提供的阈值策略，并加入工作站选项。用法见 [训练与发布](docs/TRAINING.md)；源码职责见 [架构](docs/ARCHITECTURE.md)。
- v0.3 可在“运行设置 → 分割权重”选择 **RetailVision v0.3 · 19 SKU**，同步加载最佳 DINO 映射层及 274 个训练参考实例。其独立实验拒识配置尚未校准，详情见 [v0.3 训练报告](docs/TRAINING_V0_3.md)。
- 前端依赖缺失时，先执行 `cd frontend && npm ci`。只有训练环境时，可用 `RETAILVISION_ENV=retailvision-seg ./start-web`，页面仅开放该环境实际可用的 Backend。

前端开发仍可执行 `cd frontend && npm run dev`，Vite 5178 代理 Backend 8018。需要生产商品推理时，先准备真实 Gallery 和独立校准文件，再将 `configs/inference/local.example.json` 复制为本地配置并填写路径。

## 真实模型 Fixture 验证

```bash
RETAILVISION_CONFIG=configs/inference/fixture.local.json conda run --no-capture-output -n retailvision-seg uvicorn retailvision.api.app:app --host 127.0.0.1 --port 8018
```

上传 `third_party/mmdetection/demo/demo.jpg`，目标填写 `FIXTURE_INSTANCE`。这是 COCO 长椅自参考检索，页面明确标注 Fixture，只验证本地双模型、Mask 和接口，不代表商品识别精度。

## 数据

CVAT COCO 的每个 image 增加 `capture_group`（同一批次、场景或物理样本不可跨组）。SKU 主表是 JSON 对象列表。实例优先读取 `annotation.attributes.sku_id`，缺失时 sidecar 为 `{"annotation_id":{"sku_id":"..."}}`。不允许自动猜测 SKU。

```bash
conda run -n retailvision-seg python tools/prepare_dataset.py --coco export.json --images raw_images --catalog catalog/skus.json --validate-only
conda run -n retailvision-seg python tools/prepare_dataset.py --coco export.json --images raw_images --catalog catalog/skus.json --output datasets/retail_v001
```

## 训练

```bash
conda run --no-capture-output -n retailvision-seg python tools/train.py --dataset datasets/retail_v001
```

默认依次训练分割和 DINO 并分别验证，随后用训练集重建配套图库。数据集须有独立采集组划分；现有未划分的六 SKU 数据不能用于联合验证。单独训练或断点恢复需显式指定 `--task segmentation` / `--task recognition`。新实验记录关联完整配置与数据版本。详细说明见 [训练](docs/TRAINING.md)。

## 图库、校准与推理

```bash
conda run -n retailvision-seg python tools/build_gallery.py --dataset datasets/retail_v001/recognition --output datasets/gallery_v001
conda run -n retailvision-seg python tools/evaluate.py --task calibrate --input independent_validation_scores.json --output datasets/calibration_v001.json --dataset-version val_v001 --model-version dinov2_b14_v001 --gallery-version gallery_v001 --experiment-id calibration_v001
conda run -n retailvision-seg python tools/predict.py --config configs/inference/local.json --image shelf.jpg --target LAYS_CUCUMBER_70G
```

`identity_score` 是余弦匹配分数；不是概率。校准输入须来自独立验证采集组，包括已知与未知 SKU。Top-2 为第二个不同 SKU；不足两种 SKU 时拒绝匹配。

推理配置预检：`python tools/predict.py --config configs/inference/retail_6sku_v002.json --dry-run`。只用标准库检查所选后端的本地文件、图库必需文件及配置与校准文件的 `gallery_version` / `model_version`，输出解析后的配置；不加载模型、不创建结果目录。可指定 `--backend onnxruntime` 或 `--backend tensorrt` 检查对应部署文件。正式推理复用相同预检。预检不验证权重内容、图库实际版本或 CUDA 可用性；运行时仍要求 CUDA。

## 验证

```bash
conda run -n retailvision-seg python tools/check_system.py
conda run -n retailvision-seg pytest -q
conda run -n retailvision-seg python tools/verify_models.py
cd frontend && npm run build
```

架构、数据、训练、推理、部署与许可分别见 `docs/`。本地 API 不含用户认证，仅绑定 loopback；不要直接暴露公网。

## 正式开发入口与路径

Python 业务源码统一位于 `retailvision/`。当前结构与迁移说明见 [架构](docs/ARCHITECTURE.md) 和 [结构清理记录](docs/STRUCTURE_CLEANUP.md)；ASGI 入口为 `retailvision.api.app:app`。

正式根目录为 `/home/gzh/rocco/retailvision`，所有后续算法、训练和前端工作均在此进行。

```bash
conda activate retailvision-seg  # 分割与完整本地 API
# 或 conda activate retailvision-reid  # 识别训练与特征提取
```

`tools/*.py` 和 API 入口启动时动态定位本项目根目录。相对配置、数据、图像和输出参数均相对于项目根目录；外部数据请传绝对路径。直接调用 `load_pipeline` 和模型 Adapter 时，模型与配置路径也以项目根目录为基准。

前端开发：`cd /home/gzh/rocco/retailvision/frontend && npm ci && npm run dev`。
权重分别位于 `models/segmentation/rtmdet_ins_s_coco_v001/` 与 `models/recognition/dinov2_b14_v001/`；数据入口为 `tools/prepare_dataset.py`，训练入口为 `tools/train.py`，推理入口为 `tools/predict.py`。

迁移证据见 [MIGRATION_REPORT.md](docs/MIGRATION_REPORT.md)。`docs/DELIVERY.md`、旧训练日志及 `*.raw.txt` 保留迁移前历史内容，其中的历史路径不作为运行配置使用。
