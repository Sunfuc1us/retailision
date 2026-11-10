# RetailVision

本地零售与药房商品视觉研发平台。模块化单体；二维 RGB 输入；RTMDet-Ins 分割与 DINOv2 商品身份识别分离。

## 当前状态

RetailVision 是独立的 Retail & Pharmacy Visual Intelligence Platform。DemoRoom 是公司的演示空间，非本产品名称。唯一正式开发根目录为 `/home/gzh/rocco/retailvision`；旧副本只作历史保留。官方 COCO RTMDet-Ins 权重仅用于通用实例分割验证，**尚未训练为 retail_product**。商品主表初始为空；没有真实 CVAT 零售数据、图库及独立验证集校准时，API 明确拒绝运行生产匹配。Fixture 不代表商品识别性能。

当前冻结基线：**RetailVision V1.0**（Git 标签 `v1.0.0`）。迁移后四项完整验收见 [V1.0 基线验收报告](docs/BASELINE_V1.0.md)，证据见 `docs/baseline/v1.0.0/`。此前交付与迁移记录保留为历史文档。

## 启动

所有命令从此目录执行。

```bash
conda run --no-capture-output -n retailvision-seg uvicorn backend.api.app:app --host 127.0.0.1 --port 8018
# 生产构建已存在时打开 http://127.0.0.1:8018
# 前端开发
cd frontend
npm ci
npm run dev
```

配置真实推理前，填写 `catalog/skus.json`，准备参考图库与校准文件，将 `configs/inference/local.example.json` 复制为本地配置并填写路径：

```bash
RETAILVISION_CONFIG=configs/inference/local.json conda run --no-capture-output -n retailvision-seg uvicorn backend.api.app:app --host 127.0.0.1 --port 8018
```

## 真实模型 Fixture 验证

```bash
RETAILVISION_CONFIG=configs/inference/fixture.local.json conda run --no-capture-output -n retailvision-seg uvicorn backend.api.app:app --host 127.0.0.1 --port 8018
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
conda run --no-capture-output -n retailvision-seg python tools/train.py --task segmentation --model rtmdet-ins-s --config configs/segmentation/rtmdet_ins_s.py --dataset datasets/retail_v001/segmentation
conda run --no-capture-output -n retailvision-reid python tools/train.py --task recognition --model dinov2-vitb14 --config configs/recognition/dinov2_b14.py --dataset datasets/retail_v001/recognition
```

分割数据目录须含 `manifest.json`；转换器在分割子目录也保存此文件。断点恢复增加 `--resume /path/to/last.pth`，新实验记录关联完整配置与数据版本。详细说明见 [训练](docs/TRAINING.md)。

## 图库、校准与推理

```bash
conda run -n retailvision-seg python tools/build_gallery.py --dataset datasets/retail_v001/recognition --output datasets/gallery_v001
conda run -n retailvision-seg python tools/evaluate.py --task calibrate --input independent_validation_scores.json --output datasets/calibration_v001.json --dataset-version val_v001 --model-version dinov2_b14_v001 --gallery-version gallery_v001 --experiment-id calibration_v001
conda run -n retailvision-seg python tools/predict.py --config configs/inference/local.json --image shelf.jpg --target LAYS_CUCUMBER_70G
```

`identity_score` 是余弦匹配分数；不是概率。校准输入须来自独立验证采集组，包括已知与未知 SKU。Top-2 为第二个不同 SKU；不足两种 SKU 时拒绝匹配。

## 验证

```bash
conda run -n retailvision-seg python tools/check_system.py
conda run -n retailvision-seg pytest -q
conda run -n retailvision-seg python tools/verify_models.py
cd frontend && npm run build
```

架构、数据、训练、推理、部署与许可分别见 `docs/`。本地 API 不含用户认证，仅绑定 loopback；不要直接暴露公网。

## 正式开发入口与路径

正式根目录为 `/home/gzh/rocco/retailvision`，所有后续算法、训练和前端工作均在此进行。

```bash
conda activate retailvision-seg  # 分割与完整本地 API
# 或 conda activate retailvision-reid  # 识别训练与特征提取
```

`tools/*.py` 和 API 入口启动时动态定位本项目根目录。相对配置、数据、图像和输出参数均相对于项目根目录；外部数据请传绝对路径。直接调用 `load_pipeline` 和模型 Adapter 时，模型与配置路径也以项目根目录为基准。

前端开发：`cd /home/gzh/rocco/retailvision/frontend && npm ci && npm run dev`。
权重分别位于 `models/segmentation/rtmdet_ins_s_coco_v001/` 与 `models/recognition/dinov2_b14_v001/`；数据入口为 `tools/prepare_dataset.py`，训练入口为 `tools/train.py`，推理入口为 `tools/predict.py`。

迁移证据见 [MIGRATION_REPORT.md](docs/MIGRATION_REPORT.md)。`docs/DELIVERY.md`、旧训练日志及 `*.raw.txt` 保留迁移前历史内容，其中的历史路径不作为运行配置使用。
