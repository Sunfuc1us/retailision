# RetailVision 工程交付报告

交付日期：2026-11-10。项目根目录：`/home/gzh/rocco/demoroom/retailvision`。旧 DemoRoom 文件与实验未覆盖、未删除。

## 验收结论

已完成本地平台基础实现、两个预训练模型的真实 RTX 4090 离线运行、COCO 双数据集转换、两套训练与断点恢复、统一推理/拒识接口、评估/注册模块和 Workbench 的真实 GPU 服务联调。**尚未完成零售/药房业务精度验收**：当前没有用户真实 CVAT 商品数据、SKU 主表、独立已知/未知商品验证集。

| 阶段 | 实际结果 | 证据 |
|---|---|---|
| M0 环境检查 | RTX4090 24GB、Ubuntu22.04、驱动580.178.04、初始可用磁盘约246GB，保留旧项目 | logs/nvidia-smi-initial.txt、os-release.txt |
| M1 模型基础 | 独立 Conda 环境，禁止 user site，官方本地权重，GPU 与离线验证通过 | logs/system-seg.json、system-reid.json、results/model_verification/report.json |
| M2 数据 | 6图/12实例/3采集组 Fixture 实际转换，COCO、ROI、Mask、映射与版本哈希均生成 | datasets/fixture_v001、logs/dataset-fixture-report.json |
| M3 训练 | 两种任务均真实 GPU 短训练；识别从epoch0恢复到epoch1；分割从epoch1恢复到epoch2 | work_dirs/fixture_*、logs/train-*.log |
| M4 推理 | RTMDet→Mask ROI→DINO→FAISS→目标筛选→PNG，未知向量与低分 Fixture 拒绝 | results/model_verification/combined_pipeline.json、api_real_inference.json |
| M5 评估/注册 | COCOeval、Mask IoU、识别分片/混淆矩阵/未知拒识、管线指标；不可覆盖注册与版本配置 | evaluation/、registry/、models/*/*/manifest.json、tests/ |
| M6 工作站 | 三栏React、上传/查询/运行/Overlay/Mask/透明度/Box/耗时/导出；真实GPU后端浏览器检查 | results/ui/real-gpu-overlay.png、real-gpu-mask.png、mobile-fixture.png |

## 实际测试

- `retailvision-seg`：20项自动测试通过，包含显式 CUDA、环境隔离、数据、检索/拒识、API、注册与指标测试。`logs/pytest-seg.log`。
- `retailvision-reid`：相同20项通过。`logs/pytest-reid.log`。
- 浏览器：3项通过，其中1项直接连接真实本地GPU服务，另外2项验证明确标记的响应Fixture和未配置错误状态。`logs/frontend-all-tests.log`。
- `tsc -b && vite build` 通过。`logs/frontend-build.log`。
- 两环境 `pip check` 均无依赖缺失。`logs/pip-check-*.log`。
- Python网络连接被禁止后，两模型加载、推理与真实 FastAPI 请求成功；这证明本机权重离线加载，不等于已验证全新离线安装。`tools/verify_models.py`。
- 两种 Trainer 均产生实际 GPU checkpoint 和 TensorBoard 日志。分割单类头由80类迁移，类别头形状不匹配为预期；一至两轮合成图训练未产生验证检测结果，COCO evaluator 无可报告mAP，best_checkpoint=null。没有伪造best模型。
- 识别 Fixture 为简单色块，hard triplet loss 为0、Fixture Top1为1；仅证明训练/保存/恢复执行，不证明细粒度商品训练有效。
- 训练API实际4.5ms返回202，后台训练完成，第二任务停止成功；证据见 `results/training_api_verification.json` 与 `logs/training-api-verification.log`。

## GPU 结果

- GPU：NVIDIA GeForce RTX 4090，可见显存24564MiB。
- 实际 PyTorch CUDA runtime：12.1。驱动所显示CUDA13.0为兼容上限，系统无nvcc，未改驱动。
- RTMDet-Ins-s：COCO demo图640×427，阈值0.3，输出21个实例，每个Mask为427×640。
- DINOv2 ViT-B/14：768维，L2范数0.99999994；FAISS自参考检索和真实组合管线均通过。
- 双模型验证峰值分配：710.4MiB；峰值保留：822.0MiB。含模型初始化及多次请求的验证脚本约4.05s。
- 分割Fixture训练Runner报告约751MiB；这不是复杂真实货架训练显存上限。UI截图记录了一次推理耗时，不能当成稳定性能基准。

## 环境与权重

Python3.10.22、torch2.1.2+cu121、torchvision0.16.2+cu121、MMCV2.1.0、MMEngine0.10.7、MMDetection3.3.0、NumPy1.26.4、FAISS CPU1.15.1。Node20.20.2、npm10.8.2、React18、Vite6.4.6。完整实际版本和可复现文件见 `environments/` 及 `frontend/package-lock.json`。

MMCV官方cu121/torch2.1/cp310 wheel决定组合；使用setuptools80.9.0保留torch2.1构建所需pkg_resources。旧框架维护风险已记录。原始freeze保留在*.raw.txt，复现freeze将临时MMCV本地URL改为同一官方wheel URL，保留SHA256。

| 模型 | 本地权重 |
|---|---|
| RTMDet-Ins-s，COCO80类，仅预训练 | models/segmentation/rtmdet_ins_s_coco_v001/model.pth |
| DINOv2-B/14，官方预训练 | models/recognition/dinov2_b14_v001/model.pth |

每份 manifest 包含 SHA256、官方来源、源码commit、训练数据来源、版本与许可状态；RTMDet config已展开保存。Fixture训练产物位于work_dirs，未冒充商品生产模型注册。

## 启动与操作

在项目根目录执行：

```bash
cd /home/gzh/rocco/demoroom/retailvision
# 空商品主表的正式工作站：推理会明确要求配置
conda run --no-capture-output -n retailvision-seg uvicorn backend.api.app:app --host 127.0.0.1 --port 8018
# 浏览器打开 http://127.0.0.1:8018
```

复现已经验证的真实模型样例（明确标记Fixture，COCO长椅不是商品）：

```bash
RETAILVISION_CONFIG=configs/inference/fixture.local.json conda run --no-capture-output -n retailvision-seg uvicorn backend.api.app:app --host 127.0.0.1 --port 8018
# 上传 third_party/mmdetection/demo/demo.jpg，目标输入 FIXTURE_INSTANCE
conda run -n retailvision-seg python tools/predict.py --config configs/inference/fixture.local.json --image third_party/mmdetection/demo/demo.jpg --target FIXTURE_INSTANCE
```

真实商品训练与推理：

```bash
conda run -n retailvision-seg python tools/prepare_dataset.py --coco export.json --images raw_images --catalog catalog/skus.json --output datasets/retail_v001
conda run --no-capture-output -n retailvision-seg python tools/train.py --task segmentation --model rtmdet-ins-s --config configs/segmentation/rtmdet_ins_s.py --dataset datasets/retail_v001/segmentation
conda run --no-capture-output -n retailvision-reid python tools/train.py --task recognition --model dinov2-vitb14 --config configs/recognition/dinov2_b14.py --dataset datasets/retail_v001/recognition
# --resume 指向last.pth或epoch_N.pth可恢复
conda run -n retailvision-seg python tools/build_gallery.py --dataset datasets/retail_v001/recognition --output datasets/gallery_v001
# 使用映射层时增加 --projection work_dirs/<id>/best_head.pth，重建图库并重新校准
conda run -n retailvision-seg python tools/evaluate.py --task calibrate --input independent_validation_scores.json --output datasets/calibration_v001.json --dataset-version val_v001 --model-version dinov2_b14_v001 --gallery-version gallery_v001 --experiment-id calibration_v001
conda run -n retailvision-seg python tools/predict.py --config configs/inference/local.json --image shelf.jpg --target LAYS_CUCUMBER_70G
```

生产local.json从example复制，填入训练后的单类分割模型、匹配版本的图库、主表及独立校准。后续从本地路径加载不发起模型下载。模型版本切换通过明确配置及服务重启，当前UI模型选择为只读显示。

## 未完成与风险

1. 缺少真实CVAT数据、SKU主表、独立采集组、未知商品和相似药盒样本；不能报告实际商品Top1、mAP、端到端成功率。单类retail_product真实训练待执行。
2. 自参考COCO Fixture及合成负向量只验证组合路径和拒识分支；其校准文件不可用于真实商品部署。真实阈值必须在独立验证集校准，再用独立test验收。
3. 新官方DINO仓库含非商业CELL-DINO扩展，运行时只直接导入Apache的ViT-B/14工厂，避免顶层hubconf加载额外模块。分发时需按实际所用源码保留许可。
4. RTMDet checkpoint独立商业许可声明未确认；已记录来源，不因代码Apache就默认权重同许可。见LICENSES.md。
5. 合法Polygon是否错误包围多个商品仍需人工质检；multipart标注保守报错等待审核。分组字段要由真实采集信息提供，不能用随机文件名代替。
6. API面向loopback本地使用；当前任务管理器保存本地实验，但API重启后的在途进程重新接管未实现。模型热切换、RTMDet-m对照、部分backbone微调不在当前执行范围。
7. 当前评估CLI接收模型输出/检索记录；真实困难切片、混淆矩阵和失败集须由真实test数据产出。人工标签来源真实性不能由工具代替。
8. 测试出现上游httpx TestClient弃用提醒、PyTorch旧接口提醒；不影响已运行检查。xFormers未安装，使用同一CUDA设备上的原生PyTorch attention，并非CPU回退。

完整架构、数据、训练、推理、部署与许可见同目录其余文档。所有安装失败、修复与证据保留在logs/FAILURES.md和对应日志。
