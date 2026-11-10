# RetailVision V1.0 基线验收

验收日期：2026-11-10（Asia/Shanghai）。正式根目录：`/home/gzh/rocco/retailvision`。

四项迁移后验收全部通过。此基线冻结本地工程与真实 GPU Fixture 集成能力，**不代表零售/药房业务识别精度验收**。

| 验收项 | 结果与证据 |
|---|---|
| 项目独立性 | 扫描 2430 份活动文本，旧路径运行引用为 0；符号链接未指向旧目录。环境 sys.path、editable direct_url、.pth 和 finder 无旧路径。禁止 Python 打开整个旧 DemoRoom 根目录后，真实离线 GPU 流程通过，模块来源均为新项目。见 `path-audit.json`、`environment-*.json`、`runtime-independence.json`、`offline-gpu.log`。 |
| 环境完整性 | seg、reid 各 21 passed，显式开启 CUDA 测试，无跳过；两环境 pip check 均通过。RTX 4090，torch 2.1.2+cu121，CUDA runtime 12.1。reid 单独完成离线 CUDA DINOv2 特征提取。19 份模型/历史 checkpoint 哈希与迁移清单一致，第三方源码 commit 未变且干净。见 `pytest-*.log`、`system-*.json`、`environment-reid.json`、`path-audit.json`。 |
| 前后端完整性 | npm ci、TypeScript/Vite 构建通过；4 项 Playwright 通过。默认 Backend 8018 与 Fixture Backend 8020 均从 /tmp 启动并定位到新项目，Vite 5178 代理新 8018。浏览器覆盖真实 GPU 上传推理、Box、Original/Overlay/Mask、透明度、PNG/JSON 导出、未知目标和未配置错误。桌面及 390px 手机真实 GPU 截图已检查，无横向溢出；输入标签、模式 aria-pressed 与 Canvas 标签保留。见 `services.json`、`playwright.log`、`frontend-build.log`、截图。 |
| 推理完整性 | RTMDet-Ins-s 与 DINOv2-B/14 参数均在 cuda:0；输出 21 个实例、427×640 Mask、768 维单位 embedding；FAISS 自参考检索、组合流程、未知向量拒识、真实 FastAPI POST 与 Mask GET 均通过。跨工作目录 CLI 成功。见 `report.json`、`combined_pipeline.json`、`api_real_inference.json`、`predict-from-tmp.log`。 |

以上证据均位于 [`baseline/v1.0.0/`](baseline/v1.0.0/)。网络禁止验证通过 Python socket.connect 阻断执行，证明本机权重离线加载与推理；未声称验证操作系统级断网或全新离线安装。FAISS 按既有依赖使用 CPU；两个神经网络使用真实 GPU，无 CPU 回退。

## 冻结内容

- 本地 main 首个基线提交及 annotated tag `v1.0.0`。代码、配置、前后端锁文件、文档和验收证据纳入 Git；未发布远端。
- Python 包、API 与前端版本统一为 1.0.0；两个环境仅以 `--no-deps --no-build-isolation -e .` 重绑同一项目版本，未升级第三方包、驱动或 CUDA，未重新训练。
- `assets.json` 冻结 2752 份本地模型、数据、历史训练产物、第三方源码、生产构建与 GPU Fixture 资产的大小及 SHA256。大文件继续按既有 .gitignore 留在本地，**Git 标签本身不包含这些资产**，恢复时须保留或另行备份它们。
- `seg/reid-freeze.actual.txt` 和 Conda explicit 文件记录本次实际环境；`environments/` 的可复现锁文件继续保留，历史 raw 锁文件不改。
- MMDetection：`44ebd17b145c2372c4b700bfb9cb20dbd28ab64a`；DINOv2：`7764ea0f912e53c92e82eb78a2a1631e92725fc8`。
- RTMDet 权重 SHA256：`fdc5d7ec327cf46dba079d664be98a853e80936098c1778edf5ffb726b084908`；DINOv2 权重 SHA256：`0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`。
- `AGENTS.md` 要求后续全线开发读取并遵循 stop-that-shit 技能，验收达成后结束，不扩展功能或重复验证。

历史 DELIVERY、迁移报告、日志、raw 环境文件与一次历史失败训练的 error traceback 中仍保留旧路径；这些是历史证据，不参与运行时导入或模型加载。原 DemoRoom 副本未改动。模型和数据版本未改写。

## 启动与复验

```bash
cd /home/gzh/rocco/retailvision
git show --no-patch v1.0.0
conda run --no-capture-output -n retailvision-seg uvicorn backend.api.app:app --host 127.0.0.1 --port 8018
# 打开 http://127.0.0.1:8018；默认未配置真实商品图库，明确拒绝推理。
```

真实 GPU Fixture：停止同端口默认服务后，设置 `RETAILVISION_CONFIG=configs/inference/fixture.local.json` 再启动上述 Backend。上传 `third_party/mmdetection/demo/demo.jpg`，目标 `FIXTURE_INSTANCE`。本次验收用临时服务已停止。

核心复验命令：

```bash
RETAILVISION_GPU_TEST=1 conda run --no-capture-output -n retailvision-seg pytest -q
RETAILVISION_GPU_TEST=1 conda run --no-capture-output -n retailvision-reid pytest -q
conda run --no-capture-output -n retailvision-seg python tools/verify_migration.py --forbidden-root /home/gzh/rocco/demoroom
cd frontend && npm ci && npm run build
# 浏览器测试须同时启动默认 API 8018、Fixture API 8020 与 Vite 5178：
npx playwright test --workers=1 --reporter=line
```

GPU 验证脚本会更新 Fixture 配置和 results 下的验证产物；本次冻结的证据已独立保存在 docs/baseline/v1.0.0。

## 继承限制

当前缺少真实商品 CVAT 数据、SKU 图库及独立已知/未知验证集，不能报告商品 Top1、mAP 或端到端成功率。COCO 自参考 Fixture 与合成负向量不构成商品校准。RTMDet checkpoint 独立商业许可仍待确认，沿用 LICENSES.md 的记录。现有上游弃用提醒不影响本次通过的检查。
