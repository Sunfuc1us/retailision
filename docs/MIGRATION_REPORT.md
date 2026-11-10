# RetailVision 独立迁移报告

迁移日期：2026-11-10。

## 结论

迁移完成。**唯一正式开发根目录为 `/home/gzh/rocco/retailvision`**。前后端、数据管线、训练/推理/评估代码、正式模型与Fixture checkpoint、第三方源码、锁文件和历史报告均已保留。新项目可直接开发，不再运行时读取旧 DemoRoom 路径。未训练模型、未增加模型架构、未升级依赖或驱动。

Source：`/home/gzh/rocco/demoroom/retailvision`。Destination：`/home/gzh/rocco/retailvision`。Destination在迁移前不存在，无覆盖、合并未知文件。Source保留为历史副本，不再作为开发入口。

## 文件与复制证据

使用rsync -a安全复制。排除node_modules、__pycache__、.pytest_cache、*.pyc、*.tsbuildinfo、.venv及可重建retailvision.egg-info；前端随后npm ci重建。没有复制Conda环境目录。

复制完成、路径修复前，**2984个正式/历史文件逐项校验通过**：相对路径、大小、SHA256、权限及链接信息一致。完整文件清单为 [source-before.json](migration/source-before.json)，初始副本清单为 [destination-copy.json](migration/destination-copy.json)，rsync操作清单为 [rsync-files.txt](migration/rsync-files.txt)。迁移后的最终清单为 migration/destination-final.json。差异记录见 migration/destination-diff.json；新增验证结果、缓存、Git和必要路径修复与原始复制明确区分。

| 迁移目录或文件 | 初始文件数 |
|---|---:|
| .gitignore | 1 |
| AGENTS.md | 1 |
| README.md | 1 |
| backend | 8 |
| catalog | 3 |
| configs | 8 |
| dataset | 11 |
| datasets | 44 |
| docs | 7 |
| environments | 11 |
| evaluation | 2 |
| frontend | 19 |
| inference | 11 |
| logs | 53 |
| models | 6 |
| pyproject.toml | 1 |
| registry | 2 |
| results | 27 |
| tests | 8 |
| third_party | 2693 |
| tools | 12 |
| training | 8 |
| work_dirs | 47 |

前次交付报告docs/DELIVERY.md和历史训练日志原样保留。重跑验证前的模型验证报告及UI截图备份到results/migration-before/。数据和模型版本未因迁移重命名；没有重新训练或覆盖历史checkpoint。

## SAM 3.1 / YOLOE 排除与原位置保护

源RetailVision本来就不含SAM3.1/YOLOE实验。本次没有复制父目录中的`.sam31/`、`.sam3-source/`、`yoloe-26s-seg.pt`、SAM模型、实验脚本及独立依赖；也未删除或修改它们。

迁移前后核对：源正式项目2984项内容/权限一致；SAM环境20454项、SAM源码557项、父目录models 5项均一致（清单排除可再生Python缓存）。独立YOLOE权重、SAM实验脚本/依赖/start脚本也通过sha256sum -c。见 migration/preservation.json 和 migration/excluded-files-before.sha256。

MMDetection自带的YOLOX等上游模块以及samplers/sample文件属于原官方源码，并不是YOLOE或SAM3.1；没有按模糊名称误删正式依赖。

## 模型与大文件SHA256

19份.pth/checkpoint均比较内容哈希，全部一致，未改写checkpoint内部的历史训练元数据。完整机器可读结果为 [weights-sha256.json](migration/weights-sha256.json)。

| 文件 | 源与目标SHA256 | 比较 |
|---|---|---|
| models/recognition/dinov2_b14_v001/model.pth | 0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73 | 一致 |
| models/segmentation/rtmdet_ins_s_coco_v001/model.pth | fdc5d7ec327cf46dba079d664be98a853e80936098c1778edf5ffb726b084908 | 一致 |
| work_dirs/06c9573fc3b1482881abee74467bf9ec/best.pth | 93dcd01fb9446851922045b8d67ff042855b1561bc03e04d690dc8af30f2b2ff | 一致 |
| work_dirs/06c9573fc3b1482881abee74467bf9ec/best_head.pth | a8b4d69d33cb2b9e4dea4db1161a4a23bd0414cdda2207e415160247541f19e4 | 一致 |
| work_dirs/06c9573fc3b1482881abee74467bf9ec/last.pth | 7248d8698cea21d5841085fd358f837f56e9b6af7e1092be9b90540e4d78990c | 一致 |
| work_dirs/92a98384be6a4b72b0af1ec28408ee05/best.pth | 93dcd01fb9446851922045b8d67ff042855b1561bc03e04d690dc8af30f2b2ff | 一致 |
| work_dirs/92a98384be6a4b72b0af1ec28408ee05/best_head.pth | a8b4d69d33cb2b9e4dea4db1161a4a23bd0414cdda2207e415160247541f19e4 | 一致 |
| work_dirs/92a98384be6a4b72b0af1ec28408ee05/last.pth | 7248d8698cea21d5841085fd358f837f56e9b6af7e1092be9b90540e4d78990c | 一致 |
| work_dirs/e442764fba27440b842a4045aa562a7f/best.pth | 93dcd01fb9446851922045b8d67ff042855b1561bc03e04d690dc8af30f2b2ff | 一致 |
| work_dirs/e442764fba27440b842a4045aa562a7f/best_head.pth | a8b4d69d33cb2b9e4dea4db1161a4a23bd0414cdda2207e415160247541f19e4 | 一致 |
| work_dirs/e442764fba27440b842a4045aa562a7f/last.pth | 7248d8698cea21d5841085fd358f837f56e9b6af7e1092be9b90540e4d78990c | 一致 |
| work_dirs/fixture_reid_resume_fixed/best.pth | 93dcd01fb9446851922045b8d67ff042855b1561bc03e04d690dc8af30f2b2ff | 一致 |
| work_dirs/fixture_reid_resume_fixed/best_head.pth | a8b4d69d33cb2b9e4dea4db1161a4a23bd0414cdda2207e415160247541f19e4 | 一致 |
| work_dirs/fixture_reid_resume_fixed/last.pth | 029c06e66cc9fd429848110d2add97f303c8bac341df167c03a4c7f0a1acf09a | 一致 |
| work_dirs/fixture_reid_smoke/best.pth | 93dcd01fb9446851922045b8d67ff042855b1561bc03e04d690dc8af30f2b2ff | 一致 |
| work_dirs/fixture_reid_smoke/best_head.pth | a8b4d69d33cb2b9e4dea4db1161a4a23bd0414cdda2207e415160247541f19e4 | 一致 |
| work_dirs/fixture_reid_smoke/last.pth | 7248d8698cea21d5841085fd358f837f56e9b6af7e1092be9b90540e4d78990c | 一致 |
| work_dirs/fixture_seg_resume/epoch_2.pth | e0bd5b3e83faf4f20c42ab35da08062b3775aa9c6d29aa91d1e6063c20d88866 | 一致 |
| work_dirs/fixture_seg_smoke/epoch_1.pth | b024ea4824f627750b9724317c43a1eb3bc67a707eb1df86a8c3042b90694667 | 一致 |

## 环境复用与editable重绑

复用retailvision-seg、retailvision-reid，不重建环境。Python3.10.22，torch2.1.2+cu121、torchvision0.16.2+cu121、CUDA runtime12.1、RTX4090、驱动580.178.04；seg仍使用MMCV2.1.0、MMEngine0.10.7、MMDetection3.3.0；FAISS CPU1.15.1。reid按原职责不安装MMDetection/MMCV。

seg的mmdet与retailvision通过`--no-deps --no-build-isolation -e`重绑新源码；reid仅添加同版本本地retailvision editable入口，便于从任意目录导入，不变更依赖版本。两个环境PYTHONNOUSERSITE=1保持不变，sys.path与editable direct_url不含旧路径，两个pip check均通过。

| 环境 | editable路径 |
|---|---|
| retailvision-seg / mmdet | `/home/gzh/rocco/retailvision/third_party/mmdetection` |
| retailvision-seg / retailvision | `/home/gzh/rocco/retailvision` |
| retailvision-reid / retailvision | `/home/gzh/rocco/retailvision` |

详见 migration/environment-audit.json、logs/migration-rebind.log、migration-reid-binding.log、migration-pip-check-*.log。更新的freeze和YAML在environments/，原始raw锁文件及Conda explicit文件保留。

## 路径与独立性

- project_paths.py从自身文件确定根目录；CLI和API入口统一项目相对路径。相对输入/输出按正式根目录解释，外部图片和数据可使用绝对路径。
- 模型Adapter和load_pipeline显式解析本地源代码、配置、权重、图库、校准与结果路径；训练子进程显式cwd为新根目录。
- 修复DINO模型配置、数据manifest的source位置、历史Runner可复用配置及last_checkpoint位置，未改权重或数据版本。
- 扫描Python、JSON、YAML、Shell、环境、frontend资源和符号链接，活动旧路径引用为0；全部符号链接不指向旧目录。
- 旧路径仅保留于明确标记的历史日志、交付报告、失败traceback和raw锁记录。它们不是运行配置。见 migration/path-audit.json、path-fixes.json、symlinks.json。
- 从`/tmp`直接启动新根目录predict CLI成功；正式API也从`/tmp`通过新editable导入并服务新frontend/dist。
- tools/verify_migration.py通过Python audit hook禁止打开传入的旧根目录，再执行原有离线GPU模型验证；成功，且已加载的RetailVision、MMDetection、DINOv2模块来源均在新项目。证据results/migration-independence.json。

## Backend / GPU验证

| 检查 | 实际结果 |
|---|---|
| seg自动测试 | 21 passed，含新增跨cwd路径回归 |
| reid自动测试 | 21 passed，含显式CUDA与环境隔离 |
| 两模型离线加载 | Python网络连接禁止时成功 |
| RTMDet-Ins-s | 21实例，Box与427×640独立Mask，PNG保存成功 |
| DINOv2-B/14 | 768维embedding，L2 norm 0.99999994 |
| FAISS与统一管线 | 通过，目标Fixture输出独立Mask |
| 未知拒识 | 原有低分/未知向量测试通过，浏览器未知目标返回unknown_target |
| 真实API | POST推理及GET Mask PNG成功，驻留模型复用 |
| 跨工作目录CLI | `/tmp`启动，成功输出目标Mask |

日志：logs/migration-pytest-*.log、migration-model-verification.log、migration-independence.log、migration-predict-from-tmp.log。GPU模型验证峰值分配710.4MiB，保留822.0MiB。没有将Fixture作为真实商品性能报告。

## 前端验证

完整React/TypeScript/Vite/Tailwind/Lucide源码和package-lock保留。`npm ci`成功，`npm run build`成功。原有3项Playwright测试及新增迁移功能回归共**4 passed**，实际覆盖上传、目标查询输入、GPU推理、Box、Original/Overlay/Mask、透明度、PNG/JSON导出、未知目标与未配置错误。桌面与手机截图已检查，未重新设计UI。

日志：logs/migration-npm-ci.log、migration-frontend-build.log、migration-frontend-tests.log。截图results/ui/migration-overlay.png、real-gpu-mask.png、mobile-fixture.png。

## Git

Source没有顶层Git仓库，也没有可迁移的顶层提交、分支或未提交Git状态；因此不存在丢失旧Git历史。所有源文件均复制保留。Destination新建独立main分支仓库，尚无提交，源码为未跟踪状态，未擅自创建提交。

两个third_party仓库的原有.git、commit和源码状态均保留；MMDetection commit44ebd17b145c2372c4b700bfb9cb20dbd28ab64a，DINOv2 commit7764ea0f912e53c92e82eb78a2a1631e92725fc8，均无源码修改。weights/datasets/work_dirs/third_party按现有.gitignore留在本地，通过哈希清单管理；本次不引入Git LFS。

## 启动与开发

```bash
cd /home/gzh/rocco/retailvision
conda activate retailvision-seg
uvicorn backend.api.app:app --host 127.0.0.1 --port 8018
# Workbench + API: http://127.0.0.1:8018
# 当前迁移前的旧API已停止，8018已切换为新项目。

# 前端独立开发，另一个终端
cd /home/gzh/rocco/retailvision/frontend
npm ci
npm run dev
# http://127.0.0.1:5178，代理新Backend 8018
```

真实GPU Fixture模式（先停止同端口默认服务再启动）：

```bash
cd /home/gzh/rocco/retailvision
RETAILVISION_CONFIG=configs/inference/fixture.local.json conda run --no-capture-output -n retailvision-seg uvicorn backend.api.app:app --host 127.0.0.1 --port 8018
# 上传third_party/mmdetection/demo/demo.jpg，目标FIXTURE_INSTANCE
conda run -n retailvision-seg python tools/predict.py --config configs/inference/fixture.local.json --image third_party/mmdetection/demo/demo.jpg --target FIXTURE_INSTANCE
```

数据、训练、评估命令与接口保持原样；见README、DATASET、TRAINING、INFERENCE。训练命令仅保留使用说明，本次没有执行训练。

## 未解决问题

无迁移阻塞项。继承此前限制：没有真实商品CVAT训练和独立校准数据；Fixture自参考结果不证明零售准确率；RTMDet checkpoint独立商业许可仍待确认；模型UI版本显示仍按现有配置工作，不增加热切换功能。迁移未修改这些业务能力。

旧目录、原SAM3.1/YOLOE资产继续完整保留。所有后续开发仅使用Destination。
