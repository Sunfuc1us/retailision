# RetailVision 第一次真实 CVAT COCO 数据集 Smoke Test

测试日期：2026-11-10（Asia/Shanghai）。工作目录：`/home/gzh/rocco/retailvision`。

## 结论

**通过：3 张真实图片、9 个 Polygon Annotation，生成 9 个有效实例 Mask、9 个有效商品 ROI，SKU 对应一致。** 无空 Mask、无图片丢失、无转换中的 SKU 错配。MMDetection 实际数据加载和识别图像预处理均通过。只测试数据管线；没有加载模型、训练、构建图库或执行正式 Train/Val/Test 划分。

按用户后续更正，本次临时将输入类别校验从 `retail_product` 改为 `Detail` 后测试；转换结束在 finally 中恢复，当前代码要求 `retail_product`。未增加永久类别映射配置。**当前代码直接再次导入原始 Detail JSON 仍会被类别校验拒绝**，这符合“测试后改回”的要求；此次通过依赖该明确记录的临时调整。COCO 输出仍使用内部单类 `retail_product`，类别 ID 1。

## 输入与 COCO 解析

- ZIP：`datasets/incoming/retailtest.zip`，原件未修改。
- SHA256：`b70625aaf63004116a2d2a71d26b85a8fa25cf8560a0c9716ce464084c065f75`。
- 原样解压到：`datasets/incoming/retailtest_extracted_v001/`。
- JSON：`annotations/instances_default.json`。
- 3 张图均为 1440×1080；image ID 为 1–3，annotation ID 为 1–9，每图 3 个实例。
- 1 个 Category：`id=1, name=Detail`。
- 9/9 个 Annotation 都有单一、合法、非自交、非越界 Polygon；Polygon 解码和非空校验通过。
- 原始图片没有 capture_group。本次不补造分组，使用显式 `--no-split`；输出 groups={}、split=all、capture_group=null。没有 train/val/test 目录或标注文件。

输入摘要见 [`input.json`](../results/coco_smoke_retailtest_v001/input.json)。

## 图片路径解析

JSON file_name 只有 basename，ZIP 内图片实际位于 `images/default/`。传入真实图片根目录：

```text
--images datasets/incoming/retailtest_extracted_v001/images/default
```

转换器以 `images_root / file_name` 精确定位，3/3 张图成功读取，尺寸与 COCO 声明一致。没有修改原始 JSON 的 file_name，也没有增加递归猜测。输出 segmentation/images 中 3 张图片逐字节与输入一致，SHA256 和实际路径记录于 integrity.json 的 paths 字段。

## SKU 属性解析

直接读取 `annotations[].attributes` 中的 sku_id、brand、product、flavor、specification。9/9 实例五个键齐全，sku_id 均非空且相互不同。Annotation 7、8、9 的 flavor 是空字符串，读取、转换和 SKU 元数据保存均成功。occluded=false 等原始附加属性保留在输出 COCO。

本次从标注属性生成独立主表 `datasets/incoming/retailtest_skus_v001.json`，没有修改正式 `catalog/skus.json`，没有猜测、重命名或修正 SKU ID（例如原始 `FENTA_ORANGE_330ML` 原样保留）。转换后的 `recognition/skus.json` 保存全部 9 个 SKU 的五项元数据。

以下 SKU 匹配指对原始标注的一致性检查；该主表来自同一份标注，不代表对外部业务商品主表的独立校验。

| Annotation | sku_id | brand | product | flavor | specification | Mask 像素 | ROI 宽×高 |
|---|---|---|---|---|---|---:|---|
| 1 | `VITA_YASHIXIANG_LEMON_TEA_250ML` | 维他 | 柠檬茶饮料 | 鸭屎香 | 250ml | 85953 | 298×383 |
| 2 | `VITA_BLACKCURRANT_DRINK_250ML` | 维他 | 风味饮料 | 黑加仑味 | 250ml | 71713 | 245×390 |
| 3 | `VITA_APPLE_JUICE_250ML` | 维他 | 苹果汁饮料 | 苹果味 | 250ml | 86007 | 309×396 |
| 4 | `FENTA_ORANGE_330ML` | 芬达 | 汽水 | 橙味 | 330ml | 111005 | 287×587 |
| 5 | `COKE_ZERO_330ML` | 可口可乐 | 汽水 | 原味 | 330ml | 92891 | 242×559 |
| 6 | `SPRITE_ORIGINAL_330ML` | 雪碧 | 汽水 | 柠檬味 | 330ml | 108101 | 287×580 |
| 7 | `XX_KBD_KFY_18` | 香雪 | 抗病毒口服液 | 空字符串 | 18支 | 128917 | 372×466 |
| 8 | `999_GML_KL_9` | 999 | 感冒灵颗粒 | 空字符串 | 9袋 | 60419 | 276×268 |
| 9 | `BYS_XCH_KL_10` | 白云山 | 小柴胡颗粒 | 空字符串 | 10袋 | 79569 | 315×335 |

## Segmentation 与 Mask

输出 `datasets/retailtest_smoke_v001/segmentation/annotations/all.json`：3 images、9 annotations、1 category。原始 Polygon、Bounding Box 浮点值、Annotation ID、image_id 和 attributes 全部保留；内部类别统一为 retail_product/id=1，area 使用解码 Mask 像素面积。

9 个独立全分辨率 Mask 保存在 `recognition/masks/1.png` 至 `9.png`（相对转换输出目录）。沿用现有 Mask 存储位置；分割侧使用 COCO Polygon，并由真实 MMDetection LoadAnnotations 生成实例 Mask。PNG 全部为 1440×1080，值域 {0,255}；与 pycocotools 解码的原始 Polygon 逐像素一致，空 Mask 数为 0。

实际使用 MMDetection CocoDataset + LoadImageFromFile + LoadAnnotations(with_bbox=True, with_mask=True) 读取，结果为 3 张图片、9 个 bbox、9 个有效全分辨率 Mask。模型内部 label 为 0（单类索引），与 COCO category_id=1 一致；bbox 转为 XYXY 后仍与原始标注一致。没有创建 Runner 或加载 checkpoint。

## Recognition 与 ROI

9 个 Crop 位于 `recognition/all/<sku_id>/<annotation_id>.png`。裁剪依据为 Polygon Mask 的紧边界加现有 5% padding；Mask 外像素设为 RGB(127,127,127)。逐项检查：Mask 内像素与原图完全一致，Mask 外背景像素全部为 127，裁剪尺寸全部大于 0；9/9 ROI 均为 RGB PNG，并有唯一 Annotation ID、image_id 与 SKU ID 对应。

真实调用现有 `dataset.augmentations.recognition.training_transform()` 对 9 个 ROI 进行图像预处理，均得到有限值的 3×224×224 Tensor。这只执行图像变换，没有启动训练或模型推理。

## 数据完整性与可视化

| 检查 | 实际结果 |
|---|---|
| 输入图片 / 成功解析 | 3 / 3 |
| Annotation / 有效商品实例 | 9 / 9 |
| 有效 ROI / 对应非空 SKU ID | 9 / 9 |
| SKU 元数据条目 | 9 |
| 空 Mask / 缺失图片 / 转换 SKU 错配 | 0 / 0 / 0 |
| 原图字节、Polygon、原始 Box、属性保留 | 全部通过 |
| MMDetection COCO 数据加载 | 3 图、9 实例通过 |
| Recognition ROI 图像预处理 | 9 个 3×224×224 Tensor 通过 |
| Train/Val/Test 划分 / 模型加载 / 训练 | 均未执行 |

已查看全部 3 张 Polygon Overlay、9 个 Mask 和9 个商品 ROI 的汇总图。商品轮廓与裁剪对象一致，未见空图、错商品或明显截断；ROI 保留包装文字，灰色背景与 Polygon 外区域一致。该检查验证裁剪跟随现有标注，不替代正式标注质检或商品识别精度评估。

- [Polygon Overlay 汇总](../results/coco_smoke_retailtest_v001/polygon_contact_sheet.jpg)
- [9 个实例 Mask 汇总](../results/coco_smoke_retailtest_v001/mask_contact_sheet.png)
- [9 个商品 ROI 汇总](../results/coco_smoke_retailtest_v001/roi_contact_sheet.jpg)
- 原分辨率 Overlay：`results/coco_smoke_retailtest_v001/overlays/image_1.png` 至 `image_3.png`。
- 每实例 Mask、ROI 路径及像素校验结果：[integrity.json](../results/coco_smoke_retailtest_v001/integrity.json)。

## 自动化测试

在 retailvision-seg 环境、类别恢复后运行：

```bash
conda run --no-capture-output -n retailvision-seg pytest -q \
  tests/test_dataset.py tests/test_paths.py tests/test_evaluation.py tests/test_api.py
```

**17 passed，1 条已有上游 httpx/Starlette 弃用提醒，0 failed，0 skipped。** 日志：[pytest.log](../results/coco_smoke_retailtest_v001/pytest.log)。其中 Dataset 测试 10 项，覆盖原有数据转换/分组无泄漏、sidecar、异常数据与 Mask，以及新增 no-split 回归和类别恢复回归。

新增回归明确禁止 no-split 调用分组器；检查 basename 配合 images/default 根目录、缺少 capture_group 时仅 no-split 放行、原始 bbox、SKU 元数据和空字符串保留、无正式集合产物。另有测试确认当前输入 Detail 会被 retail_product 规则拒绝，证明临时类别调整已恢复。

下游实际加载记录：[consumers.json](../results/coco_smoke_retailtest_v001/consumers.json)、[consumers.log](../results/coco_smoke_retailtest_v001/consumers.log)。

## 发现的问题、处理与最终状态

| 项目 | 原因 | 本次处理 | 最终状态 |
|---|---|---|---|
| Detail 输入类别初始被拒绝 | 原校验要求 retail_product；no-split 初次验证得到 9 条类别错误 | 按用户要求临时接受 Detail，真实 CLI 验证/转换后立即恢复 | 临时测试通过；无永久映射配置，原规则已恢复 |
| 原管线强制采集组和正式划分 | 本次 3 图无 capture_group，且明确禁止正式划分 | 增加显式 --no-split，默认分组要求不变 | 本次全部保存为 all，无伪造分组 |
| 原 bbox 会被 Mask 整数边界重写 | 原转换器无条件重算 bbox | 保留输入 bbox，缺失时才计算 | 9/9 原始 bbox 保留 |
| Recognition 缺少完整 SKU 主表输出 | 原 metadata 仅保存实例与 SKU ID 关系 | 输出实际使用 SKU 的 recognition/skus.json | 9 条五属性元数据完整 |
| 图片目录层级与 basename 不同 | CVAT ZIP 的正常布局 | 显式指定 images/default 为图片根目录 | 3/3 定位成功，无代码搜索回退 |
| 空字符串 flavor | 3 个药品未填写口味 | 原有逻辑已允许，无额外修复 | 3/3 原样保留 |

没有剩余的本次 Smoke Test 转换失败项。原始 Detail 数据在最终恢复后的默认校验下仍不直接通过；测试过程及恢复状态已记录于 [conversion-status.json](../results/coco_smoke_retailtest_v001/conversion-status.json) 和 temporary-category.patch。未来正式训练仍须准备真实采集组及独立划分，本次没有作此准备或发起训练。

## 文件位置与复现

- 原始 ZIP：`datasets/incoming/retailtest.zip`。
- 原始解压数据：`datasets/incoming/retailtest_extracted_v001/`。
- 本次独立 SKU 主表：`datasets/incoming/retailtest_skus_v001.json`。
- 双路转换产物：`datasets/retailtest_smoke_v001/`。
- Segmentation：`segmentation/images/*.jpg`、`segmentation/annotations/all.json`、`segmentation/manifest.json`。
- Recognition：`recognition/all/<sku_id>/<annotation_id>.png`、`recognition/masks/*.png`、`recognition/metadata.json`、`recognition/skus.json`。
- 本次 dataset_version：`d21d4e9b27a4ed12d7a1f067825266c765d508776ad773f3d8c3f7743b0fef62`。
- 完整日志、检查脚本、JSON 证据和可视化：`results/coco_smoke_retailtest_v001/`。

`run_conversion.py` 是本次一次性执行记录，包含临时类别替换、实际 CLI 命令及 finally 恢复；输出目录禁止覆盖，复验须使用新版本目录。`check_outputs.py` 和 `check_consumers.py` 记录完整性、可视化及下游加载检查。原始数据、SKU 主表、转换产物和 results 按既有 .gitignore 留在本地，未纳入 Git；报告和数据管线的最小修改保留在工作区，未改动 V1.0 标签及历史数据版本。
