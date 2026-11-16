# 数据契约

COCO: images(id,file_name,width,height,capture_group)，categories 中 retail_product，annotations(id,image_id,category_id,segmentation,attributes)。标注类别会统一映射为 1。默认按图片数量划分约 90% 训练集、10% 验证集，固定 seed，至少需要两个独立采集组。验证目标张数为总图片数的 10% 四舍五入，至少一张；同一采集组整体分配，实际比例可能偏离 90/10。分配先选择张数最接近目标的组，再加入能缩小张数差距的组，始终保留训练组。

分割与识别使用相同划分，输出 `segmentation/annotations/train.json`、`val.json` 和对应的识别 ROI，manifest 的 `split_image_counts` 记录实际图片数量。不自动生成测试集，数据集评估默认使用验证集；历史数据仍可显式指定 `--split test`。只有一个采集组或缺少分组信息时会报错，不随机拆散同场景照片。每款 SKU 的训练样本和验证覆盖仍须核对，90/10 划分不保证所有 SKU 都在两边出现。

SKU master: sku_id,brand,product,flavor,specification；attributes 优先，sidecar 以 annotation ID 为键。缺失 ID、未知 SKU、属性冲突均报错。sidecar 不覆盖已有有效属性。

验证范围：文件、尺寸、重复 ID/实例几何、Polygon 自交/越界/面积、Mask 非空、采集分组。多部件 Polygon 保守要求人工审核，避免错合并。算法不能凭标注判断两个相邻实体是否被单个合法 Polygon 包围，需要人工质检。

图片验证会完整解码所有帧，并拒绝 Pillow 检出的截断像素数据；仅文件头和尺寸正确不足以通过。`--validate-only` 与正式导入使用相同检查，损坏图片会在创建输出数据集前报错。

同一 capture_group 永不跨集合；采集组应包含连续视频帧、相同物理样本、重复场景，不只用图像文件名。版本哈希包含标注、SKU master、sidecar、所有原图、split seed 及划分策略，避免新旧划分共用版本。目标目录禁止覆盖。图像原件复制保留；ROI 使用灰色背景屏蔽和 5% 边缘上下文，记录原图、实例及分组关系。在线增强只用于训练，保持标签文字方向。

本仓库 tests/conftest.py 是明确标记的合成数据，不是实际 CVAT 数据。

少量真实数据的 Smoke Test 可使用 `tools/prepare_dataset.py --no-split`：输出 `segmentation/annotations/all.json` 和 `recognition/all/<sku_id>/<annotation_id>.png`，不调用分组划分器，不要求或伪造 capture_group；manifest 标记 `no_split=true`，groups 为空。此模式不生成正式训练/验证/测试集合。默认转换仍要求真实采集组并执行原有分组划分。

CVAT JSON 的 file_name 只有 basename 时，`--images` 应指向实际图片目录（例如解压后的 `images/default/`），不做递归猜测。输入提供 Bounding Box 时原值保留；缺失时才从 Mask 计算。`recognition/skus.json` 保存实际使用 SKU 的主表元数据，空 flavor/specification 等描述字段原样保留；sku_id 仍不可为空。
