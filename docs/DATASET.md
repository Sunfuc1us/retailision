# 数据契约

COCO: images(id,file_name,width,height,capture_group)，categories 中 retail_product，annotations(id,image_id,category_id,segmentation,attributes)。标注类别会统一映射为 1。必须至少三组独立采集数据，固定 seed 分组划分 train/val/test。

SKU master: sku_id,brand,product,flavor,specification；attributes 优先，sidecar 以 annotation ID 为键。缺失 ID、未知 SKU、属性冲突均报错。sidecar 不覆盖已有有效属性。

验证范围：文件、尺寸、重复 ID/实例几何、Polygon 自交/越界/面积、Mask 非空、采集分组。多部件 Polygon 保守要求人工审核，避免错合并。算法不能凭标注判断两个相邻实体是否被单个合法 Polygon 包围，需要人工质检。

同一 capture_group 永不跨集合；采集组应包含连续视频帧、相同物理样本、重复场景，不只用图像文件名。版本哈希包含标注、SKU master、sidecar、所有原图及 split seed。目标目录禁止覆盖。图像原件复制保留；ROI 使用灰色背景屏蔽和 5% 边缘上下文，记录原图、实例及分组关系。在线增强只用于训练，保持标签文字方向。

本仓库 tests/conftest.py 是明确标记的合成数据，不是实际 CVAT 数据。
