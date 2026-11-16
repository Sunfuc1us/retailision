# RetailVision 算法改進與受控對照報告（2026-11-13）

本報告記錄針對 **RTMDet 分割 → DINOv2 特徵 → 圖庫匹配 → 拒識** 流程的三項算法修復與改進實作、離線 RTX 4090 CUDA 受控對照實驗與單元測試驗證。所有歷史模型權重（`models/recognition/dinov2_b14_19sku_v003/`、`models/segmentation/retail_19sku_v003/` 等）與歷史資料集（`datasets/retail_19sku_v003/`、`datasets/retail_6sku_v002/`）均保持未覆寫；推理階段網路結構（凍結 DINOv2 ViT-B/14 + `Linear 768→256`）與推理耗時不變。

---

## 一、程式碼修復與改進內容

### 1. 識別訓練對分割誤差與背景殘留的魯棒性（第 1 項）

**原始問題**：`retailvision/data/prepare.py` 預先呼叫 `mask_crop(..., padding=0.05)` 將 GT Mask 以外像素填成純灰 `127`；`retailvision/training/recognition.py` 訓練時僅讀取這些去背裁切圖，導致訓練時從未見過實際 RTMDet 分割的裁切偏移、缺邊、溢邊與背景/鄰近商品殘留。

**實作改動**：
- [image_ops.py](../retailvision/image_ops.py)：抽取 `mask_bbox()` 與 `render_mask_crop()`，讓資料準備、訓練擾動與推理裁切共用同一套座標與填充邏輯，`mask_crop()` 保持逐像素不變。
- [augmentations.py](../retailvision/training/augmentations.py)：新增 `perturbed_mask_crop()`，以可控機率保留乾淨 `mask_crop` 樣本（預設 `clean_prob=0.35`），並對其餘樣本施加：
  - 不對稱裁切邊界偏移（`max_pad_in=0.04` 至 `max_pad_out=0.16`）；
  - 橢圓形態學腐蝕（缺邊 `1%–4%`）或膨脹（溢邊 `1.5%–6%`）與單邊局部裁缺（`3%–11%`）；
  - 實際原圖背景殘留（膨脹帶或安全比例內的未遮罩框內背景）。
  - **語義保護約束**：強制檢查裁切後可見 GT 前景面積 `>= 72%`、框內 GT 面積 `>= 80%`、未遮罩時前景佔比 `>= 40%`，不滿足則自動退回乾淨 `mask_crop`，防止裁掉主體商品或改變標籤含義。
- [recognition.py](../retailvision/training/recognition.py)：新增 `_load_roi_contexts()`，當設定檔啟用 `roi_augment` 時，從同級 `../segmentation/images/` 與 `../segmentation/annotations/{split}.json`（含 `masks/{id}.png` 後備）預裁切含 25% 上下文（最長邊限制 512 px）的原圖與 Mask 供訓練動態擾動；未設定 `roi_augment` 時保持原有行為不變。

### 2. 凍結主幹下的相似 SKU 細粒度特徵學習（第 2 項）

**原始問題**：原 `retail_19sku_v003` 僅使用 `TripletMarginLoss` + `hard` mining，在 12 張/批（每次僅覆蓋 6/19 類）下後期有效 triplet 迅速衰減（第 15 輪降為 `0`），驗證分離度不足，且缺乏全域 19 類邊界約束；此外驗證同為 `1.0` 時僅取最早達標輪次，未考慮類別間距。

**實作改動**：
- [recognition.py](../retailvision/training/recognition.py)：
  - 新增 `cosine_ce` 與 `triplet+cosine_ce` 訓練監督模式：在現有 `Linear 768→256` 後接入**僅限訓練期使用**的單位向量原型分類頭 `prototypes`（預設 `cosine_scale=16.0`、`cosine_margin=0.10`、`label_smoothing=0.05`）。
  - `best_head.pth` 與 `last_head.pth` 仍僅儲存 `head`（`768→256`），推理期完全丟棄分類原型，不增加任何推理計算或破壞部署格式。
  - 在驗證迴圈計算真類與最大錯類餘弦分差 `validation_margin`，並在 `validation_top1` 相同時以更高的 `validation_margin` 作為最佳權重選擇條件。

### 3. 圖庫多近鄰共同支持與拒識匹配（第 3 項）

**原始問題**：`Gallery.search()` 原先固定以 `np.maximum.at` 取每個 SKU 的單一最相似參考向量（1-NN max），容易因圖庫內單一異常角度/光線參考樣本拉高錯類或未知商品的相似度。

**實作改動**：
- [gallery.py](../retailvision/inference/gallery.py)：擴充 `Gallery` 支援 `aggregation='max'`（預設，保持既有行為相容）與 `aggregation='topk_mean'`（取同 SKU 內分數最高的 `min(support_k, 該SKU參考數)` 個近鄰均值，避免單一異常參考造成偶發高分，同時兼容少參考 SKU），並支援 `search(..., aggregation=..., support_k=...)` 即時切換比較。
- [factory.py](../retailvision/inference/factory.py) 與 [pipeline.py](../retailvision/inference/pipeline.py)：支援從推理設定可選 `matching` 欄位載入 `aggregation` 與 `support_k`，並在 `get_runtime_info()` 揭露目前匹配策略。

---

## 二、受控對照實驗設計（`retail_19sku_v003`）

所有訓練均固定於 **RTX 4090、離線 CUDA（禁用 Python 網路連線）、`seed=42`、20 epoch、`batch_size=12`、`lr=0.0001`、凍結 `dinov2_b14_v001` 主幹、`Linear 768→256`**，共用 `datasets/retail_19sku_v003`（訓練 46 圖 / 274 實例、留出驗證 5 圖 / 32 實例；同批次拍攝留出 `same_capture_image_holdout`，非獨立場景）。各實驗均使用各自訓練 split 的 274 個乾淨樣本重建獨立圖庫。

| 實驗 ID | 設定檔 | 關鍵變因 | 最佳輪次 | 訓練期最佳 Val Top-1 / Margin |
|---|---|---|---:|---:|
| `baseline_v003`（既有基線） | `configs/recognition/retail_19sku_v003.py` | Triplet（無 ROI 擾動） | 8 | 1.0000 / 0.1983（後測） |
| `algo_exp1_roi_triplet` | `work_dirs/algo_research/configs/exp1_roi_triplet.py` | **+ 第 1 項**：ROI/Mask 擾動 | 20 | 0.9688 / 0.2509 |
| `algo_exp2a_cosine_ce` | `work_dirs/algo_research/configs/exp2a_cosine_ce.py` | **+ 第 2 項**：純 Cosine CE | 20 | 1.0000 / 0.3644 |
| `algo_exp2b_triplet_cosine_ce` | `work_dirs/algo_research/configs/exp2b_triplet_cosine_ce.py` | **+ 第 2 項**：Triplet + Cosine CE | 20 | 1.0000 / 0.3793 |
| `algo_exp3_roi_triplet_cosine_ce` | `work_dirs/algo_research/configs/exp3_roi_triplet_cosine_ce.py` | **第 1 + 2 項結合** | 20 | 1.0000 / 0.3788 |

評測資料位於 [evaluation_report.json](../work_dirs/algo_research/evaluation_report.json)，包含三種查詢條件：
1. **乾淨 GT-ROI**（`val` 32 實例）。
2. **實際 RTMDet-Ins-s 預測 Mask**（使用 `models/segmentation/retail_19sku_v003/model.pth`、`threshold=0.3`、`clip_mask_to_bbox=True` 抽取，`32/32` 匹配 GT，平均匹配 IoU `0.9575`、最小 `0.8260`，另含 1 個未標註背景物件預測）。
3. **固定種子合成分割誤差壓力測試**（`seed=2026`，每個 val 實例 5 組隨機偏移 + 腐蝕/膨脹 + 局部裁缺 + 背景殘留，共 160 筆查詢）。

---

## 三、對照結果與核心發現

### 1. 第 1 項與第 2 項特徵學習對照（固定使用 `max` 圖庫匹配）

| 模型 | 乾淨 GT Top-1 | 乾淨 GT 平均 Margin | 相似 SKU 平均 Margin | RTMDet 預測 Top-1 | RTMDet 預測平均 Margin | 誤差壓力測試 Top-1（160筆） | 誤差壓力測試平均 Margin | 移出真類後最高錯類分（Clean / RTMDet） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_v003` | 1.0000 (32/32) | 0.1983 | 0.2035 | 0.9688 (31/32) | 0.1991 | 0.8938 (143/160) | 0.1441 | 0.9228 / 0.9066 |
| `exp1_roi_triplet` | 0.9688 (31/32) | 0.2509 | 0.2439 | 0.9688 (31/32) | 0.2548 | **0.9062 (145/160)** | 0.1805 | 0.8437 / 0.8048 |
| `exp2a_cosine_ce` | 1.0000 (32/32) | 0.3644 | 0.3396 | 0.9688 (31/32) | 0.3698 | 0.8938 (143/160) | 0.2577 | 0.6129 / 0.5985 |
| `exp2b_triplet_cosine_ce` | 1.0000 (32/32) | **0.3793** | **0.3637** | 0.9688 (31/32) | **0.3898** | 0.8688 (139/160) | 0.2719 | 0.5947 / **0.5753** |
| `exp3_roi_triplet_cosine_ce` | **1.0000 (32/32)** | 0.3788 | 0.3582 | 0.9688 (31/32) | 0.3854 | **0.9000 (144/160)** | **0.2838** | **0.5859** / 0.5937 |

細分相似商品組的乾淨 GT 平均餘弦 Margin（真類最高分減最相近同組競品）：

| 模型 | 統一三款茶飲 | 維他三款飲品 | 三養兩款火雞麵 | 巧克力系零食（奧利奧/好麗友/百奇） |
|---|---:|---:|---:|---:|
| `baseline_v003` | 0.2374 | 0.1305 | 0.1564 | 0.2375 |
| `exp1_roi_triplet` | 0.3280 | 0.1160 | 0.2284 | 0.2341 |
| `exp2b_triplet_cosine_ce` | 0.4143 | **0.3000** | **0.3556** | **0.3504** |
| `exp3_roi_triplet_cosine_ce` | **0.4501** | 0.2553 | 0.3235 | 0.3410 |

**實驗判讀**：
- **第 1 項有效提升分割誤差容忍度**：單加 `roi_augment`（`exp1`）讓 160 筆誤差壓力測試 Top-1 從 `89.38%` 升至 `90.62%`（搭 `topk_mean_2` 達 `91.87%`），平均 Margin 提升 `+25.3%`（`0.1441 → 0.1805`），10 分位數 Margin 由負轉正（`-0.0019 → +0.0066`）。但僅用 Triplet 時因樣本變難、採樣漸少，導致 1 個邊界樣本（奧利奧薄脆 `ann_id=95`）在乾淨 GT 下由勝轉微幅落敗（`0.6681` vs `0.6808`）。
- **第 2 項大幅拉開相似 SKU 類間距並壓低錯類峰值**：加入 Cosine 分類監督後（`exp2b` / `exp3`），乾淨與實際預測 Mask 的類間 Margin 幾乎翻倍（`0.1983 → 0.3793` / `0.3788`），維他系列與三養火雞麵 Margin 提升 2 倍以上；當從圖庫移除真類時，錯類最高餘弦相似度從基線 `0.9228` 劇降至 `0.575–0.595`。
- **第 1 + 2 項互補（`exp3` 綜合表現最佳）**：`exp3` 修正了 `exp1` 乾淨 GT 掉點（恢復 `32/32 = 100%`），同時修正了 `exp2b` 在分割誤差壓力測試上的退化（`86.88% → 90.00%`），並保持最高的壓力測試 Margin（`0.2838`）與最低的乾淨圖庫缺類誤匹配峰值（`0.5859`）。
- **實測真實背景誤檢物件（`unmatched_img11_pred7.png`）**：對 `old_6sku` 驗證圖中 RTMDet 檢出的 1 個未標註背景罐裝物，真類不在圖庫內時，`baseline_v003` Top-1 錯配分為 `0.4310`、`exp1` 為 `0.5071`，而 `exp2b` 降至 `0.2243`、`exp3` 降至 `0.2098`（搭配 `topk_mean_3` 降至 `0.2009`）。
- **已知共同失效案例**：五個模型在實際 RTMDet 預測 Mask 上均固定失敗同 1 筆樣本——`OREO_THIN_DARK_CHOCOLATE_62`（`ann_id=95`，對應預測裁切 `gt_95_pred_6.png`，其箱體邊緣有明顯青綠色強反光，與訓練視角差異大且即便在乾淨 GT 下基線分數也僅 `0.6216`）。

### 2. 第 3 項圖庫匹配策略比較（`max` vs `topk_mean`）

**（A）對已知 SKU 辨識與誤差測試的影響（以 `exp1` / `exp3` 與 `baseline_v003` 為例）：**

| 模型 | 匹配方式 | 乾淨 GT Top-1 | RTMDet Top-1 | 壓力測試 Top-1 | 乾淨 GT 平均 Margin | 缺類誤匹配最高分（Clean） |
|---|---|---:|---:|---:|---:|---:|
| `baseline_v003` | `max` (1-NN) | **1.0000** | 0.9688 | **0.8938** | **0.1983** | 0.9228 |
| `baseline_v003` | `topk_mean_2` | 0.9688 | 0.9688 | 0.8812 | 0.1929 | 0.9156 |
| `baseline_v003` | `topk_mean_3` | 0.9688 | 0.9688 | 0.8875 | 0.1893 | **0.9128** |
| `exp1_roi_triplet` | `max` (1-NN) | 0.9688 | 0.9688 | 0.9062 | **0.2509** | 0.8437 |
| `exp1_roi_triplet` | `topk_mean_2` | 0.9688 | 0.9688 | **0.9187** | 0.2505 | 0.8240 |
| `exp1_roi_triplet` | `topk_mean_3` | 0.9688 | 0.9688 | 0.9000 | 0.2468 | **0.8150** |
| `exp3_roi_triplet_cosine_ce` | `max` (1-NN) | **1.0000** | 0.9688 | **0.9000** | **0.3788** | 0.5859 |
| `exp3_roi_triplet_cosine_ce` | `topk_mean_2` | **1.0000** | 0.9688 | 0.8812 | 0.3769 | 0.5464 |
| `exp3_roi_triplet_cosine_ce` | `topk_mean_3` | 0.9688 | 0.9688 | 0.8875 | 0.3713 | 0.5158 |
| `exp3_roi_triplet_cosine_ce` | `topk_mean_5` | 0.9688 | 0.9688 | 0.8750 | 0.3640 | **0.4847** |

**（B）對未見過未知 SKU 的誤匹配壓抑測試（`dinov2_b14_6sku_v003` + 6-SKU 圖庫面對 13 款未見 SKU / 26 筆驗證實例）：**

| 匹配方式 | 未見 SKU 平均錯配 Top-1 | P90 錯配分 | 最高錯配分 | 超過 0.72 筆數 | 超過 0.80 筆數 |
|---|---:|---:|---:|---:|---:|
| `max` (1-NN) | 0.5869 | 0.7095 | 0.8042 | 3 / 26 | 1 / 26 |
| `topk_mean_2` | 0.5739 | 0.6901 | 0.7881 | 2 / 26 | **0 / 26** |
| `topk_mean_3` | 0.5654 | 0.6831 | 0.7826 | **1 / 26** | **0 / 26** |
| `topk_mean_5` | **0.5548** | **0.6747** | **0.7779** | **1 / 26** | **0 / 26** |

**第 3 項判讀**：`topk_mean`（特別是 `support_k=2` 或 `3`）能穩定壓低未見商品與圖庫離群參考的偶發高分（將 6-SKU 測試中超過 `0.80` 的誤接受從 1/26 降為 0/26、超過 `0.72` 從 3/26 降為 1/26，並將 `exp3` 缺類峰值再降 `0.04–0.07`）；但若 `support_k` 過大（如 `5`），會開始稀釋僅與少數角度接近的已知商品得分。

---

## 四、重要限制與使用提醒

1. **資料與驗證範圍限制**：`retail_19sku_v003` 僅有 51 張同場景照片（驗證 5 張 / 32 實例），屬於 `same_capture_image_holdout`，不代表跨門市、跨光線或不同實物的真實零售泛化精度。
2. **特徵空間分數尺度變化（不可沿用舊閾值）**：採用 Cosine 分類監督後，類別原型在球面更均勻分散，**正樣本絕對餘弦分數會同步略降**（`exp3` 乾淨平均分 `0.8270`、最低 `0.4395`，相較基線 `0.8937` / `0.6216`），但真錯類分差放大近倍（`0.1983 → 0.3788`）。因此原本基於純 Triplet 設定的 `0.80` 臨時門檻不能直接套用於新權重。
3. **尚未切換工作站預設模型**：由於仍缺少獨立採集的已知／未知 SKU 校準集，本次研究產物保留於 `work_dirs/algo_exp*` 與 `work_dirs/algo_research/`，尚未發佈覆蓋工作站現有 `retail_19sku_v003` 配置；待獨立已知／未知資料到位後，再以 `exp3`（`roi_augment + triplet+cosine_ce`）與 `topk_mean`（`support_k=2` 或 `3`）進行正式校準與發佈。

---

## 五、驗證與產物索引

- **自動化測試**：
  - `PYTHONNOUSERSITE=1 conda run -n retailvision-seg pytest -q`：**104 passed, 1 skipped**
  - `PYTHONNOUSERSITE=1 conda run -n retailvision-reid pytest tests/test_training.py tests/test_pipeline.py -q`：**26 passed**
  - `PYTHONNOUSERSITE=1 conda run -n retailvision-deploy pytest tests/test_pipeline.py -q`：**12 passed**
- **新增／更新測試**：[test_training.py](../tests/test_training.py)（前景保留約束與 ROI 上下文載入檢查）、[test_pipeline.py](../tests/test_pipeline.py)（`topk_mean` 抑制單點異常參考、少參考兼容、參數檢查與 `remove_sku` 保留設定）。
- **實驗產物目錄**：
  - 設定與評測腳本：[work_dirs/algo_research/](../work_dirs/algo_research/)（`configs/`、`extract_rtmdet_val.py`、`eval_all.py`、`evaluation_report.json`）
  - 訓練與對應圖庫：`work_dirs/algo_exp1_roi_triplet/`、`work_dirs/algo_exp2a_cosine_ce/`、`work_dirs/algo_exp2b_triplet_cosine_ce/`、`work_dirs/algo_exp3_roi_triplet_cosine_ce/`
