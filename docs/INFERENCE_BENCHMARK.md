# Inference benchmark

Engineering validation on the fixed 3-image / 9-ROI fixture; not retail accuracy.

Protocol: 50 warm-up calls, 200 measured calls per model/backend; detection threshold 0.3. GPU: NVIDIA GeForce RTX 4090.

| Model / backend | Load (ms) | Mean (ms) | P50 (ms) | P95 (ms) | Peak GPU (MiB) | GPU util avg (%) | Size (MiB) | Output parity |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| RTMDet-Ins pytorch_fp32 | 1647.7 | 30.62 | 30.40 | 32.75 | 3183 | 29.5 | 39.1 | FP32 baseline |
| DINOv2 pytorch_fp32 | 1085.2 | 6.06 | 6.03 | 6.60 | 3183 | 29.5 | 330.3 | FP32 baseline |
| RTMDet-Ins pytorch_fp16 | 224.0 | 30.49 | 30.32 | 32.82 | 2485 | 26.0 | 39.1 | matched 13/13; mean box IoU 0.99745; max box delta 3.07px; mean mask IoU 0.99912 |
| DINOv2 pytorch_fp16 | 981.6 | 6.57 | 6.49 | 7.35 | 2485 | 26.0 | 330.3 | min cosine 0.999996; max abs 0.000340 |
| RTMDet-Ins onnxruntime | 550.6 | 62.02 | 65.28 | 68.73 | 4007 | 47.8 | 41.2 | matched 13/13; mean box IoU 0.99882; max box delta 3.89px; mean mask IoU 0.99290 |
| DINOv2 onnxruntime | 297.4 | 3.77 | 3.71 | 4.07 | 4007 | 47.8 | 330.4 | min cosine 1.000000; max abs 0.000149 |
| RTMDet-Ins tensorrt | 114.5 | 16.66 | 16.54 | 18.22 | 3589 | 36.2 | 29.3 | matched 13/13; mean box IoU 0.99175; max box delta 14.88px; mean mask IoU 0.99243 |
| DINOv2 tensorrt | 122.1 | 2.38 | 2.37 | 2.73 | 3589 | 36.2 | 167.4 | min cosine 0.999912; max abs 0.001657 |

Latency includes preprocessing, CUDA execution, output transfer, and model postprocessing. Models remain loaded during warm-up and measurement. GPU memory/utilization are process-external samples while both models are resident; memory also includes desktop/other GPU processes. Load times include source imports and model construction.

Instance matching uses same category and maximum global Box IoU, retaining pairs with IoU ≥ 0.5; unmatched instances remain explicit. NMS can select different overlapping detections near score ties.

Full Box/Mask and embedding parity details: `work_dirs/deployment/benchmark.json`.

## Interpretation

The dataset contains 3 images and 9 labeled product instances. The source COCO pretrained model produces 13 detections at threshold 0.3 for this backend comparison. All 13 are matched by the alternative backends. Box and Mask outputs have measured numerical differences, including a TensorRT maximum Box coordinate delta of 14.88 pixels; these results establish fixture engineering consistency, not a retail accuracy guarantee. Source clipping uses the padded image canvas; masks are cropped to the original image.

DINOv2 timings cover one ROI. Full unified inference performs one embedding per detected instance and adds FAISS search and mask-file writing. UI screenshots capture first requests after backend switches; warmed 200-call model measurements above support the performance comparison. Model load times are one observed load per adapter and include import costs, with the first PyTorch import cold.

TensorRT improves both measured model latencies in this fixture. PyTorch FP16 does not show a useful speed improvement at batch 1 here. ORT RTMDet is slower than PyTorch because this adapter returns dense mask outputs through CPU memory; native TensorRT retains dense masks on GPU until instance selection. ORT CUDA execution remains a verified deployment alternative.

## Functional acceptance — 2026-11-10

| Check | Result | Evidence |
|---|---|---|
| Two independent ONNX exports and checker | Passed | Versioned ONNX artifacts and manifests under `models/` |
| ONNX loading and CUDA inference | Both models passed | `work_dirs/deployment/benchmark.json` |
| Native TensorRT builds / FP16 inference | Both models passed | CUDA 12.4 engine manifests; benchmark JSON |
| trtexec loading and repeated execution | Both engines passed | `work_dirs/deployment/trtexec-rtmdet.log`, `trtexec-dinov2.log` |
| RTMDet Box / category / nonempty Mask comparison | 13/13 instances matched at threshold 0.3 | Full matched Box/Mask data in benchmark JSON |
| DINOv2 embeddings | All 9 real ROI crops compared | Embedding errors/cosines in benchmark JSON |
| Unified PyTorch / ORT / TensorRT → FAISS → target SKU and Mask | All three passed real HTTP requests | `work_dirs/deployment/ui/{pytorch,onnxruntime,tensorrt}.json` |
| TensorRT → PyTorch fallback | Passed | `work_dirs/deployment/fallback-pytorch.json` |
| PyTorch FP16 through config factory and full pipeline | Passed with Python networking blocked | `work_dirs/deployment/fp16_unified.json` |
| Backend switching, Mask download, desktop/mobile Workbench | 1 real browser test passed; screenshots inspected | `work_dirs/deployment/playwright.log`, `ui/` |
| API / pipeline / path automated checks | 6 passed; 1 Starlette deprecation warning | `work_dirs/deployment/pytest.log` |
| Training environments / checkpoint preservation | Both still PyTorch 2.1.2+cu121 and CUDA available; original hashes preserved | `environment-seg.json`, `environment-reid.json`, model manifests |
| Deployment → training environment switch | Library search path restored | `work_dirs/deployment/environment-switch.log` |

The unified HTTP/Workbench checks use the existing MMDetection demo photo and a real DINOv2 self-reference FAISS Gallery with a synthetic negative vector. The dedicated fixture policy uses explicit engineering thresholds, not a retail validation calibration. Original Gallery and calibration files are preserved. No new model training was started, and no source checkpoint was replaced. No Metric Learning head is present in this fixture.

Acceptance summary: `work_dirs/deployment/acceptance.json`. [Desktop screenshot](../work_dirs/deployment/ui/tensorrt-desktop.png) · [Mobile screenshot](../work_dirs/deployment/ui/mobile.png).
