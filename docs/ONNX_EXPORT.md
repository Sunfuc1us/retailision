# ONNX export

Validated on 2026-11-10 with local pretrained RTMDet-Ins-s COCO and DINOv2 ViT-B/14 weights. This proves deployment behavior on fixtures, not retail recognition accuracy. Checkpoints remain unchanged.

## RTMDet-Ins

Use MMDeploy v1.3.1 (`bc75c9d6c8940aa03d0e1e5b5962bd930478ba77`) and the project configuration, which preserves instance masks and matches the source NMS IoU 0.6 and `nms_pre=1000`.

```bash
conda activate retailvision-deploy
python tools/deploy/mmdeploy_deploy.py \
  configs/deployment/rtmdet_ins_onnxruntime_static_640.py \
  third_party/mmdetection/configs/rtmdet/rtmdet-ins_s_8xb32-300e_coco.py \
  models/segmentation/rtmdet_ins_s_coco_v001/model.pth \
  datasets/retailtest_smoke_v001/segmentation/images/20261110-141818.474-1.jpg \
  --work-dir models/segmentation/rtmdet_ins_s_coco_v001/deployment/onnx \
  --device cuda --dump-info
python tools/deploy/validate_rtmdet_backend.py --backend onnxruntime
```

The static input is `input: [1,3,640,640]`, normalized BGR. Preserve aspect ratio when resizing, then pad to 640; forcing an explicit resize shape in MMDeploy's `create_input` stretches these images and breaks parity. The adapter uses the source test pipeline and data preprocessor. Outputs are `dets` (xyxy boxes plus score), `labels`, and `masks`. The adapter restores Box coordinates to source-image units, applies the source configuration's padded-canvas clipping, and crops binary masks to the true original image dimensions. Category IDs are retained.

`export_postprocess_mask=True` remains enabled. TensorRT's TopK limit is below MMDeploy's default 5000, so the export uses 1000, matching this source model's configuration. No mask output is removed.

## DINOv2

```bash
conda activate retailvision-deploy
python tools/deploy/export_dinov2.py \
  --checkpoint models/recognition/dinov2_b14_v001/model.pth \
  --output models/recognition/dinov2_b14_v001/deployment/model.onnx
python tools/deploy/validate_dinov2_onnx.py
```

The graph takes RGB `images: [1,3,224,224]`, using resize 224×224 and ImageNet mean/std normalization. It returns `embedding: [1,768]`, L2 normalized. Opset 17 and static batch 1 are the validated configuration. `--dynamic-batch` exports a dynamic batch graph; its engine profile must be built separately.

The export adapter expands Attention into scaled QK matrix multiplication, Softmax and AV multiplication while reusing all learned parameters. It checks this adaptation against the source PyTorch forward before ONNX export. The original SDPA export ran correctly in ORT but produced incorrect TensorRT embeddings; the explicit formulation fixes the observed TensorRT 10.0.1 optimization incompatibility.

An existing Linear Metric Learning head checkpoint in the project's `best_head.pth` format can be included with `--projection PATH`. The head runs before L2 normalization and changes the exported embedding dimension. No learned head is present in this fixture, so the delivered model exports the pretrained backbone embedding.

The exporter runs ONNX checker and writes `model.manifest.json`. Runtime adapters read actual tensor names and reject missing CUDA execution. Full numerical comparisons on the nine real ROI crops are in [INFERENCE_BENCHMARK.md](INFERENCE_BENCHMARK.md).

## ONNX Runtime CUDA

The selected ORT 1.18.0 CUDA 12 build supports cuDNN 8, matching PyTorch 2.1.2's cuDNN 8.9.2. Install from the official CUDA 12 feed as described in [DEPLOYMENT.md](DEPLOYMENT.md); the same-version PyPI wheel tested here expected CUDA 11. See the [official CUDA provider matrix](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

`inference/backends/onnx_cuda.py` preloads the environment's CUDA libraries before session construction. Both adapters require `CUDAExecutionProvider` as the first provider; CPU fallback is rejected. ORT can place shape and postprocessing operations on CPU; supported model computation uses CUDA. Session-level CPU fallback is rejected.
