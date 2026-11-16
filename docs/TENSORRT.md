# TensorRT deployment

## Environment and artifacts

Native TensorRT Python API 10.0.1 runs in `retailvision-deploy`. `tools/deploy/trtexec` invokes NVIDIA's `24.05-py3` container (TensorRT 10.0.1.6, CUDA 12.4.1). The host driver is 580.178.04 and the target GPU is RTX 4090, compute capability 8.9. The container is required only for the CLI; the application uses native Python TensorRT. Native supplemental CUDA libraries are pinned to the CUDA 12.4.1 component versions, matching this container release.

Engines are reproducible deployment artifacts, retained alongside the original PyTorch checkpoint. Engine metadata records GPU/compute capability, CUDA runtime and PyTorch CUDA build, TensorRT version, source ONNX path/hash, precision, static input shape or dynamic profiles, workspace, TF32 setting, file hash/size, and UTC build time. Rebuild for incompatible TensorRT or GPU versions.

## Build FP16 engines

Export ONNX first using [ONNX_EXPORT.md](ONNX_EXPORT.md).

```bash
conda activate retailvision-deploy
python tools/deploy/build_tensorrt.py \
  --onnx models/segmentation/rtmdet_ins_s_coco_v001/deployment/onnx/end2end.onnx \
  --output models/segmentation/rtmdet_ins_s_coco_v001/deployment/tensorrt/end2end.engine \
  --input-name input --precision fp16 --workspace-gib 6
python tools/deploy/build_tensorrt.py \
  --onnx models/recognition/dinov2_b14_v001/deployment/model.onnx \
  --output models/recognition/dinov2_b14_v001/deployment/model_fp16.engine \
  --input-name images --precision fp16 --workspace-gib 6
python tools/deploy/validate_rtmdet_backend.py --backend tensorrt
python tools/deploy/validate_dinov2_onnx.py --backend tensorrt \
  --output work_dirs/deployment/dinov2_tensorrt_parity.json
```

The tested inputs are static `[1,3,640,640]` for RTMDet and `[1,3,224,224]` for DINOv2. For an independently exported dynamic graph, provide all of `--min-shape`, `--opt-shape`, `--max-shape`. `--precision fp32 --disable-tf32` is available for numerical diagnosis. Unsupported graph operators fail with parser diagnostics; `--plugin PATH` loads an explicitly required plugin.

The delivered RTMDet graph uses standard ONNX operators and TensorRT's NMS support, with all three outputs (`dets`, `labels`, `masks`). It requires no custom MMDeploy plugin library. `pre_top_k=1000` avoids the default TopK=5000 parser failure and matches the source configuration.

## Resident inference

The Python adapters retain engines and execution contexts on CUDA. The TensorRT 10 adapter uses tensor names and `execute_async_v3`, avoiding MMDeploy 1.3.1 runtime wrappers written for older TensorRT binding APIs. RTMDet's data-dependent NMS outputs use persistent, aligned `IOutputAllocator` buffers that are reused across calls. Creating a new output allocator on every call caused memory growth during the initial 200-call test and has been corrected. DINOv2 uses the engine's actual input/output dtypes and CUDA tensor addresses. RTMDet keeps dense mask outputs on GPU, filters selected instances there, and transfers only the final masks; it avoids copying all candidate masks to CPU and back.

DINOv2 must use the explicit Attention ONNX export. The original SDPA-derived graph produced embeddings nearly orthogonal to the source in TensorRT, despite ORT parity passing; changing opset or disabling TF32 did not fix it. The export adaptation reuses the source weights and restores parity. The nine-ROI FP16 validation reached minimum cosine 0.999912 against PyTorch FP32.

## trtexec

```bash
tools/deploy/trtexec --help
tools/deploy/trtexec \
  --loadEngine=models/segmentation/rtmdet_ins_s_coco_v001/deployment/tensorrt/end2end.engine \
  --warmUp=500 --duration=10
tools/deploy/trtexec \
  --loadEngine=models/recognition/dinov2_b14_v001/deployment/model_fp16.engine \
  --warmUp=500 --duration=10
```

The wrapper mounts the project at `/workspace` and uses `--pull=never`, so inference cannot implicitly download a container. Default random-input trtexec timing verifies engine execution; use the fixture benchmark for model parity and application latency. [NVIDIA's container release notes](https://docs.nvidia.com/deeplearning/frameworks/container-release-notes/) document the bundled versions; [TensorRT installation documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-pip.html) describes native Python packages.
