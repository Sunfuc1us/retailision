# Deployment

## Validated environment

The deployment environment is isolated from `retailvision-seg` and `retailvision-reid`. Their PyTorch versions and original checkpoints are preserved. No product training is started by this deployment stack.

| Component | Actual version / configuration |
|---|---|
| GPU | NVIDIA RTX 4090 24 GB, compute capability 8.9 |
| NVIDIA driver | 580.178.04 |
| Python | 3.10 |
| PyTorch / torchvision | 2.1.2+cu121 / 0.16.2+cu121 |
| PyTorch CUDA / cuDNN | CUDA 12.1 / cuDNN 8.9.2 |
| MMCV / MMEngine / MMDetection | 2.1.0 / 0.10.7 / 3.3.0 |
| MMDeploy source | v1.3.1, bc75c9d6c8940aa03d0e1e5b5962bd930478ba77 |
| ONNX | 1.16.2 |
| ONNX Runtime GPU | 1.18.0, official CUDA 12 wheel |
| TensorRT Python / native libraries | 10.0.1 |
| Additional standalone CUDA runtime | 12.4.1 component set: cudart 12.4.127, cuFFT 11.2.1.3, cuRAND 10.3.5.147, nvJitLink 12.4.127 |
| trtexec | NVIDIA TensorRT 24.05 container, TensorRT 10.0.1.6 / CUDA 12.4.1 |

The driver's reported maximum CUDA version is a driver capability, not an installed toolkit version. The standalone runtime components are pinned to the [CUDA 12.4 Update 1 component table](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-toolkit-release-notes/index.html), matching the TensorRT 24.05 CUDA release. The selected ORT build uses CUDA 12 and cuDNN 8, matching the existing cuDNN major version. CUDA 12 minor-version compatibility is documented in the [ORT CUDA provider matrix](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html). NVIDIA documents the matching CLI image in its [container release notes](https://docs.nvidia.com/deeplearning/frameworks/container-release-notes/).

## Reproduce the environment

For a machine with the existing `retailvision-seg` environment and project sources/weights:

```bash
bash environments/install-deploy.sh
conda activate retailvision-deploy
python -m pip check
trtexec --help
```

The installer creates a new `retailvision-deploy` clone, pins deployment dependencies, installs the CUDA 12 ORT wheel from its official feed, and exposes CUDA library paths during activation and restores the previous path on deactivation. The deployment-to-training environment switch was verified. It does not replace an existing deployment environment. Full installed snapshots are `environments/deploy-freeze.txt` and `environments/deploy-conda-explicit.txt`.

MMDeploy v1.3.1 requires protobuf ≤3.20.2. The cloned TensorBoard 2.21 dependency requires a newer protobuf; TensorBoard is removed from the deployment environment to resolve that concrete conflict. TensorBoard in the two training environments is preserved. `pip check` passes in the deployment environment.

The CUDA 12 ORT wheel must be installed from its versioned feed:

```bash
python -m pip install --no-deps \
  --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
  onnxruntime-gpu==1.18.0
```

The same-version PyPI wheel tested here expected CUDA 11. Do not upgrade PyTorch to resolve that wheel mismatch. TensorRT pip packages provide the native Python runtime; the version-matched `trtexec` command uses Docker with NVIDIA GPU support.

## Export, build and benchmark

See [ONNX_EXPORT.md](ONNX_EXPORT.md) for both independent model exporters and [TENSORRT.md](TENSORRT.md) for engine construction, loading and compatibility fixes.

```bash
conda activate retailvision-deploy
python tools/deploy/benchmark.py --warmup 50 --repeats 200
```

This uses the same three real dataset images and nine ROI crops for PyTorch FP32, PyTorch FP16, ORT CUDA, and TensorRT FP16. Measured latency, P50/P95, model load time, memory, utilization, model size and output differences are in [INFERENCE_BENCHMARK.md](INFERENCE_BENCHMARK.md). File hashes and engine build metadata accompany each model version.

## Backend selection and Workbench

The project Web entry point is `./start-web` (default port 8018), which builds the frontend and starts the API. With no explicit environment/config override it selects the real three-image trained Checkpoint. Use `./start-web --model coco_pretrained` for the exported COCO model, or `./start-web --config PATH` for another complete inference bundle. Startup prints the exact config and Checkpoint paths. Models load on the first prediction.

Workbench's segmentation-weight selector obtains model options from `GET /api/inference/models`. `configs/inference/model_profiles.json` pairs each weight with its compatible architecture, recognition weights, Gallery and policy. The three-image model currently exposes PyTorch only. Selecting another model updates the SKU list and permitted Backends, clears stale results, and causes a real model reload on the next prediction. API clients can pass `model_id=retailtest_smoke_v001|coco_pretrained` alongside `backend`; omitting it preserves the startup configuration default. Results include the actual selected `model_id` and `model_name`. `GET /api/inference/backends?model_id=...` returns availability for that model.


`configs/inference/deployment.fixture.json` is a complete three-backend fixture configuration using the existing real pretrained checkpoints, the existing self-reference FAISS CPU Gallery, and explicit engineering-only thresholds from `configs/deployment/fixture_calibration.json`. These thresholds are not a retail validation calibration. Original Gallery files and the original calibration remain unchanged.

```bash
conda activate retailvision-deploy
RETAILVISION_CONFIG=configs/inference/deployment.fixture.json \
  uvicorn backend.api.app:app --host 127.0.0.1 --port 8020
```

Open `http://127.0.0.1:8020`. Workbench obtains model and per-model Backend choices from `GET /api/inference/models`; `GET /api/inference/backends?model_id=...` exposes the same availability check separately. These checks inspect configured artifacts, runtime packages and CUDA availability. It displays **PyTorch**, **ONNX Runtime GPU**, and **TensorRT** for the validated fixture configuration. Results show the backend and total inference time.

The existing prediction endpoint accepts the additional form field `backend=pytorch|onnxruntime|tensorrt`. It shares image inputs, target product, detection threshold, SKU Gallery, FAISS, rejection policy, masks and result schema across backends. PyTorch remains the default; selecting it explicitly provides the reference/fallback path. Backend failures remain visible rather than silently reporting another backend's output.

For another model version, configure the `deployment` section with `seg_deploy_config`, `seg_model`, and `rec_model` paths as in the fixture JSON. Keep that version's Gallery and calibration consistent with its embeddings.

PyTorch FP16 inference uses `precision: "fp16"` in the segmentation and recognition sections. RTMDet uses CUDA autocast for network computation and promotes head outputs to FP32 for MMCV's NMS. DINOv2 uses inference-time autocast. No training `AmpOptimWrapper` is involved. ORT runs the validated FP32 ONNX graphs on CUDA; TensorRT uses mixed FP16 engines with FP32 external tensor contracts.

## Artifact layout and scope

Existing checkpoint versions retain their names (`rtmdet_ins_s_coco_v001`, `dinov2_b14_v001`). Their deployment subdirectories contain exported graphs, engines and build manifests; root-level ONNX/engine aliases provide the uniform loading layout. DINOv2's `backbone.pth` aliases the original `model.pth`. No `head.pth` is fabricated when no learned projection head exists.

All model loading and inference are offline. Exports, engine builds and verification use real local weights and CUDA. Fixture results establish engineering integration; they do not establish retail SKU recognition accuracy or generalization.
