#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Run once to create a new deployment environment. Existing environments are not replaced.
conda create -y -n retailvision-deploy --clone retailvision-seg
conda run -n retailvision-deploy python -m pip uninstall -y tensorboard
conda run -n retailvision-deploy python -m pip install \
  onnx==1.16.2 protobuf==3.20.2 coloredlogs==15.0.1 flatbuffers==25.12.19 \
  aenum==3.1.17 grpcio==1.85.1 multiprocess==0.70.19 prettytable==3.18.0 terminaltables==3.1.10 \
  nvidia-cuda-runtime-cu12==12.4.127 nvidia-curand-cu12==10.3.5.147 \
  nvidia-cufft-cu12==11.2.1.3 nvidia-nvjitlink-cu12==12.4.127
conda run -n retailvision-deploy python -m pip install --no-deps \
  --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
  onnxruntime-gpu==1.18.0
conda run -n retailvision-deploy python -m pip install --extra-index-url https://pypi.nvidia.com \
  tensorrt-cu12==10.0.1 tensorrt-cu12-bindings==10.0.1 tensorrt-cu12-libs==10.0.1
DEPLOY_PREFIX="$(conda run -n retailvision-deploy python -c 'import sys; print(sys.prefix)')"
mkdir -p "${DEPLOY_PREFIX}/etc/conda/activate.d"
mkdir -p "${DEPLOY_PREFIX}/etc/conda/deactivate.d"
cp environments/deploy-cuda-libs.sh "${DEPLOY_PREFIX}/etc/conda/activate.d/retailvision-cuda-libs.sh"
cp environments/deploy-cuda-libs-deactivate.sh "${DEPLOY_PREFIX}/etc/conda/deactivate.d/retailvision-cuda-libs.sh"
if [ ! -d third_party/mmdeploy ]; then
  git clone --depth 1 --branch v1.3.1 https://github.com/open-mmlab/mmdeploy.git third_party/mmdeploy
fi
test "$(git -C third_party/mmdeploy describe --tags --exact-match)" = v1.3.1
conda run -n retailvision-deploy python -m pip check
conda run -n retailvision-deploy python -c 'import torch; assert torch.__version__ == "2.1.2+cu121"; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())'
# trtexec is provided by tools/deploy/trtexec through this matching container.
docker pull nvcr.io/nvidia/tensorrt:24.05-py3
ln -s "${PWD}/tools/deploy/trtexec" "${DEPLOY_PREFIX}/bin/trtexec"
