#!/usr/bin/env bash
set -euo pipefail
# Run from RetailVision root. Official MMCV ABI wheel determines this combination.
conda create -y -n retailvision-seg python=3.10 pip
conda env config vars set -n retailvision-seg PYTHONNOUSERSITE=1
conda run --no-capture-output -n retailvision-seg python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
conda run --no-capture-output -n retailvision-seg python -m pip install setuptools==80.9.0 'numpy<2' 'mmengine>=0.7.1,<1' 'opencv-python<4.12' pycocotools shapely fastapi uvicorn python-multipart faiss-cpu scikit-learn tensorboard pytest httpx pytorch-metric-learning
conda run --no-capture-output -n retailvision-seg python -m pip install https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/mmcv-2.1.0-cp310-cp310-manylinux1_x86_64.whl
conda run --no-capture-output -n retailvision-seg python -m pip install --no-build-isolation -e third_party/mmdetection -e .
