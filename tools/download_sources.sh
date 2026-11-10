#!/usr/bin/env bash
set -euo pipefail
# Idempotency is intentionally explicit: never replace existing source or weights.
git clone https://github.com/open-mmlab/mmdetection.git third_party/mmdetection
git -C third_party/mmdetection checkout 44ebd17b145c2372c4b700bfb9cb20dbd28ab64a
git clone https://github.com/facebookresearch/dinov2.git third_party/dinov2
git -C third_party/dinov2 checkout 7764ea0f912e53c92e82eb78a2a1631e92725fc8
mkdir -p models/segmentation/rtmdet_ins_s_coco_v001 models/recognition/dinov2_b14_v001
test ! -e models/segmentation/rtmdet_ins_s_coco_v001/model.pth
curl --fail -L https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_s_8xb32-300e_coco/rtmdet-ins_s_8xb32-300e_coco_20221121_212604-fdc5d7ec.pth -o models/segmentation/rtmdet_ins_s_coco_v001/model.pth
test ! -e models/recognition/dinov2_b14_v001/model.pth
curl --fail -L https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth -o models/recognition/dinov2_b14_v001/model.pth
