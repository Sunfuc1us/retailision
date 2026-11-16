_base_ = [
    '../../third_party/mmdeploy/configs/mmdet/instance-seg/instance-seg_rtmdet-ins_onnxruntime_static-640x640.py'
]

# Match the source RTMDet model's nms_pre=1000 and stay within TensorRT TopK limits.
# Instance mask postprocessing remains enabled by the inherited config.
codebase_config = dict(post_processing=dict(pre_top_k=1000, iou_threshold=0.6, export_postprocess_mask=True))
