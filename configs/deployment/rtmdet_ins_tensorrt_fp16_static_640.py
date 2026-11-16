_base_ = [
    '../../third_party/mmdeploy/configs/mmdet/instance-seg/instance-seg_rtmdet-ins_tensorrt_static-640x640.py'
]

codebase_config = dict(post_processing=dict(pre_top_k=1000, iou_threshold=0.6, export_postprocess_mask=True))
backend_config = dict(common_config=dict(fp16_mode=True, max_workspace_size=4 << 30))
