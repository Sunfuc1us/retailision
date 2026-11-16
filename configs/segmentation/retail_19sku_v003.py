"""Supervised v0.3 training from local COCO weights, with image holdout validation."""
_base_ = './rtmdet_ins_s.py'

load_from = 'models/segmentation/rtmdet_ins_s_coco_v001/model.pth'
train_dataloader = dict(
    dataset=dict(pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=False),
        dict(type='RandomResize', scale=(960, 960), ratio_range=(0.8, 1.2), keep_ratio=True),
        dict(type='RandomCrop', crop_size=(960, 960), recompute_bbox=True, allow_negative_crop=False),
        dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
        dict(type='YOLOXHSVRandomAug'),
        dict(type='Pad', size=(960, 960), pad_val=dict(img=(114, 114, 114))),
        dict(type='PackDetInputs'),
    ]),
)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(960, 960), keep_ratio=True),
    dict(type='Pad', size=(960, 960), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
train_cfg = dict(_delete_=True, type='EpochBasedTrainLoop', max_epochs=50, val_interval=5)
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    loss_scale=dict(init_scale=128.0),
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.0001),
    clip_grad=dict(max_norm=10, norm_type=2),
)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, begin=0, end=25, by_epoch=False),
    dict(type='CosineAnnealingLR', eta_min=0.00005, begin=0, end=50, T_max=50, by_epoch=True, convert_to_iter_based=True),
]
custom_hooks = []
default_hooks = dict(
    checkpoint=dict(interval=5, max_keep_ckpts=2, save_best='coco/segm_mAP'),
    logger=dict(interval=10),
)
