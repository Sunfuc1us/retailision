"""v0.2: all 15 annotated images for training; no independent validation."""
_base_ = './rtmdet_ins_s.py'

train_dataloader = dict(
    _delete_=True,
    batch_size=3,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root='datasets/retail_6sku_v002/segmentation/',
        ann_file='annotations/all.json',
        data_prefix=dict(img='images/'),
        metainfo=dict(classes=('retail_product',)),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=False),
            dict(type='RandomResize', scale=(960, 960), ratio_range=(0.8, 1.2), keep_ratio=True),
            dict(type='RandomCrop', crop_size=(960, 960), recompute_bbox=True, allow_negative_crop=False),
            dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
            dict(type='YOLOXHSVRandomAug'),
            dict(type='Pad', size=(960, 960), pad_val=dict(img=(114, 114, 114))),
            dict(type='PackDetInputs'),
        ],
    ),
)
train_cfg = dict(_delete_=True, type='IterBasedTrainLoop', max_iters=500, val_interval=501)
val_cfg = None
val_dataloader = None
val_evaluator = None
test_cfg = None
test_evaluator = None
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(960, 960), keep_ratio=True),
    dict(type='Pad', size=(960, 960), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]
test_dataloader = None
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    loss_scale=dict(init_scale=128.0),
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.0001),
    clip_grad=dict(max_norm=10, norm_type=2),
)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, begin=0, end=25, by_epoch=False),
    dict(type='CosineAnnealingLR', eta_min=0.00005, begin=25, end=500, T_max=475, by_epoch=False),
]
custom_hooks = []
default_hooks = dict(
    checkpoint=dict(_delete_=True, type='CheckpointHook', by_epoch=False, interval=250, max_keep_ckpts=2, save_last=True),
    logger=dict(interval=10, log_metric_by_epoch=False),
)
log_processor = dict(type='LogProcessor', window_size=1, by_epoch=False)
randomness = dict(seed=42)
