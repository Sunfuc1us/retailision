_base_ = '../../third_party/mmdetection/configs/rtmdet/rtmdet-ins_s_8xb32-300e_coco.py'
model = dict(backbone=dict(init_cfg=None), bbox_head=dict(num_classes=1))
load_from = 'models/segmentation/rtmdet_ins_s_coco_v001/model.pth'
train_cfg = dict(max_epochs=30, val_interval=1)
optim_wrapper = dict(type='AmpOptimWrapper',optimizer=dict(lr=0.001))
param_scheduler = [dict(type='LinearLR',start_factor=1e-5,by_epoch=False,begin=0,end=100),dict(type='CosineAnnealingLR',eta_min=0.00005,begin=0,end=30,T_max=30,by_epoch=True,convert_to_iter_based=True)]
custom_hooks = [dict(type='EMAHook',ema_type='ExpMomentumEMA',momentum=0.0002,update_buffers=True,priority=49)]
vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(vis_backends=vis_backends)
default_hooks = dict(checkpoint=dict(interval=1,save_best='coco/segm_mAP'))
