import json
from pathlib import Path

def train(config: str, dataset: str, experiment, resume: str | None) -> dict:
    import torch
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmdet.utils import register_all_modules
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    register_all_modules();cfg=Config.fromfile(config);root=Path(dataset).resolve()
    for split,loader in [('train','train_dataloader'),('val','val_dataloader'),('test','test_dataloader')]:
        if split=='test' and not (root/'annotations/test.json').is_file():
            cfg.test_dataloader=None;cfg.test_evaluator=None;cfg.test_cfg=None
            continue
        cfg[loader].batch_size=4 if split=='train' else 1
        cfg[loader].num_workers=2
        cfg[loader].dataset.update(data_root=str(root)+'/',ann_file=f'annotations/{split}.json',data_prefix=dict(img='images/'),metainfo=dict(classes=('retail_product',)))
        if split!='train':cfg[f'{split}_evaluator'].ann_file=str(root/f'annotations/{split}.json')
    cfg.work_dir=str(experiment.path);cfg.randomness=dict(seed=42)
    cfg.val_evaluator.outfile_prefix=str(experiment.path/'validation')
    if resume:cfg.resume=True;cfg.load_from=resume
    experiment.data['hyperparameters']=dict(batch_size=cfg.train_dataloader.batch_size,optimizer=cfg.optim_wrapper.to_dict(),scheduler=[x.to_dict() for x in cfg.param_scheduler],epochs=cfg.train_cfg.max_epochs);experiment.save()
    cfg.dump(str(experiment.path/'resolved_config.py'))
    runner=Runner.from_cfg(cfg);runner.train()
    checkpoints=sorted(experiment.path.glob('best*.pth'))
    scalars={k:float(v.current()) for k,v in runner.message_hub.log_scalars.items()}
    validation={k:v for k,v in scalars.items() if k.startswith('val/coco/')}
    predictions=experiment.path/'validation.bbox.json'
    if not validation and predictions.is_file() and json.loads(predictions.read_text())==[]:
        # MMDetection returns no metrics for empty detections; these are zero AP,
        # not a missing validation run. Reuse the project's empty-result evaluator.
        from retailvision.evaluation import segmentation
        validation=segmentation(str(root/'annotations/val.json'),str(predictions))
    marker=experiment.path/'last_checkpoint'
    return dict(best_checkpoint=str(checkpoints[-1]) if checkpoints else None,
        last_checkpoint=marker.read_text().strip() if marker.is_file() else None,
        training_metrics={k:v for k,v in scalars.items() if k.startswith('train/')},validation_metrics=validation,
        validation_performed=bool(validation),validation_status='completed' if validation else 'no_metrics',device='cuda:0')
