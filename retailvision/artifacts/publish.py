"""Publish a completed joint experiment as one Workbench model choice."""
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from retailvision.artifacts.config import preflight
from retailvision.artifacts.registry import Registry
from retailvision.paths import project_path


def read(path):
    return json.loads(Path(path).read_text())


def publish(experiment: str, version: str, policy: str, name: str | None = None, dataset: str | None = None) -> dict:
    import torch
    from mmengine.config import Config

    if not version or Path(version).name != version or version in ('.', '..'):
        raise ValueError('invalid release version')
    run = project_path(experiment)
    record = read(run / 'experiment.json')
    stages = record.get('stages', {})
    if record['status'] != 'completed' or set(stages) != {'segmentation', 'recognition'}:
        raise ValueError('publish requires a completed joint experiment')
    if any(s['status'] != 'completed' or s.get('validation_status') != 'completed' for s in stages.values()):
        raise ValueError('both stages must have completed validation')
    if any(s['dataset_version'] != record['dataset_version'] for s in stages.values()):
        raise ValueError('stage dataset versions do not match')
    seg, rec = stages['segmentation'], stages['recognition']
    dataset = dataset or record.get('dataset')
    if not dataset:raise ValueError('legacy experiment requires an explicit dataset path')
    data_root = project_path(dataset)
    if read(data_root / 'manifest.json')['dataset_version'] != record['dataset_version']:
        raise ValueError('publication dataset version does not match experiment')
    if not seg.get('best_checkpoint') or not rec.get('projection_checkpoint'):
        raise ValueError('validated best checkpoints are required')
    seg_checkpoint = project_path(seg['best_checkpoint'])
    projection = project_path(rec['projection_checkpoint'])
    gallery_source = project_path(record['gallery'])
    gallery_meta = read(gallery_source / 'manifest.json')
    if gallery_meta['dataset_version'] != record['dataset_version'] or gallery_meta['split'] != 'train':
        raise ValueError('gallery must use this experiment training split')
    if any(r['split'] != 'train' for r in gallery_meta['references']):
        raise ValueError('validation references cannot be published')
    recognition_meta = read(data_root / 'recognition/metadata.json')
    training_refs = {(r['image_id'], r['annotation_id']) for r in recognition_meta['items'] if r['split'] == 'train'}
    if any((r['image_id'], r['annotation_id']) not in training_refs for r in gallery_meta['references']):
        raise ValueError('gallery contains references outside the training split')
    source_projection = project_path(gallery_meta['model']['projection'])
    if hashlib.sha256(source_projection.read_bytes()).digest() != hashlib.sha256(projection.read_bytes()).digest():
        raise ValueError('gallery projection does not match selected checkpoint')
    recognition = rec['hyperparameters']
    if project_path(recognition['checkpoint']).resolve() != project_path(gallery_meta['model']['checkpoint']).resolve():
        raise ValueError('gallery backbone does not match recognition training')
    gallery_version = f'gallery_{version}'
    rejection = read(project_path(policy))
    for key, expected in dict(dataset_version=record['dataset_version'], model_version=version, gallery_version=gallery_version).items():
        if rejection.get(key) != expected:
            raise ValueError(f'policy {key} does not match release')
    for key in ('similarity', 'margin'):
        if not isinstance(rejection.get(key), (float, int)):
            raise ValueError(f'policy {key} must be numeric')

    profiles_path = project_path('configs/inference/model_profiles.json')
    profiles = read(profiles_path)
    model_dirs = [project_path(f'models/{task}/{version}') for task in ('segmentation', 'recognition')]
    gallery = project_path(f'datasets/{gallery_version}')
    config_path = project_path(f'configs/inference/{version}.json')
    policy_path = project_path(f'configs/inference/{version}_policy.json')
    outputs = [*model_dirs, gallery, config_path, policy_path]
    if any(p.exists() for p in outputs) or any(p['id'] == version for p in profiles):
        raise FileExistsError('release version already exists')

    # Resolve the best checkpoint's metrics rather than reporting the last epoch.
    checkpoint = torch.load(seg_checkpoint, map_location='cpu', weights_only=False)
    cfg = Config.fromfile(str(seg_checkpoint.parent / 'resolved_config.py'))
    cfg.test_dataloader = cfg.val_dataloader.copy()
    epoch = checkpoint['meta']['epoch']
    scalar_file = next(seg_checkpoint.parent.glob('*/vis_data/scalars.json'))
    rows = [json.loads(line) for line in scalar_file.read_text().splitlines()]
    best = next(r for r in rows if 'coco/segm_mAP' in r and r['step'] == epoch)
    metrics = {k: v for k, v in best.items() if k.startswith('coco/')}
    common = dict(training_dataset=record['dataset_version'], experiment_id=record['experiment_id'],
                  validation_scope=record.get('validation_scope', 'independent_capture_groups'),
                  independent_validation=record.get('independent_validation', True),
                  fixture=record.get('fixture', False), calibration_required=not rejection.get('calibrated', False),
                  production_eligible=rejection.get('production_eligible', False), license='See initial model manifest')
    registry = Registry(str(project_path('models')))
    try:
        with tempfile.TemporaryDirectory() as temporary:
            # Keep the original training config in the experiment; publish the loadable config.
            inference_config = Path(temporary) / 'inference_config.py'
            cfg.dump(str(inference_config))
            registry.register('segmentation', version, str(seg_checkpoint), str(inference_config),
                              dict(common, model_name=name or version, architecture='RTMDet-Ins-s',
                                   initial_checkpoint=cfg.get('load_from'),
                                   training_config_source=str(seg_checkpoint.parent / 'resolved_config.py'),
                                   evaluation_metrics=metrics, selected_epoch=epoch))
            recognition_config = Path(temporary) / 'recognition.py'
            recognition_config.write_text('config = ' + repr(recognition) + '\n')
            registry.register('recognition', version, str(projection), str(recognition_config),
                              dict(common, model_name=name or version, architecture='Frozen DINOv2 ViT-B/14 + projection',
                                   artifact_type='projection_head', base_checkpoint=recognition['checkpoint'],
                                   evaluation_metrics=rec['validation_metrics']))
        shutil.copytree(gallery_source, gallery)
        shutil.copy2(data_root / 'recognition/skus.json', gallery / 'catalog.json')
        gallery_meta.update(gallery_version=gallery_version)
        gallery_meta['model']['projection'] = str(model_dirs[1] / 'model.pth')
        (gallery / 'manifest.json').write_text(json.dumps(gallery_meta, indent=2))
        settings = dict(experiment=record['experiment_id'], fixture=record.get('fixture', False),
                        segmentation=dict(config=str(model_dirs[0] / 'inference_config.py'), checkpoint=str(model_dirs[0] / 'model.pth')),
                        recognition=dict(source=recognition['source'], checkpoint=recognition['checkpoint'], projection=str(model_dirs[1] / 'model.pth')),
                        gallery=str(gallery), catalog=str(gallery / 'catalog.json'), calibration=str(policy_path),
                        gallery_version=gallery_version, model_version=version, results='results', roi=dict(clip_mask_to_bbox=True))
        policy_path.write_text(json.dumps(rejection, indent=2))
        config_path.write_text(json.dumps(settings, indent=2))
        preflight(config_path)
        profiles.append(dict(id=version, name=name or version, config=str(config_path)))
        temporary = profiles_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(profiles, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(profiles_path)
    except BaseException:
        for path in outputs:
            if path.is_dir():shutil.rmtree(path)
            elif path.exists():path.unlink()
        raise
    return dict(model_id=version, config=str(config_path), gallery=str(gallery), validation_metrics=dict(segmentation=metrics, recognition=rec['validation_metrics']))
