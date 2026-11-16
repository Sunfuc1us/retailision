import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from retailvision import paths
from retailvision.artifacts.config import preflight
from retailvision.artifacts.publish import publish
from retailvision.data.prepare import prepare
from retailvision.inference.gallery import Gallery


@pytest.fixture
def release(data, tmp_path, monkeypatch):
    """Synthetic files exercise publication and rollback, not model accuracy."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(paths, 'PROJECT_ROOT', tmp_path)
    manifest = prepare(**data)
    root = Path(data['output'])
    metadata = json.loads((root / 'recognition/metadata.json').read_text())
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'hubconf.py').touch()
    torch.save({}, source / 'backbone.pth')
    torch.save({'head': {}}, source / 'head.pth')
    torch.save({'meta': {'epoch': 1}}, source / 'best.pth')
    (source / 'resolved_config.py').write_text('val_dataloader = dict(dataset=dict(pipeline=[]))\n')
    log = source / 'logs/vis_data'
    log.mkdir(parents=True)
    (log / 'scalars.json').write_text(json.dumps({'step': 1, 'coco/segm_mAP': 0.25}) + '\n')
    refs = [r for r in metadata['items'] if r['split'] == 'train']
    gallery = Gallery()
    gallery.build_index(np.ones((len(refs), 2), dtype='float32'), [r['sku_id'] for r in refs])
    gallery.save(str(source / 'gallery'))
    (source / 'gallery/manifest.json').write_text(json.dumps(dict(dataset_version=manifest['dataset_version'],
        split='train', references=refs, model=dict(checkpoint=str(source / 'backbone.pth'), projection=str(source / 'head.pth')))))
    run = tmp_path / 'run'
    run.mkdir()
    common = dict(status='completed', validation_status='completed', dataset_version=manifest['dataset_version'])
    record = dict(status='completed', experiment_id='fixture', fixture=True, dataset=str(root),
        dataset_version=manifest['dataset_version'], gallery=str(source / 'gallery'), stages=dict(
            segmentation=dict(common, best_checkpoint=str(source / 'best.pth')),
            recognition=dict(common, projection_checkpoint=str(source / 'head.pth'), validation_metrics=dict(best_top1=0.5),
                hyperparameters=dict(source=str(source), checkpoint=str(source / 'backbone.pth')))))
    (run / 'experiment.json').write_text(json.dumps(record))
    policy = tmp_path / 'policy.json'
    policy.write_text(json.dumps(dict(dataset_version=manifest['dataset_version'], gallery_version='gallery_fixture_release',
        model_version='fixture_release', similarity=0.8, margin=0.01, calibrated=False)))
    (tmp_path / 'configs/inference').mkdir(parents=True)
    (tmp_path / 'configs/inference/model_profiles.json').write_text('[]')
    return dict(experiment=str(run), version='fixture_release', policy=str(policy))


def test_publish_binds_weights_gallery_and_policy_without_overwriting(release):
    result = publish(**release)
    settings = preflight(result['config'])
    assert settings['recognition']['projection'].endswith('models/recognition/fixture_release/model.pth')
    assert result['validation_metrics']['segmentation']['coco/segm_mAP'] == 0.25
    assert json.loads(Path('configs/inference/model_profiles.json').read_text())[0]['id'] == 'fixture_release'
    files = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('models').rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError):publish(**release)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in files.items())


def test_failed_publication_leaves_no_partial_release(release):
    Path('source/hubconf.py').unlink()
    with pytest.raises(ValueError, match='hubconf'):publish(**release)
    assert not Path('models/segmentation/fixture_release').exists()
    assert not Path('models/recognition/fixture_release').exists()
    assert not Path('datasets/gallery_fixture_release').exists()
    assert not Path('configs/inference/fixture_release.json').exists()
    assert json.loads(Path('configs/inference/model_profiles.json').read_text()) == []


def test_publish_rejects_validation_references_before_writing(release):
    path = Path('source/gallery/manifest.json')
    metadata = json.loads(path.read_text())
    metadata['references'][0]['split'] = 'val'
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='validation references'):publish(**release)
    assert not Path('models').exists()


def test_publish_anchors_all_outputs_when_called_from_another_directory(release, tmp_path, monkeypatch):
    other = tmp_path / 'other'
    other.mkdir()
    monkeypatch.chdir(other)
    result = publish(**release)
    assert Path(result['config']).is_file()
    assert (tmp_path / 'models/segmentation/fixture_release/model.pth').is_file()
    assert not (other / 'models').exists()
