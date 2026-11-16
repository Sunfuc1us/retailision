import json
import subprocess
import sys
from pathlib import Path

import pytest

from retailvision.artifacts.config import preflight
from retailvision.paths import PROJECT_ROOT


@pytest.fixture
def config(tmp_path):
    for name in ('seg.py', 'seg.pth', 'rec.pth', 'hubconf.py', 'embeddings.npy', 'skus.json', 'catalog.json'):
        (tmp_path / name).touch()
    policy = dict(gallery_version='gallery-v1', model_version='model-v1')
    (tmp_path / 'policy.json').write_text(json.dumps(policy))
    settings = dict(
        segmentation=dict(config=str(tmp_path / 'seg.py'), checkpoint=str(tmp_path / 'seg.pth')),
        recognition=dict(source=str(tmp_path), checkpoint=str(tmp_path / 'rec.pth')),
        gallery=str(tmp_path), catalog=str(tmp_path / 'catalog.json'),
        calibration=str(tmp_path / 'policy.json'), results=str(tmp_path / 'new-results'), **policy)
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(settings))
    return path, settings


@pytest.mark.parametrize('failure', ['checkpoint', 'empty', 'version', 'gallery', 'projection'])
def test_preflight_rejects_invalid_inputs(config, failure):
    path, settings = config
    if failure == 'checkpoint':
        Path(settings['segmentation']['checkpoint']).unlink()
    elif failure == 'empty':
        settings['recognition']['source'] = ''
    elif failure == 'version':
        settings['model_version'] = 'wrong'
    elif failure == 'gallery':
        (path.parent / 'embeddings.npy').unlink()
    else:
        settings['recognition']['projection'] = str(path.parent / 'absent.pth')
    path.write_text(json.dumps(settings))
    with pytest.raises(ValueError):
        preflight(path)
    assert not Path(settings['results']).exists()


def test_deployment_checks_selected_artifacts_only(config):
    path, settings = config
    Path(settings['segmentation']['checkpoint']).unlink()
    artifacts = dict(seg_deploy_config=str(path.parent / 'seg.py'),
                     seg_model=str(path.parent / 'seg.onnx'), rec_model=str(path.parent / 'rec.onnx'))
    settings['deployment'] = dict(onnxruntime=artifacts)
    path.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match='seg_model'):
        preflight(path, 'onnxruntime')
    Path(artifacts['seg_model']).touch()
    Path(artifacts['rec_model']).touch()
    assert preflight(path, 'onnxruntime')['deployment']['onnxruntime'] == artifacts


def test_factory_preflights_before_model_construction(config, monkeypatch):
    from retailvision.inference import factory
    path, settings = config
    settings['model_version'] = 'wrong'
    path.write_text(json.dumps(settings))

    def unexpected_model(*args, **kwargs):
        raise AssertionError('model constructed before preflight')

    monkeypatch.setattr(factory, 'RTMDet', unexpected_model)
    with pytest.raises(ValueError, match='model_version mismatch'):
        factory.load_pipeline(str(path))


def test_dry_run_needs_no_model_libraries_or_output(config, tmp_path):
    path, settings = config
    # Fail on any attempted heavy-library import, including indirect imports.
    script = '''
import runpy, sys
class NoModels:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'numpy', 'PIL', 'faiss', 'mmdet'):
            raise AssertionError('unexpected import: ' + fullname)
sys.meta_path.insert(0, NoModels())
sys.path.insert(0, sys.argv[1])
sys.argv = [sys.argv[1] + '/predict.py', '--config', sys.argv[2], '--dry-run']
runpy.run_path(sys.argv[0], run_name='__main__')
'''
    result = subprocess.run([sys.executable, '-c', script, str(PROJECT_ROOT / 'tools'), str(path)],
                            cwd=tmp_path, text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)['config'] == settings
    assert not Path(settings['results']).exists()


def test_preflight_rejects_gallery_from_other_projection(config):
    path,settings=config
    head=path.parent/'head.pth';head.touch()
    settings['recognition']['projection']=str(head)
    path.write_text(json.dumps(settings))
    (path.parent/'manifest.json').write_text(json.dumps(dict(model=dict(checkpoint=settings['recognition']['checkpoint'],projection=None))))
    with pytest.raises(ValueError,match='projection mismatch'):
        preflight(path)


def test_preflight_requires_recalibration_after_gallery_revision(config):
    path,settings=config
    (path.parent/'manifest.json').write_text(json.dumps(dict(gallery_version='new-revision')))
    with pytest.raises(ValueError,match='gallery version mismatch'):
        preflight(path)
