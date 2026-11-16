"""Read local inference paths and version contracts without loading model libraries."""
import json
from pathlib import Path

from retailvision.paths import project_path


def preflight(config: str | Path, backend: str = 'pytorch') -> dict:
    if backend not in ('pytorch', 'onnxruntime', 'tensorrt'):
        raise ValueError(f'unsupported inference backend: {backend}')
    settings = json.loads(project_path(config).read_text())

    def path_field(section: dict, key: str, *, directory: bool = False, output: bool = False):
        value = section.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{key}: non-empty local path required')
        path = project_path(value).resolve()
        if output:
            if path.exists() and not path.is_dir():
                raise ValueError(f'{key}: output must be a directory: {path}')
        elif not (path.is_dir() if directory else path.is_file()):
            raise ValueError(f'{key}: missing {"directory" if directory else "file"}: {path}')
        section[key] = str(path)
        return path

    seg, rec = settings['segmentation'], settings['recognition']
    path_field(seg, 'config')
    if backend == 'pytorch':
        path_field(seg, 'checkpoint')
        path_field(rec, 'checkpoint')
        source = path_field(rec, 'source', directory=True)
        if not (source / 'hubconf.py').is_file():
            raise ValueError(f'source: local DINOv2 hubconf.py missing: {source}')
        if rec.get('projection') is not None:
            path_field(rec, 'projection')
        for section in (seg, rec):
            if section.get('precision', 'fp32') not in ('fp32', 'fp16'):
                raise ValueError(f'unsupported precision: {section["precision"]}')
    else:
        artifacts = settings.get('deployment', {}).get(backend)
        if not artifacts:
            raise ValueError(f'{backend} artifacts are not configured')
        for key in ('seg_deploy_config', 'seg_model', 'rec_model'):
            path_field(artifacts, key)
    gallery = path_field(settings, 'gallery', directory=True)
    for name in ('embeddings.npy', 'skus.json'):
        if not (gallery / name).is_file():
            raise ValueError(f'gallery: missing file: {gallery / name}')
    manifest_path=gallery/'manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if manifest.get('gallery_version') and manifest['gallery_version']!=settings.get('gallery_version'):
        raise ValueError('gallery version mismatch; recalibrate the updated gallery')
    model=manifest.get('model',{})
    if rec.get('projection') and not model:
        raise ValueError('projection requires a gallery manifest identifying its feature model')
    if model:
        for key in ('checkpoint','projection'):
            actual=model.get(key);expected=rec.get(key)
            if (str(project_path(actual).resolve()) if actual else None)!=(str(project_path(expected).resolve()) if expected else None):
                raise ValueError(f'gallery recognition {key} mismatch; rebuild gallery with the selected model')
    path_field(settings, 'catalog')
    calibration = path_field(settings, 'calibration')
    path_field(settings, 'results', output=True)
    policy = json.loads(calibration.read_text())
    for key in ('gallery_version', 'model_version'):
        if not settings.get(key) or policy.get(key) != settings[key]:
            raise ValueError(f'calibration {key} mismatch')
    return settings
