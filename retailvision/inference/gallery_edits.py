"""Explicit gallery edits preserve a snapshot and invalidate calibration."""
import json
import shutil
from pathlib import Path
from uuid import uuid4

import numpy as np

from retailvision.inference.gallery import Gallery


def snapshot(path: Path):
    destination = path.parent / 'gallery_history' / uuid4().hex
    shutil.copytree(path, destination)


def invalidate(path: Path):
    manifest_path = path / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    revision = uuid4().hex
    manifest.update(gallery_version=revision, calibration_required=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return revision


def add_reference(path: Path, pipe, image, sku_id: str):
    embedding = np.asarray(pipe.recognition.extract_embedding(image), dtype='float32').reshape(1, -1)
    snapshot(path)
    pipe.gallery.add_sku(sku_id, embedding)
    reference = path / 'references' / sku_id / f'{uuid4().hex[:8]}.png'
    reference.parent.mkdir(parents=True, exist_ok=True)
    image.save(reference)
    pipe.gallery.save(str(path), overwrite=True)
    revision = invalidate(path)
    pipe.policy = None
    return dict(gallery_version=revision, calibration_required=True, sku_id=sku_id,
                reference_path=str(reference.relative_to(path)), **pipe.gallery.summary())


def remove_sku(path: Path, sku_id: str, pipe=None):
    if not (path / 'embeddings.npy').is_file():
        raise FileNotFoundError('gallery not found')
    gallery = Gallery.load(str(path))
    if sku_id not in gallery.ids:
        raise KeyError('SKU not in gallery')
    gallery.remove_sku(sku_id)
    if not gallery.ids:
        raise ValueError('cannot remove last SKU from gallery')
    snapshot(path)
    gallery.save(str(path), overwrite=True)
    revision = invalidate(path)
    if pipe is not None:
        pipe.gallery, pipe.policy = gallery, None
    return dict(deleted=sku_id, gallery_version=revision, calibration_required=True, **gallery.summary())
