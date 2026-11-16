import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

def polygon_mask(polygons: list[list[float]], width: int, height: int) -> np.ndarray:
    rle = coco_mask.merge(coco_mask.frPyObjects(polygons, height, width))
    return coco_mask.decode(rle).astype(bool)

def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return inclusive-exclusive [left, top, right, bottom] bounds of a non-empty 2D mask."""
    if mask.ndim != 2:
        raise ValueError('invalid or empty instance mask')
    ys = np.flatnonzero(mask.any(axis=1))
    xs = np.flatnonzero(mask.any(axis=0))
    if not len(xs) or not len(ys):
        raise ValueError('invalid or empty instance mask')
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

def render_mask_crop(image: Image.Image, mask: np.ndarray, box: tuple[int, int, int, int], *, apply_mask: bool = True, fill: int = 127) -> Image.Image:
    left, top, right, bottom = box
    pixels = np.array(image.crop(box).convert('RGB'))
    if apply_mask:
        pixels[~mask[top:bottom, left:right]] = fill
    return Image.fromarray(pixels)

def mask_crop(image: Image.Image, mask: np.ndarray, padding: float = .05) -> Image.Image:
    if mask.shape != (image.height, image.width):
        raise ValueError('invalid or empty instance mask')
    x1, y1, x2, y2 = mask_bbox(mask)
    pad = int(max(x2 - x1, y2 - y1) * padding)
    box = (max(0, x1 - pad), max(0, y1 - pad), min(image.width, x2 + pad), min(image.height, y2 + pad))
    return render_mask_crop(image, mask, box)
