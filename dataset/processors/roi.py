import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

def polygon_mask(polygons: list[list[float]], width: int, height: int) -> np.ndarray:
    rle = coco_mask.merge(coco_mask.frPyObjects(polygons, height, width))
    return coco_mask.decode(rle).astype(bool)

def mask_crop(image: Image.Image, mask: np.ndarray, padding: float = .05) -> Image.Image:
    if mask.shape != (image.height, image.width) or not mask.any():
        raise ValueError('invalid or empty instance mask')
    ys, xs = np.where(mask)
    pad = int(max(xs.max()-xs.min()+1, ys.max()-ys.min()+1)*padding)
    box = (max(0,int(xs.min())-pad),max(0,int(ys.min())-pad),min(image.width,int(xs.max())+1+pad),min(image.height,int(ys.max())+1+pad))
    pixels = np.array(image.convert('RGB'))
    pixels[~mask] = 127
    return Image.fromarray(pixels).crop(box)
