import json
from pathlib import Path
from PIL import Image, ImageFile
from shapely.geometry import Polygon
from retailvision.image_ops import polygon_mask
from retailvision.data.catalog import Catalog

def validate(coco: str | Path, images: str | Path, catalog: str | Path, sidecar: str | Path | None = None, *, require_capture_group: bool = True) -> dict:
    data = json.loads(Path(coco).read_text())
    master = Catalog(catalog).items
    side = json.loads(Path(sidecar).read_text()) if sidecar else {}
    errors: list[str] = []
    records = []
    ids: dict[str,set] = {}
    for kind in ('images','annotations','categories'):
        ids[kind] = set()
        for item in data.get(kind, []):
            if item['id'] in ids[kind]: errors.append(f'duplicate {kind} id {item["id"]}')
            ids[kind].add(item['id'])
    categories = {x['id']: x['name'] for x in data['categories']}
    image_map = {x['id']: x for x in data['images']}
    root = Path(images).resolve()
    for im in data['images']:
        p = (root / im['file_name']).resolve()
        if not p.is_relative_to(root):
            errors.append(f'unsafe image path {im["id"]}');continue
        try:
            with Image.open(p) as img:
                if img.size != (im['width'],im['height']): errors.append(f'image dimensions mismatch {im["id"]}')
                allow_truncated = ImageFile.LOAD_TRUNCATED_IMAGES
                try:
                    ImageFile.LOAD_TRUNCATED_IMAGES = False
                    for frame in range(getattr(img, 'n_frames', 1)):
                        img.seek(frame)
                        img.load()
                finally:
                    ImageFile.LOAD_TRUNCATED_IMAGES = allow_truncated
        except (OSError,ValueError): errors.append(f'missing or invalid image {im["id"]}')
        if require_capture_group and not im.get('capture_group'): errors.append(f'image {im["id"]}: capture_group required (batch/scene/physical sample)')
    fingerprints=set()
    for a in data['annotations']:
        try:
            im = image_map[a['image_id']]
            if categories.get(a['category_id']) != 'retail_product': raise ValueError('category must be retail_product')
            attrs=a.get('attributes') or {}
            sku=attrs.get('sku_id') or side.get(str(a['id']),{}).get('sku_id')
            if not sku: raise ValueError('missing attributes.sku_id; supply annotation-ID sidecar metadata')
            if sku not in master: raise ValueError(f'unknown sku_id {sku}')
            for key in ('brand','product','flavor','specification'):
                if attrs.get(key) and attrs[key] != master[sku].get(key): raise ValueError(f'SKU master conflict: {key}')
            polys=a['segmentation']
            if not isinstance(polys,list) or not polys: raise ValueError('polygon list required')
            for coords in polys:
                if len(coords)<6 or len(coords)%2: raise ValueError('invalid polygon coordinates')
                poly=Polygon(zip(coords[::2],coords[1::2]))
                if not poly.is_valid or poly.area<=0: raise ValueError('invalid/self-intersecting polygon')
                if not all(0<=x<=im['width'] and 0<=y<=im['height'] for x,y in zip(coords[::2],coords[1::2])): raise ValueError('polygon out of bounds')
            if len(polys)>1: raise ValueError('multipart polygon requires manual merged-instance review before import')
            mask=polygon_mask(polys,im['width'],im['height'])
            if not mask.any(): raise ValueError('empty mask')
            fingerprint=(a['image_id'],mask.tobytes())
            if fingerprint in fingerprints: raise ValueError('duplicate instance geometry')
            fingerprints.add(fingerprint)
            records.append(dict(annotation=a,image=im,sku_id=sku))
        except (KeyError,ValueError,TypeError) as e: errors.append(f'annotation {a.get("id")}: {e}')
    return dict(valid=not errors,errors=errors,records=records,image_count=len(image_map))
