import hashlib,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
from retailvision.data.validate import validate
from retailvision.image_ops import polygon_mask,mask_crop
from retailvision.data.split import split_groups
from retailvision.data.catalog import Catalog

def prepare(coco: str, images: str, catalog: str, output: str, sidecar: str | None = None, seed: int = 42, *, no_split: bool = False, validation_images: list[str] | None = None) -> dict:
    if no_split and validation_images is not None:raise ValueError('no_split and validation_images are mutually exclusive')
    report=validate(coco,images,catalog,sidecar,require_capture_group=not no_split and validation_images is None)
    if not report['valid']: raise ValueError(json.dumps(report['errors'],ensure_ascii=False))
    data=json.loads(Path(coco).read_text())
    if validation_images is not None:
        held_out=set(validation_images);names={i['file_name'] for i in data['images']}
        if not held_out or not held_out<names:raise ValueError('validation_images must be a nonempty proper subset of image filenames')
    splits={} if no_split or validation_images is not None else split_groups([i['capture_group'] for i in data['images']],seed)
    if no_split:image_splits={i['id']:'all' for i in data['images']}
    elif validation_images is not None:image_splits={i['id']:'val' if i['file_name'] in held_out else 'train' for i in data['images']}
    else:image_splits={i['id']:splits[i['capture_group']] for i in data['images']}
    digest=hashlib.sha256()
    for p in (coco,catalog,*([sidecar] if sidecar else [])): digest.update(Path(p).read_bytes())
    for im in sorted(data['images'],key=lambda x:x['id']): digest.update((Path(images)/im['file_name']).read_bytes())
    digest.update(str(seed).encode())
    if validation_images is not None:digest.update(json.dumps(sorted(held_out)).encode());digest.update(b'image_holdout_v1')
    elif no_split: digest.update(b'no_split')
    else: digest.update(b'grouped_train90_val10_v1')
    version=digest.hexdigest()
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    meta=[]
    image_counts={}
    for split in (('all',) if no_split else ('train','val')):
        ims=[i for i in data['images'] if image_splits[i['id']]==split]
        image_counts[split]=len(ims)
        imids={i['id'] for i in ims}; anns=[]
        for i in ims:
            dest=out/'segmentation/images'/i['file_name'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(Path(images)/i['file_name'],dest)
        for rec in report['records']:
            a=rec['annotation'];im=rec['image']
            if im['id'] not in imids: continue
            mask=polygon_mask(a['segmentation'],im['width'],im['height'])
            ys,xs=np.where(mask);a=dict(a,category_id=1,area=int(mask.sum()),bbox=a.get('bbox',[int(xs.min()),int(ys.min()),int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)]),iscrowd=0)
            anns.append(a)
            rel=Path(split)/rec['sku_id']/f'{a["id"]}.png'
            # SKU identifiers are directory names: reject path traversal.
            if Path(rec['sku_id']).name!=rec['sku_id'] or rec['sku_id'] in ('.','..'): raise ValueError('unsafe sku_id')
            dest=out/'recognition'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            with Image.open(Path(images)/im['file_name']) as rgb: mask_crop(rgb,mask).save(dest)
            mp=out/'recognition/masks'/f'{a["id"]}.png';mp.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(mask.astype('uint8')*255).save(mp)
            meta.append(dict(path=str(rel),sku_id=rec['sku_id'],image_id=im['id'],annotation_id=a['id'],capture_group=im.get('capture_group'),split=split))
        ap=out/f'segmentation/annotations/{split}.json';ap.parent.mkdir(parents=True,exist_ok=True)
        ap.write_text(json.dumps(dict(images=ims,annotations=anns,categories=[dict(id=1,name='retail_product')]),ensure_ascii=False))
    manifest=dict(dataset_version=version,seed=seed,groups=splits,no_split=no_split,source=str(Path(coco).resolve()),fixture=data.get('fixture',False))
    if validation_images is not None:
        manifest.update(split_unit='image',validation_images=sorted(held_out),independent_validation=False,validation_scope='same_capture_image_holdout')
    manifest['split_image_counts']=image_counts
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'segmentation/manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'recognition/metadata.json').write_text(json.dumps(dict(**manifest,items=meta),indent=2))
    master=Catalog(catalog).items
    (out/'recognition/skus.json').write_text(json.dumps([master[sku] for sku in sorted({item['sku_id'] for item in meta})],ensure_ascii=False,indent=2))
    return manifest
