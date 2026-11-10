import hashlib,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
from dataset.validators.coco import validate
from dataset.processors.roi import polygon_mask,mask_crop
from dataset.splitters.grouped import split_groups

def prepare(coco: str, images: str, catalog: str, output: str, sidecar: str | None = None, seed: int = 42) -> dict:
    report=validate(coco,images,catalog,sidecar)
    if not report['valid']: raise ValueError(json.dumps(report['errors'],ensure_ascii=False))
    data=json.loads(Path(coco).read_text())
    splits=split_groups([i['capture_group'] for i in data['images']],seed)
    digest=hashlib.sha256()
    for p in (coco,catalog,*([sidecar] if sidecar else [])): digest.update(Path(p).read_bytes())
    for im in sorted(data['images'],key=lambda x:x['id']): digest.update((Path(images)/im['file_name']).read_bytes())
    digest.update(str(seed).encode());version=digest.hexdigest()
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    meta=[]
    for split in ('train','val','test'):
        ims=[i for i in data['images'] if splits[i['capture_group']]==split]
        imids={i['id'] for i in ims}; anns=[]
        for i in ims:
            dest=out/'segmentation/images'/i['file_name'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(Path(images)/i['file_name'],dest)
        for rec in report['records']:
            a=rec['annotation'];im=rec['image']
            if im['id'] not in imids: continue
            mask=polygon_mask(a['segmentation'],im['width'],im['height'])
            ys,xs=np.where(mask);a=dict(a,category_id=1,area=int(mask.sum()),bbox=[int(xs.min()),int(ys.min()),int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)],iscrowd=0)
            anns.append(a)
            rel=Path(split)/rec['sku_id']/f'{a["id"]}.png'
            # SKU identifiers are directory names: reject path traversal.
            if Path(rec['sku_id']).name!=rec['sku_id'] or rec['sku_id'] in ('.','..'): raise ValueError('unsafe sku_id')
            dest=out/'recognition'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            with Image.open(Path(images)/im['file_name']) as rgb: mask_crop(rgb,mask).save(dest)
            mp=out/'recognition/masks'/f'{a["id"]}.png';mp.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(mask.astype('uint8')*255).save(mp)
            meta.append(dict(path=str(rel),sku_id=rec['sku_id'],image_id=im['id'],annotation_id=a['id'],capture_group=im['capture_group'],split=split))
        ap=out/f'segmentation/annotations/{split}.json';ap.parent.mkdir(parents=True,exist_ok=True)
        ap.write_text(json.dumps(dict(images=ims,annotations=anns,categories=[dict(id=1,name='retail_product')]),ensure_ascii=False))
    manifest=dict(dataset_version=version,seed=seed,groups=splits,source=str(Path(coco).resolve()),fixture=data.get('fixture',False))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'segmentation/manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'recognition/metadata.json').write_text(json.dumps(dict(**manifest,items=meta),indent=2))
    return manifest
