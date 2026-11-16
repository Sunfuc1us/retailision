import _bootstrap
import argparse,json,runpy
from pathlib import Path
import numpy as np
from PIL import Image
from retailvision.inference.dino import DINOv2
from retailvision.inference.gallery import Gallery
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',required=True);p.add_argument('--projection');p.add_argument('--config',default='configs/recognition/dinov2_b14.py');p.add_argument('--split',default='train');p.add_argument('--overwrite',action='store_true');a=p.parse_args()
c=runpy.run_path(a.config)['config']
m=DINOv2(c['source'],c['checkpoint'],a.projection);m.load();meta=json.loads((Path(a.dataset)/'metadata.json').read_text());xs=[];ids=[];refs=[]
target_split='all' if meta.get('no_split') and a.split=='train' else a.split
for r in meta['items']:
 if target_split!='all' and r['split']!=target_split:continue
 with Image.open(Path(a.dataset)/r['path']) as im:xs.append(m.extract_embedding(im.convert('RGB')))
 ids.append(r['sku_id']);refs.append(r)
g=Gallery();g.build_index(np.stack(xs),ids);g.save(a.output,overwrite=a.overwrite)
(Path(a.output)/'manifest.json').write_text(json.dumps(dict(gallery_version=Path(a.output).name,dataset_version=meta['dataset_version'],split=target_split,references=refs,model=m.get_model_info()),indent=2))
