import _bootstrap
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from inference.recognition.dino import DINOv2
from inference.retrieval.gallery import Gallery
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',required=True);p.add_argument('--projection');a=p.parse_args()
m=DINOv2('third_party/dinov2','models/recognition/dinov2_b14_v001/model.pth',a.projection);m.load();meta=json.loads((Path(a.dataset)/'metadata.json').read_text());xs=[];ids=[];refs=[]
for r in meta['items']:
 if r['split']!='train':continue
 with Image.open(Path(a.dataset)/r['path']) as im:xs.append(m.extract_embedding(im.convert('RGB')))
 ids.append(r['sku_id']);refs.append(r)
g=Gallery();g.build_index(np.stack(xs),ids);g.save(a.output)
(Path(a.output)/'manifest.json').write_text(json.dumps(dict(dataset_version=meta['dataset_version'],references=refs,model=m.get_model_info()),indent=2))
