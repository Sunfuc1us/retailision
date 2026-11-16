import _bootstrap
import argparse,json
from retailvision.artifacts.config import preflight
p=argparse.ArgumentParser()
p.add_argument('--config',required=True)
p.add_argument('--image')
p.add_argument('--dry-run',action='store_true',help='Validate local inference paths and versions without loading models')
p.add_argument('--target')
p.add_argument('--threshold',type=float,default=.8)
p.add_argument('--backend',choices=['pytorch','onnxruntime','tensorrt'],default='pytorch')
p.add_argument('--diagnose',action='store_true')
p.add_argument('--top-k',type=int,default=5)
a=p.parse_args()
if a.dry_run:
    print(json.dumps(dict(backend=a.backend,config=preflight(a.config,a.backend)),indent=2,ensure_ascii=False))
    raise SystemExit(0)
if not a.image:p.error('--image is required unless --dry-run is used')
from PIL import Image
from retailvision.inference.factory import load_pipeline
pipe=load_pipeline(a.config,a.backend)
img=Image.open(a.image).convert('RGB')
if a.diagnose or not a.target:
    res=pipe.diagnose(img,a.threshold,top_k=a.top_k)
else:
    res=pipe.predict(img,a.target,a.threshold)
print(json.dumps(res,indent=2,ensure_ascii=False))
