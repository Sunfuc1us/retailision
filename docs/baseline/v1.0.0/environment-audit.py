from pathlib import Path
import os,sys,site,json,importlib.metadata as md,hashlib,subprocess
root=Path('/home/gzh/rocco/retailvision');legacy='/home/gzh/rocco/demoroom'
import project_paths,torch
assert project_paths.PROJECT_ROOT==root
assert not site.ENABLE_USER_SITE
assert not any(legacy in p for p in sys.path)
packages={}
for name in ('retailvision','torch','torchvision','faiss-cpu','mmcv','mmengine','mmdet'):
 try:
  d=md.distribution(name);packages[name]={'version':d.version,'direct_url':d.read_text('direct_url.json')}
 except md.PackageNotFoundError:packages[name]=None
assert legacy not in json.dumps(packages)
# Audit environment .pth files and editable finder modules, not only sys.path.
for folder in site.getsitepackages():
 for pattern in ('*.pth','__editable__*.py'):
  for p in Path(folder).glob(pattern):assert legacy not in p.read_text(),str(p)
assert torch.cuda.is_available()
r=dict(executable=sys.executable,project_root=str(root),user_site=site.ENABLE_USER_SITE,sys_path=sys.path,packages=packages,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name())
if 'retailvision-reid' in sys.executable:
 import socket,numpy as np
 from PIL import Image
 def deny(event,args):
  if event=='open' and args and isinstance(args[0],(str,bytes)) and legacy in os.fsdecode(args[0]):raise RuntimeError('legacy access denied')
 sys.addaudithook(deny)
 def offline(*a,**kw):raise RuntimeError('network prohibited')
 socket.socket.connect=offline
 from inference.recognition.dino import DINOv2
 rec=DINOv2('third_party/dinov2','models/recognition/dinov2_b14_v001/model.pth');rec.load()
 e=rec.extract_embedding(Image.open(root/'third_party/mmdetection/demo/demo.jpg'))
 assert next(rec.model.parameters()).is_cuda and e.shape==(768,) and np.isfinite(e).all()
 r['offline_dino_inference']={'device':str(next(rec.model.parameters()).device),'embedding_dim':len(e),'norm':float(np.linalg.norm(e))}
print(json.dumps(r,indent=2))
