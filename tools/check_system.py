import json,platform,shutil,subprocess,sys
from pathlib import Path
r=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),disk_free_gb=shutil.disk_usage('.').free/2**30)
r['nvidia_smi']=subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout
try:
 import torch
 r.update(torch=torch.__version__,cuda_runtime=torch.version.cuda,cuda_available=torch.cuda.is_available())
 if torch.cuda.is_available():
  x=torch.ones((1024,1024),device='cuda');y=x@x;torch.cuda.synchronize()
  r.update(gpu=torch.cuda.get_device_name(),vram_bytes=torch.cuda.get_device_properties(0).total_memory,allocated_bytes=torch.cuda.max_memory_allocated())
except ImportError:r['torch']='not installed'
print(json.dumps(r,indent=2))
