from pathlib import Path
from project_paths import project_path
import numpy as np
from PIL import Image

class DINOv2:
    def __init__(self, source: str, checkpoint: str, projection: str | None = None):
        self.source=str(project_path(source));self.checkpoint=str(project_path(checkpoint));self.projection=str(project_path(projection)) if projection else None;self.model=None;self.head=None
    def load(self) -> None:
        import torch
        from torchvision import transforms
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required; CPU fallback disabled')
        if not Path(self.source,'hubconf.py').is_file() or not Path(self.checkpoint).is_file(): raise FileNotFoundError('local DINOv2 source/weights missing')
        import sys
        source=str(Path(self.source).resolve())
        if source not in sys.path:sys.path.insert(0,source)
        from dinov2.hub.backbones import dinov2_vitb14
        self.model=dinov2_vitb14(pretrained=False)
        self.model.load_state_dict(torch.load(self.checkpoint,map_location='cpu',weights_only=True))
        self.model=self.model.eval().cuda()
        if self.projection:
            state=torch.load(self.projection,map_location='cpu',weights_only=True)
            self.head=torch.nn.Linear(768,state['head']['weight'].shape[0]).cuda()
            self.head.load_state_dict(state['head']);self.head.eval()
        self.transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    def extract_embedding(self, image: Image.Image) -> np.ndarray:
        import torch
        if self.model is None: raise RuntimeError('model not loaded')
        with torch.inference_mode():
            e=self.model(self.transform(image.convert('RGB')).unsqueeze(0).cuda())
            if self.head is not None:e=self.head(e)
            e=torch.nn.functional.normalize(e,dim=-1)
        torch.cuda.synchronize()
        return e[0].cpu().numpy().astype('float32')
    def unload(self) -> None:
        import torch
        self.model=None;self.head=None;torch.cuda.empty_cache()
    def get_model_info(self) -> dict:
        return dict(architecture='DINOv2 ViT-B/14',checkpoint=self.checkpoint,projection=self.projection,loaded=self.model is not None,device='cuda:0')
