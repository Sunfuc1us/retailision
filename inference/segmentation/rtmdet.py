from pathlib import Path
from project_paths import project_path
import numpy as np
from PIL import Image
from inference.interfaces import Instance

class RTMDet:
    def __init__(self, config: str, checkpoint: str):
        self.config=str(project_path(config));self.checkpoint=str(project_path(checkpoint));self.model=None
    def load(self) -> None:
        import torch
        from mmdet.apis import init_detector
        from mmengine.config import Config
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required; CPU fallback disabled')
        if not Path(self.checkpoint).is_file(): raise FileNotFoundError(self.checkpoint)
        cfg=Config.fromfile(self.config)
        cfg.model.backbone.init_cfg=None
        self.model=init_detector(cfg,self.checkpoint,device='cuda:0')
    def predict(self, image: Image.Image, threshold: float = .3) -> list[Instance]:
        import torch
        from mmdet.apis import inference_detector
        if self.model is None: raise RuntimeError('model not loaded')
        with torch.inference_mode(): pred=inference_detector(self.model,np.array(image.convert('RGB'))[:,:,::-1].copy()).pred_instances
        torch.cuda.synchronize()
        return [Instance(b.tolist(),m.cpu().numpy().astype(bool),float(s)) for b,m,s in zip(pred.bboxes,pred.masks,pred.scores) if float(s)>=threshold]
    def unload(self) -> None:
        import torch
        self.model=None;torch.cuda.empty_cache()
    def get_model_info(self) -> dict:
        return dict(architecture='RTMDet-Ins-s',checkpoint=self.checkpoint,config=self.config,loaded=self.model is not None,device='cuda:0')
