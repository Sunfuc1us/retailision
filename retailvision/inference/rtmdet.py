from pathlib import Path
from retailvision.paths import project_path
import numpy as np
from PIL import Image
from retailvision.inference.interfaces import Instance

class RTMDet:
    def __init__(self, config: str, checkpoint: str, precision: str = 'fp32'):
        if precision not in ('fp32', 'fp16'): raise ValueError(f'unsupported RTMDet precision: {precision}')
        self.config=str(project_path(config));self.checkpoint=str(project_path(checkpoint));self.model=None;self.precision=precision
    def load(self) -> None:
        import torch
        from mmdet.apis import init_detector
        from mmengine.config import Config
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required; CPU fallback disabled')
        if not Path(self.checkpoint).is_file(): raise FileNotFoundError(self.checkpoint)
        cfg=Config.fromfile(self.config)
        cfg.model.backbone.init_cfg=None
        self.model=init_detector(cfg,self.checkpoint,device='cuda:0')
        if self.precision=='fp16':
            # MMCV's CUDA NMS requires float32 scores and boxes. Keep the CNN
            # under autocast, then promote head outputs before decoding/NMS.
            def fp32_predictions(module, inputs, outputs):
                scores,boxes,kernels,mask_features=outputs
                return ([x.float() for x in scores],[x.float() for x in boxes],
                        [x.float() for x in kernels],mask_features.float())
            self.model.bbox_head.register_forward_hook(fp32_predictions)
    def predict(self, image: Image.Image, threshold: float = .3) -> list[Instance]:
        import torch
        from mmdet.apis import inference_detector
        if self.model is None: raise RuntimeError('model not loaded')
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=self.precision == 'fp16'):
            pred=inference_detector(self.model,np.array(image.convert('RGB'))[:,:,::-1].copy()).pred_instances
        torch.cuda.synchronize()
        return [Instance(b.tolist(),m.cpu().numpy().astype(bool),float(s),int(label))
                for b,m,s,label in zip(pred.bboxes,pred.masks,pred.scores,pred.labels) if float(s)>=threshold]
    def unload(self) -> None:
        import torch
        self.model=None;torch.cuda.empty_cache()
    def get_model_info(self) -> dict:
        return dict(architecture='RTMDet-Ins-s',checkpoint=self.checkpoint,config=self.config,loaded=self.model is not None,device='cuda:0',precision=self.precision,backend='pytorch')
