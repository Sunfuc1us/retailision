import json
from pathlib import Path
from retailvision.paths import project_path
from retailvision.artifacts.config import preflight
from retailvision.inference.rtmdet import RTMDet
from retailvision.inference.dino import DINOv2
from retailvision.inference.gallery import Gallery
from retailvision.inference.rejection import RejectionPolicy
from retailvision.inference.pipeline import VisionPipeline
from retailvision.data.catalog import Catalog

def load_pipeline(config: str, backend: str = 'pytorch') -> VisionPipeline:
    c=preflight(config, backend)
    if backend == 'pytorch':
        seg=RTMDet(**c['segmentation']);rec=DINOv2(**c['recognition'])
    elif backend in ('onnxruntime', 'tensorrt'):
        from retailvision.inference.backends.rtmdet_mmdeploy import RTMDetMMDeploy
        artifacts = c.get('deployment', {}).get(backend)
        if not artifacts:
            raise ValueError(f'{backend} artifacts are not configured')
        seg = RTMDetMMDeploy(c['segmentation']['config'], str(project_path(artifacts['seg_deploy_config'])),
                             str(project_path(artifacts['seg_model'])), backend)
        from retailvision.inference.backends.dinov2_engine import DINOv2Engine
        rec = DINOv2Engine(str(project_path(artifacts['rec_model'])), backend,
                           input_dtype='float16' if backend == 'tensorrt' else 'float32')
    else:
        raise ValueError(f'unsupported inference backend: {backend}')
    matching=c.get('matching',{})
    g=Gallery.load(c['gallery'],aggregation=matching.get('aggregation','max'),support_k=matching.get('support_k',1))
    policy_data=json.loads(Path(c['calibration']).read_text())
    policy=RejectionPolicy(**{k:policy_data[k] for k in ('similarity','margin','dataset_version','gallery_version','model_version')},
                           sku_similarity=policy_data.get('sku_similarity',{}))
    seg.load();rec.load()
    pipeline=VisionPipeline(seg,rec,g,Catalog(c['catalog']),policy,c['results'],
                            clip_mask_to_bbox=c.get('roi',{}).get('clip_mask_to_bbox',False))
    pipeline.backend=backend
    return pipeline
