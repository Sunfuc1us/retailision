import json
from pathlib import Path
from project_paths import project_path
from inference.segmentation.rtmdet import RTMDet
from inference.recognition.dino import DINOv2
from inference.retrieval.gallery import Gallery
from inference.retrieval.rejection import RejectionPolicy
from inference.pipeline import VisionPipeline
from catalog.store import Catalog

def load_pipeline(config: str) -> VisionPipeline:
    c=json.loads(project_path(config).read_text())
    c['segmentation']={k:str(project_path(v)) for k,v in c['segmentation'].items()}
    c['recognition']={k:str(project_path(v)) if v is not None else None for k,v in c['recognition'].items()}
    for key in ('gallery','calibration','catalog','results'):c[key]=str(project_path(c[key]))
    seg=RTMDet(**c['segmentation']);rec=DINOv2(**c['recognition'])
    g=Gallery.load(c['gallery']);policy_data=json.loads(Path(c['calibration']).read_text())
    for k in ('gallery_version','model_version'):
        if policy_data[k]!=c[k]:raise ValueError(f'calibration {k} mismatch')
    policy=RejectionPolicy(**{k:policy_data[k] for k in ('similarity','margin','dataset_version','gallery_version','model_version')})
    seg.load();rec.load()
    return VisionPipeline(seg,rec,g,Catalog(c['catalog']),policy,c['results'])
