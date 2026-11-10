import _bootstrap
import hashlib,json,subprocess
from pathlib import Path
from datetime import datetime,timezone
from mmengine.config import Config
models=[('segmentation','rtmdet_ins_s_coco_v001','RTMDet-Ins-s','third_party/mmdetection','Apache-2.0 code; checkpoint separate permission unconfirmed','COCO','https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_s_8xb32-300e_coco/rtmdet-ins_s_8xb32-300e_coco_20221121_212604-fdc5d7ec.pth'),('recognition','dinov2_b14_v001','DINOv2 ViT-B/14','third_party/dinov2','Apache-2.0 model card','LVD-142M','https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth')]
for task,version,name,repo,license,dataset,url in models:
 out=Path('models')/task/version
 if (out/'manifest.json').exists():raise FileExistsError(out/'manifest.json')
 if task=='segmentation':
  cfg=Config.fromfile('third_party/mmdetection/configs/rtmdet/rtmdet-ins_s_8xb32-300e_coco.py');cfg.model.backbone.init_cfg=None;cfg.dump(str(out/'config.py'));conf='config.py'
 else:
  conf='config.json';(out/conf).write_text(json.dumps(dict(architecture='dinov2_vitb14',source=str(Path(repo).resolve()),embedding_dim=768)))
 manifest=dict(task=task,model_name=name,model_version=version,architecture=name,checkpoint='model.pth',training_dataset=dataset,training_config=conf,license=license,evaluation_metrics={},created_at=datetime.now(timezone.utc).isoformat(),sha256=hashlib.sha256((out/'model.pth').read_bytes()).hexdigest(),source_url=url,source_commit=subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip(),scope='pretrained only; not retail trained')
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
