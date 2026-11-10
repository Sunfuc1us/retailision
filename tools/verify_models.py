import _bootstrap
import json,socket
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from PIL import Image
from inference.segmentation.rtmdet import RTMDet
from inference.recognition.dino import DINOv2
from inference.retrieval.gallery import Gallery
# Deliberately disable Python networking to verify local-only loading.
def no_network(*a,**kw):raise RuntimeError('network prohibited during offline verification')
socket.socket.connect=no_network
assert torch.cuda.is_available(),'CUDA required'
torch.cuda.reset_peak_memory_stats();start=perf_counter()
seg=RTMDet('third_party/mmdetection/configs/rtmdet/rtmdet-ins_s_8xb32-300e_coco.py','models/segmentation/rtmdet_ins_s_coco_v001/model.pth');seg.load()
assert next(seg.model.parameters()).device.type=='cuda'
im=Image.open('third_party/mmdetection/demo/demo.jpg').convert('RGB');instances=seg.predict(im,.3)
assert instances and all(x.mask.shape==(im.height,im.width) and x.mask.any() for x in instances)
rec=DINOv2('third_party/dinov2','models/recognition/dinov2_b14_v001/model.pth');rec.load();e=rec.extract_embedding(im)
assert next(rec.model.parameters()).device.type=='cuda'
assert e.shape==(768,) and np.isfinite(e).all()
g=Gallery();g.build_index(np.stack([e,-e]),['FIXTURE_SAME_IMAGE','FIXTURE_NEGATIVE_VECTOR']);assert g.search(e)[0][0]=='FIXTURE_SAME_IMAGE'
out=Path('results/model_verification');out.mkdir(parents=True,exist_ok=True)
Image.fromarray(instances[0].mask.astype('uint8')*255).save(out/'mask.png')
# Real adapter composition on same-image references: integration only, never retail accuracy.
from dataset.processors.roi import mask_crop
from inference.pipeline import VisionPipeline
from inference.retrieval.rejection import RejectionPolicy,calibrate
from catalog.store import Catalog
ref=rec.extract_embedding(mask_crop(im,instances[0].mask))
gallery=Gallery();gallery.build_index(np.stack([ref,-ref]),['FIXTURE_INSTANCE','FIXTURE_NEGATIVE'])
(out/'catalog.json').write_text(json.dumps([dict(sku_id='FIXTURE_INSTANCE',product='COCO instance fixture'),dict(sku_id='FIXTURE_NEGATIVE',product='negative vector fixture')]))
cs=gallery.search(ref);unknown=gallery.search(np.roll(ref,1))
cal=calibrate([dict(known=True,correct=True,score=cs[0][1],margin=cs[0][1]-cs[1][1]),dict(known=False,correct=False,score=unknown[0][1],margin=unknown[0][1]-unknown[1][1])],'fixture_self_reference','fixture','fixture',0)
fixture_gallery=out/'gallery'
if not fixture_gallery.exists():gallery.save(str(fixture_gallery))
(out/'calibration.json').write_text(json.dumps(dict(fixture=True,**cal),indent=2))
config=dict(fixture=True,segmentation=dict(config=seg.config,checkpoint=seg.checkpoint),recognition=dict(source=rec.source,checkpoint=rec.checkpoint),gallery=str(fixture_gallery),catalog=str(out/'catalog.json'),calibration=str(out/'calibration.json'),gallery_version='fixture',model_version='fixture',results='results')
Path('configs/inference/fixture.local.json').write_text(json.dumps(config,indent=2))
policy=RejectionPolicy(**{k:cal[k] for k in ('similarity','margin','dataset_version','gallery_version','model_version')})
pipe=VisionPipeline(seg,rec,gallery,Catalog(out/'catalog.json'),policy,'results')
combined=pipe.predict(im,'FIXTURE_INSTANCE')
assert combined['target_found'] and combined['instances']
assert not policy.accepts(unknown)
# Exercise the real resident adapters through the actual FastAPI endpoint.
import io,importlib
from fastapi.testclient import TestClient
api=importlib.import_module('backend.api.app');api.runtime=pipe
buffer=io.BytesIO();im.save(buffer,format='PNG')
with TestClient(api.app) as client:
    response=client.post('/api/inference/predict',files={'image':('fixture.png',buffer.getvalue(),'image/png')},data={'target_product':'FIXTURE_INSTANCE'})
    assert response.status_code==200 and response.json()['target_found']
    assert client.get(response.json()['instances'][0]['mask_path']).status_code==200
    (out/'api_real_inference.json').write_text(json.dumps(dict(fixture=True,**response.json()),indent=2))
(out/'combined_pipeline.json').write_text(json.dumps(dict(fixture=True,scope='self-reference integration, synthetic negative; not accuracy or production calibration',**combined),indent=2))
r=dict(fixture=True,scope='COCO image model smoke test; not retail SKU accuracy',offline_python_network_blocked=True,gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,segmentation_device=str(next(seg.model.parameters()).device),recognition_device=str(next(rec.model.parameters()).device),faiss_device='cpu',segmentation_instances=len(instances),mask_shape=list(instances[0].mask.shape),embedding_shape=list(e.shape),embedding_norm=float(np.linalg.norm(e)),faiss_self_retrieval=True,combined_real_pipeline_target_found=combined['target_found'],fixture_unknown_rejected=True,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),total_seconds=perf_counter()-start)
(out/'report.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
