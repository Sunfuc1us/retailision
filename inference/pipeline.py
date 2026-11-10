from pathlib import Path
from time import perf_counter
from uuid import uuid4
from PIL import Image
import numpy as np
from catalog.store import Catalog
from dataset.processors.roi import mask_crop
from inference.interfaces import SegmentationAdapter,RecognitionAdapter
from inference.retrieval.gallery import Gallery
from inference.retrieval.rejection import RejectionPolicy

class VisionPipeline:
    def __init__(self, segmentation: SegmentationAdapter, recognition: RecognitionAdapter, gallery: Gallery, catalog: Catalog, policy: RejectionPolicy | None, results: str):
        self.segmentation=segmentation;self.recognition=recognition;self.gallery=gallery;self.catalog=catalog;self.policy=policy;self.results=Path(results)
    def predict(self, image: Image.Image, target_product: str, detection_threshold: float = .3) -> dict:
        start=perf_counter();candidates=self.catalog.resolve(target_product)
        if len(candidates)!=1:return dict(success=True,target_found=False,reason='ambiguous_target' if candidates else 'unknown_target',candidates=candidates)
        sku=candidates[0]['sku_id']
        if self.policy is None:return dict(success=True,target_found=False,reason='calibration_required',sku_id=sku)
        instances=self.segmentation.predict(image,detection_threshold);seg_end=perf_counter()
        rid=uuid4().hex;out=self.results/rid;out.mkdir(parents=True)
        matches=[];all_results=[]
        for i,inst in enumerate(instances,1):
            nearest=self.gallery.search(self.recognition.extract_embedding(mask_crop(image,inst.mask)))
            accepted=self.policy.accepts(nearest);identity=nearest[0][0] if accepted else None
            item=dict(instance_id=i,sku_id=identity,identity_score=nearest[0][1],bbox=inst.bbox,detection_score=inst.score,accepted=accepted)
            all_results.append(item)
            if identity==sku:
                mp=out/f'mask_{i:03}.png';Image.fromarray(inst.mask.astype('uint8')*255).save(mp)
                item['mask_path']=f'/api/results/{rid}/mask_{i:03}.png';matches.append(item)
        end=perf_counter()
        result=dict(success=True,target_found=bool(matches),sku_id=sku,identity_score=max((x['identity_score'] for x in matches),default=None),instances=matches,candidates=all_results,result_id=rid,timing=dict(segmentation_ms=(seg_end-start)*1000,recognition_ms=(end-seg_end)*1000,total_ms=(end-start)*1000))
        if not matches:result['reason']='no_confident_match'
        import json
        (out/'result.json').write_text(json.dumps(result,indent=2));return result
