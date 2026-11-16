from pathlib import Path
import math
from time import perf_counter
from uuid import uuid4
from PIL import Image
import numpy as np
from retailvision.data.catalog import Catalog
from retailvision.image_ops import mask_crop
from retailvision.inference.interfaces import SegmentationAdapter,RecognitionAdapter
from retailvision.inference.gallery import Gallery
from retailvision.inference.rejection import RejectionPolicy

def mask_within_box(mask: np.ndarray, bbox: list[float]) -> np.ndarray:
    """Remove disconnected mask pixels far outside the predicted Box."""
    x1,y1,x2,y2=map(float,bbox)
    padding=.05*max(x2-x1,y2-y1)
    height,width=mask.shape
    left=max(0,min(width,math.floor(x1-padding)));top=max(0,min(height,math.floor(y1-padding)))
    right=max(0,min(width,math.ceil(x2+padding)));bottom=max(0,min(height,math.ceil(y2+padding)))
    bounded=np.zeros_like(mask,dtype=bool)
    bounded[top:bottom,left:right]=mask[top:bottom,left:right]
    return bounded

class VisionPipeline:
    def __init__(self, segmentation: SegmentationAdapter, recognition: RecognitionAdapter, gallery: Gallery, catalog: Catalog, policy: RejectionPolicy | None, results: str, clip_mask_to_bbox: bool = False):
        self.segmentation=segmentation;self.recognition=recognition;self.gallery=gallery;self.catalog=catalog;self.policy=policy;self.results=Path(results);self.backend='pytorch';self.clip_mask_to_bbox=clip_mask_to_bbox;self.compact_masks=False
    def unload(self):
        self.segmentation.unload();self.recognition.unload()
    def get_runtime_info(self) -> dict:
        seg_info=self.segmentation.get_model_info() if hasattr(self.segmentation,'get_model_info') else {}
        rec_info=self.recognition.get_model_info() if hasattr(self.recognition,'get_model_info') else {}
        return dict(backend=self.backend,model_id=getattr(self,'model_id',None),model_name=getattr(self,'model_name',None),
                    clip_mask_to_bbox=self.clip_mask_to_bbox,segmentation=seg_info,recognition=rec_info,
                    gallery=dict(ntotal=len(self.gallery.ids),unique_skus=len(set(self.gallery.ids)),
                                 aggregation=getattr(self.gallery,'aggregation','max'),
                                 support_k=getattr(self.gallery,'support_k',1)),
                    policy=vars(self.policy) if self.policy else None)
    def _inspect_instances(self, image: Image.Image, detection_threshold: float, out: Path, rid: str, top_k: int = 5) -> tuple[list[dict], float, float]:
        start=perf_counter();instances=self.segmentation.predict(image,detection_threshold);seg_end=perf_counter()
        all_results=[];pending=[]
        for i,inst in enumerate(instances,1):
            raw_mask=inst.mask;raw_pixels=int(raw_mask.sum());mask=raw_mask
            if self.clip_mask_to_bbox:
                if mask.shape!=(image.height,image.width):raise ValueError('instance mask shape does not match image')
                mask=mask_within_box(mask,inst.bbox)
                if not mask.any():
                    all_results.append(dict(instance_id=i,sku_id=None,top1_sku=None,identity_score=None,margin=None,top_k=[],bbox=inst.bbox,
                        category_id=inst.label,detection_score=inst.score,raw_mask_pixels=raw_pixels,mask_pixels=0,
                        bbox_clipped=True,accepted=False,reason='empty_mask_in_bbox'))
                    continue
            mask_pixels=int(mask.sum())
            mask_meta={}
            if self.compact_masks:
                ys=np.flatnonzero(mask.any(axis=1));xs=np.flatnonzero(mask.any(axis=0))
                left,top,right,bottom=int(xs[0]),int(ys[0]),int(xs[-1])+1,int(ys[-1])+1
                Image.fromarray(mask[top:bottom,left:right].astype('uint8')*255).save(out/f'mask_{i:03}.crop.png',compress_level=1)
                mask_meta=dict(mask_preview_path=f'/api/results/{rid}/mask_{i:03}.crop.png',mask_origin=[left,top],image_size=[image.width,image.height])
            else:
                Image.fromarray(mask.astype('uint8')*255).save(out/f'mask_{i:03}.png',compress_level=1)
            pending.append((i,inst,raw_pixels,mask_pixels,mask_meta,mask_crop(image,mask)))
        crops=[entry[-1] for entry in pending]
        embeddings=(self.recognition.extract_embeddings(crops) if hasattr(self.recognition,'extract_embeddings')
                    else [self.recognition.extract_embedding(crop) for crop in crops])
        for (i,inst,raw_pixels,mask_pixels,mask_meta,_),embedding in zip(pending,embeddings):
            emb=np.asarray(embedding,dtype='float32')
            nearest=self.gallery.search(emb,k=top_k) if self.gallery.index is not None else []
            top1_sku=nearest[0][0] if nearest else None
            top1_score=float(nearest[0][1]) if nearest else None
            margin=float(nearest[0][1]-nearest[1][1]) if len(nearest)>=2 else (top1_score if top1_score is not None else 0.0)
            accepted=self.policy.accepts(nearest) if (self.policy is not None and nearest) else False
            identity=top1_sku if accepted else None
            reason=None
            if self.policy is not None and not accepted:
                reason='insufficient_candidates' if len(nearest)<2 else ('below_similarity' if nearest[0][1]<self.policy.similarity_for(nearest[0][0]) else 'below_margin')
            item=dict(instance_id=i,sku_id=identity,top1_sku=top1_sku,identity_score=top1_score,margin=margin,
                      top_k=[dict(sku_id=s,score=float(sc)) for s,sc in nearest],bbox=inst.bbox,
                      category_id=inst.label,detection_score=inst.score,raw_mask_pixels=raw_pixels,
                      mask_pixels=mask_pixels,bbox_clipped=bool(self.clip_mask_to_bbox),
                      embedding_dim=int(emb.size),embedding_norm=float(np.linalg.norm(emb)),
                      mask_path=f'/api/results/{rid}/mask_{i:03}.png',accepted=accepted,
                      **mask_meta,**({'reason':reason} if reason else {}))
            all_results.append(item)
        all_results.sort(key=lambda item:item['instance_id'])
        end=perf_counter()
        return all_results,(seg_end-start)*1000,(end-seg_end)*1000
    def diagnose(self, image: Image.Image, detection_threshold: float = .8, top_k: int = 5) -> dict:
        import json
        rid=uuid4().hex;out=self.results/rid;out.mkdir(parents=True)
        all_results,seg_ms,rec_ms=self._inspect_instances(image,detection_threshold,out,rid,top_k=top_k)
        result=dict(success=True,candidate_count=len(all_results),candidates=all_results,result_id=rid,
                    backend=self.backend,model_id=getattr(self,'model_id',None),model_name=getattr(self,'model_name',None),
                    runtime=self.get_runtime_info(),timing=dict(segmentation_ms=seg_ms,recognition_ms=rec_ms,total_ms=seg_ms+rec_ms))
        (out/'result.json').write_text(json.dumps(result,indent=2));return result
    def predict(self, image: Image.Image, target_product: str, detection_threshold: float = .8) -> dict:
        import json
        candidates=self.catalog.resolve(target_product)
        if len(candidates)!=1:return dict(success=True,target_found=False,reason='ambiguous_target' if candidates else 'unknown_target',candidates=candidates)
        sku=candidates[0]['sku_id']
        if self.policy is None:return dict(success=True,target_found=False,reason='calibration_required',sku_id=sku)
        rid=uuid4().hex;out=self.results/rid;out.mkdir(parents=True)
        all_results,seg_ms,rec_ms=self._inspect_instances(image,detection_threshold,out,rid)
        matches=[x for x in all_results if x['sku_id']==sku]
        result=dict(success=True,target_found=bool(matches),sku_id=sku,identity_score=max((x['identity_score'] for x in matches),default=None),instances=matches,candidates=all_results,result_id=rid,backend=self.backend,model_id=getattr(self,'model_id',None),model_name=getattr(self,'model_name',None),timing=dict(segmentation_ms=seg_ms,recognition_ms=rec_ms,total_ms=seg_ms+rec_ms))
        if not matches:result['reason']='no_confident_match'
        (out/'result.json').write_text(json.dumps(result,indent=2));return result
