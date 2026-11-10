import numpy as np
from sklearn.metrics import confusion_matrix

def recognition(rows: list[dict]) -> dict:
    known=[r for r in rows if r['known']];unknown=[r for r in rows if not r['known']]
    def acc(rs,k):return sum(r['truth'] in r['predictions'][:k] for r in rs)/len(rs) if rs else None
    labels=sorted({r['truth'] for r in known}|{r['predictions'][0] for r in known if r['predictions']})
    return dict(top1=acc(known,1),top5=acc(known,5),retrieval_recall_at_5=acc(known,5),similar_sku_confusion_rate=(1-acc([r for r in known if 'similar_sku' in r.get('tags',[])],1)) if any('similar_sku' in r.get('tags',[]) for r in known) else None,unknown_rejection_rate=sum(not r['accepted'] for r in unknown)/len(unknown) if unknown else None,slices={tag:dict(count=len(sub),top1=acc(sub,1)) for tag in ['same_brand_different_flavor','same_flavor_different_size','similar_medicine','occlusion','small_object','lighting'] for sub in [[r for r in known if tag in r.get('tags',[])]]},confusion_matrix=dict(labels=labels,values=confusion_matrix([r['truth'] for r in known],[r['predictions'][0] for r in known],labels=labels).tolist()) if known else {},failure_samples=[r for r in rows if (r['known'] and r['truth'] not in r['predictions'][:1]) or (not r['known'] and r['accepted'])])

def segmentation(coco_path: str, predictions_path: str) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt=COCO(coco_path);dt=gt.loadRes(predictions_path);out={}
    for kind in ('bbox','segm'):
        ev=COCOeval(gt,dt,kind);ev.evaluate();ev.accumulate();ev.summarize()
        precision=ev.eval['precision'][0,:,:,0,-1]
        out[kind]=dict(map50_95=float(ev.stats[0]),map50=float(ev.stats[1]),precision_at_iou50=float(precision[precision>=0].mean()),recall=float(ev.stats[8]),small_object_recall=float(ev.stats[9]))
    ious=[]
    for image_id in gt.getImgIds():
        truth=gt.loadAnns(gt.getAnnIds(imgIds=[image_id]));preds=sorted(dt.loadAnns(dt.getAnnIds(imgIds=[image_id])),key=lambda a:a['score'],reverse=True);used=set()
        for pred in preds:
            matches=[(mask_iou(gt.annToMask(t),dt.annToMask(pred)),t['id']) for t in truth if t['id'] not in used and t['category_id']==pred['category_id']]
            if matches:
                score,tid=max(matches)
                if score>=.5:ious.append(score);used.add(tid)
    out['matched_mask_iou']=float(np.mean(ious)) if ious else None
    return out

def mask_iou(a: np.ndarray,b: np.ndarray) -> float:
    union=np.logical_or(a,b).sum();return float(np.logical_and(a,b).sum()/union) if union else 1.

def pipeline(rows: list[dict]) -> dict:
    if not rows:raise ValueError('empty evaluation')
    return {**{k:sum(bool(r[k]) for r in rows)/len(rows) for k in ('localization_success','identification_success','end_to_end_success')},'latency_ms_mean':float(np.mean([r['latency_ms'] for r in rows])),'latency_ms_p95':float(np.percentile([r['latency_ms'] for r in rows],95)),'peak_gpu_memory_bytes':max(r['gpu_memory_bytes'] for r in rows),'failure_samples':[r for r in rows if not r['end_to_end_success']]}
