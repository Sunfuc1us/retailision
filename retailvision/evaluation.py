import numpy as np
from sklearn.metrics import confusion_matrix

def recognition(rows: list[dict]) -> dict:
    known=[r for r in rows if r['known']];unknown=[r for r in rows if not r['known']]
    def acc(rs,k):return sum(r['truth'] in r['predictions'][:k] for r in rs)/len(rs) if rs else None
    labels=sorted({r['truth'] for r in known}|{r['predictions'][0] for r in known if r['predictions']})
    return dict(top1=acc(known,1),top5=acc(known,5),retrieval_recall_at_5=acc(known,5),similar_sku_confusion_rate=(1-acc([r for r in known if 'similar_sku' in r.get('tags',[])],1)) if any('similar_sku' in r.get('tags',[]) for r in known) else None,unknown_rejection_rate=sum(not r['accepted'] for r in unknown)/len(unknown) if unknown else None,slices={tag:dict(count=len(sub),top1=acc(sub,1)) for tag in ['same_brand_different_flavor','same_flavor_different_size','similar_medicine','occlusion','small_object','lighting'] for sub in [[r for r in known if tag in r.get('tags',[])]]},confusion_matrix=dict(labels=labels,values=confusion_matrix([r['truth'] for r in known],[r['predictions'][0] for r in known],labels=labels).tolist()) if known else {},failure_samples=[r for r in rows if (r['known'] and r['truth'] not in r['predictions'][:1]) or (not r['known'] and r['accepted'])])

def segmentation(coco_path: str, predictions_path: str) -> dict:
    import json
    from pathlib import Path
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt=COCO(coco_path);raw_preds=json.loads(Path(predictions_path).read_text())
    if not raw_preds:
        empty=dict(map50_95=0.,map50=0.,precision_at_iou50=0.,recall=0.,small_object_recall=0.)
        return dict(bbox=dict(empty),segm=dict(empty),matched_mask_iou=None)
    dt=gt.loadRes(raw_preds);out={}
    for kind in ('bbox','segm'):
        ev=COCOeval(gt,dt,kind);ev.evaluate();ev.accumulate();ev.summarize()
        precision=ev.eval['precision'][0,:,:,0,-1]
        valid=precision[precision>=0]
        out[kind]=dict(map50_95=float(ev.stats[0]),map50=float(ev.stats[1]),precision_at_iou50=float(valid.mean()) if valid.size else 0.,recall=float(ev.stats[8]),small_object_recall=float(ev.stats[9]))
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

def evaluate_dataset(pipe, dataset_dir: str, split: str = 'val', threshold: float = .3, work_dir: str | None = None) -> dict:
    import json,tempfile
    from pathlib import Path
    from PIL import Image
    from pycocotools import mask as mask_utils
    from retailvision.image_ops import polygon_mask
    root=Path(dataset_dir);meta=json.loads((root/'recognition/metadata.json').read_text())
    actual_split='all' if meta.get('no_split') and not (root/f'segmentation/annotations/{split}.json').is_file() else split
    coco_path=root/f'segmentation/annotations/{actual_split}.json'
    coco_data=json.loads(coco_path.read_text())
    ann_sku={r['annotation_id']:r['sku_id'] for r in meta['items']}
    known_skus=set(pipe.gallery.ids)
    roi_rows=[]
    for item in meta['items']:
        if actual_split!='all' and item['split']!=actual_split:continue
        with Image.open(root/'recognition'/item['path']) as crop:
            emb=pipe.recognition.extract_embedding(crop.convert('RGB'))
        nearest=pipe.gallery.search(emb,k=5) if pipe.gallery.index is not None else []
        accepted=pipe.policy.accepts(nearest) if (pipe.policy is not None and nearest) else False
        roi_rows.append(dict(known=item['sku_id'] in known_skus,truth=item['sku_id'],
                             predictions=[s for s,_ in nearest],accepted=accepted,annotation_id=item['annotation_id']))
    coco_preds=[];pipe_rows=[]
    anns_by_img={}
    for a in coco_data['annotations']:anns_by_img.setdefault(a['image_id'],[]).append(a)
    for im in coco_data['images']:
        with Image.open(root/'segmentation/images'/im['file_name']) as rgb:
            diag=pipe.diagnose(rgb.convert('RGB'),detection_threshold=threshold,top_k=5)
        res_dir=pipe.results/diag['result_id']
        pred_masks=[]
        for c in diag['candidates']:
            mp=res_dir/f"mask_{c['instance_id']:03}.png"
            if not mp.is_file():continue
            m=(np.asarray(Image.open(mp))>0)
            pred_masks.append((c,m))
            rle=mask_utils.encode(np.asfortranarray(m.astype('uint8')));rle['counts']=rle['counts'].decode()
            x1,y1,x2,y2=c['bbox']
            coco_preds.append(dict(image_id=im['id'],category_id=1,bbox=[x1,y1,x2-x1,y2-y1],segmentation=rle,score=c['detection_score']))
        for gt_ann in anns_by_img.get(im['id'],[]):
            gt_mask=polygon_mask(gt_ann['segmentation'],im['width'],im['height'])
            gt_sku=ann_sku.get(gt_ann['id'])
            best_iou=0.;best_cand=None
            for c,pm in pred_masks:
                iou_val=mask_iou(gt_mask,pm)
                if iou_val>best_iou:best_iou=iou_val;best_cand=c
            loc_ok=best_iou>=.5
            id_ok=bool(loc_ok and best_cand and best_cand.get('top1_sku')==gt_sku)
            e2e_ok=bool(loc_ok and best_cand and best_cand.get('sku_id')==gt_sku)
            pipe_rows.append(dict(image_id=im['id'],annotation_id=gt_ann['id'],sku_id=gt_sku,matched_iou=best_iou,
                                  localization_success=loc_ok,identification_success=id_ok,end_to_end_success=e2e_ok,
                                  latency_ms=diag['timing']['total_ms'],gpu_memory_bytes=0))
    with tempfile.TemporaryDirectory(dir=work_dir) as td:
        pp=Path(td)/'predictions.json';pp.write_text(json.dumps(coco_preds))
        seg_metrics=segmentation(str(coco_path),str(pp))
    return dict(dataset_version=meta['dataset_version'],split=actual_split,threshold=threshold,
                segmentation=seg_metrics,recognition=recognition(roi_rows),
                pipeline=pipeline(pipe_rows) if pipe_rows else None)
