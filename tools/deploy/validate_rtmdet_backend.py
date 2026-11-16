"""Compare exported RTMDet-Ins boxes and instance masks with the source checkpoint."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
MODEL='third_party/mmdetection/configs/rtmdet/rtmdet-ins_s_8xb32-300e_coco.py'
CHECKPOINT='models/segmentation/rtmdet_ins_s_coco_v001/model.pth'
ORT_CONFIG='configs/deployment/rtmdet_ins_onnxruntime_static_640.py'
TRT_CONFIG='configs/deployment/rtmdet_ins_tensorrt_fp16_static_640.py'
ORT_MODEL='models/segmentation/rtmdet_ins_s_coco_v001/deployment/onnx/end2end.onnx'
TRT_MODEL='models/segmentation/rtmdet_ins_s_coco_v001/deployment/tensorrt/end2end.engine'

def load(name):
    if name.startswith('pytorch'):
        from retailvision.inference.rtmdet import RTMDet
        return RTMDet(MODEL,CHECKPOINT,'fp16' if name.endswith('fp16') else 'fp32')
    from retailvision.inference.backends.rtmdet_mmdeploy import RTMDetMMDeploy
    if name=='onnxruntime':return RTMDetMMDeploy(MODEL,ORT_CONFIG,ORT_MODEL,name)
    return RTMDetMMDeploy(MODEL,TRT_CONFIG,TRT_MODEL,name)

def bbox_iou(a,b):
    a=np.asarray(a);b=np.asarray(b)
    wh=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))
    inter=wh[0]*wh[1]
    area_a=max(0,a[2]-a[0])*max(0,a[3]-a[1]);area_b=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return float(inter/max(area_a+area_b-inter,1e-8))

def compare_instances(reference, candidate):
    from scipy.optimize import linear_sum_assignment
    overlaps=np.array([[bbox_iou(a.bbox,b.bbox) if a.label==b.label else 0.
                        for b in candidate] for a in reference],dtype=np.float64)
    pairs=[];used_reference=set();used_candidate=set()
    if reference and candidate:
        indices=linear_sum_assignment(-overlaps)
        for i,j in zip(*indices):
            if overlaps[i,j]<.5:continue
            a,b=reference[i],candidate[j]
            ma,mb=a.mask.astype(bool),b.mask.astype(bool);union=np.logical_or(ma,mb).sum()
            used_reference.add(int(i));used_candidate.add(int(j))
            pairs.append(dict(box_max_abs=float(np.max(np.abs(np.asarray(a.bbox)-np.asarray(b.bbox)))),
                bbox_iou=float(overlaps[i,j]),mask_iou=float(np.logical_and(ma,mb).sum()/union) if union else 1.,
                reference_box=a.bbox,backend_box=b.bbox,reference_label=a.label,backend_label=b.label,
                reference_score=a.score,backend_score=b.score,
                reference_mask_shape=list(ma.shape),backend_mask_shape=list(mb.shape)))
    return dict(reference_count=len(reference),backend_count=len(candidate),matched=pairs,
                unmatched_reference=len(reference)-len(used_reference),unmatched_backend=len(candidate)-len(used_candidate))

def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['pytorch_fp32','pytorch_fp16','onnxruntime','tensorrt'],required=True)
    p.add_argument('--output');p.add_argument('--threshold',type=float,default=.3);args=p.parse_args()
    import torch
    ref=load('pytorch_fp32');run=load(args.backend)
    ref.load();run.load();paths=sorted((ROOT/'datasets/retailtest_smoke_v001/segmentation/images').glob('*'))
    rows=[]
    for path in paths:
        image=Image.open(path).convert('RGB')
        with torch.inference_mode(): a=ref.predict(image,args.threshold);b=run.predict(image,args.threshold)
        rows.append(dict(image=str(path),**compare_instances(a,b)))
    ref.unload();run.unload()
    report=dict(architecture='RTMDet-Ins-s COCO pretrained',backend=args.backend,score_threshold=args.threshold,
        images=len(rows),mask_output_present=all('mask_iou' in pair for row in rows for pair in row['matched']),results=rows)
    output=Path(args.output or f'work_dirs/deployment/rtmdet_{args.backend}_parity.json');output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
