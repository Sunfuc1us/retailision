"""Repeated end-to-end model latency and parity benchmark on the fixed smoke fixture."""
import argparse
import json
import subprocess
import threading
import time
import sys
from datetime import datetime,timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
SEG_CONFIG='third_party/mmdetection/configs/rtmdet/rtmdet-ins_s_8xb32-300e_coco.py'
SEG_CKPT='models/segmentation/rtmdet_ins_s_coco_v001/model.pth'
DINO_CKPT='models/recognition/dinov2_b14_v001/model.pth'
ORT_DEPLOY='configs/deployment/rtmdet_ins_onnxruntime_static_640.py'
TRT_DEPLOY='configs/deployment/rtmdet_ins_tensorrt_fp16_static_640.py'
SEG_ONNX='models/segmentation/rtmdet_ins_s_coco_v001/deployment/onnx/end2end.onnx'
SEG_ENGINE='models/segmentation/rtmdet_ins_s_coco_v001/deployment/tensorrt/end2end.engine'
DINO_ONNX='models/recognition/dinov2_b14_v001/deployment/model.onnx'
DINO_ENGINE='models/recognition/dinov2_b14_v001/deployment/model_fp16.engine'

class GPUUtilSampler:
    def __init__(self): self.values=[]; self.memory=[]; self.stop=threading.Event(); self.thread=None
    def __enter__(self):
        def sample():
            while not self.stop.is_set():
                p=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True)
                try:
                    fields=p.stdout.strip().splitlines()[0].split(',')
                    self.values.append(float(fields[0]));self.memory.append(int(fields[1]))
                except (ValueError,IndexError): pass
                self.stop.wait(1)
        self.thread=threading.Thread(target=sample,daemon=True);self.thread.start();return self
    def __exit__(self,*_): self.stop.set();self.thread.join()

def percentiles(values):
    a=np.asarray(values,dtype=np.float64)
    return dict(mean_ms=float(a.mean()),p50_ms=float(np.percentile(a,50)),p95_ms=float(np.percentile(a,95)))

def make_models(backend):
    from retailvision.inference.dino import DINOv2
    if backend.startswith('pytorch'):
        precision='fp16' if backend.endswith('fp16') else 'fp32'
        from retailvision.inference.rtmdet import RTMDet
        return RTMDet(SEG_CONFIG,SEG_CKPT,precision),DINOv2('third_party/dinov2',DINO_CKPT,precision=precision)
    from retailvision.inference.backends.rtmdet_mmdeploy import RTMDetMMDeploy
    from retailvision.inference.backends.dinov2_engine import DINOv2Engine
    if backend=='onnxruntime':
        return RTMDetMMDeploy(SEG_CONFIG,ORT_DEPLOY,SEG_ONNX,backend),DINOv2Engine(DINO_ONNX,backend)
    return RTMDetMMDeploy(SEG_CONFIG,TRT_DEPLOY,SEG_ENGINE,backend),DINOv2Engine(DINO_ENGINE,backend,input_dtype='float16')

def serialized_size(backend):
    if backend.startswith('pytorch'): return dict(segmentation=(ROOT/SEG_CKPT).stat().st_size,recognition=(ROOT/DINO_CKPT).stat().st_size)
    seg=SEG_ONNX if backend=='onnxruntime' else SEG_ENGINE
    dino=DINO_ONNX if backend=='onnxruntime' else DINO_ENGINE
    return dict(segmentation=(ROOT/seg).stat().st_size,recognition=(ROOT/dino).stat().st_size)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--warmup',type=int,default=50)
    p.add_argument('--repeats',type=int,default=200)
    p.add_argument('--output',default='docs/INFERENCE_BENCHMARK.md')
    p.add_argument('--json-output',default='work_dirs/deployment/benchmark.json')
    p.add_argument('--threshold',type=float,default=.3)
    args=p.parse_args()
    if args.warmup<1 or args.repeats<20: p.error('use at least 1 warm-up and 20 measured repeats (recommended 50/200)')
    import torch
    from tools.deploy.validate_rtmdet_backend import compare_instances
    image_paths=sorted((ROOT/'datasets/retailtest_smoke_v001/segmentation/images').glob('*'))
    roi_paths=sorted((ROOT/'datasets/retailtest_smoke_v001/recognition/all').rglob('*.png'))
    images=[Image.open(path).convert('RGB') for path in image_paths]
    rois=[Image.open(path).convert('RGB') for path in roi_paths]
    if len(images)!=3 or len(rois)!=9: p.error(f'expected 3 fixture images and 9 ROI crops, found {len(images)} and {len(rois)}')
    backends=['pytorch_fp32','pytorch_fp16','onnxruntime','tensorrt']
    reports={};baseline_seg=baseline_dino=None
    for backend in backends:
        seg,rec=make_models(backend);torch.cuda.reset_peak_memory_stats()
        start=perf_counter();seg.load();torch.cuda.synchronize();seg_load_ms=(perf_counter()-start)*1000
        start=perf_counter();rec.load();torch.cuda.synchronize();rec_load_ms=(perf_counter()-start)*1000
        seg_times=[];dino_times=[];predictions=[];embeddings=[]
        torch.cuda.reset_peak_memory_stats()
        with GPUUtilSampler() as util:
            for i in range(args.warmup):
                seg.predict(images[i%len(images)],args.threshold)
                rec.extract_embedding(rois[i%len(rois)])
            torch.cuda.synchronize()
            for i in range(args.repeats):
                image=images[i%len(images)];roi=rois[i%len(rois)]
                t=perf_counter();pred=seg.predict(image,args.threshold);torch.cuda.synchronize();seg_times.append((perf_counter()-t)*1000)
                t=perf_counter();emb=rec.extract_embedding(roi);torch.cuda.synchronize();dino_times.append((perf_counter()-t)*1000)
                if i<len(images): predictions.append(pred)
                if i<len(rois): embeddings.append(emb)
        gpu_util=float(np.mean(util.values)) if util.values else None
        gpu_mem=max(util.memory) if util.memory else None
        reports[backend]=dict(load_ms=dict(segmentation=seg_load_ms,recognition=rec_load_ms),segmentation=percentiles(seg_times),recognition=percentiles(dino_times),
            peak_gpu_memory_mib=gpu_mem,torch_allocator_peak_mib=int(torch.cuda.max_memory_allocated()/1024**2),
            gpu_utilization_mean_percent=gpu_util,file_size_bytes=serialized_size(backend),
            detections=[len(x) for x in predictions])
        if backend=='pytorch_fp32': baseline_seg,baseline_dino=predictions,embeddings
        else:
            seg_parity=[]
            for ref,got in zip(baseline_seg or [],predictions):
                seg_parity.append(compare_instances(ref,got))
            dino_parity=[]
            for a,b in zip(baseline_dino or [],embeddings):
                dino_parity.append(dict(max_abs=float(np.max(np.abs(a-b))),cosine=float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))))
            reports[backend]['rtmdet_output_comparison']=seg_parity
            reports[backend]['dinov2_output_comparison']=dino_parity
        seg.unload();rec.unload();torch.cuda.empty_cache()
        print(f'{backend}: segmentation mean {reports[backend]["segmentation"]["mean_ms"]:.2f} ms; recognition mean {reports[backend]["recognition"]["mean_ms"]:.2f} ms',flush=True)
    out=Path(args.json_output);out.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(protocol=dict(warmup=args.warmup,repeats=args.repeats,images=len(images),rois=len(rois),score_threshold=args.threshold),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        environment=dict(torch=torch.__version__,pytorch_cuda_build=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
                         onnx=version('onnx'),onnxruntime=version('onnxruntime-gpu'),tensorrt=version('tensorrt-cu12'),
                         standalone_cuda_runtime=version('nvidia-cuda-runtime-cu12')),
        hardware=torch.cuda.get_device_name(),results=reports)
    out.write_text(json.dumps(payload,indent=2)+'\n')
    md=['# Inference benchmark','','Engineering validation on the fixed 3-image / 9-ROI fixture; not retail accuracy.','',
        f"Protocol: {args.warmup} warm-up calls, {args.repeats} measured calls per model/backend; detection threshold {args.threshold}. GPU: {payload['hardware']}.",'',
        '| Model / backend | Load (ms) | Mean (ms) | P50 (ms) | P95 (ms) | Peak GPU (MiB) | GPU util avg (%) | Size (MiB) | Output parity |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for name,row in reports.items():
        for kind,label in [('segmentation','RTMDet-Ins'),('recognition','DINOv2')]:
            t=row[kind];parity='FP32 baseline'
            if name!='pytorch_fp32':
                if kind=='recognition':
                    comparisons=row['dinov2_output_comparison']
                    parity=f"min cosine {min(x['cosine'] for x in comparisons):.6f}; max abs {max(x['max_abs'] for x in comparisons):.6f}"
                else:
                    comparisons=row['rtmdet_output_comparison'];pairs=[p for image in comparisons for p in image['matched']]
                    parity=f"matched {len(pairs)}/{sum(x['reference_count'] for x in comparisons)}; mean box IoU {np.mean([x['bbox_iou'] for x in pairs]):.5f}; max box delta {max(x['box_max_abs'] for x in pairs):.2f}px; mean mask IoU {np.mean([x['mask_iou'] for x in pairs]):.5f}"
            md.append(f"| {label} {name} | {row['load_ms'][kind]:.1f} | {t['mean_ms']:.2f} | {t['p50_ms']:.2f} | {t['p95_ms']:.2f} | {row['peak_gpu_memory_mib']} | {row['gpu_utilization_mean_percent'] or 0:.1f} | {row['file_size_bytes'][kind]/1024**2:.1f} | {parity} |")
    md+=['', 'Latency includes preprocessing, CUDA execution, output transfer, and model postprocessing. Models remain loaded during warm-up and measurement. GPU memory/utilization are process-external samples while both models are resident; memory also includes desktop/other GPU processes. Load times include source imports and model construction.', '', 'Instance matching uses same category and maximum global Box IoU, retaining pairs with IoU ≥ 0.5; unmatched instances remain explicit. NMS can select different overlapping detections near score ties.', '',f'Full Box/Mask and embedding parity details: `{out}`.','']
    Path(args.output).write_text('\n'.join(md))
    print(json.dumps(payload,indent=2))

if __name__=='__main__':main()
