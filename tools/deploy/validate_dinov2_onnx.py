"""Compare real DINOv2 checkpoint embeddings with ONNX Runtime CUDA outputs."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='models/recognition/dinov2_b14_v001/model.pth')
    parser.add_argument('--onnx', default='models/recognition/dinov2_b14_v001/deployment/model.onnx')
    parser.add_argument('--rois', default='datasets/retailtest_smoke_v001/recognition/all')
    parser.add_argument('--output', default='work_dirs/deployment/dinov2_onnx_parity.json')
    parser.add_argument('--backend', choices=('onnxruntime','tensorrt'), default='onnxruntime')
    parser.add_argument('--engine', default='models/recognition/dinov2_b14_v001/deployment/model_fp16.engine')
    args = parser.parse_args()
    import torch
    from torchvision import transforms
    from retailvision.inference.dino import DINOv2
    crops = sorted(Path(args.rois).rglob('*.png'))
    if not crops:
        parser.error(f'no ROI crops in {args.rois}')
    model = DINOv2('third_party/dinov2', args.checkpoint)
    load_start = time.perf_counter(); model.load(); torch.cuda.synchronize(); load_ms = (time.perf_counter()-load_start)*1e3
    from retailvision.inference.backends.onnx_cuda import preload_cuda12
    preload_cuda12()
    import onnxruntime as ort
    ort_session = ort.InferenceSession(args.onnx, providers=['CUDAExecutionProvider'])
    if ort_session.get_providers()[0] != 'CUDAExecutionProvider':
        raise RuntimeError(f'CUDA provider not selected: {ort_session.get_providers()}')
    if args.backend=='onnxruntime':
        session=ort_session;provider=session.get_providers()[0]
    else:
        from retailvision.inference.backends.dinov2_engine import DINOv2Engine
        session=DINOv2Engine(args.engine,'tensorrt',input_dtype='float32');session.load();provider='TensorRT CUDA engine'
    tx = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    rows=[]
    for path in crops:
        image=Image.open(path).convert('RGB'); tensor=tx(image).unsqueeze(0)
        with torch.inference_mode():
            ref=model.model(tensor.cuda()); ref=torch.nn.functional.normalize(ref.float(),dim=-1)[0].cpu().numpy()
        ort_got=ort_session.run(None,{ort_session.get_inputs()[0].name:tensor.numpy()})[0][0]
        if args.backend=='onnxruntime': got=ort_got
        else: got=session.extract_embedding(image)
        row=dict(roi=str(path),max_abs=float(np.max(np.abs(ref-got))),
                         cosine=float(np.dot(ref,got)/(np.linalg.norm(ref)*np.linalg.norm(got))),
                         ort_max_abs=float(np.max(np.abs(ref-ort_got))),
                         ort_self_cosine=float(np.dot(ref,ort_got)/(np.linalg.norm(ref)*np.linalg.norm(ort_got))),
                         ort_cosine=float(np.dot(ort_got,got)/(np.linalg.norm(ort_got)*np.linalg.norm(got))))
        rows.append(row)
    report=dict(model='DINOv2 ViT-B/14 pretrained checkpoint',backend=args.backend,
        provider=provider,roi_count=len(rows),torch_load_ms=load_ms,
        max_abs_error=max(x['max_abs'] for x in rows),min_cosine=min(x['cosine'] for x in rows),
        mean_cosine=float(np.mean([x['cosine'] for x in rows])),
        mean_ort_cosine=float(np.mean([x['ort_cosine'] for x in rows])),results=rows)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    model.unload()
    if args.backend=='tensorrt':session.unload()

if __name__=='__main__': main()
