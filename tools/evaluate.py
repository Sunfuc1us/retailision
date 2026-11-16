import _bootstrap
import argparse,json
from pathlib import Path
from retailvision import evaluation as metrics
from retailvision.inference.rejection import calibrate
p=argparse.ArgumentParser()
p.add_argument('--task',choices=['segmentation','recognition','pipeline','calibrate','dataset'],required=True)
p.add_argument('--input')
p.add_argument('--ground-truth')
p.add_argument('--config')
p.add_argument('--dataset')
p.add_argument('--split',default='val')
p.add_argument('--threshold',type=float,default=.3)
p.add_argument('--backend',choices=['pytorch','onnxruntime','tensorrt'],default='pytorch')
p.add_argument('--output',required=True)
p.add_argument('--dataset-version')
p.add_argument('--model-version')
p.add_argument('--experiment-id',default='manual')
p.add_argument('--gallery-version')
p.add_argument('--max-far',type=float,default=.01)
a=p.parse_args()
if a.task=='dataset':
    if not a.config or not a.dataset:p.error('--task dataset requires --config and --dataset')
    from retailvision.inference.factory import load_pipeline
    pipe=load_pipeline(a.config,a.backend)
    r=metrics.evaluate_dataset(pipe,a.dataset,split=a.split,threshold=a.threshold)
    ds_ver=a.dataset_version or r['dataset_version']
    mod_ver=a.model_version or getattr(pipe.policy,'model_version','unregistered')
else:
    if not a.input or not a.dataset_version or not a.model_version:
        p.error('--input, --dataset-version and --model-version are required')
    rows=json.loads(Path(a.input).read_text())
    ds_ver=a.dataset_version;mod_ver=a.model_version
    if a.task=='segmentation':r=metrics.segmentation(a.ground_truth,a.input)
    elif a.task=='calibrate':r=calibrate(rows,a.dataset_version,a.gallery_version,a.model_version,a.max_far)
    else:r=getattr(metrics,a.task)(rows)
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
with out.open('x') as f:json.dump({**r,'dataset_version':ds_ver,'model_version':mod_ver,'experiment_id':a.experiment_id,'evaluation_config':vars(a)},f,indent=2)
