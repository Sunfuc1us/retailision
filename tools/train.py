import _bootstrap
import argparse,traceback
from training.common.logging import configure_logging
from training.common.experiment import Experiment
p=argparse.ArgumentParser();p.add_argument('--task',choices=['segmentation','recognition'],required=True);p.add_argument('--model',required=True);p.add_argument('--config',required=True);p.add_argument('--dataset',required=True);p.add_argument('--resume');p.add_argument('--experiment-id')
a=p.parse_args();configure_logging('logs/training.log');expected={'segmentation':'rtmdet-ins-s','recognition':'dinov2-vitb14'}
if a.model!=expected[a.task]:p.error('unsupported task/model combination')
e=Experiment(a.task,a.model,a.config,a.dataset,a.experiment_id)
try:
 if a.task=='segmentation':from training.segmentation.trainer import train
 else:from training.recognition.trainer import train
 e.finish('completed',**train(a.config,a.dataset,e,a.resume))
except BaseException:
 e.finish('failed',error=traceback.format_exc());raise
