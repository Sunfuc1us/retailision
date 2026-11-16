import _bootstrap
import argparse,traceback
from retailvision.logging import configure_logging
from retailvision.training.experiment import Experiment
p=argparse.ArgumentParser()
p.add_argument('--task',choices=['both','segmentation','recognition'],default='both')
p.add_argument('--model');p.add_argument('--config');p.add_argument('--dataset',required=True)
p.add_argument('--recognition-config',default='configs/recognition/dinov2_b14.py')
p.add_argument('--resume');p.add_argument('--experiment-id')
p.add_argument('--no-validation',action='store_true',help='Recognition-only experiment on explicitly unsplit data; save last weights without validation or best selection')
a=p.parse_args();configure_logging('logs/training.log')
from retailvision.training.options import TrainingRequest
try:
 options=TrainingRequest(**{k:v for k,v in vars(a).items() if k!='experiment_id'})
 options.check_paths()
except ValueError as error:p.error(str(error))
for key,value in options.model_dump().items():setattr(a,key,value)
e=Experiment(a.task,a.model,a.config,a.dataset,a.experiment_id)
try:
 if a.task=='both':
  from retailvision.training.joint import train
  result=train(a.config,a.dataset,e,a.recognition_config)
 elif a.task=='segmentation':
  from retailvision.training.segmentation import train
  result=train(a.config,a.dataset,e,a.resume)
 else:
  from retailvision.training.recognition import train
  result=train(a.config,a.dataset,e,a.resume,no_validation=a.no_validation)
 e.finish('completed',**result)
except BaseException:
 e.finish('failed',error=traceback.format_exc());raise
