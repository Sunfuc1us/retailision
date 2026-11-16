import json,os,signal,subprocess,sys
from pathlib import Path
from uuid import uuid4
from retailvision.paths import PROJECT_ROOT
from retailvision.training.options import TrainingRequest

class Jobs:
    def __init__(self):self.processes={}
    def start(self, req: TrainingRequest) -> str:
        for p in self.processes.values():
            if p.poll() is None:raise ValueError('a training job is already running')
        req.check_paths()
        config=req.config
        eid=uuid4().hex
        state=Path('work_dirs')/eid;state.mkdir(parents=True)
        from retailvision.training.experiment import now
        (state/'experiment.json').write_text(json.dumps(dict(experiment_id=eid,task=req.task,status='queued',start_time=None,end_time=None,queued_at=now()),indent=2))
        log=Path('logs')/f'{eid}.log';log.parent.mkdir(exist_ok=True)
        env='retailvision-reid' if req.task=='recognition' else 'retailvision-seg'
        command=[os.environ.get('CONDA_EXE','conda'),'run','--no-capture-output','-n',env,'python','tools/train.py','--task',req.task,'--config',config,'--dataset',req.dataset,'--experiment-id',eid]
        if req.model:command+=['--model',req.model]
        if req.task=='both':command+=['--recognition-config',req.recognition_config]
        if req.resume:command+=['--resume',req.resume]
        if req.no_validation:command+=['--no-validation']
        with log.open('w') as f:self.processes[eid]=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,cwd=PROJECT_ROOT)
        return eid
    def stop(self,eid: str) -> dict:
        p=self.processes.get(eid)
        if p is None:raise KeyError(eid)
        if p.poll() is None:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=15)
        path=Path('work_dirs')/eid/'experiment.json'
        if path.exists():
            data=json.loads(path.read_text());data['status']='stopped'
            from retailvision.training.experiment import now
            data['end_time']=now();path.write_text(json.dumps(data,indent=2))
        return dict(experiment_id=eid,status='stopped')
