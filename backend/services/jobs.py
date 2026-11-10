import json,os,signal,subprocess,sys
from pathlib import Path
from uuid import uuid4
from project_paths import PROJECT_ROOT
from backend.schemas.requests import TrainingRequest

class Jobs:
    def __init__(self):self.processes={}
    def start(self, req: TrainingRequest) -> str:
        for p in self.processes.values():
            if p.poll() is None:raise ValueError('a training job is already running')
        for p in (req.config,req.dataset):
            if not Path(p).exists():raise ValueError(f'missing path: {p}')
        eid=uuid4().hex
        state=Path('work_dirs')/eid;state.mkdir(parents=True)
        from training.common.experiment import now
        (state/'experiment.json').write_text(json.dumps(dict(experiment_id=eid,task=req.task,status='queued',start_time=None,end_time=None,queued_at=now()),indent=2))
        log=Path('logs')/f'{eid}.log';log.parent.mkdir(exist_ok=True)
        env='retailvision-seg' if req.task=='segmentation' else 'retailvision-reid'
        command=['conda','run','--no-capture-output','-n',env,'python','tools/train.py','--task',req.task,'--model',req.model,'--config',req.config,'--dataset',req.dataset,'--experiment-id',eid]
        if req.resume:command+=['--resume',req.resume]
        with log.open('w') as f:self.processes[eid]=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,cwd=PROJECT_ROOT)
        return eid
    def stop(self,eid: str) -> dict:
        p=self.processes.get(eid)
        if p is None:raise KeyError(eid)
        if p.poll() is None:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=15)
        path=Path('work_dirs')/eid/'experiment.json'
        if path.exists():
            data=json.loads(path.read_text());data['status']='stopped'
            from training.common.experiment import now
            data['end_time']=now();path.write_text(json.dumps(data,indent=2))
        return dict(experiment_id=eid,status='stopped')
