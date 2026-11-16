import json
from datetime import datetime,timezone
from pathlib import Path
from uuid import uuid4

def now() -> str:return datetime.now(timezone.utc).isoformat()

class Experiment:
    def __init__(self, task: str, model: str, config: str, dataset: str, experiment_id: str | None = None):
        self.id=experiment_id or uuid4().hex;self.path=Path('work_dirs')/self.id
        self.path.mkdir(parents=True,exist_ok=bool(experiment_id))
        manifest=Path(dataset)/('manifest.json' if task in ('segmentation','both') else 'metadata.json')
        source_manifest=json.loads(manifest.read_text())
        self.data=dict(fixture=source_manifest.get('fixture',False),experiment_id=self.id,task=task,model_name=model,model_version='unregistered',dataset_version=json.loads(manifest.read_text())['dataset_version'],config=Path(config).read_text(),hyperparameters={},start_time=now(),end_time=None,status='running',training_metrics={},validation_metrics={},best_checkpoint=None)
        self.data['dataset']=str(Path(dataset).resolve())
        for key in ('validation_scope','independent_validation','split_unit'):
            if key in source_manifest:self.data[key]=source_manifest[key]
        self.save()
    def save(self) -> None:
        tmp=self.path/'experiment.tmp';tmp.write_text(json.dumps(self.data,indent=2));tmp.replace(self.path/'experiment.json')
    def finish(self, status: str, **values) -> None:
        self.data.update(status=status,end_time=now(),**values);self.save()
