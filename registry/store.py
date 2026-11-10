import hashlib,json,shutil
from pathlib import Path
from datetime import datetime,timezone

class Registry:
    def __init__(self, root: str = 'models'):self.root=Path(root)
    def register(self, task: str, version: str, checkpoint: str, config: str, metadata: dict) -> Path:
        if task not in ('segmentation','recognition') or Path(version).name!=version:raise ValueError('invalid model path')
        required={'model_name','architecture','training_dataset','license','evaluation_metrics'}
        if not required<=metadata.keys():raise ValueError('incomplete manifest')
        out=self.root/task/version;out.mkdir(parents=True,exist_ok=False)
        shutil.copy2(checkpoint,out/'model.pth');shutil.copy2(config,out/Path(config).name)
        manifest=dict(**metadata,model_version=version,checkpoint='model.pth',training_config=Path(config).name,sha256=hashlib.sha256((out/'model.pth').read_bytes()).hexdigest(),created_at=datetime.now(timezone.utc).isoformat())
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2));return out
    def list(self) -> list[dict]:return [json.loads(p.read_text()) for p in sorted(self.root.glob('*/*/manifest.json'))]
