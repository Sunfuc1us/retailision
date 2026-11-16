"""Local inference model choices: weights travel with their config, gallery and policy."""
import json
import os
from dataclasses import dataclass
from pathlib import Path
from retailvision.paths import project_path


@dataclass(frozen=True)
class ModelProfile:
    id: str
    name: str
    config: Path

    def settings(self) -> dict:
        return json.loads(self.config.read_text())


def profiles() -> list[ModelProfile]:
    entries=json.loads(project_path('configs/inference/model_profiles.json').read_text())
    choices=[ModelProfile(x['id'],x['name'],project_path(x['config']).resolve()) for x in entries]
    configured=os.getenv('RETAILVISION_CONFIG')
    if configured:
        path=project_path(configured).resolve()
        if all(x.config!=path for x in choices):
            choices.append(ModelProfile('configured','启动配置 · '+path.stem,path))
    return choices


def resolve_profile(model_id: str | None = None) -> ModelProfile:
    choices=profiles()
    if model_id is not None:
        for choice in choices:
            if choice.id==model_id:return choice
        raise ValueError(f'unknown inference model: {model_id}')
    configured=os.getenv('RETAILVISION_CONFIG')
    if not configured:raise RuntimeError('Local models, SKU gallery and validation calibration must be configured')
    path=project_path(configured).resolve()
    return next(x for x in choices if x.config==path)


def available_backends(profile: ModelProfile) -> list[str]:
    if not profile.config.is_file():return []
    from retailvision.artifacts.config import preflight
    try:
        import torch
        if not torch.cuda.is_available():return []
    except (ImportError,OSError):return []
    available=[]
    for name in ('pytorch','onnxruntime','tensorrt'):
        try:
            preflight(profile.config,name)
            if name!='pytorch':
                module=__import__(name)
                if name=='onnxruntime' and 'CUDAExecutionProvider' not in module.get_available_providers():continue
        except (ImportError,OSError,ValueError,KeyError):continue
        available.append(name)
    return available


def describe(profile: ModelProfile) -> dict:
    config=profile.settings() if profile.config.is_file() else {}
    rec=config.get('recognition',{})
    return dict(id=profile.id,name=profile.name,config=str(profile.config),
                checkpoint=str(project_path(config['segmentation']['checkpoint'])) if config.get('segmentation',{}).get('checkpoint') else None,
                recognition_version=config.get('model_version',Path(rec.get('checkpoint','')).parent.name),
                recognition_checkpoint=str(project_path(rec['checkpoint'])) if rec.get('checkpoint') else None,
                fixture=config.get('fixture',False),backends=available_backends(profile))


def result_roots() -> list[Path]:
    roots=[project_path('results')]
    for profile in profiles():
        if profile.config.is_file():
            output=profile.settings().get('results')
            if output:
                path=project_path(output)
                if path not in roots:roots.append(path)
    return roots
