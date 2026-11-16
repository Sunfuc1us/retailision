"""Train both stages against one versioned train/validation split."""
import json,os,runpy,subprocess,sys
from pathlib import Path
from retailvision.paths import PROJECT_ROOT,project_path
from retailvision.training.recognition import training_rows


def train(config: str, dataset: str, experiment, recognition_config: str) -> dict:
    root=project_path(dataset)
    manifest=json.loads((root/'manifest.json').read_text())
    recognition=json.loads((root/'recognition/metadata.json').read_text())
    segmentation=json.loads((root/'segmentation/manifest.json').read_text())
    if manifest.get('no_split'):raise ValueError('joint training requires independent train/val capture groups; unsplit data cannot be used for joint validation')
    if not (manifest['dataset_version']==recognition['dataset_version']==segmentation['dataset_version']):
        raise ValueError('segmentation and recognition must use the same dataset version')
    training_rows(recognition,runpy.run_path(recognition_config)['config']['batch_size'])
    split_ids={}
    for split in ('train','val'):
        images=json.loads((root/f'segmentation/annotations/{split}.json').read_text())['images']
        if not images:raise ValueError(f'joint training requires nonempty {split} images')
        split_ids[split]={im['id'] for im in images}
        if manifest.get('split_unit')=='image':
            held_out=set(manifest['validation_images'])
            if any((im['file_name'] in held_out)!=(split=='val') for im in images):
                raise ValueError('image does not match the shared train/val image split')
        elif any(manifest['groups'].get(im.get('capture_group'))!=split for im in images):
            raise ValueError('image capture group does not match the shared train/val split')
        if {x['image_id'] for x in recognition['items'] if x['split']==split}-{im['id'] for im in images}:
            raise ValueError('recognition instances do not match the segmentation split')
    if split_ids['train']&split_ids['val']:raise ValueError('training and validation image IDs overlap')
    env=dict(os.environ,PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    conda=os.environ.get('CONDA_EXE','conda')
    stages={}
    experiment.data.update(recognition_config=Path(recognition_config).read_text(),stages=stages,validation_status='pending',
        validation_scope=manifest.get('validation_scope','independent_capture_groups'),independent_validation=manifest.get('independent_validation',True))
    experiment.save()
    for task,model,cfg,environment in (
        ('segmentation','rtmdet-ins-s',config,'retailvision-seg'),
        ('recognition','dinov2-vitb14',recognition_config,'retailvision-reid')):
        child_id=f'{experiment.id}_{task}'
        stages[task]=dict(experiment_id=child_id,status='running');experiment.save()
        command=[conda,'run','--no-capture-output','-n',environment,'python','tools/train.py',
            '--task',task,'--model',model,'--config',str(cfg),'--dataset',str(root/task),'--experiment-id',child_id]
        try:
            subprocess.run(command,cwd=PROJECT_ROOT,env=env,check=True)
            result=json.loads((PROJECT_ROOT/'work_dirs'/child_id/'experiment.json').read_text())
            if result['status']!='completed':raise RuntimeError(f'{task} training did not complete')
            if result.get('validation_status')!='completed':raise RuntimeError(f'{task} validation produced no metrics')
        except BaseException:
            stages[task]['status']='failed';experiment.data['validation_status']='failed';experiment.save();raise
        stages[task]=result;experiment.save()
    gallery=experiment.path/'gallery'
    subprocess.run([conda,'run','--no-capture-output','-n','retailvision-reid','python','tools/build_gallery.py',
        '--dataset',str(root/'recognition'),'--config',str(recognition_config),'--projection',stages['recognition']['projection_checkpoint'],
        '--output',str(gallery)],cwd=PROJECT_ROOT,env=env,check=True)
    return dict(stages=stages,gallery=str(gallery),calibration_required=True,validation_status='completed',
        validation_metrics={task:result['validation_metrics'] for task,result in stages.items()})
