import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from retailvision.training.options import TrainingRequest
from retailvision.training.recognition import training_rows
from retailvision.training.joint import train as train_joint
from retailvision.data.prepare import prepare


def metadata(split=True):
    return dict(no_split=not split,items=[dict(sku_id=sku,split=part)
        for sku in ('A','B') for part in (['train','train','val'] if split else ['all']*3)])


def test_recognition_validation_is_required_by_default():
    with pytest.raises(ValueError,match='independent validation'):
        training_rows(metadata(False),4)
    train,val=training_rows(metadata(),4)
    assert len(train)==4 and len(val)==2


def test_explicit_unsplit_training_has_no_validation_samples():
    train,val=training_rows(metadata(False),4,no_validation=True)
    assert len(train)==6 and val==[]
    with pytest.raises(ValueError,match='explicitly unsplit'):
        training_rows(metadata(),4,no_validation=True)


@pytest.mark.parametrize('batch_size',[2,3,6])
def test_recognition_batches_require_positive_and_negative_examples(batch_size):
    with pytest.raises(ValueError,match='batch_size'):
        training_rows(metadata(),batch_size)


def test_validation_sku_requires_training_references():
    meta=metadata();meta['items'].append(dict(sku_id='UNKNOWN',split='val'))
    with pytest.raises(ValueError,match='no training references'):
        training_rows(meta,4)


def experiment(tmp_path):
    return SimpleNamespace(id='test_joint',path=tmp_path/'experiment',data={},save=lambda:None)


def test_joint_rejects_unsplit_data_before_launch(data,tmp_path,monkeypatch):
    prepare(**data,no_split=True)
    import retailvision.training.joint as joint
    monkeypatch.setattr(joint.subprocess,'run',lambda *a,**kw:pytest.fail('must not start training'))
    with pytest.raises(ValueError,match='independent train/val'):
        train_joint('configs/segmentation/fixture_smoke.py',data['output'],experiment(tmp_path),'configs/recognition/fixture_smoke.py')


def test_joint_checks_shared_dataset_version_before_launch(data,tmp_path,monkeypatch):
    prepare(**data)
    path=Path(data['output'])/'recognition/metadata.json'
    meta=json.loads(path.read_text());meta['dataset_version']='other';path.write_text(json.dumps(meta))
    import retailvision.training.joint as joint
    monkeypatch.setattr(joint.subprocess,'run',lambda *a,**kw:pytest.fail('must not start training'))
    with pytest.raises(ValueError,match='same dataset version'):
        train_joint('configs/segmentation/fixture_smoke.py',data['output'],experiment(tmp_path),'configs/recognition/fixture_smoke.py')


def test_joint_rejects_image_holdout_mismatch_before_launch(data,tmp_path,monkeypatch):
    prepare(**data,validation_images=['0.png'])
    path=Path(data['output'])/'segmentation/annotations/val.json'
    annotations=json.loads(path.read_text());annotations['images'][0]['file_name']='1.png'
    path.write_text(json.dumps(annotations))
    import retailvision.training.joint as joint
    monkeypatch.setattr(joint.subprocess,'run',lambda *a,**kw:pytest.fail('must not start training'))
    with pytest.raises(ValueError,match='shared train/val image split'):
        train_joint('configs/segmentation/fixture_smoke.py',data['output'],experiment(tmp_path),'configs/recognition/fixture_smoke.py')


def test_joint_stops_before_recognition_when_segmentation_fails(data,tmp_path,monkeypatch):
    import subprocess
    import retailvision.training.joint as joint
    prepare(**data);calls=[]
    def fail(command,**kwargs):
        calls.append(command);raise subprocess.CalledProcessError(1,command)
    monkeypatch.setattr(joint.subprocess,'run',fail)
    exp=experiment(tmp_path)
    with pytest.raises(subprocess.CalledProcessError):
        train_joint('configs/segmentation/fixture_smoke.py',data['output'],exp,'configs/recognition/fixture_smoke.py')
    assert len(calls)==1 and calls[0][calls[0].index('--task')+1]=='segmentation'
    assert exp.data['stages']['segmentation']['status']=='failed'


def test_training_api_defaults_to_joint_and_keeps_single_stage_options():
    req=TrainingRequest(dataset='datasets/prepared')
    assert req.task=='both' and req.no_validation is False
    assert TrainingRequest(task='recognition',model='dinov2-vitb14',config='recognition.py',dataset='data').task=='recognition'


def test_training_job_passes_both_configs_and_uses_segmentation_environment(tmp_path,monkeypatch):
    from retailvision.training.jobs import Jobs
    import retailvision.training.jobs as job_module
    monkeypatch.chdir(tmp_path);monkeypatch.setenv('CONDA_EXE','/conda')
    config=tmp_path/'seg.py';config.write_text('config={}');recognition=tmp_path/'rec.py';recognition.write_text('config={}')
    dataset=tmp_path/'data';dataset.mkdir();calls=[]
    def launch(command,**kwargs):
        calls.append((command,kwargs));return SimpleNamespace(poll=lambda:None)
    monkeypatch.setattr(job_module.subprocess,'Popen',launch)
    jobs=Jobs();req=TrainingRequest(config=str(config),recognition_config=str(recognition),dataset=str(dataset))
    eid=jobs.start(req);command,options=calls[0]
    assert command[0]=='/conda' and command[command.index('-n')+1]=='retailvision-seg'
    assert command[command.index('--task')+1]=='both'
    assert command[command.index('--recognition-config')+1]==str(recognition)
    assert options['start_new_session'] is True
    assert json.loads((tmp_path/'work_dirs'/eid/'experiment.json').read_text())['status']=='queued'
    with pytest.raises(ValueError,match='already running'):jobs.start(req)


def test_perturbed_mask_crop_preserves_clean_and_foreground_retention():
    import numpy as np
    from PIL import Image
    from retailvision.image_ops import mask_crop
    from retailvision.training.augmentations import perturbed_mask_crop

    arr=np.full((120,120,3),30,dtype=np.uint8)
    arr[30:90,30:90]=[220,40,40]
    img=Image.fromarray(arr)
    mask=np.zeros((120,120),dtype=bool);mask[30:90,30:90]=True
    clean=mask_crop(img,mask,padding=0.05)
    assert np.array_equal(np.asarray(perturbed_mask_crop(img,mask,np.random.default_rng(0),clean_prob=1.0)),np.asarray(clean))
    rng=np.random.default_rng(7)
    crops=[np.asarray(perturbed_mask_crop(img,mask,rng,clean_prob=0.0)) for _ in range(24)]
    assert all(c.size>0 and c.ndim==3 for c in crops)
    # Center of the product remains visible (red channel >> blue/green) across all perturbed samples.
    for c in crops:
        h,w,_=c.shape
        center=c[h//3:2*h//3,w//3:2*w//3]
        assert float(center[:,:,0].mean())>150.0


def test_load_roi_contexts_from_prepared_dataset(data,tmp_path):
    import shutil
    from retailvision.training.recognition import _load_roi_contexts
    prepare(**data)
    rec_dir=Path(data['output'])/'recognition'
    meta=json.loads((rec_dir/'metadata.json').read_text())
    train_rows=[r for r in meta['items'] if r['split']=='train']
    contexts=_load_roi_contexts(rec_dir,train_rows)
    assert len(contexts)==len(train_rows)
    assert all(rgb.width>0 and mask.shape==(rgb.height,rgb.width) and mask.any() for rgb,mask in contexts)
    isolated=tmp_path/'isolated_rec';shutil.copytree(rec_dir,isolated)
    with pytest.raises(FileNotFoundError,match='segmentation'):
        _load_roi_contexts(isolated,train_rows)
