import pytest
from retailvision.artifacts.registry import Registry

def test_immutable(tmp_path):
    cp=tmp_path/'c.pth';cp.write_bytes(b'fixture');cfg=tmp_path/'config.py';cfg.write_text('fixture=True')
    r=Registry(str(tmp_path/'models'));kw=dict(task='recognition',version='v001',checkpoint=str(cp),config=str(cfg),metadata=dict(model_name='fixture',architecture='fixture',training_dataset='fixture',license='fixture',evaluation_metrics={}))
    out=r.register(**kw);assert len(r.list())==1
    dep=out/'deployment/onnx';dep.mkdir(parents=True);(dep/'model.onnx').write_bytes(b'onnx')
    item=r.get('recognition','v001')
    assert item['deployments']['onnxruntime']=='deployment/onnx/model.onnx'
    with pytest.raises(FileExistsError):r.register(**kw)
