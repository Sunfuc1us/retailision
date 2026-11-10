import pytest
from registry.store import Registry

def test_immutable(tmp_path):
    cp=tmp_path/'c.pth';cp.write_bytes(b'fixture');cfg=tmp_path/'config.py';cfg.write_text('fixture=True')
    r=Registry(str(tmp_path/'models'));kw=dict(task='recognition',version='v001',checkpoint=str(cp),config=str(cfg),metadata=dict(model_name='fixture',architecture='fixture',training_dataset='fixture',license='fixture',evaluation_metrics={}))
    r.register(**kw);assert len(r.list())==1
    with pytest.raises(FileExistsError):r.register(**kw)
