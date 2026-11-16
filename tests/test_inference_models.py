import io,json
from pathlib import Path
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from retailvision.api.app import app
from retailvision.artifacts import profiles as models


@pytest.fixture
def choices(tmp_path,monkeypatch):
    profiles=[]
    for name in ('trained','reference'):
        catalog=tmp_path/f'{name}_catalog.json'
        catalog.write_text(json.dumps([{'sku_id':name,'product':name}]))
        config=tmp_path/f'{name}.json'
        config.write_text(json.dumps(dict(fixture=True,catalog=str(catalog),results=str(tmp_path/name),
            segmentation=dict(config='absent.py',checkpoint=f'{name}.pth'),
            recognition=dict(checkpoint='dinov2.pth'))))
        profiles.append(models.ModelProfile(name,name,config))
    monkeypatch.setattr(models,'profiles',lambda:profiles)
    monkeypatch.setenv('RETAILVISION_CONFIG',str(profiles[0].config))
    return profiles


def test_model_options_and_catalog_follow_selected_weights(choices):
    client=TestClient(app)
    result=client.get('/api/inference/models').json()
    assert result['default_model']=='trained'
    assert [x['id'] for x in result['models']]==['trained','reference']
    assert all(x['backends']==[] for x in result['models'])
    assert result['models'][0]['checkpoint'].endswith('trained.pth')
    assert client.get('/api/catalog').json()[0]['sku_id']=='trained'
    assert client.get('/api/catalog?model_id=reference').json()[0]['sku_id']=='reference'
    assert client.get('/api/catalog/trained?model_id=reference').status_code==404
    assert client.get('/api/catalog?model_id=unknown').status_code==422


def test_unknown_model_and_missing_export_are_explicit(choices):
    client=TestClient(app);buffer=io.BytesIO();Image.new('RGB',(4,4)).save(buffer,format='PNG')
    files={'image':('input.png',buffer.getvalue(),'image/png')}
    assert client.post('/api/inference/predict',files=files,data=dict(target_product='trained',model_id='unknown')).status_code==422
    response=client.post('/api/inference/predict',files=files,data=dict(target_product='trained',model_id='trained',backend='tensorrt'))
    assert response.status_code==422 and 'no tensorrt export' in response.json()['detail']


def test_saved_masks_remain_accessible_after_model_switch(choices,monkeypatch):
    directory=Path(choices[0].settings()['results'])/'saved-result';directory.mkdir(parents=True)
    (directory/'result.json').write_text(json.dumps(dict(model_id='trained')))
    Image.new('L',(4,4),255).save(directory/'mask_001.png')
    monkeypatch.setenv('RETAILVISION_CONFIG',str(choices[1].config))
    client=TestClient(app)
    assert client.get('/api/results/saved-result').json()['model_id']=='trained'
    response=client.get('/api/results/saved-result/mask_001.png')
    assert response.status_code==200 and response.headers['content-type']=='image/png'


def test_sku_and_gallery_management_and_diagnose(choices,tmp_path,monkeypatch):
    import numpy as np
    from retailvision.data.catalog import Catalog
    from retailvision.inference.interfaces import Instance
    from retailvision.inference.pipeline import VisionPipeline
    from retailvision.inference.gallery import Gallery
    from retailvision.inference.rejection import RejectionPolicy
    import retailvision.api.app as api_mod
    gdir=tmp_path/'trained_gallery'
    g=Gallery();g.build_index(np.eye(2,dtype='float32'),['trained','EXTRA'])
    g.save(str(gdir))
    cfg=choices[0].settings();cfg['gallery']=str(gdir);choices[0].config.write_text(json.dumps(cfg))
    class Seg:
        def predict(self,image,threshold):return [Instance([0,0,4,4],np.ones((4,4),dtype=bool),.9)]
        def get_model_info(self):return {'architecture':'RTMDet'}
        def unload(self):pass
    class Rec:
        def extract_embedding(self,image):return np.array([0.,1.],dtype='float32')
        def get_model_info(self):return {'architecture':'DINOv2'}
        def unload(self):pass
    pipe=VisionPipeline(Seg(),Rec(),Gallery.load(str(gdir)),Catalog(cfg['catalog']),RejectionPolicy(.5,.1,'v','g','m'),cfg['results'])
    pipe.model_id='trained';pipe.model_name='trained'
    monkeypatch.setattr(api_mod.runtime,'pipe',pipe)
    monkeypatch.setattr(api_mod.runtime,'key',(str(choices[0].config),'pytorch'))
    client=TestClient(app)
    assert client.get('/api/inference/runtime').json()['loaded'] is True
    created=client.post('/api/catalog',json=dict(sku_id='NEW_SKU',product='New Drink',brand='Vita',model_id='trained'))
    assert created.status_code==201 and client.get('/api/catalog/NEW_SKU?model_id=trained').status_code==200
    buf=io.BytesIO();Image.new('RGB',(4,4)).save(buf,format='PNG')
    files={'image':('ref.png',buf.getvalue(),'image/png')}
    ref_res=client.post('/api/gallery/references',files=files,data=dict(sku_id='NEW_SKU',model_id='trained'))
    assert ref_res.status_code==201 and ref_res.json()['counts']['NEW_SKU']==1
    assert client.get('/api/gallery?model_id=trained').json()['ntotal']==3
    diag=client.post('/api/inference/diagnose',files=files,data=dict(model_id='trained',detection_threshold='0.3'))
    assert diag.status_code==200 and diag.json()['candidate_count']==1
    del_g=client.delete('/api/gallery/NEW_SKU?model_id=trained')
    assert del_g.status_code==200 and 'NEW_SKU' not in del_g.json()['counts']
    assert client.delete('/api/catalog/NEW_SKU?model_id=trained').status_code==200
