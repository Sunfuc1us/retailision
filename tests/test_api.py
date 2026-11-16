import io
from fastapi.testclient import TestClient
from PIL import Image
from retailvision.api.app import app
client=TestClient(app)
def test_health_models_invalid():
    assert client.get('/api/health').json()['product']=='RetailVision'
    assert client.get('/api/models').status_code==200
    assert client.get('/api/catalog/missing').status_code==404
    assert client.post('/api/inference/predict',files={'image':('bad.png',b'bad')},data={'target_product':'x'}).status_code==422
    assert client.post('/api/inference/predict').status_code==422

def test_unconfigured_model_is_explicit(monkeypatch):
    monkeypatch.delenv('RETAILVISION_CONFIG',raising=False)
    assert client.get('/api/inference/backends').json()=={'available':[]}
    b=io.BytesIO();Image.new('RGB',(5,5)).save(b,format='PNG')
    assert client.post('/api/inference/predict',files={'image':('x.png',b.getvalue())},data={'target_product':'x'}).status_code==503

def test_dataset_api_uses_grouped_train_validation_split(data):
    from pathlib import Path
    import json
    response=client.post('/api/datasets/prepare',json=data)
    assert response.status_code==200
    assert response.json()['split_image_counts']=={'train':4,'val':2}
    out=Path(data['output'])
    metadata=json.loads((out/'recognition/metadata.json').read_text())
    assert {row['split'] for row in metadata['items']}=={'train','val'}
    segmentation={}
    for split in ('train','val'):
        segmentation[split]=json.loads((out/f'segmentation/annotations/{split}.json').read_text())
        assert {row['image_id'] for row in metadata['items'] if row['split']==split}=={image['id'] for image in segmentation[split]['images']}
    assert {im['capture_group'] for im in segmentation['train']['images']}.isdisjoint(im['capture_group'] for im in segmentation['val']['images'])


def test_dataset_api_supports_explicit_image_holdout_and_rejects_conflicting_modes(data):
    assert client.post('/api/datasets/prepare',json=dict(data,no_split=True,validation_images=['0.png'])).status_code==422
    response=client.post('/api/datasets/prepare',json=dict(data,validation_images=['0.png']))
    assert response.status_code==200
    assert response.json()['split_image_counts']=={'train':5,'val':1}
    assert response.json()['independent_validation'] is False
