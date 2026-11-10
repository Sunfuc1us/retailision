import io
from fastapi.testclient import TestClient
from PIL import Image
from backend.api.app import app
client=TestClient(app)
def test_health_models_invalid():
    assert client.get('/api/health').json()['product']=='RetailVision'
    assert client.get('/api/models').status_code==200
    assert client.get('/api/catalog/missing').status_code==404
    assert client.post('/api/inference/predict',files={'image':('bad.png',b'bad')},data={'target_product':'x'}).status_code==422
    assert client.post('/api/inference/predict').status_code==422

def test_unconfigured_model_is_explicit(monkeypatch):
    monkeypatch.delenv('RETAILVISION_CONFIG',raising=False)
    b=io.BytesIO();Image.new('RGB',(5,5)).save(b,format='PNG')
    assert client.post('/api/inference/predict',files={'image':('x.png',b.getvalue())},data={'target_product':'x'}).status_code==503
