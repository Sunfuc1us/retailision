import json
from pathlib import Path
import pytest
from dataset.validators.coco import validate
from dataset.converters.prepare import prepare
from dataset.processors.roi import polygon_mask

def test_coco_roi_and_group_leakage(data):
    assert validate(data['coco'],data['images'],data['catalog'])['valid']
    manifest=prepare(**data);assert manifest['fixture']
    p=Path(data['output']);meta=json.loads((p/'recognition/metadata.json').read_text());seen={}
    for item in meta['items']:
        assert (p/'recognition'/item['path']).exists()
        seen.setdefault(item['capture_group'],set()).add(item['split'])
    assert all(len(s)==1 for s in seen.values())
    assert len(meta['items'])==12
    for split in ('train','val','test'):
        assert json.loads((p/f'segmentation/annotations/{split}.json').read_text())['categories']==[dict(id=1,name='retail_product')]
    with pytest.raises(FileExistsError):prepare(**data)

def test_missing_sku_and_sidecar(data,tmp_path):
    p=Path(data['coco']);c=json.loads(p.read_text());c['annotations'][0]['attributes']={};p.write_text(json.dumps(c))
    assert any('sidecar' in x for x in validate(data['coco'],data['images'],data['catalog'])['errors'])
    side=tmp_path/'side.json';side.write_text(json.dumps({'0':{'sku_id':'FIXTURE_A'}}));assert validate(data['coco'],data['images'],data['catalog'],str(side))['valid']

@pytest.mark.parametrize('change',['duplicate','invalid','conflict','missing_image','group'])
def test_reject_invalid(data,change):
    p=Path(data['coco']);c=json.loads(p.read_text())
    if change=='duplicate':c['annotations'].append(c['annotations'][0])
    if change=='invalid':c['annotations'][0]['segmentation']=[[0,0,20,20,20,0,0,20]]
    if change=='conflict':c['annotations'][0]['attributes']['brand']='wrong'
    if change=='missing_image':c['images'][0]['file_name']='absent.png'
    if change=='group':c['images'][0].pop('capture_group')
    p.write_text(json.dumps(c));assert not validate(data['coco'],data['images'],data['catalog'])['valid']

def test_mask():
    m=polygon_mask([[1,1,5,1,5,6,1,6]],10,10);assert m.sum()==20
