import json
from pathlib import Path
import pytest
from retailvision.data.validate import validate
from retailvision.data.prepare import prepare
from retailvision.image_ops import polygon_mask
from retailvision.data.split import split_groups

@pytest.mark.parametrize('image_count,validation_count',[(2,1),(10,1),(15,2),(16,2),(25,3),(30,3),(35,4),(100,10)])
def test_default_split_scales_validation_by_image_count(image_count,validation_count):
    groups=[f'capture_{i}' for i in range(image_count)]
    splits=split_groups(groups)
    assert sum(s=='val' for s in splits.values())==validation_count
    assert sum(s=='train' for s in splits.values())==image_count-validation_count
    assert splits==split_groups(list(reversed(groups)))

def test_split_uses_image_counts_not_group_counts():
    groups=['large']*90+['validation']*10
    assert split_groups(groups)=={'large':'train','validation':'val'}
    # Whole groups take precedence over an exact 90/10 image ratio.
    groups=['large']*95+['small']*5
    assert split_groups(groups)=={'large':'train','small':'val'}

@pytest.mark.parametrize('groups',[[],['one'],['same']*15])
def test_split_rejects_missing_independent_capture_groups(groups):
    with pytest.raises(ValueError,match='two independent capture groups'):
        split_groups(groups)

def test_coco_roi_and_group_leakage(data):
    assert validate(data['coco'],data['images'],data['catalog'])['valid']
    manifest=prepare(**data);assert manifest['fixture']
    p=Path(data['output']);meta=json.loads((p/'recognition/metadata.json').read_text());seen={}
    for item in meta['items']:
        assert (p/'recognition'/item['path']).exists()
        seen.setdefault(item['capture_group'],set()).add(item['split'])
    assert all(len(s)==1 for s in seen.values())
    assert len(meta['items'])==12
    for split in ('train','val'):
        assert json.loads((p/f'segmentation/annotations/{split}.json').read_text())['categories']==[dict(id=1,name='retail_product')]
    assert manifest['split_image_counts']=={'train':4,'val':2}
    assert not (p/'segmentation/annotations/test.json').exists()
    with pytest.raises(FileExistsError):prepare(**data)

def test_single_capture_group_fails_before_creating_dataset(data):
    p=Path(data['coco']);c=json.loads(p.read_text())
    for image in c['images']:image['capture_group']='same_capture'
    p.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='two independent capture groups'):prepare(**data)
    assert not Path(data['output']).exists()


def test_explicit_image_holdout_preserves_capture_group_and_has_no_roi_leakage(data):
    p=Path(data['coco']);c=json.loads(p.read_text())
    for image in c['images']:image['capture_group']='same_capture'
    p.write_text(json.dumps(c))
    manifest=prepare(**data,validation_images=['0.png'])
    assert manifest['split_image_counts']=={'train':5,'val':1}
    assert manifest['split_unit']=='image' and manifest['independent_validation'] is False
    assert manifest['validation_scope']=='same_capture_image_holdout' and manifest['groups']=={}
    meta=json.loads((Path(data['output'])/'recognition/metadata.json').read_text())
    assert all(row['capture_group']=='same_capture' for row in meta['items'])
    train_ids={r['image_id'] for r in meta['items'] if r['split']=='train'}
    val_ids={r['image_id'] for r in meta['items'] if r['split']=='val'}
    assert train_ids==set(range(1,6)) and val_ids=={0}
    assert not train_ids&val_ids


@pytest.mark.parametrize('held_out',[[],['missing.png'],[f'{i}.png' for i in range(6)]])
def test_explicit_image_holdout_rejects_empty_unknown_or_all_images(data,held_out):
    with pytest.raises(ValueError,match='nonempty proper subset'):
        prepare(**data,validation_images=held_out)
    assert not Path(data['output']).exists()


def test_image_holdout_changes_dataset_version_when_validation_images_change(data):
    first=prepare(**data,validation_images=['0.png'])
    data['output']+='-other'
    second=prepare(**data,validation_images=['1.png'])
    assert first['dataset_version']!=second['dataset_version']

def test_missing_sku_and_sidecar(data,tmp_path):
    p=Path(data['coco']);c=json.loads(p.read_text());c['annotations'][0]['attributes']={};p.write_text(json.dumps(c))
    assert any('sidecar' in x for x in validate(data['coco'],data['images'],data['catalog'])['errors'])
    side=tmp_path/'side.json';side.write_text(json.dumps({'0':{'sku_id':'FIXTURE_A'}}));assert validate(data['coco'],data['images'],data['catalog'],str(side))['valid']

@pytest.mark.parametrize('allow_truncated', [False, True])
def test_truncated_pixels_rejected_before_dataset_creation(data, monkeypatch, allow_truncated):
    from PIL import Image, ImageFile
    image = Path(data['images']) / '0.png'
    Image.new('RGB', (100, 80), 'red').save(image, format='JPEG')
    image.write_bytes(image.read_bytes()[:-20])
    with Image.open(image) as header:
        assert header.size == (100, 80)
    monkeypatch.setattr(ImageFile, 'LOAD_TRUNCATED_IMAGES', allow_truncated)
    report = validate(data['coco'], data['images'], data['catalog'])
    assert 'missing or invalid image 0' in report['errors']
    assert ImageFile.LOAD_TRUNCATED_IMAGES is allow_truncated
    with pytest.raises(ValueError, match='missing or invalid image 0'):
        prepare(**data)
    assert not Path(data['output']).exists()

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

def test_unsplit_cvat_paths_metadata_and_boxes(data,monkeypatch):
    import importlib
    converter=importlib.import_module('retailvision.data.prepare')
    def forbidden_split(*args,**kwargs):
        raise AssertionError('no-split conversion must not call the splitter')
    monkeypatch.setattr(converter,'split_groups',forbidden_split)
    root=Path(data['images']);nested=root/'default';nested.mkdir()
    for image in root.glob('*.png'):image.rename(nested/image.name)
    p=Path(data['coco']);c=json.loads(p.read_text())
    for image in c['images']:image.pop('capture_group')
    c['annotations'][0]['bbox']=[10.25,10.25,29.5,54.5]
    p.write_text(json.dumps(c))
    assert any('missing or invalid image' in e for e in validate(data['coco'],root,data['catalog'],require_capture_group=False)['errors'])
    data['images']=str(nested)
    assert any('capture_group required' in e for e in validate(data['coco'],nested,data['catalog'])['errors'])
    assert validate(data['coco'],nested,data['catalog'],require_capture_group=False)['valid']
    manifest=prepare(**data,no_split=True)
    out=Path(data['output']);converted=json.loads((out/'segmentation/annotations/all.json').read_text())
    assert manifest['no_split'] and manifest['groups']=={}
    assert len(converted['images'])==6 and len(converted['annotations'])==12
    assert converted['annotations'][0]['bbox']==c['annotations'][0]['bbox']
    assert [p.name for p in (out/'segmentation/annotations').iterdir()]==['all.json']
    meta=json.loads((out/'recognition/metadata.json').read_text())
    assert all(row['split']=='all' and row['capture_group'] is None for row in meta['items'])
    assert all((out/'recognition'/row['path']).is_file() for row in meta['items'])
    skus=json.loads((out/'recognition/skus.json').read_text())
    assert {s['sku_id'] for s in skus}=={'FIXTURE_A','FIXTURE_B'}
    assert all(s['flavor']=='' and s['specification']=='' for s in skus)
    assert not any((out/'recognition'/s).exists() for s in ('train','val','test'))

def test_category_remains_retail_product(data):
    p=Path(data['coco']);c=json.loads(p.read_text());c['categories'][0]['name']='Detail';p.write_text(json.dumps(c))
    report=validate(data['coco'],data['images'],data['catalog'],require_capture_group=False)
    assert not report['valid'] and any('category must be retail_product' in e for e in report['errors'])

@pytest.mark.parametrize('mode',['RGB','L','RGBA'])
@pytest.mark.parametrize('touch_edge',[False,True])
def test_local_mask_crop_preserves_pixels_and_padding(mode,touch_edge):
    import numpy as np
    from PIL import Image
    from retailvision.image_ops import mask_crop
    image=Image.fromarray(np.arange(80*60*3,dtype='uint8').reshape(60,80,3)).convert(mode)
    mask=np.zeros((60,80),dtype=bool)
    left,top=(0,0) if touch_edge else (20,15)
    mask[top:top+30,left:left+20]=True
    mask[top+5:top+10,left+5:left+10]=False
    # The existing ROI contract: 5% context and gray pixels outside the mask.
    padding=1
    box=(max(0,left-padding),max(0,top-padding),left+20+padding,top+30+padding)
    expected=np.array(image.convert('RGB'));expected[~mask]=127
    expected=Image.fromarray(expected).crop(box)
    actual=mask_crop(image,mask)
    assert actual.mode=='RGB' and actual.size==expected.size
    assert np.array_equal(np.array(actual),np.array(expected))


def test_local_mask_crop_rejects_empty_and_mismatched_masks():
    import numpy as np
    from PIL import Image
    from retailvision.image_ops import mask_crop
    image=Image.new('RGB',(10,10))
    for mask in (np.zeros((10,10),dtype=bool),np.ones((5,5),dtype=bool)):
        with pytest.raises(ValueError,match='invalid or empty'):mask_crop(image,mask)
