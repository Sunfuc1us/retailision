import json
import numpy as np
import pytest
from PIL import Image
from retailvision.data.catalog import Catalog
from retailvision.inference.interfaces import Instance
from retailvision.inference.pipeline import VisionPipeline
from retailvision.inference.gallery import Gallery
from retailvision.inference.rejection import RejectionPolicy,calibrate

class FixtureSeg:
    def predict(self,image,threshold):return [Instance([0,0,10,10],np.ones((10,10),dtype=bool),.9)]
class FixtureRec:
    def __init__(self,e):self.e=np.array(e,dtype='float32')
    def extract_embedding(self,image):return self.e

def test_gallery_sku_margin_and_updates(tmp_path):
    g=Gallery();g.build_index(np.array([[1,0],[1,.01],[0,1]],dtype='float32'),['A','A','B']);r=g.search(np.array([1,0]));assert len(r)==2 and r[0][0]=='A'
    g.save(str(tmp_path/'g'));g=Gallery.load(str(tmp_path/'g'));g.remove_sku('A');assert g.ids==['B'];g.add_sku('C',np.array([[1,0]]));assert g.search(np.array([1,0]))[0][0]=='C'

def test_target_mask_and_unknown_rejection(tmp_path):
    c=tmp_path/'c.json';c.write_text(json.dumps([dict(sku_id='A',product='snack'),dict(sku_id='B',product='snack')]))
    g=Gallery();g.build_index(np.eye(2,dtype='float32'),['A','B']);policy=RejectionPolicy(.8,.2,'fixture','g','m')
    p=VisionPipeline(FixtureSeg(),FixtureRec([1,0]),g,Catalog(c),policy,str(tmp_path/'results'))
    result=p.predict(Image.new('RGB',(10,10)),'A');assert result['target_found'];assert list((tmp_path/'results').glob('*/mask_001.png'))
    assert p.predict(Image.new('RGB',(10,10)),'snack')['reason']=='ambiguous_target'
    p.recognition=FixtureRec([1,1]);assert not p.predict(Image.new('RGB',(10,10)),'A')['target_found']
    p.policy=None;assert p.predict(Image.new('RGB',(10,10)),'A')['reason']=='calibration_required'

def test_calibration():
    r=calibrate([dict(known=True,correct=True,score=.9,margin=.4),dict(known=False,correct=False,score=.7,margin=.1)],'fixture','g','m',0)
    assert r['calibration_metrics']['unknown_false_accept_rate']==0

def test_bbox_bounded_mask_controls_roi_and_saved_mask(tmp_path):
    class LeakySeg:
        def predict(self,image,threshold):
            mask=np.zeros((20,20),dtype=bool);mask[7:14,7:14]=True;mask[1,1]=True
            return [Instance([7,7,14,14],mask,.9)]
    class CaptureRec(FixtureRec):
        def extract_embedding(self,image):self.crop_size=image.size;return self.e
    catalog=tmp_path/'catalog.json';catalog.write_text(json.dumps([dict(sku_id='A',product='a'),dict(sku_id='B',product='b')]))
    gallery=Gallery();gallery.build_index(np.eye(2,dtype='float32'),['A','B'])
    policy=RejectionPolicy(.8,.2,'fixture','g','m');image=Image.new('RGB',(20,20))
    rec=CaptureRec([1,0]);pipeline=VisionPipeline(LeakySeg(),rec,gallery,Catalog(catalog),policy,str(tmp_path/'bounded'),clip_mask_to_bbox=True)
    result=pipeline.predict(image,'A');assert result['target_found'] and rec.crop_size==(7,7)
    saved=np.asarray(Image.open(next((tmp_path/'bounded').glob('*/mask_001.png'))))
    assert saved[1,1]==0 and saved[8,8]==255
    legacy=CaptureRec([1,0]);pipeline=VisionPipeline(LeakySeg(),legacy,gallery,Catalog(catalog),policy,str(tmp_path/'legacy'))
    assert pipeline.predict(image,'A')['target_found'] and legacy.crop_size==(13,13)
    assert np.asarray(Image.open(next((tmp_path/'legacy').glob('*/mask_001.png'))))[1,1]==255


def test_empty_mask_within_box_does_not_abort_other_candidates(tmp_path):
    class Seg:
        def predict(self,image,threshold):
            outside=np.zeros((20,20),dtype=bool);outside[0,0]=True
            valid=np.zeros((20,20),dtype=bool);valid[8:14,8:14]=True
            return [Instance([8,8,14,14],outside,.9),Instance([8,8,14,14],valid,.8)]
    catalog=tmp_path/'catalog.json';catalog.write_text(json.dumps([dict(sku_id='A',product='a'),dict(sku_id='B',product='b')]))
    gallery=Gallery();gallery.build_index(np.eye(2,dtype='float32'),['A','B'])
    pipeline=VisionPipeline(Seg(),FixtureRec([1,0]),gallery,Catalog(catalog),RejectionPolicy(.8,.2,'fixture','g','m'),str(tmp_path/'results'),clip_mask_to_bbox=True)
    result=pipeline.predict(Image.new('RGB',(20,20)),'A')
    assert result['target_found'] and len(result['instances'])==1
    assert result['candidates'][0]['reason']=='empty_mask_in_bbox' and not result['candidates'][0]['accepted']
    assert result['candidates'][1]['top_k'][0]['sku_id']=='A'
    assert result['candidates'][1]['embedding_dim']==2 and result['candidates'][1]['mask_pixels']==36


def test_pipeline_diagnose_and_rejection_reasons(tmp_path):
    catalog=tmp_path/'catalog.json';catalog.write_text(json.dumps([dict(sku_id='A',product='a'),dict(sku_id='B',product='b')]))
    gallery=Gallery();gallery.build_index(np.eye(2,dtype='float32'),['A','B'])
    pipeline=VisionPipeline(FixtureSeg(),FixtureRec([.8,.6]),gallery,Catalog(catalog),RejectionPolicy(.85,.2,'fixture','g','m'),str(tmp_path/'diag'))
    pred=pipeline.predict(Image.new('RGB',(10,10)),'A')
    assert not pred['target_found'] and pred['candidates'][0]['reason']=='below_similarity'
    assert pred['candidates'][0]['top1_sku']=='A' and len(pred['candidates'][0]['top_k'])==2
    assert (tmp_path/'diag'/pred['result_id']/'mask_001.png').is_file()
    diag=pipeline.diagnose(Image.new('RGB',(10,10)),top_k=2)
    assert diag['candidate_count']==1 and diag['runtime']['gallery']['unique_skus']==2


def test_v002_rejects_unknown_and_ambiguous_skus_despite_high_detection_score(tmp_path):
    from retailvision.paths import project_path
    settings=json.loads(project_path('configs/inference/retail_6sku_v002_policy.json').read_text())
    policy=RejectionPolicy(**{k:settings[k] for k in ('similarity','margin','dataset_version','gallery_version','model_version')},sku_similarity=settings['sku_similarity'])
    # Reviewed mixed-shelf feedback: Oreo is closest to blackcurrant, but is unknown.
    assert not policy.accepts([('VITA_BLACKCURRANT_DRINK_250ML',.665),('OTHER',.653)])
    # High cosine similarity alone must not accept an ambiguous identity.
    assert not policy.accepts([('A',.95),('B',.949)])
    assert policy.accepts([('VITA_APPLE_JUICE_250ML',.7615),('OTHER',.6473)])
    assert policy.accepts([('COKE_ZERO_330ML',.9221),('OTHER',.9101)])
    assert policy.accepts([('COKE_ZERO_330ML',.7875),('OTHER',.7800)])
    assert not policy.accepts([('VITA_YASHIXIANG_LEMON_TEA_250ML',.7677),('OTHER',.6845)])
    assert policy.accepts([('VITA_YASHIXIANG_LEMON_TEA_250ML',.8394),('OTHER',.5900)])
    class HighScoreSeg(FixtureSeg):
        def predict(self,image,threshold):return [Instance([0,0,10,10],np.ones((10,10),dtype=bool),.999)]
    catalog=tmp_path/'catalog.json';catalog.write_text(json.dumps([dict(sku_id='A',product='a'),dict(sku_id='B',product='b')]))
    gallery=Gallery();gallery.build_index(np.array([[1,0,0],[0,1,0]],dtype='float32'),['A','B'])
    pipeline=VisionPipeline(HighScoreSeg(),FixtureRec([.65,.6,np.sqrt(1-.65**2-.6**2)]),gallery,Catalog(catalog),policy,str(tmp_path/'results'))
    result=pipeline.predict(Image.new('RGB',(10,10)),'A')
    assert not result['target_found'] and result['candidates'][0]['reason']=='below_similarity'
    assert result['candidates'][0]['detection_score']==.999


def test_per_sku_rejection_reason_uses_the_same_threshold_as_acceptance(tmp_path):
    catalog=tmp_path/'catalog.json';catalog.write_text(json.dumps([dict(sku_id='A',product='a'),dict(sku_id='B',product='b')]))
    gallery=Gallery();gallery.build_index(np.eye(2,dtype='float32'),['A','B'])
    policy=RejectionPolicy(.7,.01,'v','g','m',sku_similarity={'A':.85})
    pipe=VisionPipeline(FixtureSeg(),FixtureRec([.8,.6]),gallery,Catalog(catalog),policy,str(tmp_path/'results'))
    result=pipe.predict(Image.new('RGB',(10,10)),'A')
    assert result['candidates'][0]['reason']=='below_similarity' and not result['target_found']


def test_chinese_name_resolves_before_inference_and_broad_keyword_is_not_guessed(tmp_path):
    catalog=tmp_path/'catalog.json';catalog.write_text(json.dumps([
        dict(sku_id='APPLE',brand='维他',product='苹果汁饮料',flavor='苹果味',specification='250ml'),
        dict(sku_id='BERRY',brand='维他',product='风味饮料',flavor='黑加仑味',specification='250ml')]))
    gallery=Gallery();gallery.build_index(np.eye(2,dtype='float32'),['APPLE','BERRY'])
    pipe=VisionPipeline(FixtureSeg(),FixtureRec([1,0]),gallery,Catalog(catalog),RejectionPolicy(.8,.2,'v','g','m'),str(tmp_path/'results'))
    result=pipe.predict(Image.new('RGB',(10,10)),'维他苹果汁')
    assert result['target_found'] and result['sku_id']=='APPLE'
    assert pipe.predict(Image.new('RGB',(10,10)),'维他')['reason']=='ambiguous_target'


def test_compact_mask_download_restores_original_pixels(tmp_path, monkeypatch):
    import importlib
    from fastapi.testclient import TestClient
    api=importlib.import_module('retailvision.api.app')
    from retailvision.artifacts import profiles as inference_models
    c=tmp_path/'catalog.json';c.write_text(json.dumps([dict(sku_id='A',product='a'),dict(sku_id='B',product='b')]))
    class LocalSeg:
        def predict(self,image,threshold):
            mask=np.zeros((80,100),dtype=bool);mask[20:50,30:60]=True;mask[25:30,40:45]=False
            return [Instance([30,20,60,50],mask,.9)]
    gallery=Gallery();gallery.build_index(np.eye(2,dtype='float32'),['A','B'])
    pipe=VisionPipeline(LocalSeg(),FixtureRec([1,0]),gallery,Catalog(c),RejectionPolicy(.8,.2,'v','g','m'),str(tmp_path/'results'))
    pipe.compact_masks=True
    result=pipe.predict(Image.new('RGB',(100,80)),'A');item=result['instances'][0]
    monkeypatch.setattr(inference_models,'result_roots',lambda:[tmp_path/'results'])
    response=TestClient(api.app).get(item['mask_path'])
    import io
    restored=np.array(Image.open(io.BytesIO(response.content)))
    expected=LocalSeg().predict(None,None)[0].mask.astype('uint8')*255
    assert response.status_code==200 and np.array_equal(restored,expected)
    assert item['mask_origin']==[30,20]
    assert Image.open(tmp_path/'results'/result['result_id']/'mask_001.crop.png').size==(30,30)


def test_gallery_exact_sku_aggregation_matches_reference():
    rng=np.random.default_rng(4)
    vectors=rng.normal(size=(100,8)).astype('float32');ids=[str(i%7) for i in range(100)]
    gallery=Gallery();gallery.build_index(vectors,ids)
    query=rng.normal(size=8).astype('float32');query/=np.linalg.norm(query)
    expected=sorted([(sku,max(float(v@query) for v,s in zip(gallery.vectors,ids) if s==sku)) for sku in set(ids)],key=lambda x:-x[1])[:5]
    actual=gallery.search(query)
    assert [x[0] for x in actual]==[x[0] for x in expected]
    assert np.allclose([x[1] for x in actual],[x[1] for x in expected],atol=1e-6)


def test_gallery_topk_mean_reduces_single_outlier_reference():
    # SKU A has one outlier near [1,0] and two orthogonal vectors; SKU B has three consistent vectors.
    v_a=np.array([[1.,0.],[0.,1.],[0.,1.]],dtype='float32')
    v_b=np.array([[0.9,0.43589],[0.9,0.43589],[0.9,0.43589]],dtype='float32')
    g=Gallery(aggregation='topk_mean',support_k=3)
    g.build_index(np.concatenate([v_a,v_b]),['A','A','A','B','B','B'])
    q=np.array([1.,0.],dtype='float32')
    # Single-max ranks A (1.0) ahead of B (0.9); top-3 mean ranks consistent B (0.9) ahead of A (1/3).
    assert g.search(q,aggregation='max')[0][0]=='A'
    res=g.search(q)
    assert res[0][0]=='B' and np.isclose(res[0][1],0.9,atol=1e-5)
    assert res[1][0]=='A' and np.isclose(res[1][1],1./3.,atol=1e-5)
    # Works cleanly when an SKU has fewer than support_k references and preserves settings across remove_sku.
    g.add_sku('C',np.array([[0.95,0.31225]],dtype='float32'))
    assert g.search(q)[0][0]=='C'
    g.remove_sku('A')
    assert g.aggregation=='topk_mean' and g.support_k==3
    with pytest.raises(ValueError):Gallery(aggregation='invalid')
    with pytest.raises(ValueError):Gallery(aggregation='topk_mean',support_k=0)
