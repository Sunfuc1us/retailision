import json
import numpy as np
from PIL import Image
from catalog.store import Catalog
from inference.interfaces import Instance
from inference.pipeline import VisionPipeline
from inference.retrieval.gallery import Gallery
from inference.retrieval.rejection import RejectionPolicy,calibrate

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
