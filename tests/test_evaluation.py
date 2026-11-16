import pytest
import numpy as np
from retailvision.evaluation import recognition,pipeline,mask_iou
from retailvision.inference.rejection import calibrate

def test_slices_unknown_and_failures():
    r=recognition([dict(known=True,truth='a',predictions=['b','a'],accepted=True,tags=['same_brand_different_flavor']),dict(known=False,truth='unknown',predictions=['b'],accepted=False)])
    assert r['top1']==0 and r['top5']==1 and r['unknown_rejection_rate']==1
    assert r['slices']['same_brand_different_flavor']['count']==1
    assert r['slices']['similar_medicine']['top1'] is None
    assert len(r['failure_samples'])==1

def test_pipeline_and_iou():
    r=pipeline([dict(localization_success=True,identification_success=False,end_to_end_success=False,latency_ms=5,gpu_memory_bytes=100)])
    assert r['end_to_end_success']==0 and r['peak_gpu_memory_bytes']==100
    assert mask_iou(np.ones((2,2)),np.zeros((2,2)))==0

def test_calibration_requires_unknown():
    with pytest.raises(ValueError):calibrate([dict(known=True,score=.9,margin=.1,correct=True)],'v','g','m')

def test_coco_segmentation_metrics(data,tmp_path):
    import json
    from pathlib import Path
    from pycocotools import mask as mask_utils
    from retailvision.data.prepare import prepare
    from retailvision.image_ops import polygon_mask
    from retailvision.evaluation import segmentation
    prepare(**data)
    cp=Path(data['output'])/'segmentation/annotations/val.json';c=json.loads(cp.read_text());preds=[]
    ims={x['id']:x for x in c['images']}
    for a in c['annotations']:
        im=ims[a['image_id']];mask=np.asfortranarray(polygon_mask(a['segmentation'],im['width'],im['height']).astype('uint8'));rle=mask_utils.encode(mask);rle['counts']=rle['counts'].decode()
        preds.append(dict(image_id=a['image_id'],category_id=1,bbox=a['bbox'],segmentation=rle,score=1.))
    pp=tmp_path/'fixture_predictions.json';pp.write_text(json.dumps(preds));m=segmentation(str(cp),str(pp))
    assert m['segm']['map50_95']==pytest.approx(1.)
    assert m['matched_mask_iou']==1.
    empty_pp=tmp_path/'empty_predictions.json';empty_pp.write_text('[]');empty_m=segmentation(str(cp),str(empty_pp))
    assert empty_m['segm']['map50_95']==0. and empty_m['matched_mask_iou'] is None

def test_evaluate_dataset_end_to_end(data,tmp_path):
    import json
    from pathlib import Path
    from retailvision.data.catalog import Catalog
    from retailvision.data.prepare import prepare
    from retailvision.image_ops import polygon_mask
    from retailvision.evaluation import evaluate_dataset
    from retailvision.inference.interfaces import Instance
    from retailvision.inference.pipeline import VisionPipeline
    from retailvision.inference.gallery import Gallery
    from retailvision.inference.rejection import RejectionPolicy
    prepare(**data)
    out=Path(data['output']);test_coco=json.loads((out/'segmentation/annotations/val.json').read_text())
    class PerfectSeg:
        def predict(self,image,threshold):
            anns=[a for a in test_coco['annotations'] if a['image_id']==test_coco['images'][0]['id']]
            return [Instance([a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]],
                             polygon_mask(a['segmentation'],image.width,image.height),.95) for a in anns]
    class ConstRec:
        def extract_embedding(self,image):return np.array([1.,0.],dtype='float32')
    g=Gallery();g.build_index(np.eye(2,dtype='float32'),['FIXTURE_A','FIXTURE_B'])
    pipe=VisionPipeline(PerfectSeg(),ConstRec(),g,Catalog(data['catalog']),RejectionPolicy(.5,.1,'v','g','m'),str(tmp_path/'eval_results'))
    report=evaluate_dataset(pipe,str(out),threshold=.3)
    assert report['split']=='val'
    assert 'segmentation' in report and 'recognition' in report and report['pipeline'] is not None
    assert report['recognition']['top5']==1.0
