import pytest
import numpy as np
from evaluation.metrics import recognition,pipeline,mask_iou
from inference.retrieval.rejection import calibrate

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
    from dataset.converters.prepare import prepare
    from dataset.processors.roi import polygon_mask
    from evaluation.metrics import segmentation
    prepare(**data)
    cp=Path(data['output'])/'segmentation/annotations/test.json';c=json.loads(cp.read_text());preds=[]
    ims={x['id']:x for x in c['images']}
    for a in c['annotations']:
        im=ims[a['image_id']];mask=np.asfortranarray(polygon_mask(a['segmentation'],im['width'],im['height']).astype('uint8'));rle=mask_utils.encode(mask);rle['counts']=rle['counts'].decode()
        preds.append(dict(image_id=a['image_id'],category_id=1,bbox=a['bbox'],segmentation=rle,score=1.))
    pp=tmp_path/'fixture_predictions.json';pp.write_text(json.dumps(preds));m=segmentation(str(cp),str(pp))
    assert m['segm']['map50_95']==pytest.approx(1.)
    assert m['matched_mask_iou']==1.
