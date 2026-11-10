from project_paths import PROJECT_ROOT,project_path
from inference.recognition.dino import DINOv2
from inference.segmentation.rtmdet import RTMDet

def test_checkout_paths_from_another_working_directory(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert project_path('models')==PROJECT_ROOT/'models'
    assert project_path(tmp_path)==tmp_path
    segmentation=RTMDet('configs/segmentation/rtmdet_ins_s.py','models/segmentation/rtmdet_ins_s_coco_v001/model.pth')
    recognition=DINOv2('third_party/dinov2','models/recognition/dinov2_b14_v001/model.pth')
    assert segmentation.config==str(PROJECT_ROOT/'configs/segmentation/rtmdet_ins_s.py')
    assert recognition.source==str(PROJECT_ROOT/'third_party/dinov2')
