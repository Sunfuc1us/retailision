import json
import pytest
from PIL import Image,ImageDraw
@pytest.fixture
def data(tmp_path):
    images=tmp_path/'images';images.mkdir();ims=[];anns=[]
    catalog=[dict(sku_id=s,brand='fixture',product=s,flavor='',specification='') for s in ('FIXTURE_A','FIXTURE_B')]
    for i in range(6):
        img=Image.new('RGB',(100,80),'white');draw=ImageDraw.Draw(img);draw.rectangle((10,10,40,65),fill='red');draw.rectangle((55,10,90,65),fill='blue');img.save(images/f'{i}.png')
        ims.append(dict(id=i,file_name=f'{i}.png',width=100,height=80,capture_group=f'batch_{i//2}'))
        for j,(sku,x1,x2) in enumerate([('FIXTURE_A',10,40),('FIXTURE_B',55,90)]):anns.append(dict(id=i*2+j,image_id=i,category_id=1,segmentation=[[x1,10,x2,10,x2,65,x1,65]],attributes=dict(sku_id=sku)))
    coco=dict(fixture=True,images=ims,annotations=anns,categories=[dict(id=1,name='retail_product')]);cp=tmp_path/'coco.json';cp.write_text(json.dumps(coco));cat=tmp_path/'catalog.json';cat.write_text(json.dumps(catalog))
    return dict(coco=str(cp),images=str(images),catalog=str(cat),output=str(tmp_path/'prepared'))
