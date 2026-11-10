from pydantic import BaseModel,Field
from typing import Literal
class DatasetRequest(BaseModel):
    coco:str
    images:str
    catalog:str='catalog/skus.json'
    sidecar:str|None=None
    output:str|None=None
    seed:int=42
class TrainingRequest(BaseModel):
    task:Literal['segmentation','recognition']
    model:str
    config:str
    dataset:str
    resume:str|None=None
