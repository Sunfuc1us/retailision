from pydantic import BaseModel,Field,model_validator
from typing import Literal
class DatasetRequest(BaseModel):
    coco:str
    images:str
    catalog:str='catalog/skus.json'
    sidecar:str|None=None
    output:str|None=None
    seed:int=42
    no_split:bool=False
    validation_images:list[str]|None=None

    @model_validator(mode='after')
    def check_split(self):
        if self.no_split and self.validation_images is not None:raise ValueError('no_split and validation_images are mutually exclusive')
        return self
class SKURegisterRequest(BaseModel):
    sku_id:str=Field(min_length=1)
    product:str=Field(min_length=1)
    brand:str=''
    flavor:str=''
    specification:str=''
    model_id:str|None=None
class ModelRegisterRequest(BaseModel):
    task:Literal['segmentation','recognition']
    version:str=Field(min_length=1)
    checkpoint:str=Field(min_length=1)
    config:str=Field(min_length=1)
    metadata:dict
