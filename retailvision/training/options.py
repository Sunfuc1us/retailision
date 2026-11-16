from pathlib import Path
from pydantic import BaseModel,model_validator
from typing import Literal

class TrainingRequest(BaseModel):
    task:Literal['both','segmentation','recognition']='both'
    model:str|None=None
    config:str|None=None
    recognition_config:str='configs/recognition/dinov2_b14.py'
    dataset:str
    resume:str|None=None
    no_validation:bool=False

    @model_validator(mode='after')
    def resolve(self):
        expected={'both':'rtmdet-ins-s+dinov2-vitb14','segmentation':'rtmdet-ins-s','recognition':'dinov2-vitb14'}
        self.model=self.model or expected[self.task]
        self.config=self.config or ('configs/recognition/dinov2_b14.py' if self.task=='recognition' else 'configs/segmentation/rtmdet_ins_s.py')
        if self.model!=expected[self.task]:raise ValueError('unsupported task/model combination')
        if self.no_validation and self.task!='recognition':raise ValueError('--no-validation is only supported for recognition experiments')
        if self.resume and self.task=='both':raise ValueError('resume individual training stages')
        return self

    def check_paths(self):
        for value in (self.config,self.dataset,*([self.recognition_config] if self.task=='both' else []),*([self.resume] if self.resume else [])):
            if not Path(value).exists():raise ValueError(f'missing path: {value}')
