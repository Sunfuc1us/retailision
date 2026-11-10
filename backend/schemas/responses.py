from pydantic import BaseModel

class InstanceResult(BaseModel):
    instance_id:int
    sku_id:str
    identity_score:float
    bbox:list[float]
    mask_path:str
    detection_score:float
    accepted:bool

class Timing(BaseModel):
    segmentation_ms:float
    recognition_ms:float
    total_ms:float

class PredictionResponse(BaseModel):
    success:bool
    target_found:bool
    sku_id:str|None=None
    identity_score:float|None=None
    reason:str|None=None
    instances:list[InstanceResult]|None=None
    candidates:list[dict]|None=None
    result_id:str|None=None
    timing:Timing|None=None
