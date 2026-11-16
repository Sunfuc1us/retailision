from pydantic import BaseModel

class TopKMatch(BaseModel):
    sku_id:str
    score:float

class InstanceResult(BaseModel):
    instance_id:int
    sku_id:str
    top1_sku:str|None=None
    identity_score:float
    margin:float|None=None
    top_k:list[TopKMatch]|None=None
    bbox:list[float]
    category_id:int=0
    mask_path:str
    mask_preview_path:str|None=None
    mask_origin:list[int]|None=None
    image_size:list[int]|None=None
    detection_score:float
    raw_mask_pixels:int|None=None
    mask_pixels:int|None=None
    bbox_clipped:bool|None=None
    embedding_dim:int|None=None
    embedding_norm:float|None=None
    accepted:bool
    reason:str|None=None

class Timing(BaseModel):
    segmentation_ms:float
    recognition_ms:float
    total_ms:float

class InferenceModelOption(BaseModel):
    id:str
    name:str
    config:str
    checkpoint:str|None=None
    recognition_version:str
    recognition_checkpoint:str|None=None
    fixture:bool
    backends:list[str]

class InferenceModelsResponse(BaseModel):
    default_model:str|None=None
    loaded_model:str|None=None
    models:list[InferenceModelOption]

class PredictionResponse(BaseModel):
    success:bool
    target_found:bool
    sku_id:str|None=None
    identity_score:float|None=None
    reason:str|None=None
    instances:list[InstanceResult]|None=None
    candidates:list[dict]|None=None
    result_id:str|None=None
    backend:str|None=None
    model_id:str|None=None
    model_name:str|None=None
    timing:Timing|None=None
