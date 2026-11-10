from project_paths import activate_project_root
activate_project_root()
import io,json,os,subprocess
from pathlib import Path
from threading import Lock
from fastapi import FastAPI,File,Form,HTTPException,UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image,UnidentifiedImageError
from backend.schemas.requests import DatasetRequest,TrainingRequest
from backend.services.jobs import Jobs
from backend.schemas.responses import PredictionResponse
from catalog.store import Catalog
from registry.store import Registry
from dataset.validators.coco import validate
from dataset.converters.prepare import prepare

from training.common.logging import configure_logging
configure_logging('logs/api.log')
app=FastAPI(title='RetailVision',version='1.0.0');jobs=Jobs();runtime=None;runtime_lock=Lock()
def read_result(path: Path):
    if not path.is_file():raise HTTPException(404,'not found')
    return json.loads(path.read_text())
@app.get('/api/health')
def health():return dict(status='ok',product='RetailVision',runtime_loaded=runtime is not None,configured=bool(os.getenv('RETAILVISION_CONFIG')),fixture=json.loads(Path(os.environ['RETAILVISION_CONFIG']).read_text()).get('fixture',False) if os.getenv('RETAILVISION_CONFIG') else False)
@app.get('/api/system/gpu')
def gpu():
    p=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
    return dict(available=p.returncode==0,gpus=p.stdout.strip().splitlines(),error=p.stderr or None)
@app.get('/api/models')
def models():return Registry().list()
@app.get('/api/catalog')
def catalog(q: str = ''):
    c=Catalog('catalog/skus.json');return c.resolve(q) if q else list(c.items.values())
@app.get('/api/catalog/{sku_id}')
def sku(sku_id: str):
    c=Catalog('catalog/skus.json')
    if sku_id not in c.items:raise HTTPException(404,'unknown SKU')
    return c.items[sku_id]
@app.post('/api/inference/predict',response_model=PredictionResponse,response_model_exclude_none=True)
def predict(image: UploadFile=File(...),target_product: str=Form(...,min_length=1),detection_threshold: float=Form(.3,ge=0,le=1)):
    global runtime
    try:
        data=image.file.read(25*1024*1024+1)
        if len(data)>25*1024*1024:raise HTTPException(413,'image exceeds 25MB')
        rgb=Image.open(io.BytesIO(data));rgb.load();rgb=rgb.convert('RGB')
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError):raise HTTPException(422,'invalid image')
    with runtime_lock:
        if runtime is None:
            cfg=os.getenv('RETAILVISION_CONFIG')
            if not cfg:raise HTTPException(503,'Local models, SKU gallery and validation calibration must be configured')
            try:
                from inference.factory import load_pipeline
                runtime=load_pipeline(cfg)
            except Exception as e:raise HTTPException(503,str(e))
        return runtime.predict(rgb,target_product,detection_threshold)
@app.post('/api/datasets/validate')
def dataset_validate(req: DatasetRequest):
    try:r=validate(req.coco,req.images,req.catalog,req.sidecar);r.pop('records');return r
    except (ValueError,OSError,KeyError) as e:raise HTTPException(422,str(e))
@app.post('/api/datasets/prepare')
def dataset_prepare(req: DatasetRequest):
    if not req.output:raise HTTPException(422,'output required')
    try:return prepare(req.coco,req.images,req.catalog,req.output,req.sidecar,req.seed)
    except (ValueError,OSError) as e:raise HTTPException(422,str(e))
@app.post('/api/training/start',status_code=202)
def start(req: TrainingRequest):
    try:return dict(experiment_id=jobs.start(req),status='queued')
    except ValueError as e:raise HTTPException(422,str(e))
@app.get('/api/experiments')
def experiments():return [json.loads(p.read_text()) for p in sorted(Path('work_dirs').glob('*/experiment.json'))]
@app.get('/api/experiments/{experiment_id}')
@app.get('/api/training/{experiment_id}')
def experiment(experiment_id: str):return read_result(Path('work_dirs')/experiment_id/'experiment.json')
@app.post('/api/training/{experiment_id}/stop')
def stop(experiment_id: str):
    try:return jobs.stop(experiment_id)
    except KeyError:raise HTTPException(404,'job not owned by this server')
@app.get('/api/results/{result_id}')
def result(result_id: str):return read_result(Path('results')/result_id/'result.json')
@app.get('/api/results/{result_id}/{filename}')
def mask(result_id: str,filename: str):
    if not filename.startswith('mask_') or not filename.endswith('.png'):raise HTTPException(404)
    p=Path('results')/result_id/filename
    if not p.is_file():raise HTTPException(404)
    return FileResponse(p)
if Path('frontend/dist').exists():app.mount('/',StaticFiles(directory='frontend/dist',html=True),name='workbench')
