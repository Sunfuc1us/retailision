from retailvision.paths import activate_project_root
activate_project_root()
import io,json,os,subprocess
from pathlib import Path
from retailvision.inference.runtime import Runtime
from retailvision.inference import gallery_edits
from retailvision.paths import project_path
from typing import Literal
from fastapi import FastAPI,File,Form,HTTPException,UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from PIL import Image,UnidentifiedImageError
from retailvision.api.requests import DatasetRequest,SKURegisterRequest,ModelRegisterRequest
from retailvision.training.jobs import Jobs
from retailvision.training.options import TrainingRequest
from retailvision.api.responses import PredictionResponse,InferenceModelsResponse
from retailvision.data.catalog import Catalog
from retailvision.artifacts.registry import Registry
from retailvision.data.validate import validate
from retailvision.data.prepare import prepare

from retailvision.logging import configure_logging
configure_logging('logs/api.log')
app=FastAPI(title='RetailVision',version='1.0.0');jobs=Jobs();runtime=Runtime()
def read_result(path: Path):
    if not path.is_file():raise HTTPException(404,'not found')
    return json.loads(path.read_text())
@app.get('/api/health')
def health():return dict(status='ok',product='RetailVision',runtime_loaded=runtime.pipe is not None,configured=bool(os.getenv('RETAILVISION_CONFIG')),fixture=json.loads(Path(os.environ['RETAILVISION_CONFIG']).read_text()).get('fixture',False) if os.getenv('RETAILVISION_CONFIG') else False)
@app.get('/api/system/gpu')
def gpu():
    p=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
    return dict(available=p.returncode==0,gpus=p.stdout.strip().splitlines(),error=p.stderr or None)
@app.get('/api/models')
def models():return Registry().list()
@app.post('/api/models/register',status_code=201)
def register_model(req: ModelRegisterRequest):
    try:
        out=Registry().register(req.task,req.version,req.checkpoint,req.config,req.metadata)
        return json.loads((out/'manifest.json').read_text())
    except FileExistsError:raise HTTPException(409,'model version already exists')
    except (ValueError,OSError) as e:raise HTTPException(422,str(e))
def selected_profile(model_id: str | None = None):
    from retailvision.artifacts.profiles import resolve_profile
    try:return resolve_profile(model_id)
    except ValueError as e:raise HTTPException(422,str(e))
    except RuntimeError as e:raise HTTPException(503,str(e))

@app.get('/api/inference/models',response_model=InferenceModelsResponse)
def inference_models():
    from retailvision.artifacts.profiles import profiles,describe,resolve_profile
    try:default=resolve_profile().id
    except RuntimeError:default=None
    return dict(default_model=default,loaded_model=getattr(runtime.pipe,'model_id',None),models=[describe(x) for x in profiles()])

@app.get('/api/inference/runtime')
def inference_runtime():return runtime.info()

@app.get('/api/inference/backends')
def inference_backends(model_id: str | None = None):
    from retailvision.artifacts.profiles import available_backends,resolve_profile
    try:profile=resolve_profile(model_id)
    except RuntimeError:return {'available':[]}
    except ValueError as e:raise HTTPException(422,str(e))
    return {'available':available_backends(profile)}

def selected_catalog(model_id: str | None = None):
    from retailvision.paths import project_path
    if model_id is None and not os.getenv('RETAILVISION_CONFIG'):return Catalog('catalog/skus.json')
    return Catalog(project_path(selected_profile(model_id).settings()['catalog']))

@app.get('/api/catalog')
def catalog(q: str = '',model_id: str | None = None):
    c=selected_catalog(model_id);return c.resolve(q) if q else list(c.items.values())
@app.post('/api/catalog',status_code=201)
def register_sku(req: SKURegisterRequest):
    try:
        with runtime.lock:
            c=selected_catalog(req.model_id)
            entry=c.register(req.sku_id,req.product,req.brand,req.flavor,req.specification)
            c.save()
            if runtime.pipe is not None and Path(runtime.pipe.catalog.path).resolve()==c.path.resolve():
                runtime.pipe.catalog=c
            return entry
    except ValueError as e:raise HTTPException(422,str(e))
@app.get('/api/catalog/{sku_id}')
def sku(sku_id: str,model_id: str | None = None):
    c=selected_catalog(model_id)
    if sku_id not in c.items:raise HTTPException(404,'unknown SKU')
    return c.items[sku_id]
@app.delete('/api/catalog/{sku_id}')
def delete_sku(sku_id: str,model_id: str | None = None):
    with runtime.lock:
        c=selected_catalog(model_id)
        if sku_id not in c.items:raise HTTPException(404,'unknown SKU')
        c.remove(sku_id);c.save()
        if runtime.pipe is not None and Path(runtime.pipe.catalog.path).resolve()==c.path.resolve():
            runtime.pipe.catalog=c
        return dict(deleted=sku_id)

@app.get('/api/gallery')
def gallery_info(model_id: str | None = None):
    from retailvision.paths import project_path
    from retailvision.inference.gallery import Gallery
    profile=selected_profile(model_id)
    gpath=project_path(profile.settings()['gallery'])
    if not (gpath/'embeddings.npy').is_file():return dict(ntotal=0,unique_skus=0,counts={})
    return Gallery.load(str(gpath)).summary()

def _parse_image(image: UploadFile) -> Image.Image:
    try:
        data=image.file.read(25*1024*1024+1)
        if len(data)>25*1024*1024:raise HTTPException(413,'image exceeds 25MB')
        rgb=Image.open(io.BytesIO(data));rgb.load();return rgb.convert('RGB')
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError):raise HTTPException(422,'invalid image')

def _ensure_runtime(model_id: str | None, backend: str):
    try:return runtime.ensure(selected_profile(model_id),backend)
    except ValueError as error:raise HTTPException(422,str(error))
    except RuntimeError as error:raise HTTPException(503,str(error))

@app.post('/api/gallery/references',status_code=201)
def add_gallery_reference(image: UploadFile=File(...),sku_id: str=Form(...,min_length=1),model_id: str | None=Form(None),backend: Literal['pytorch','onnxruntime','tensorrt']=Form('pytorch')):
    rgb=_parse_image(image)
    if sku_id not in selected_catalog(model_id).items:raise HTTPException(404,'unknown SKU; register SKU in catalog first')
    with runtime.lock:
        pipe=_ensure_runtime(model_id,backend)
        path=project_path(selected_profile(model_id).settings()['gallery'])
        return gallery_edits.add_reference(path,pipe,rgb,sku_id)

@app.delete('/api/gallery/{sku_id}')
def remove_gallery_sku(sku_id: str,model_id: str | None = None):
    profile=selected_profile(model_id)
    with runtime.lock:
        pipe=runtime.pipe if getattr(runtime.pipe,'model_id',None)==profile.id else None
        try:return gallery_edits.remove_sku(project_path(profile.settings()['gallery']),sku_id,pipe)
        except (FileNotFoundError,KeyError) as error:raise HTTPException(404,str(error))
        except ValueError as error:raise HTTPException(422,str(error))

@app.post('/api/inference/diagnose')
def diagnose(image: UploadFile=File(...),detection_threshold: float=Form(.8,ge=0,le=1),top_k: int=Form(5,ge=1,le=20),backend: Literal['pytorch','onnxruntime','tensorrt']=Form('pytorch'),model_id: str | None=Form(None)):
    rgb=_parse_image(image)
    with runtime.lock:
        pipe=_ensure_runtime(model_id,backend)
        return pipe.diagnose(rgb,detection_threshold,top_k=top_k)

@app.post('/api/inference/predict',response_model=PredictionResponse,response_model_exclude_none=True)
def predict(image: UploadFile=File(...),target_product: str=Form(...,min_length=1),detection_threshold: float=Form(.8,ge=0,le=1),backend: Literal['pytorch','onnxruntime','tensorrt']=Form('pytorch'),model_id: str | None=Form(None)):
    rgb=_parse_image(image)
    with runtime.lock:
        pipe=_ensure_runtime(model_id,backend)
        return pipe.predict(rgb,target_product,detection_threshold)
@app.post('/api/datasets/validate')
def dataset_validate(req: DatasetRequest):
    try:r=validate(req.coco,req.images,req.catalog,req.sidecar,require_capture_group=not req.no_split and req.validation_images is None);r.pop('records');return r
    except (ValueError,OSError,KeyError) as e:raise HTTPException(422,str(e))
@app.post('/api/datasets/prepare')
def dataset_prepare(req: DatasetRequest):
    if not req.output:raise HTTPException(422,'output required')
    try:return prepare(req.coco,req.images,req.catalog,req.output,req.sidecar,req.seed,no_split=req.no_split,validation_images=req.validation_images)
    except (ValueError,OSError) as e:raise HTTPException(422,str(e))
@app.post('/api/training/start',status_code=202)
def start(req: TrainingRequest):
    try:return dict(experiment_id=jobs.start(req),status='queued')
    except ValueError as e:raise HTTPException(422,str(e))
def _experiment_paths() -> list[Path]:
    root=Path('work_dirs')
    return sorted({*root.glob('*/experiment.json'),*root.glob('experiments/*/experiment.json')})
@app.get('/api/experiments')
def experiments():return [json.loads(p.read_text()) for p in _experiment_paths()]
@app.get('/api/experiments/{experiment_id}')
@app.get('/api/training/{experiment_id}')
def experiment(experiment_id: str):
    if Path(experiment_id).name!=experiment_id:raise HTTPException(404,'not found')
    for candidate in (Path('work_dirs')/experiment_id/'experiment.json',Path('work_dirs/experiments')/experiment_id/'experiment.json'):
        if candidate.is_file():return read_result(candidate)
    raise HTTPException(404,'not found')
@app.post('/api/training/{experiment_id}/stop')
def stop(experiment_id: str):
    try:return jobs.stop(experiment_id)
    except KeyError:raise HTTPException(404,'job not owned by this server')
def result_file(result_id: str,filename: str) -> Path:
    from retailvision.artifacts.profiles import result_roots
    for root in result_roots():
        candidate=root/result_id/filename
        if candidate.is_file():return candidate
    raise HTTPException(404,'not found')
@app.get('/api/results/{result_id}')
def result(result_id: str):return read_result(result_file(result_id,'result.json'))
@app.get('/api/results/{result_id}/{filename}')
def mask(result_id: str,filename: str):
    if not filename.startswith('mask_') or not filename.endswith('.png'):raise HTTPException(404)
    if Path(filename).name!=filename:raise HTTPException(404)
    try:return FileResponse(result_file(result_id,filename))
    except HTTPException:
        data=read_result(result_file(result_id,'result.json'))
        item=next((x for x in data.get('candidates',[]) if x.get('mask_path')==f'/api/results/{result_id}/{filename}'),None)
        if not item or not item.get('mask_preview_path'):raise HTTPException(404)
        with Image.open(result_file(result_id,Path(item['mask_preview_path']).name)) as crop:
            full=Image.new('L',tuple(item['image_size']));full.paste(crop,tuple(item['mask_origin']))
            buffer=io.BytesIO();full.save(buffer,format='PNG',compress_level=1)
        return Response(buffer.getvalue(),media_type='image/png')
if Path('frontend/dist').exists():app.mount('/',StaticFiles(directory='frontend/dist',html=True),name='workbench')
