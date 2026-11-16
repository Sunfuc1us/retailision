"""Build the local Workbench and start its API with an explicit model configuration."""
import _bootstrap
import argparse
import os
from pathlib import Path
import subprocess
from retailvision.paths import PROJECT_ROOT,project_path
from retailvision.artifacts.profiles import profiles,available_backends,ModelProfile


def main():
    choices=profiles()
    parser=argparse.ArgumentParser(description='RetailVision Web Workbench：本地模型权重选择与推理')
    selection=parser.add_mutually_exclusive_group()
    selection.add_argument('--model',choices=[x.id for x in choices],help='选择配套权重、模型配置、SKU 图库和阈值；默认 retail_6sku_v002')
    selection.add_argument('--config',help='指定其他完整推理 JSON 配置；也可使用 RETAILVISION_CONFIG')
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=8018)
    args=parser.parse_args()
    if args.config:
        profile=ModelProfile('configured',Path(args.config).stem,project_path(args.config).resolve())
    elif args.model:
        profile=next(x for x in choices if x.id==args.model)
    elif os.getenv('RETAILVISION_CONFIG'):
        configured=project_path(os.environ['RETAILVISION_CONFIG']).resolve()
        profile=next(x for x in choices if x.config==configured)
    else:
        profile=next(x for x in choices if x.id=='retail_6sku_v002')
    if not available_backends(profile):
        parser.error(f'模型配置没有可用 CUDA Backend，请检查权重、图库与环境：{profile.config}')
    frontend=PROJECT_ROOT/'frontend'
    if not (frontend/'node_modules').is_dir():
        parser.error('前端依赖未安装；请先执行 cd frontend && npm ci')
    subprocess.run(['npm','run','build'],cwd=frontend,check=True)
    os.environ['RETAILVISION_CONFIG']=str(profile.config)
    config=profile.settings()
    print(f'\nRetailVision: http://{args.host}:{args.port}\n模型: {profile.name}\n推理配置: {profile.config}\n分割权重: {project_path(config["segmentation"]["checkpoint"])}\n模型将在首次推理时加载。',flush=True)
    import uvicorn
    uvicorn.run('retailvision.api.app:app',host=args.host,port=args.port)


if __name__=='__main__':main()
