"""Run MMDeploy's CLI after preloading CUDA shared libraries for ORT CUDA EP."""
import runpy
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'third_party/mmdeploy'))
from retailvision.inference.backends.onnx_cuda import preload_cuda12
preload_cuda12()
sys.argv[0]=str(ROOT/'third_party/mmdeploy/tools/deploy.py')
runpy.run_path(sys.argv[0],run_name='__main__')
