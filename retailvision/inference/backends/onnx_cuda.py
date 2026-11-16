"""Load CUDA 12 runtime DSOs from the isolated PyTorch/NVIDIA pip environment."""
import ctypes
from pathlib import Path


def preload_cuda12():
    import torch
    site = Path(torch.__file__).resolve().parents[1]
    candidates = [
        site/'torch/lib/libcudnn.so.8',
        site/'torch/lib/libcublasLt.so.12',
        site/'torch/lib/libcublas.so.12',
        site/'nvidia/cuda_runtime/lib/libcudart.so.12',
        site/'nvidia/curand/lib/libcurand.so.10',
        site/'nvidia/cufft/lib/libcufft.so.11',
    ]
    loaded=[]
    for path in candidates:
        if path.is_file():
            loaded.append(ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL))
    return loaded
