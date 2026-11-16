export _RETAILVISION_PREVIOUS_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
export _RETAILVISION_LD_LIBRARY_PATH_WAS_SET="${LD_LIBRARY_PATH+x}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.10/site-packages/torch/lib:${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/curand/lib:${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cufft/lib:${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
