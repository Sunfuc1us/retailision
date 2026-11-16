"""Build a TensorRT engine from ONNX and record the immutable build inputs."""
import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--input-name', required=True)
    parser.add_argument('--min-shape', help='e.g. 1x3x224x224; omit for static input')
    parser.add_argument('--opt-shape', help='e.g. 1x3x224x224')
    parser.add_argument('--max-shape', help='e.g. 4x3x224x224')
    parser.add_argument('--workspace-gib', type=int, default=4)
    parser.add_argument('--plugin', action='append', default=[])
    parser.add_argument('--precision', choices=('fp16','fp32'), default='fp16')
    parser.add_argument('--disable-tf32', action='store_true',
                        help='Use IEEE FP32 math instead of TensorFloat-32 on supported GPUs')
    args = parser.parse_args()

    import tensorrt as trt
    onnx_path, output = Path(args.onnx).resolve(), Path(args.output).resolve()
    if not onnx_path.is_file():
        parser.error(f'ONNX file does not exist: {onnx_path}')
    output.parent.mkdir(parents=True, exist_ok=True)
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, '')
    for plugin in args.plugin:
        import ctypes
        ctypes.CDLL(str(Path(plugin).resolve()), mode=ctypes.RTLD_GLOBAL)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser_onnx = trt.OnnxParser(network, logger)
    if not parser_onnx.parse(onnx_path.read_bytes()):
        errors = '\n'.join(str(parser_onnx.get_error(i)) for i in range(parser_onnx.num_errors))
        raise RuntimeError(f'TensorRT could not parse ONNX operators:\n{errors}')
    input_names=[network.get_input(i).name for i in range(network.num_inputs)]
    if args.input_name not in input_names:
        parser.error(f'input {args.input_name!r} is not present in ONNX; expected {input_names}')
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gib * (1 << 30))
    if args.precision=='fp16':config.set_flag(trt.BuilderFlag.FP16)
    if args.disable_tf32:config.clear_flag(trt.BuilderFlag.TF32)
    if args.min_shape or args.opt_shape or args.max_shape:
        if not all((args.min_shape, args.opt_shape, args.max_shape)):
            parser.error('provide min, opt, and max shapes together')
        shape = lambda value: tuple(int(v) for v in value.lower().split('x'))
        profile = builder.create_optimization_profile()
        profile.set_shape(args.input_name, shape(args.min_shape), shape(args.opt_shape), shape(args.max_shape))
        config.add_optimization_profile(profile)
    build_started=perf_counter()
    plan = builder.build_serialized_network(network, config)
    build_seconds=perf_counter()-build_started
    if plan is None:
        raise RuntimeError('TensorRT engine build failed; inspect parser/builder errors above')
    output.write_bytes(bytes(plan))
    import torch
    import ctypes
    cuda_library=Path(torch.__file__).resolve().parents[1]/'nvidia/cuda_runtime/lib/libcudart.so.12'
    cuda_version=ctypes.c_int()
    if ctypes.CDLL(str(cuda_library)).cudaRuntimeGetVersion(ctypes.byref(cuda_version))!=0:
        raise RuntimeError('Cannot query CUDA runtime version')
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                         capture_output=True, text=True, check=False).stdout.strip()
    manifest_path = output.parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.pop('validation',None)
    manifest['engine'] = dict(path=str(output), sha256=sha256(output), size_bytes=output.stat().st_size,
        gpu_architecture=f'{gpu}; compute_capability={torch.cuda.get_device_capability()}',
        cuda_runtime=f'{cuda_version.value//1000}.{(cuda_version.value%1000)//10}',
        pytorch_cuda_build=torch.version.cuda, tensorrt_version=trt.__version__,
        onnx_source=str(onnx_path), onnx_sha256=sha256(onnx_path), precision=args.precision.upper(),
        input_name=args.input_name, min_shape=args.min_shape, opt_shape=args.opt_shape,
        max_shape=args.max_shape, workspace_gib=args.workspace_gib,
        input_shapes={network.get_input(i).name:list(network.get_input(i).shape) for i in range(network.num_inputs)},
        outputs={network.get_output(i).name:dict(shape=list(network.get_output(i).shape),dtype=str(network.get_output(i).dtype)) for i in range(network.num_outputs)},
        tf32_enabled=bool(config.get_flag(trt.BuilderFlag.TF32)),
        build_seconds=build_seconds,
        build_time_utc=datetime.now(timezone.utc).isoformat())
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest['engine'], indent=2))


if __name__ == '__main__':
    main()
