"""DINOv2 embedding inference adapters for ONNX Runtime and native TensorRT engines."""
from pathlib import Path
import numpy as np
from PIL import Image


class DINOv2Engine:
    def __init__(self, model_path: str, backend: str, input_dtype: str = 'float32',
                 projection_output: int | None = None):
        self.model_path = Path(model_path)
        self.backend = backend
        self.input_dtype = input_dtype
        self.projection_output = projection_output
        self.session = None
        self.engine = None
        self.runtime = None
        self.context = None
        self.stream = None
        self.input_name = None
        self.output_name = None

    def load(self):
        if self.backend == 'onnxruntime':
            from retailvision.inference.backends.onnx_cuda import preload_cuda12
            preload_cuda12()
            import onnxruntime as ort
            self.session = ort.InferenceSession(str(self.model_path), providers=['CUDAExecutionProvider'])
            if self.session.get_providers()[0] != 'CUDAExecutionProvider':
                raise RuntimeError(f'ONNX Runtime did not select CUDAExecutionProvider: {self.session.get_providers()}')
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.embedding_dim=self.projection_output or int(self.session.get_outputs()[0].shape[-1])
        elif self.backend == 'tensorrt':
            import tensorrt as trt
            import torch
            self.logger = trt.Logger(trt.Logger.WARNING)
            self.runtime = trt.Runtime(self.logger)
            with self.model_path.open('rb') as f:
                self.engine = self.runtime.deserialize_cuda_engine(f.read())
            if self.engine is None:
                raise RuntimeError(f'Cannot deserialize TensorRT engine {self.model_path}')
            self.context = self.engine.create_execution_context()
            self.stream = torch.cuda.Stream()
            inputs = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
                      if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
            outputs = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
                       if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
            if len(inputs) != 1 or len(outputs) != 1:
                raise RuntimeError(f'Expected one embedding input/output, got {inputs}/{outputs}')
            self.input_name, self.output_name = inputs[0], outputs[0]
            self.embedding_dim=self.projection_output or int(self.engine.get_tensor_shape(self.output_name)[-1])
        else:
            raise ValueError(f'Unsupported DINOv2 engine backend: {self.backend}')

    @staticmethod
    def preprocess(image: Image.Image) -> np.ndarray:
        import torch
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225))])
        return transform(image.convert('RGB')).unsqueeze(0).numpy()

    def extract_embedding(self, image: Image.Image) -> np.ndarray:
        x = self.preprocess(image)
        if self.backend == 'onnxruntime':
            y = self.session.run([self.output_name], {self.input_name: x.astype(self.input_dtype)})[0]
        else:
            import torch
            dtype = torch.float16 if self.engine.get_tensor_dtype(self.input_name).name == 'HALF' else torch.float32
            tensor = torch.from_numpy(x).to(device='cuda', dtype=dtype)
            self.stream.wait_stream(torch.cuda.current_stream())
            tensor.record_stream(self.stream)
            self.context.set_input_shape(self.input_name, tuple(tensor.shape))
            self.context.set_tensor_address(self.input_name, tensor.data_ptr())
            shape = tuple(self.context.get_tensor_shape(self.output_name))
            output_dtype = torch.float16 if self.engine.get_tensor_dtype(self.output_name).name == 'HALF' else torch.float32
            output = torch.empty(shape, dtype=output_dtype, device='cuda')
            output.record_stream(self.stream)
            self.context.set_tensor_address(self.output_name, output.data_ptr())
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError('TensorRT DINOv2 execution failed')
            self.stream.synchronize()
            y = output.float().cpu().numpy()
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(y))
        if y.shape != (self.embedding_dim,) or not np.isfinite(y).all() or norm == 0:
            raise RuntimeError(f'Invalid DINOv2 embedding shape={y.shape}, norm={norm}')
        return y / norm

    def unload(self):
        self.session = self.engine = self.context = self.runtime = None

    def get_model_info(self):
        return dict(architecture='DINOv2 ViT-B/14', model_file=str(self.model_path),
                    backend=self.backend, precision=self.input_dtype)
