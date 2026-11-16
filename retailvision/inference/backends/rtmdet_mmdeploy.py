"""RTMDet-Ins ONNX/TensorRT adapter with MMDeploy preprocessing and mask rescaling."""
from pathlib import Path
import numpy as np
from PIL import Image
from retailvision.inference.interfaces import Instance


class _TensorRTOutputAllocator:
    """CUDA allocator callback for TensorRT's data-dependent NMS output shapes."""
    def __new__(cls, trt, torch, dtype, stream):
        class Allocator(trt.IOutputAllocator):
            def __init__(self):
                trt.IOutputAllocator.__init__(self)
                self.buffer=None;self.shape=None;self.dtype=dtype;self.torch=torch;self.stream=stream;self.offset=0;self.byte_size=0
            def reallocate_output(self,name,memory,size,alignment):
                alignment=max(int(alignment),1)
                if self.buffer is None or int(size)>self.byte_size:
                    self.buffer=self.torch.empty(int(size)+alignment,dtype=self.torch.uint8,device='cuda')
                    self.buffer.record_stream(self.stream)
                    base=int(self.buffer.data_ptr());aligned=(base+alignment-1)//alignment*alignment
                    self.offset=aligned-base;self.byte_size=int(size)
                return int(self.buffer.data_ptr())+self.offset
            def notify_shape(self,name,shape):self.shape=tuple(shape)
        return Allocator()


class RTMDetMMDeploy:
    def __init__(self, model_config: str, deploy_config: str, artifact: str, backend: str):
        self.model_config=str(Path(model_config).resolve())
        self.deploy_config=str(Path(deploy_config).resolve())
        self.artifact=str(Path(artifact).resolve())
        self.backend=backend
        self.processor=self.model_cfg=self.deploy_cfg=self.data_preprocessor=None
        self.session=self.runtime=self.engine=self.context=None
        self.output_allocators={}

    def load(self):
        import torch
        import sys
        from retailvision.paths import project_path
        source=str(project_path('third_party/mmdeploy'))
        if source not in sys.path:sys.path.insert(0,source)
        from mmdeploy.apis.utils import build_task_processor
        from mmdeploy.utils import load_config
        self.deploy_cfg,self.model_cfg=load_config(self.deploy_config,self.model_config)
        self.processor=build_task_processor(self.model_cfg,self.deploy_cfg,'cuda')
        from mmdet.registry import MODELS
        self.data_preprocessor=MODELS.build(self.model_cfg.model.data_preprocessor).cuda()
        if self.backend=='onnxruntime':
            from retailvision.inference.backends.onnx_cuda import preload_cuda12
            preload_cuda12()
            import onnxruntime as ort
            self.session=ort.InferenceSession(self.artifact,providers=['CUDAExecutionProvider'])
            if self.session.get_providers()[0]!='CUDAExecutionProvider':
                raise RuntimeError(f'ONNX Runtime CUDA provider unavailable: {self.session.get_providers()}')
            self.input_name=self.session.get_inputs()[0].name
            self.output_names=[x.name for x in self.session.get_outputs()]
        elif self.backend=='tensorrt':
            import tensorrt as trt
            self.logger=trt.Logger(trt.Logger.WARNING)
            trt.init_libnvinfer_plugins(self.logger,'')
            self.runtime=trt.Runtime(self.logger)
            self.engine=self.runtime.deserialize_cuda_engine(Path(self.artifact).read_bytes())
            if self.engine is None:raise RuntimeError(f'Cannot deserialize TensorRT engine {self.artifact}')
            self.context=self.engine.create_execution_context()
            self.input_names=[];self.output_names=[]
            for i in range(self.engine.num_io_tensors):
                name=self.engine.get_tensor_name(i)
                if self.engine.get_tensor_mode(name)==trt.TensorIOMode.INPUT:self.input_names.append(name)
                else:self.output_names.append(name)
            if len(self.input_names)!=1 or not {'dets','labels','masks'}<=set(self.output_names):
                raise RuntimeError(f'Expected one input and dets/labels/masks outputs, got {self.input_names}/{self.output_names}')
            self.input_name=self.input_names[0]
            self.stream=torch.cuda.Stream()
        else:raise ValueError(f'Unsupported RTMDet deployment backend: {self.backend}')

    def _execute(self, inputs):
        import torch
        tensor=inputs['inputs']
        if isinstance(tensor,(list,tuple)):tensor=torch.stack(tensor)
        tensor=tensor.contiguous()
        if self.backend=='onnxruntime':
            outputs=self.session.run(self.output_names,{self.input_name:tensor.detach().cpu().numpy()})
            return dict(zip(self.output_names,outputs))
        import torch
        import tensorrt as trt
        dtype=torch.float16 if self.engine.get_tensor_dtype(self.input_name)==trt.DataType.HALF else torch.float32
        tensor=tensor.to(dtype=dtype)
        self.stream.wait_stream(torch.cuda.current_stream())
        tensor.record_stream(self.stream)
        if not self.context.set_input_shape(self.input_name,tuple(tensor.shape)):
            raise RuntimeError(f'TensorRT rejected RTMDet input shape {tuple(tensor.shape)}')
        self.context.set_tensor_address(self.input_name,tensor.data_ptr())
        buffers={};allocators={}
        dtype_map={trt.DataType.FLOAT:torch.float32,trt.DataType.HALF:torch.float16,
                   trt.DataType.INT32:torch.int32,trt.DataType.INT64:torch.int64,
                   trt.DataType.BOOL:torch.bool}
        for name in self.output_names:
            shape=tuple(self.engine.get_tensor_shape(name))
            dtype=dtype_map[self.engine.get_tensor_dtype(name)]
            if any(x<0 for x in shape):
                allocator=self.output_allocators.get(name)
                if allocator is None:
                    allocator=_TensorRTOutputAllocator(trt,torch,dtype,self.stream)
                    self.output_allocators[name]=allocator
                    self.context.set_output_allocator(name,allocator)
                allocators[name]=allocator
            else:
                output=torch.empty(shape,dtype=dtype,device='cuda')
                self.context.set_tensor_address(name,output.data_ptr());buffers[name]=output
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError('TensorRT RTMDet execution failed')
        self.stream.synchronize()
        for name,allocator in allocators.items():
            if allocator.buffer is None or allocator.shape is None:
                raise RuntimeError(f'TensorRT did not allocate dynamic output {name}')
            size=int(np.prod(allocator.shape))*torch.empty((),dtype=allocator.dtype).element_size()
            buffers[name]=allocator.buffer[allocator.offset:allocator.offset+size].view(allocator.dtype).reshape(allocator.shape)
        return buffers

    def predict(self,image:Image.Image,threshold:float=.3)->list[Instance]:
        if self.processor is None:raise RuntimeError('RTMDet deployment adapter is not loaded')
        bgr=np.asarray(image.convert('RGB'))[:,:,::-1].copy()
        # Keep the source pipeline's aspect ratio; it resizes then pads to 640.
        # Passing an explicit (640, 640) here stretches the image vertically.
        inputs,_=self.processor.create_input(bgr)
        inputs=self.data_preprocessor(inputs,training=False)
        outputs=self._execute(inputs)
        import torch
        def host(value):
            return value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else np.asarray(value)
        dets=host(outputs['dets'])[0]
        labels=host(outputs['labels'])[0]
        masks=outputs['masks']
        if masks.ndim==4:masks=masks[0]
        selected=np.flatnonzero((dets[:,4]>=threshold)&(labels>=0))
        if not len(selected):return []
        dets,labels=dets[selected],labels[selected]
        mask_indices=torch.as_tensor(selected,device=masks.device) if isinstance(masks,torch.Tensor) else selected
        masks=masks[mask_indices]
        meta=inputs['data_samples'][0].metainfo
        scale=np.asarray(meta.get('scale_factor',(1.,1.)),dtype=np.float32).reshape(-1)
        if scale.size==2:scale=np.tile(scale,2)
        # Source bbox decoding clips to img_shape, including the configured
        # padding canvas. Rescale that same canvas rather than clipping to ori_shape.
        canvas_h,canvas_w=meta['img_shape'][:2]
        clipped=dets[:,:4].copy()
        clipped[:,[0,2]]=np.clip(clipped[:,[0,2]],0,canvas_w)
        clipped[:,[1,3]]=np.clip(clipped[:,[1,3]],0,canvas_h)
        boxes=clipped/scale.reshape(1,4)
        ori_h,ori_w=meta['ori_shape'][:2]
        factor=meta.get('scale_factor',(1.,1.))
        factor=np.asarray(factor).reshape(-1)
        if factor.size==4:factor=factor[:2]
        out_h=int(np.ceil(masks.shape[-2]/factor[1]));out_w=int(np.ceil(masks.shape[-1]/factor[0]))
        import torch
        resized=torch.nn.functional.interpolate(torch.as_tensor(masks,dtype=torch.float32,device='cuda').unsqueeze(0),
            size=(out_h,out_w),mode='nearest')[0,:,:ori_h,:ori_w]>=.5
        results=[]
        for i,(box,score,label) in enumerate(zip(boxes,dets[:,4],labels)):
            if i>=len(resized):raise RuntimeError('TensorRT/ONNX detection and mask output counts differ')
            results.append(Instance(box.tolist(),resized[i].cpu().numpy().astype(bool),float(score),int(label)))
        return results

    def unload(self):
        import torch
        self.processor=self.session=self.context=self.engine=self.runtime=self.data_preprocessor=None
        self.output_allocators={}
        torch.cuda.empty_cache()

    def get_model_info(self):
        return dict(architecture='RTMDet-Ins-s',backend=self.backend,model_file=self.artifact,
                    deployment_config=self.deploy_config,loaded=self.processor is not None,
                    device='cuda:0',precision='fp32' if self.backend=='onnxruntime' else 'fp16')
