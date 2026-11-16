"""Export normalized DINOv2 ViT-B/14 embeddings to ONNX without changing the checkpoint."""
import argparse, json, sys
from pathlib import Path
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from retailvision.inference.dino import DINOv2


class EmbeddingModel(nn.Module):
    def __init__(self, model, projection=None):
        super().__init__()
        self.model = model
        self.projection = projection

    def forward(self, images):
        embeddings = self.model(images)
        if self.projection is not None:
            embeddings = self.projection(embeddings)
        return torch.nn.functional.normalize(embeddings.float(), dim=-1)


class ExportAttention(nn.Module):
    """Express SDPA with basic ONNX operators, using the original learned weights."""
    def __init__(self, attention):
        super().__init__()
        self.qkv=attention.qkv
        self.proj=attention.proj
        self.proj_drop=attention.proj_drop
        self.num_heads=attention.num_heads
        self.scale=attention.scale

    def forward(self, x):
        batch,tokens,channels=x.shape
        qkv=self.qkv(x).reshape(batch,tokens,3,self.num_heads,channels//self.num_heads).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]
        attention=((q*self.scale)@k.transpose(-2,-1)).softmax(dim=-1)
        output=(attention@v).transpose(1,2).reshape(batch,tokens,channels)
        return self.proj_drop(self.proj(output))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='models/recognition/dinov2_b14_v001/model.pth')
    parser.add_argument('--projection', help='Optional exported Linear head checkpoint: best_head.pth')
    parser.add_argument('--output', default='models/recognition/dinov2_b14_v001/deployment/model.onnx')
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--dynamic-batch', action='store_true')
    args = parser.parse_args()
    torch.cuda.set_device(0)
    adapter = DINOv2('third_party/dinov2', args.checkpoint, args.projection)
    adapter.load()
    sample = torch.randn(1, 3, 224, 224, device='cuda')
    with torch.inference_mode():
        reference=adapter.model(sample)
        for block in adapter.model.blocks:
            block.attn=ExportAttention(block.attn)
        adapted=adapter.model(sample)
    attention_delta=float((reference-adapted).abs().max())
    if not torch.allclose(reference,adapted,atol=2e-4,rtol=2e-4):
        raise RuntimeError(f'Export attention differs from source model: max_abs={attention_delta}')
    projection = adapter.head
    wrapper = EmbeddingModel(adapter.model, projection).eval().cuda()
    destination = ROOT / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes={'images': {0: 'batch'}, 'embedding': {0: 'batch'}} if args.dynamic_batch else None
    torch.onnx.export(wrapper, sample, str(destination), input_names=['images'], output_names=['embedding'],
                      dynamic_axes=dynamic_axes, opset_version=args.opset, do_constant_folding=True)
    import onnx
    model = onnx.load(str(destination), load_external_data=True)
    onnx.checker.check_model(model)
    result = dict(model='DINOv2-ViT-B/14', checkpoint=str((ROOT / args.checkpoint).resolve()),
                  projection=str((ROOT / args.projection).resolve()) if args.projection else None,
                  onnx=str(destination.resolve()), opset=args.opset, dynamic_batch=args.dynamic_batch,
                  input_shape=['batch', 3, 224, 224] if args.dynamic_batch else [1, 3, 224, 224],
                  output_shape=['batch', projection.out_features if projection is not None else 768] if args.dynamic_batch else [1, projection.out_features if projection is not None else 768],
                  output='L2-normalized embedding', file_bytes=destination.stat().st_size,
                  torch_version=torch.__version__, device='cuda:0')
    result['attention_implementation']='explicit scaled QK matmul, softmax, AV matmul'
    result['source_attention_max_abs']=attention_delta
    destination.with_suffix('.manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
