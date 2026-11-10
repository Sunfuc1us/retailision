import os
import pytest

def test_isolated_environment():
    import sys
    assert 'retailvision-' in sys.executable
    import site
    assert not site.ENABLE_USER_SITE, 'user site-packages must be disabled'
    assert not any('/.local/' in p for p in sys.path)

@pytest.mark.skipif(os.getenv('RETAILVISION_GPU_TEST')!='1',reason='set RETAILVISION_GPU_TEST=1 for explicit CUDA tests')
def test_cuda():
    import torch
    assert torch.cuda.is_available()
    assert '4090' in torch.cuda.get_device_name()
    assert torch.cuda.get_device_properties(0).total_memory>=23*2**30
    x=torch.ones((64,64),device='cuda');assert float((x@x)[0,0])==64
