# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
import pytest
import numpy as np

from slangpy import DeviceType
from slangpy.testing import helpers

if sys.platform == "darwin":
    pytest.skip("Torch CUDA tests not available on macOS", allow_module_level=True)

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")

CUDA_TYPES = [DeviceType.cuda] if DeviceType.cuda in helpers.DEFAULT_DEVICE_TYPES else []
if not CUDA_TYPES:
    pytest.skip("CUDA device type required", allow_module_level=True)

from slangpy.types import NDBuffer

# Force torch type registration so pack() can find torch.Tensor
import slangpy.torchintegration.torchtensormarshall  # noqa: F401
from slangpy import pack


# ---------------------------------------------------------------------------
# Slang source snippets
# ---------------------------------------------------------------------------

ADD_SHADER = r"""
void add_buffers(float a, float b, out float c) {
    c = a + b;
}
"""

SCALE_SHADER = r"""
void scale(float a, float factor, out float result) {
    result = a * factor;
}
"""

ADD_RETURN_SHADER = r"""
float add_return(float a, float b) {
    return a + b;
}
"""

READ_FIRST_SHADER = r"""
float read_first(Tensor<float, 1> buf) {
    return buf[0];
}
"""


# ---------------------------------------------------------------------------
# Mixed torch.Tensor + NDBuffer in a single call
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_type", CUDA_TYPES)
def test_torch_input_ndbuffer_output(device_type: DeviceType):
    """torch.Tensor input, NDBuffer output."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "add_buffers", ADD_SHADER)

    a = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda", dtype=torch.float32)
    b = NDBuffer.from_numpy(device, np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32))
    c = NDBuffer.zeros(device, shape=(4,), dtype=float)

    func(a, b, c)

    assert np.allclose(c.to_numpy(), [11.0, 22.0, 33.0, 44.0])


@pytest.mark.parametrize("device_type", CUDA_TYPES)
def test_ndbuffer_input_torch_output(device_type: DeviceType):
    """NDBuffer input, torch.Tensor output."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "add_buffers", ADD_SHADER)

    a = NDBuffer.from_numpy(device, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    b = torch.tensor([10.0, 20.0, 30.0], device="cuda", dtype=torch.float32)
    c = torch.zeros(3, device="cuda", dtype=torch.float32)

    func(a, b, c)
    torch.cuda.synchronize()

    assert torch.allclose(c, torch.tensor([11.0, 22.0, 33.0], device="cuda"))


@pytest.mark.parametrize("device_type", CUDA_TYPES)
def test_torch_with_scalar_broadcast(device_type: DeviceType):
    """torch.Tensor + scalar broadcast."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "scale", SCALE_SHADER)

    t = torch.tensor([2.0, 4.0, 6.0], device="cuda", dtype=torch.float32)
    out = torch.zeros(3, device="cuda", dtype=torch.float32)

    func(t, 3.0, out)
    torch.cuda.synchronize()

    assert torch.allclose(out, torch.tensor([6.0, 12.0, 18.0], device="cuda"))


@pytest.mark.parametrize("device_type", CUDA_TYPES)
def test_torch_return_value(device_type: DeviceType):
    """torch.Tensor inputs with a Slang return value."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "add_return", ADD_RETURN_SHADER)

    a = torch.tensor([1.0, 2.0], device="cuda", dtype=torch.float32)
    b = torch.tensor([3.0, 4.0], device="cuda", dtype=torch.float32)

    result = func(a, b)
    assert isinstance(result, torch.Tensor)
    assert torch.allclose(result, torch.tensor([4.0, 6.0], device="cuda"))


# ---------------------------------------------------------------------------
# build_shader_object path (pack + torch.Tensor)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_type", CUDA_TYPES)
def test_pack_torch_tensor(device_type: DeviceType):
    """pack() a torch.Tensor and pass it through a Slang function."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "read_first", READ_FIRST_SHADER)

    t = torch.tensor([42.0, 1.0, 2.0], device="cuda", dtype=torch.float32)
    packed = pack(func.module, t)
    result = func(packed)

    assert result == pytest.approx(42.0)


@pytest.mark.parametrize("device_type", CUDA_TYPES)
def test_pack_torch_tensor_different_values(device_type: DeviceType):
    """pack() multiple torch tensors with different values."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "read_first", READ_FIRST_SHADER)

    for val in [0.0, -1.0, 99.5, 1e6]:
        t = torch.tensor([val, 0.0], device="cuda", dtype=torch.float32)
        packed = pack(func.module, t)
        result = func(packed)
        assert result == pytest.approx(val)


# ---------------------------------------------------------------------------
# 2D torch tensors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_type", CUDA_TYPES)
def test_torch_2d_add(device_type: DeviceType):
    """2D torch.Tensor inputs."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "add_buffers", ADD_SHADER)

    a = torch.arange(6, device="cuda", dtype=torch.float32).reshape(2, 3)
    b = torch.ones(2, 3, device="cuda", dtype=torch.float32) * 10
    c = torch.zeros(2, 3, device="cuda", dtype=torch.float32)

    func(a, b, c)
    torch.cuda.synchronize()

    expected = a + b
    assert torch.allclose(c, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
