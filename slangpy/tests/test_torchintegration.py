# SPDX-License-Identifier: Apache-2.0
import pytest
from slangpy.backend import DeviceType, Device
from slangpy.torchintegration import TorchModule
from slangpy.core.module import Module
import slangpy.tests.helpers as helpers
import hashlib
import os

from slangpy.torchintegration.torchfunction import TorchFunction

try:
    import torch
except ImportError:
    pytest.skip("Pytorch not installed", allow_module_level=True)

TEST_CODE = """
import tensor;
[Differentiable]
float square(float x) {
    return x * x;
}
"""


def get_test_tensors(device: Device, N: int = 4):
    weights = torch.randn((5, 8), dtype=torch.float32,
                          device=torch.device('cuda'), requires_grad=True)
    biases = torch.randn((5, ), dtype=torch.float32,
                         device=torch.device('cuda'), requires_grad=True)
    x = torch.randn((8, ), dtype=torch.float32, device=torch.device('cuda'), requires_grad=False)

    return weights, biases, x


def load_test_module(device_type: DeviceType):
    device = helpers.get_device(device_type, cuda_interop=True)
    return TorchModule.load_from_file(device, "test_torchintegration.slang")


def compare_tensors(a: torch.Tensor, b: torch.Tensor):
    assert a.shape == b.shape, f"Tensor shape {a.shape} does not match expected shape {b.shape}"
    err = torch.max(torch.abs(a - b)).item()
    assert err < 1e-4, f"Tensor deviates by {err} from reference"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_add_floats(device_type: DeviceType):
    torch.autograd.grad_mode.set_multithreading_enabled(False)

    module = load_test_module(device_type)

    a = torch.randn((10,), dtype=torch.float32, device=torch.device('cuda'))
    b = torch.randn((10,), dtype=torch.float32, device=torch.device('cuda'))

    func = module.add
    assert isinstance(func, TorchFunction)

    res = func(a, b)
    assert isinstance(res, torch.Tensor)

    compare_tensors(a+b, res)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_polynomial(device_type: DeviceType):
    torch.autograd.grad_mode.set_multithreading_enabled(False)

    module = load_test_module(device_type)

    a = 2.0
    b = 4.0
    c = 1.0
    x = torch.randn((10,), dtype=torch.float32, device=torch.device('cuda'), requires_grad=True)

    res = module.polynomial(a, b, c, x)
    assert isinstance(res, torch.Tensor)

    compare_tensors(a*x*x+b*x+c, res)

    res.backward(torch.ones_like(res))

    compare_tensors(2*a*x+b, x.grad)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
