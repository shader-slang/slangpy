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
@pytest.mark.parametrize("extra_dims", [0, 1, 3, 5])
def test_add_floats(device_type: DeviceType, extra_dims: int):
    torch.autograd.grad_mode.set_multithreading_enabled(False)

    module = load_test_module(device_type)

    extra_shape = (5,) * extra_dims

    a = torch.randn((10,), dtype=torch.float32, device=torch.device('cuda'))
    b = torch.randn((10,), dtype=torch.float32, device=torch.device('cuda'))

    func = module.add
    assert isinstance(func, TorchFunction)

    res = func(a, b)
    assert isinstance(res, torch.Tensor)

    compare_tensors(a+b, res)


ADD_TESTS = [
    ('add', ()),
    ('add_vectors', (3,)),
    ('add_vectors_generic<4>', (4,)),
    ('add_arrays', (5,))
]


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("extra_dims", [0, 1, 3, 5])
@pytest.mark.parametrize("func_and_shape", ADD_TESTS)
def test_add_values(device_type: DeviceType, extra_dims: int, func_and_shape: tuple[str, tuple[int]]):
    torch.autograd.grad_mode.set_multithreading_enabled(False)

    module = load_test_module(device_type)

    func_name = func_and_shape[0]
    val_shape = func_and_shape[1]
    extra_shape = (5,) * extra_dims

    a = torch.randn(extra_shape+val_shape, dtype=torch.float32,
                    device=torch.device('cuda'), requires_grad=True)
    b = torch.randn(extra_shape+val_shape, dtype=torch.float32,
                    device=torch.device('cuda'), requires_grad=True)

    res = module[func_name](a, b)
    assert isinstance(res, torch.Tensor)

    compare_tensors(a+b, res)

    # Not much to check for backwards pass of an 'add', but call it
    # so we at least catch any exceptions that fire.
    res.backward(torch.ones_like(res))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("extra_dims", [0, 1, 3, 5])
@pytest.mark.parametrize("func_and_shape", ADD_TESTS)
def test_add_values_fail(device_type: DeviceType, extra_dims: int, func_and_shape: tuple[str, tuple[int]]):
    torch.autograd.grad_mode.set_multithreading_enabled(False)

    module = load_test_module(device_type)

    func_name = func_and_shape[0]
    val_shape = func_and_shape[1]
    if len(val_shape) == 0:
        pytest.skip("No shape to fail")

    extra_shape = (5,) * extra_dims

    val_shape = val_shape[0:-1] + (val_shape[-1] + 1,)

    a = torch.randn(extra_shape+val_shape, dtype=torch.float32, device=torch.device('cuda'))
    b = torch.randn(extra_shape+val_shape, dtype=torch.float32, device=torch.device('cuda'))

    with pytest.raises(ValueError, match="does not match expected shape"):
        res = module.add_vectors(a, b)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("extra_dims", [0, 1, 3, 5])
def test_add_vectors_generic_explicit(device_type: DeviceType, extra_dims: int):
    torch.autograd.grad_mode.set_multithreading_enabled(False)

    module = load_test_module(device_type)

    extra_shape = (5,) * extra_dims

    a = torch.randn(extra_shape+(3,), dtype=torch.float32, device=torch.device('cuda'))
    b = torch.randn(extra_shape+(3,), dtype=torch.float32, device=torch.device('cuda'))

    # Can't currently infer generic vector from tensor shape, but explicit type map should work
    res = module.add_vectors_generic.map('float3', 'float3')(a, b)
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


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_polynomial_outparam(device_type: DeviceType):
    torch.autograd.grad_mode.set_multithreading_enabled(False)

    module = load_test_module(device_type)

    a = 2.0
    b = 4.0
    c = 1.0
    x = torch.randn((10,), dtype=torch.float32, device=torch.device('cuda'), requires_grad=True)
    res = torch.zeros_like(x)

    module.polynomial_outparam(a, b, c, x, res)

    compare_tensors(a*x*x+b*x+c, res)

    res.backward(torch.ones_like(res))

    compare_tensors(2*a*x+b, x.grad)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
