# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import hashlib
import os
import sys

from slangpy import DeviceType, Device, Module
from slangpy.testing import helpers

try:
    import torch
except ImportError:
    pytest.skip("Pytorch not installed", allow_module_level=True)

if sys.platform == "darwin":
    pytest.skip("PyTorch requires CUDA, that is not available on macOS", allow_module_level=True)

TEST_CODE = """
[Differentiable]
float square(float x) {
    return x * x;
}
"""


def get_test_tensors(device: Device, N: int = 4):
    weights = torch.randn(
        (5, 8), dtype=torch.float32, device=torch.device("cuda"), requires_grad=True
    )
    biases = torch.randn((5,), dtype=torch.float32, device=torch.device("cuda"), requires_grad=True)
    x = torch.randn((8,), dtype=torch.float32, device=torch.device("cuda"), requires_grad=False)

    return weights, biases, x


def get_module(device: Device):
    path = os.path.split(__file__)[0] + "/test_tensor.slang"
    module_source = open(path, "r").read()
    module = device.load_module_from_source(
        hashlib.sha256(module_source.encode()).hexdigest()[0:16], module_source
    )
    return Module.load_from_module(device, module)


def compare_tensors(a: torch.Tensor, b: torch.Tensor):
    assert a.shape == b.shape, f"Tensor shape {a.shape} does not match expected shape {b.shape}"
    err = torch.max(torch.abs(a - b)).item()
    assert err < 1e-4, f"Tensor deviates by {err} from reference\n\nA:\n{a}\n\nB:\n{b}"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_basic_tensor_arguments(device_type: DeviceType):

    device = helpers.get_torch_device(device_type)
    module = helpers.create_module(device, TEST_CODE)

    a = torch.randn((8, 5), dtype=torch.float32, device=torch.device("cuda"), requires_grad=True)
    b = module.square(a)

    compare_tensors(b, a * a)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_basic_autograd(device_type: DeviceType):

    device = helpers.get_torch_device(device_type)
    module = helpers.create_module(device, TEST_CODE)

    a = torch.randn((8, 5), dtype=torch.float32, device=torch.device("cuda"), requires_grad=True)
    b = module.square(a)
    b.sum().backward()

    compare_tensors(b, a * a)
    assert a.grad is not None
    compare_tensors(a.grad, 2 * a)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tensor_arguments(device_type: DeviceType):

    device = helpers.get_torch_device(device_type)
    module = get_module(device)

    weights, biases, x = get_test_tensors(device)
    y = module.matrix_vector_direct(weights, biases, x)
    reference = torch.nn.functional.linear(x, weights, biases)
    compare_tensors(y, reference)

    y_grad = torch.randn_like(y)
    y.backward(y_grad)

    assert weights.grad is not None
    assert biases.grad is not None
    compare_tensors(weights.grad, torch.outer(y_grad, x))
    compare_tensors(biases.grad, y_grad)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tensor_interfaces(device_type: DeviceType):

    device = helpers.get_torch_device(device_type)
    module = get_module(device)

    weights, biases, x = get_test_tensors(device)
    y = module.matrix_vector_interfaces(weights, biases, x)
    reference = torch.nn.functional.linear(x, weights, biases)
    compare_tensors(y, reference)

    y_grad = torch.randn_like(y)
    y.backward(y_grad)

    assert weights.grad is not None
    assert biases.grad is not None
    compare_tensors(weights.grad, torch.outer(y_grad, x))
    compare_tensors(biases.grad, y_grad)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tensor_generic(device_type: DeviceType):

    device = helpers.get_torch_device(device_type)
    module = get_module(device)

    weights, biases, x = get_test_tensors(device)

    y = module["matrix_vector_generic<8, 5>"](weights, biases, x)
    reference = torch.nn.functional.linear(x, weights, biases)
    compare_tensors(y, reference)

    y_grad = torch.randn_like(y)
    y.backward(y_grad)

    assert weights.grad is not None
    assert biases.grad is not None
    compare_tensors(weights.grad, torch.outer(y_grad, x))
    compare_tensors(biases.grad, y_grad)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
