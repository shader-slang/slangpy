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
def test_autograd_after_no_grad_call(device_type: DeviceType, torch_bridge_mode: str):
    # Regression for #1052: a no-grad call must not poison the call-data cache
    # for a later requires_grad call of the same ndim/dtype. The torch cache
    # signature previously omitted requires_grad, so the first no-grad call
    # cached a non-autograd dispatch that the grad call then reused, silently
    # dropping the autograd hook (grad_fn=None -> backward() fails). Runs in
    # both native and Python-fallback bridge modes via torch_bridge_mode.
    device = helpers.get_torch_device(device_type)
    module = helpers.create_module(device, TEST_CODE)

    # First call: same ndim/dtype but no grad required -> builds the cache entry.
    a_nograd = torch.randn(
        (8, 5), dtype=torch.float32, device=torch.device("cuda"), requires_grad=False
    )
    b_nograd = module.square(a_nograd)
    compare_tensors(b_nograd, a_nograd * a_nograd)
    assert not b_nograd.requires_grad

    # Second call: identical ndim/dtype but requires grad -> must attach autograd.
    a = torch.randn((8, 5), dtype=torch.float32, device=torch.device("cuda"), requires_grad=True)
    b = module.square(a)

    assert b.requires_grad, "autograd hook was dropped: output has requires_grad=False"
    assert b.grad_fn is not None, "autograd hook was dropped: output has no grad_fn"

    b.sum().backward()
    compare_tensors(b, a * a)
    assert a.grad is not None
    compare_tensors(a.grad, 2 * a)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tensor_arguments(device_type: DeviceType):
    if device_type != DeviceType.cuda:
        pytest.skip("Test currently unreliable on GFX pipelines")

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
    y = module.matrix_vector_interfaces_diff(weights, biases, x)
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

    y = module["matrix_vector_generic_diff<8, 5>"](weights, biases, x)
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
