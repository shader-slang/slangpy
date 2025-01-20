# SPDX-License-Identifier: Apache-2.0
import pytest
from slangpy.backend import DeviceType, Device
import slangpy.tests.helpers as helpers
import hashlib
import os

try:
    import torch
except ImportError:
    pytest.skip("Pytorch not installed", allow_module_level=True)

TEST_CODE = """
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


def get_module(device: Device):
    from slangpy.torchintegration import TorchModule
    path = os.path.split(__file__)[0] + "/test_tensor.slang"
    module_source = open(path, "r").read()
    module = device.load_module_from_source(
        hashlib.sha256(module_source.encode()).hexdigest()[0:16], module_source
    )
    return TorchModule.load_from_module(device, module)


def compare_tensors(a: torch.Tensor, b: torch.Tensor):
    assert a.shape == b.shape, f"Tensor shape {a.shape} does not match expected shape {b.shape}"
    err = torch.max(torch.abs(a - b)).item()
    assert err < 1e-4, f"Tensor deviates by {err} from reference"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_missing_cuda_interop(device_type: DeviceType):
    from slangpy.torchintegration import TorchModule
    torch.autograd.grad_mode.set_multithreading_enabled(False)

    device = helpers.get_device(device_type, use_cache=False, cuda_interop=False)
    with pytest.raises(RuntimeError, match=r"Cuda interop must be enabled for torch support.*"):
        module = TorchModule(helpers.create_module(device, TEST_CODE))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_missing_torch_context(device_type: DeviceType):
    torch.autograd.grad_mode.set_multithreading_enabled(False)

    device = helpers.get_device(device_type, use_cache=False, cuda_interop=True)
    module = helpers.create_module(device, TEST_CODE)

    a = torch.randn((8, 5), dtype=torch.float32, device=torch.device('cuda'), requires_grad=True)
    with pytest.raises(ValueError, match=r"Tensor types can not be directly passed to SlangPy"):
        b = module.square(a)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_basic_tensor_arguments(device_type: DeviceType):
    from slangpy.torchintegration import TorchModule
    torch.autograd.grad_mode.set_multithreading_enabled(False)

    device = helpers.get_device(device_type, use_cache=False, cuda_interop=True)
    module = TorchModule(helpers.create_module(device, TEST_CODE))

    a = torch.randn((8, 5), dtype=torch.float32, device=torch.device('cuda'), requires_grad=True)
    b = module.square(a)

    compare_tensors(b, a * a)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_basic_autograd(device_type: DeviceType):
    from slangpy.torchintegration import TorchModule
    torch.autograd.grad_mode.set_multithreading_enabled(False)

    device = helpers.get_device(device_type, use_cache=False, cuda_interop=True)
    module = TorchModule(helpers.create_module(device, TEST_CODE))

    a = torch.randn((8, 5), dtype=torch.float32, device=torch.device('cuda'), requires_grad=True)
    b = module.square(a)
    b.sum().backward()

    compare_tensors(b, a * a)
    assert a.grad is not None
    compare_tensors(a.grad, 2 * a)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tensor_arguments(device_type: DeviceType):
    torch.autograd.grad_mode.set_multithreading_enabled(False)

    device = helpers.get_device(device_type, use_cache=False, cuda_interop=True)
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
    torch.autograd.grad_mode.set_multithreading_enabled(False)

    device = helpers.get_device(device_type, use_cache=False, cuda_interop=True)
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
    torch.autograd.grad_mode.set_multithreading_enabled(False)

    device = helpers.get_device(device_type, use_cache=False, cuda_interop=True)
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
