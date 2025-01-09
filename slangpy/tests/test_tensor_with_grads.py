# SPDX-License-Identifier: Apache-2.0
import pytest
from slangpy.backend import DeviceType, Device
from slangpy.types import Tensor
import slangpy.tests.helpers as helpers
import numpy as np
from typing import Any
import os


def get_test_tensors(device: Device, N: int = 4):
    np.random.seed(0)

    np_weights = np.random.randn(8, 8).astype(np.float32)
    np_biases = np.random.randn(8).astype(np.float32)
    np_x = np.random.randn(8).astype(np.float32)
    np_result = np.tile(np_weights.dot(np_x) + np_biases, (N, 1))

    weights = Tensor.from_numpy(np_weights, device).broadcast_to((N, 8, 8))
    biases = Tensor.from_numpy(np_biases, device).broadcast_to((N, 8))
    x = Tensor.from_numpy(np_x, device).broadcast_to((N, 8))

    weights = weights.with_grads(None, Tensor.zeros_like(weights))
    biases = biases.with_grads(None, Tensor.zeros_like(biases))

    return weights, biases, x, np_result


def get_func(device: Device, name: str):
    path = os.path.split(__file__)[0] + "/test_tensor.slang"
    return helpers.create_function_from_module(device, name, open(path, "r").read())


def compare_tensors(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]):
    assert a.shape == b.shape, f"Tensor shape {a.shape} does not match expected shape {b.shape}"
    err = np.max(np.abs(a - b))
    assert err < 1e-4, f"Tensor deviates by {err} from reference"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_differentiable_interface_parameters(device_type: DeviceType):
    device = helpers.get_device(device_type)

    func_base = get_func(device, "matrix_vector_interfaces")
    func = func_base.return_type(Tensor)
    weights, biases, x, np_result = get_test_tensors(device)

    assert weights.grad_out is not None
    assert biases.grad_out is not None

    y: Tensor = func(weights, biases, x)
    compare_tensors(y.to_numpy(), np_result)

    y.grad_in = Tensor.zeros_like(y)
    y.grad_in.storage.from_numpy(np.random.rand(
        *y.shape, *y.dtype.shape.as_tuple()).astype(np.float32))

    func.bwds_diff(weights, biases, x, y)

    weight_grad_ref = y.grad_in.to_numpy()[..., np.newaxis] * x.to_numpy()[..., np.newaxis, :]
    bias_grad_ref = y.grad_in.to_numpy()

    compare_tensors(weights.grad_out.to_numpy(), weight_grad_ref)
    compare_tensors(biases.grad_out.to_numpy(), bias_grad_ref)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])