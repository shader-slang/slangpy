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

    np_weights = np.random.randn(5, 8).astype(np.float32)
    np_biases = np.random.randn(5).astype(np.float32)
    np_x = np.random.randn(8).astype(np.float32)
    np_result = np.tile(np_weights.dot(np_x) + np_biases, (N, 1))

    biases = Tensor.numpy(device, np_biases).broadcast_to((N, 5))
    x = Tensor.numpy(device, np_x).broadcast_to((N, 8))
    weights = Tensor.numpy(device, np_weights, ).broadcast_to((N, 5, 8))

    return weights, biases, x, np_result


def get_func(device: Device, name: str):
    path = os.path.split(__file__)[0] + "/test_tensor.slang"
    return helpers.create_function_from_module(device, name, open(path, "r").read())


def compare_tensors(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]):
    assert a.shape == b.shape, f"Tensor shape {a.shape} does not match expected shape {b.shape}"
    err = np.max(np.abs(a - b))
    assert err < 1e-4, f"Tensor deviates by {err} from reference"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tensor_arguments(device_type: DeviceType):
    device = helpers.get_device(device_type)

    func = get_func(device, "matrix_vector_direct").return_type(Tensor)
    weights, biases, x, np_result = get_test_tensors(device)
    y = func(weights, biases, x)
    compare_tensors(y.to_numpy(), np_result)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tensor_parameters(device_type: DeviceType):
    device = helpers.get_device(device_type)

    func = get_func(device, "matrix_vector_tensorized").return_type(Tensor)
    weights, biases, x, np_result = get_test_tensors(device)
    y = func(weights, biases, x)
    compare_tensors(y.to_numpy(), np_result)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_interface_parameters(device_type: DeviceType):
    device = helpers.get_device(device_type)

    func = get_func(device, "matrix_vector_interfaces").return_type(Tensor)
    weights, biases, x, np_result = get_test_tensors(device)
    y = func(weights, biases, x)
    compare_tensors(y.to_numpy(), np_result)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_generic_call(device_type: DeviceType):
    device = helpers.get_device(device_type)

    func = get_func(device, "matrix_vector_generic<8, 5>").return_type(Tensor)
    weights, biases, x, np_result = get_test_tensors(device)
    y = func(weights, biases, x)
    compare_tensors(y.to_numpy(), np_result)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
