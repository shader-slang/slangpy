# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import numpy as np
import os

from slangpy import DeviceType, Device
from slangpy.types import Tensor
from slangpy.testing import helpers

from typing import Any


def get_test_tensors(device: Device, din: int = 5, dout: int = 8, N: int = 4):
    np.random.seed(0)

    np_weights = np.random.randn(din, dout).astype(np.float32)
    np_biases = np.random.randn(din).astype(np.float32)
    np_x = np.random.randn(dout).astype(np.float32)
    np_result = np.tile(np_weights.dot(np_x) + np_biases, (N, 1))

    biases = Tensor.from_numpy(device, np_biases).broadcast_to((N, din))
    x = Tensor.from_numpy(device, np_x).broadcast_to((N, dout))
    weights = Tensor.from_numpy(
        device,
        np_weights,
    ).broadcast_to((N, din, dout))

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
def test_matrix_parameters(device_type: DeviceType):
    device = helpers.get_device(device_type)

    func = get_func(device, "matrix_vector_matrices").return_type(Tensor)
    weights, biases, x, np_result = get_test_tensors(device, din=3, dout=4)
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
