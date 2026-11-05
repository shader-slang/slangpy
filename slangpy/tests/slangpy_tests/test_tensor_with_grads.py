# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import numpy as np
import os

from slangpy import DeviceType, Device, Feature
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
    weights = Tensor.from_numpy(device, np_weights).broadcast_to((N, din, dout))

    weights = weights.with_grads()
    biases = biases.with_grads()

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

    func_base = get_func(device, "matrix_vector_interfaces_diff")
    func = func_base.return_type(Tensor)
    weights, biases, x, np_result = get_test_tensors(device)

    assert weights.grad_out is not None
    assert biases.grad_out is not None

    y: Tensor = func(weights, biases, x)
    compare_tensors(y.to_numpy(), np_result)

    y.grad_in = Tensor.zeros_like(y)
    y.grad_in.storage.copy_from_numpy(
        np.random.rand(*y.shape, *y.dtype.shape.as_tuple()).astype(np.float32)
    )  # type: ignore

    func.bwds(weights, biases, x, y)

    weight_grad_ref = y.grad_in.to_numpy()[..., np.newaxis] * x.to_numpy()[..., np.newaxis, :]
    bias_grad_ref = y.grad_in.to_numpy()

    compare_tensors(weights.grad_out.to_numpy(), weight_grad_ref)
    compare_tensors(biases.grad_out.to_numpy(), bias_grad_ref)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_differentiable_matrix_parameters(device_type: DeviceType):
    device = helpers.get_device(device_type)

    func_base = get_func(device, "matrix_vector_matrices")
    func = func_base.return_type(Tensor)
    weights, biases, x, np_result = get_test_tensors(device, din=3, dout=4)

    assert weights.grad_out is not None
    assert biases.grad_out is not None

    y: Tensor = func(weights, biases, x)
    compare_tensors(y.to_numpy(), np_result)

    y.grad_in = Tensor.zeros_like(y)
    y.grad_in.storage.copy_from_numpy(
        np.random.rand(*y.shape, *y.dtype.shape.as_tuple()).astype(np.float32)
    )  # type: ignore

    func.bwds(weights, biases, x, y)

    weight_grad_ref = y.grad_in.to_numpy()[..., np.newaxis] * x.to_numpy()[..., np.newaxis, :]
    bias_grad_ref = y.grad_in.to_numpy()

    compare_tensors(weights.grad_out.to_numpy(), weight_grad_ref)
    compare_tensors(biases.grad_out.to_numpy(), bias_grad_ref)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_fail_shared_inout_grad_buffers(device_type: DeviceType):
    device = helpers.get_device(device_type)

    function = helpers.create_function_from_module(
        device,
        "inc",
        r"""
[Differentiable]
void inc(float amount, inout float val) { val += amount; }
""",
    )

    amount = Tensor.from_numpy(device, np.array([1.0], dtype=np.float32))
    val = Tensor.zeros(device, (1,), "float").with_grads()

    function(amount, val)
    assert np.allclose(val.to_numpy(), amount.to_numpy())

    with pytest.raises(Exception, match="inout param"):
        function.bwds(amount, val)


# Each function has a half input whose gradient is accumulated with float16 atomic adds.
HALF_INPUT_GRAD_FUNCS = {
    "half_to_half": ("[Differentiable] half f(half x) { return x * x; }", ()),
    "half_to_float": ("[Differentiable] float f(half x) { return float(x * x); }", ()),
    "half3_to_float": (
        "[Differentiable] float f(half3 x) { return float(dot(x, x)); }",
        (3,),
    ),
}


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("func_name", HALF_INPUT_GRAD_FUNCS.keys())
def test_half_input_grads(device_type: DeviceType, func_name: str):
    device = helpers.get_device(device_type)
    source, element_shape = HALF_INPUT_GRAD_FUNCS[func_name]
    function = helpers.create_function_from_module(device, "f", source)

    x_np = np.arange(1, 1 + 4 * int(np.prod(element_shape)), dtype=np.float16)
    x_np = x_np.reshape((4,) + element_shape)
    x = Tensor.from_numpy(device, x_np).with_grads(zero=True)
    y = function(x).with_grads()
    y.grad.copy_from_numpy(np.ones(4, dtype=y.to_numpy().dtype))

    if helpers.has_half_atomic_add(device):
        function.bwds(x, y)
        assert np.allclose(x.grad.to_numpy(), 2 * x_np.astype(np.float32))
    else:
        with pytest.raises(ValueError, match="Feature.atomic_half"):
            function.bwds(x, y)


# half2 gradients use the packed fp16x2 atomic, which only Feature.atomic_half guarantees.
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_half2_input_grads(device_type: DeviceType):
    device = helpers.get_device(device_type)
    if not device.has_feature(Feature.atomic_half):
        pytest.skip("Device does not report Feature.atomic_half")
    function = helpers.create_function_from_module(
        device, "f", "[Differentiable] float f(half2 x) { return float(dot(x, x)); }"
    )

    x_np = np.arange(1, 9, dtype=np.float16).reshape((4, 2))
    x = Tensor.zeros(device, (4,), "half2").with_grads(zero=True)
    x.copy_from_numpy(x_np)
    y = Tensor.zeros(device, (4,), "float").with_grads()
    y.grad.copy_from_numpy(np.ones(4, dtype=np.float32))

    function.bwds(x, _result=y)
    assert np.allclose(x.grad.to_numpy().reshape((4, 2)), 2 * x_np.astype(np.float32))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_half_difftensor_param_grads(device_type: DeviceType):
    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device,
        "f",
        "[Differentiable] float f(IDiffTensor<half, 1> t, int i) { return float(t.load({i})) * 3.0; }",
    )

    t = Tensor.from_numpy(device, np.arange(4, dtype=np.float16)).with_grads(zero=True)
    idx = np.arange(4, dtype=np.int32)
    y = Tensor.zeros(device, (4,), "float").with_grads()
    y.grad.copy_from_numpy(np.ones(4, dtype=np.float32))

    if helpers.has_half_atomic_add(device):
        function.bwds(t, idx, _result=y)
        assert np.allclose(t.grad.to_numpy(), 3.0)
    else:
        with pytest.raises(ValueError, match="Feature.atomic_half"):
            function.bwds(t, idx, _result=y)


# A half result only reads its incoming gradient, so it needs no atomics on any device.
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_half_result_grads(device_type: DeviceType):
    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device, "f", "[Differentiable] half f(float x) { return half(x * x); }"
    )

    x_np = np.arange(1, 5, dtype=np.float32)
    x = Tensor.from_numpy(device, x_np).with_grads(zero=True)
    y = function(x).with_grads()
    y.grad.copy_from_numpy(np.ones(4, dtype=np.float16))

    function.bwds(x, y)
    assert np.allclose(x.grad.to_numpy(), 2 * x_np)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
