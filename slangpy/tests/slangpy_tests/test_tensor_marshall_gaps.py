# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Tests targeting coverage gaps in builtin/tensor.py.

Exercises:
- TensorMarshall grad validation error paths
- is_nested_array helper
- build_shader_object with differentiable tensors via pack()
"""

import pytest
import numpy as np

from slangpy import DeviceType
from slangpy.testing import helpers
from slangpy.types.tensor import Tensor
from slangpy import pack, BufferUsage


DIFF_POLY_SHADER = r"""
[Differentiable]
float polynomial(float a, float b) {
    return a * a + b;
}
"""

SIMPLE_READ_SHADER = r"""
float read_first(Tensor<float, 1> buf) { return buf[0]; }
"""

DEVICE_TYPES = helpers.DEFAULT_DEVICE_TYPES


# ============================================================================
# is_nested_array coverage
# ============================================================================


def test_is_nested_array():
    """Directly test is_nested_array with various SlangType instances."""
    from slangpy.builtin.tensor import is_nested_array

    device = helpers.get_device(DEVICE_TYPES[0])
    func = helpers.create_function_from_module(device, "read_first", SIMPLE_READ_SHADER)
    layout = func.module.layout

    scalar_type = layout.find_type_by_name("float")
    assert is_nested_array(scalar_type) is True

    vec_type = layout.find_type_by_name("float2")
    assert is_nested_array(vec_type) is True

    mat_type = layout.find_type_by_name("float2x2")
    assert is_nested_array(mat_type) is True

    tensor_type = layout.find_type_by_name("Tensor<float,1>")
    if tensor_type is not None:
        assert is_nested_array(tensor_type) is False


# ============================================================================
# Grad validation errors via functional API
# ============================================================================


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_grad_out_dtype_must_match_derivative(device_type: DeviceType):
    """Attaching a grad_out with wrong dtype raises when passed to a differentiable function."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "polynomial", DIFF_POLY_SHADER)

    primal = Tensor.zeros(device, shape=(4,), dtype="float")
    bad_grad = Tensor.zeros(device, shape=(4,), dtype="int")

    tensor = primal.with_grads(grad_out=bad_grad)
    with pytest.raises(ValueError, match="[Ii]nvalid element type"):
        func(tensor, 1.0)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_grad_in_dtype_must_match_derivative(device_type: DeviceType):
    """Attaching a grad_in with wrong dtype raises when passed to a differentiable function."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "polynomial", DIFF_POLY_SHADER)

    primal = Tensor.zeros(
        device, shape=(4,), dtype="float",
        usage=BufferUsage.shader_resource | BufferUsage.unordered_access,
    )
    bad_grad = Tensor.zeros(device, shape=(4,), dtype="int")

    tensor = primal.with_grads(grad_in=bad_grad)
    with pytest.raises(ValueError, match="[Ii]nvalid element type"):
        func(tensor, 1.0)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_grad_in_requires_writable_tensor(device_type: DeviceType):
    """Attaching grad_in to a read-only tensor raises when passed to a differentiable function."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "polynomial", DIFF_POLY_SHADER)

    primal = Tensor.zeros(
        device, shape=(4,), dtype="float",
        usage=BufferUsage.shader_resource,
    )
    grad = Tensor.zeros(
        device, shape=(4,), dtype="float",
        usage=BufferUsage.shader_resource | BufferUsage.unordered_access,
    )

    tensor = primal.with_grads(grad_in=grad)
    with pytest.raises(ValueError, match="writable"):
        func(tensor, 1.0)


# ============================================================================
# TensorMarshall properties via pack()
# ============================================================================


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_tensor_marshall_properties_via_pack(device_type: DeviceType):
    """pack() exposes marshall properties: has_derivative, is_writable, repr."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "read_first", SIMPLE_READ_SHADER)

    tensor = Tensor.zeros(
        device, shape=(3,), dtype="float",
        usage=BufferUsage.shader_resource | BufferUsage.unordered_access,
    )
    packed = pack(func.module, tensor)
    assert packed is not None
    r = repr(packed)
    assert "NativePackedArg" in r

    grad_out = Tensor.zeros(
        device, shape=(3,), dtype="float",
        usage=BufferUsage.shader_resource | BufferUsage.unordered_access,
    )
    tensor_with_grad = tensor.with_grads(grad_out=grad_out)
    packed_grad = pack(func.module, tensor_with_grad)
    assert packed_grad is not None


# ============================================================================
# create_tensor_marshall error path
# ============================================================================


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_create_tensor_marshall_rejects_bad_type(device_type: DeviceType):
    """create_tensor_marshall raises for unsupported type."""
    from slangpy.builtin.tensor import create_tensor_marshall

    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "read_first", SIMPLE_READ_SHADER)
    layout = func.module.layout

    with pytest.raises(ValueError, match="unsupported"):
        create_tensor_marshall(layout, "not a tensor")


# ============================================================================
# build_shader_object with derivatives via pack()
# ============================================================================


@pytest.mark.xfail(
    reason="TensorMarshall.build_shader_object derivative path references 'primal' field "
    "that doesn't exist on the shader object type; likely needs DiffTensor infrastructure",
    raises=RuntimeError,
)
@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_pack_tensor_with_grads(device_type: DeviceType):
    """pack() a Tensor with gradients to exercise build_shader_object derivative path."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "read_first", SIMPLE_READ_SHADER)

    data = np.array([42.0, 1.0, 2.0], dtype=np.float32)
    tensor = Tensor.from_numpy(device, data)
    grad_tensor = Tensor.zeros(
        device, shape=(3,), dtype="float",
        usage=BufferUsage.shader_resource | BufferUsage.unordered_access,
    )
    tensor_with_grads = tensor.with_grads(grad_out=grad_tensor)

    packed = pack(func.module, tensor_with_grads)
    assert packed is not None
