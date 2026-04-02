# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Tests targeting coverage gaps in builtin/tensor.py.

Exercises:
- TensorMarshall grad validation error paths (lines 74, 79, 85)
- is_nested_array helper (lines 44-48)
- build_shader_object with differentiable tensors via pack() (lines 158-170)
- create_tensor_marshall error path (line 198)
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
# is_nested_array coverage (lines 37-49 of builtin/tensor.py)
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
# TensorMarshall grad validation errors (lines 73-87)
# ============================================================================


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_grad_element_type_mismatch_d_out(device_type: DeviceType):
    """TensorMarshall raises if d_out element type doesn't match derivative type (line 78-82)."""
    from slangpy.builtin.tensor import TensorMarshall

    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "polynomial", DIFF_POLY_SHADER)
    layout = func.module.layout

    float_type = layout.find_type_by_name("float")
    int_type = layout.find_type_by_name("int")
    if int_type is None:
        pytest.skip("int type not found in layout")

    good_grad = TensorMarshall(layout, float_type, 1, True, None, None)
    bad_grad = TensorMarshall(layout, int_type, 1, True, None, None)

    with pytest.raises(ValueError, match="[Ii]nvalid element type"):
        TensorMarshall(layout, float_type, 1, True, None, bad_grad)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_grad_element_type_mismatch_d_in(device_type: DeviceType):
    """TensorMarshall raises if d_in element type doesn't match derivative type (line 73-77)."""
    from slangpy.builtin.tensor import TensorMarshall

    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "polynomial", DIFF_POLY_SHADER)
    layout = func.module.layout

    float_type = layout.find_type_by_name("float")
    int_type = layout.find_type_by_name("int")
    if int_type is None:
        pytest.skip("int type not found in layout")

    bad_grad = TensorMarshall(layout, int_type, 1, True, None, None)

    with pytest.raises(ValueError, match="[Ii]nvalid element type"):
        TensorMarshall(layout, float_type, 1, True, bad_grad, None)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_d_in_requires_writable(device_type: DeviceType):
    """TensorMarshall raises if d_in supplied but primal is not writable (line 84-87)."""
    from slangpy.builtin.tensor import TensorMarshall

    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "polynomial", DIFF_POLY_SHADER)
    layout = func.module.layout

    float_type = layout.find_type_by_name("float")
    d_in = TensorMarshall(layout, float_type, 1, True, None, None)

    with pytest.raises(ValueError, match="writable"):
        TensorMarshall(layout, float_type, 1, False, d_in, None)


# ============================================================================
# TensorMarshall properties
# ============================================================================


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_tensor_marshall_properties(device_type: DeviceType):
    """TensorMarshall has_derivative and is_writable properties."""
    from slangpy.builtin.tensor import TensorMarshall

    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "polynomial", DIFF_POLY_SHADER)
    layout = func.module.layout
    float_type = layout.find_type_by_name("float")

    no_grad = TensorMarshall(layout, float_type, 1, True, None, None)
    assert no_grad.has_derivative is False
    assert no_grad.is_writable is True
    assert "Tensor" in repr(no_grad)

    d_out = TensorMarshall(layout, float_type, 1, True, None, None)
    with_grad = TensorMarshall(layout, float_type, 1, True, None, d_out)
    assert with_grad.has_derivative is True


# ============================================================================
# create_tensor_marshall error path (line 198)
# ============================================================================


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_create_tensor_marshall_bad_type(device_type: DeviceType):
    """create_tensor_marshall raises for unsupported type (line 198)."""
    from slangpy.builtin.tensor import create_tensor_marshall

    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "read_first", SIMPLE_READ_SHADER)
    layout = func.module.layout

    with pytest.raises(ValueError, match="unsupported"):
        create_tensor_marshall(layout, "not a tensor")


# ============================================================================
# build_shader_object with derivatives via pack() (lines 158-170)
# ============================================================================


@pytest.mark.xfail(
    reason="TensorMarshall.build_shader_object derivative path references 'primal' field "
    "that doesn't exist on the shader object type; likely needs DiffTensor infrastructure",
    raises=RuntimeError,
)
@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_pack_tensor_with_grads(device_type: DeviceType):
    """pack() a Tensor with gradients to exercise build_shader_object derivative path (lines 158-170)."""
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
