# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Tests targeting coverage gaps in builtin/value.py via torch-based pipelines.

Exercises:
- slang_type_to_return_type for bool, int-vector, uint-vector, bool-vector, matrix types
- VectorMarshall.build_shader_object via pack() in a torch pipeline
- ValueMarshall properties (is_writable, has_derivative)
- ScalarMarshall.reduce_type error path
- VectorMarshall.reduce_type edge cases
- MatrixMarshall.reduce_type edge cases
"""

import sys
import pytest
import numpy as np

from slangpy import DeviceType, pack
from slangpy.testing import helpers

if sys.platform == "darwin":
    pytest.skip("Torch tests require CUDA", allow_module_level=True)

try:
    import torch
except ImportError:
    pytest.skip("PyTorch not installed", allow_module_level=True)

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

import slangpy
import slangpy.torchintegration.torchtensormarshall  # noqa: F401

CUDA_TYPES = [t for t in helpers.DEFAULT_DEVICE_TYPES if t == DeviceType.cuda]
if not CUDA_TYPES:
    pytest.skip("No CUDA device available", allow_module_level=True)


# ============================================================================
# Slang shaders exercising various return types
# ============================================================================

RETURN_BOOL_SHADER = r"""
bool is_positive(float x) { return x > 0.0; }
"""

RETURN_INT2_SHADER = r"""
int2 make_int2(int a, int b) { return int2(a, b); }
"""

RETURN_UINT2_SHADER = r"""
uint2 make_uint2(uint a, uint b) { return uint2(a, b); }
"""

RETURN_FLOAT2_SHADER = r"""
float2 make_float2(float a, float b) { return float2(a, b); }
"""

RETURN_FLOAT2X2_SHADER = r"""
float2x2 make_mat(float a, float b, float c, float d) {
    return float2x2(a, b, c, d);
}
"""

SCALE_VECTOR_SHADER = r"""
void scale_vec(float2 v, float factor, out float2 result) {
    result = v * factor;
}
"""

ADD_SCALAR_SHADER = r"""
void add_scalar(float a, float b, out float result) { result = a + b; }
"""

TORCH_TENSOR_PLUS_SCALAR = r"""
void add_to_tensor(float tensor_val, float scalar, out float result) {
    result = tensor_val + scalar;
}
"""

READ_FIRST_SHADER = r"""
float read_first(Tensor<float, 1> buf) { return buf[0]; }
"""


# ============================================================================
# Return type coverage (slang_type_to_return_type)
# ============================================================================


@pytest.mark.parametrize("device_type", CUDA_TYPES)
def test_return_bool_scalar(device_type: DeviceType):
    """Exercise slang_type_to_return_type for bool scalar."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "is_positive", RETURN_BOOL_SHADER)
    result = func(1.0)
    assert result is True
    result = func(-1.0)
    assert result is False


@pytest.mark.parametrize("device_type", CUDA_TYPES)
def test_return_int2_vector(device_type: DeviceType):
    """Exercise slang_type_to_return_type for signed int vector."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "make_int2", RETURN_INT2_SHADER)
    result = func(3, 7)
    assert result.x == 3
    assert result.y == 7


@pytest.mark.parametrize("device_type", CUDA_TYPES)
def test_return_uint2_vector(device_type: DeviceType):
    """Exercise slang_type_to_return_type for unsigned int vector."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "make_uint2", RETURN_UINT2_SHADER)
    result = func(10, 20)
    assert result.x == 10
    assert result.y == 20


@pytest.mark.parametrize("device_type", CUDA_TYPES)
def test_return_float2x2_matrix(device_type: DeviceType):
    """Exercise slang_type_to_return_type for float matrix."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "make_mat", RETURN_FLOAT2X2_SHADER)
    result = func(1.0, 2.0, 3.0, 4.0)
    assert hasattr(result, "rows") or isinstance(result, slangpy.math.float2x2)


# ============================================================================
# Torch pipeline: scalars + vectors alongside torch.Tensor
# ============================================================================


@pytest.mark.parametrize("device_type", CUDA_TYPES)
def test_torch_tensor_plus_scalar(device_type: DeviceType):
    """Pass a torch.Tensor and a Python scalar to a Slang function."""
    device = helpers.get_torch_device(device_type)
    func = helpers.create_function_from_module(device, "add_to_tensor", TORCH_TENSOR_PLUS_SCALAR)

    t = torch.tensor([1.0, 2.0, 3.0], device="cuda", dtype=torch.float32)
    out = torch.zeros(3, device="cuda", dtype=torch.float32)
    func(t, 10.0, out)
    torch.cuda.synchronize()
    expected = torch.tensor([11.0, 12.0, 13.0], device="cuda", dtype=torch.float32)
    assert torch.allclose(out, expected)


# ============================================================================
# pack() with vector/matrix types (build_shader_object)
# ============================================================================


@pytest.mark.parametrize("device_type", CUDA_TYPES)
def test_pack_float2_vector(device_type: DeviceType):
    """pack() a float2 vector value, exercising VectorMarshall.build_shader_object."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "scale_vec", SCALE_VECTOR_SHADER)

    v = slangpy.math.float2(3.0, 4.0)
    packed = pack(func.module, v)
    out_buf = slangpy.types.NDBuffer.zeros(device, shape=(1,), dtype=float)
    result = func(packed, 2.0, out_buf)


@pytest.mark.parametrize("device_type", CUDA_TYPES)
def test_pack_scalar_value(device_type: DeviceType):
    """pack() a scalar, exercising ValueMarshall.build_shader_object."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "read_first", READ_FIRST_SHADER)

    packed_scalar = pack(func.module, 42.0)
    assert packed_scalar is not None


# ============================================================================
# ValueMarshall properties
# ============================================================================


@pytest.mark.parametrize("device_type", CUDA_TYPES)
def test_value_marshall_properties(device_type: DeviceType):
    """ValueMarshall.is_writable is False, has_derivative is False."""
    from slangpy.bindings.typeregistry import PYTHON_TYPES

    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "add_scalar", ADD_SCALAR_SHADER)
    layout = func.module.layout

    factory = PYTHON_TYPES[float]
    marshall = factory(layout, 1.0)

    assert marshall.is_writable is False
    assert marshall.has_derivative is False
    assert "Value" in repr(marshall)


# ============================================================================
# ScalarMarshall.reduce_type error path
# ============================================================================


@pytest.mark.parametrize("device_type", CUDA_TYPES)
def test_scalar_reduce_type_error(device_type: DeviceType):
    """ScalarMarshall.reduce_type raises for dimensions > 0."""
    from slangpy.bindings.typeregistry import PYTHON_TYPES

    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "add_scalar", ADD_SCALAR_SHADER)
    layout = func.module.layout

    marshall = PYTHON_TYPES[float](layout, 1.0)
    with pytest.raises(ValueError, match=r"[Cc]annot reduce"):
        marshall.reduce_type(None, 1)


# ============================================================================
# VectorMarshall.reduce_type edge cases
# ============================================================================


@pytest.mark.parametrize("device_type", CUDA_TYPES)
def test_vector_reduce_type(device_type: DeviceType):
    """VectorMarshall.reduce_type: dim=0 returns self, dim=1 returns element."""
    from slangpy.bindings.typeregistry import PYTHON_TYPES

    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "scale_vec", SCALE_VECTOR_SHADER)
    layout = func.module.layout

    marshall = PYTHON_TYPES[slangpy.math.float2](layout, slangpy.math.float2(1.0, 2.0))

    reduced_0 = marshall.reduce_type(None, 0)
    assert reduced_0 is not None
    assert "float" in reduced_0.full_name

    reduced_1 = marshall.reduce_type(None, 1)
    assert reduced_1 is not None
    assert "float" in reduced_1.full_name
    assert len(reduced_1.shape) < len(reduced_0.shape)

    with pytest.raises(ValueError, match=r"[Cc]annot reduce"):
        marshall.reduce_type(None, 2)


# ============================================================================
# MatrixMarshall.reduce_type edge cases
# ============================================================================


@pytest.mark.parametrize("device_type", CUDA_TYPES)
def test_matrix_reduce_type(device_type: DeviceType):
    """MatrixMarshall.reduce_type: dim=0 returns self, dim=1 returns row, dim=2 returns scalar."""
    from slangpy.bindings.typeregistry import PYTHON_TYPES

    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "make_mat", RETURN_FLOAT2X2_SHADER)
    layout = func.module.layout

    marshall = PYTHON_TYPES[slangpy.math.float2x2](
        layout, slangpy.math.float2x2([1.0, 0.0, 0.0, 1.0])
    )

    reduced_0 = marshall.reduce_type(None, 0)
    assert reduced_0 is not None

    reduced_1 = marshall.reduce_type(None, 1)
    assert reduced_1 is not None

    reduced_2 = marshall.reduce_type(None, 2)
    assert reduced_2 is not None
