# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Tests targeting C++ coverage gaps identified in Tier 1 of the coverage plan.

Covers:
- FunctionNode.__repr__ (slangpyfunction.cpp, ~8 lines)
- PackedArg.__repr__ (slangpypackedarg.cpp, ~8 lines)
- get_texture_shape() standalone function (slangpyresources.cpp, ~16 lines)
- SignatureBuilder.bytes property (slangpy.cpp, ~3 lines)
- Shape.__eq__ with non-Shape/non-list type (slangpy.cpp, 1 line)
- StridedBufferView negative indexing + out-of-bounds (slangpystridedbufferview.cpp, 2 lines)
- NativeSlangType.__repr__ with null reflection (slangpy.cpp, 2 lines)
- StridedBufferView.copy_from_torch CPU fallback (slangpystridedbufferview.cpp, 3 lines)
"""

import sys
import pytest
import numpy as np

from slangpy import (
    DeviceType,
    Format,
    Module,
    TextureType,
    TextureDesc,
    TextureUsage,
    pack,
)
from slangpy.slangpy import Shape, SignatureBuilder, get_texture_shape, NativeSlangType
from slangpy.types import Tensor
from slangpy.testing import helpers

MODULE_SRC = r"""
int identity(int x) { return x; }
"""


# ============================================================================
# repr() coverage
# ============================================================================


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_function_node_repr(device_type: DeviceType):
    """FunctionNode.__repr__ exercises NativeFunctionNode::to_string (~8 lines)."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "identity", MODULE_SRC)
    r = repr(func)
    assert "NativeFunctionNode" in r
    assert "type" in r
    assert "data_type" in r


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_packed_arg_repr(device_type: DeviceType):
    """PackedArg.__repr__ exercises NativePackedArg::to_string (~8 lines)."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "identity", MODULE_SRC)
    pa = pack(func.module, 42)
    r = repr(pa)
    assert "NativePackedArg" in r
    assert "python_type" in r
    assert "python_object_type" in r


# ============================================================================
# get_texture_shape() standalone function
# ============================================================================


def _create_texture(device, tex_type, width=16, height=16, depth=16, array_length=1):
    desc = TextureDesc()
    desc.type = tex_type
    desc.format = Format.rgba32_float
    desc.usage = TextureUsage.shader_resource
    desc.width = width
    desc.mip_count = 1
    desc.array_length = array_length
    if tex_type in (
        TextureType.texture_2d,
        TextureType.texture_2d_array,
        TextureType.texture_cube,
        TextureType.texture_cube_array,
    ):
        desc.height = height
    if tex_type == TextureType.texture_3d:
        desc.height = height
        desc.depth = depth
    return device.create_texture(desc)


@pytest.mark.parametrize(
    "tex_type,expected_shape",
    [
        (TextureType.texture_1d, (16,)),
        (TextureType.texture_2d, (16, 16)),
        (TextureType.texture_3d, (16, 16, 16)),
        (TextureType.texture_cube, (6, 16, 16)),
    ],
    ids=["1d", "2d", "3d", "cube"],
)
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_get_texture_shape(device_type, tex_type, expected_shape):
    """get_texture_shape() free function (slangpyresources.cpp, ~16 lines)."""
    device = helpers.get_device(device_type)
    texture = _create_texture(device, tex_type)
    shape = get_texture_shape(texture)
    assert shape.as_tuple() == expected_shape


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_get_texture_shape_array_types(device_type):
    """get_texture_shape() with array texture types."""
    device = helpers.get_device(device_type)

    tex_1d_arr = _create_texture(device, TextureType.texture_1d_array, array_length=4)
    assert get_texture_shape(tex_1d_arr).as_tuple() == (4, 16)

    tex_2d_arr = _create_texture(device, TextureType.texture_2d_array, array_length=3)
    assert get_texture_shape(tex_2d_arr).as_tuple() == (3, 16, 16)

    tex_cube_arr = _create_texture(device, TextureType.texture_cube_array, array_length=2)
    assert get_texture_shape(tex_cube_arr).as_tuple() == (2, 6, 16, 16)


# ============================================================================
# SignatureBuilder.bytes property
# ============================================================================


def test_signature_builder_bytes():
    """SignatureBuilder.bytes returns raw bytes (slangpy.cpp, ~3 lines)."""
    sb = SignatureBuilder()
    sb.add("hello")
    sb.add("world")
    b = sb.bytes
    assert isinstance(b, bytes)
    assert b"hello" in b
    assert b"world" in b
    assert sb.str.encode() == b


# ============================================================================
# Shape.__eq__ with non-Shape/non-list type
# ============================================================================


def test_shape_eq_incompatible_type():
    """Shape.__eq__ returns False for non-Shape/non-list types (slangpy.cpp, 1 line)."""
    s = Shape([1, 2, 3])
    assert s != "not a shape"
    assert s != 42
    assert s != 3.14


# ============================================================================
# NativeSlangType.__repr__ with null reflection
# ============================================================================


def test_slangtype_repr_no_reflection():
    """NativeSlangType.__repr__ without type_reflection (slangpy.cpp, ~2 lines)."""
    st = NativeSlangType()
    st.shape = Shape([2, 3])
    r = repr(st)
    assert "NativeSlangType" in r
    assert "type_reflection = None" in r
    assert "shape" in r


# ============================================================================
# StridedBufferView negative indexing + out-of-bounds
# ============================================================================


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_strided_buffer_view_negative_index(device_type: DeviceType):
    """StridedBufferView negative index wrapping (slangpystridedbufferview.cpp, 1 line)."""
    device = helpers.get_device(device_type)
    tensor = Tensor.empty(device, dtype="float", shape=(4,))
    data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    tensor.copy_from_numpy(data)
    device.wait_for_idle()
    last = tensor[-1].to_numpy().item()
    assert last == pytest.approx(40.0)
    second_last = tensor[-2].to_numpy().item()
    assert second_last == pytest.approx(30.0)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_strided_buffer_view_out_of_bounds(device_type: DeviceType):
    """StridedBufferView out-of-bounds raises IndexError (slangpystridedbufferview.cpp, 1 line)."""
    device = helpers.get_device(device_type)
    tensor = Tensor.empty(device, dtype="float", shape=(4,))
    with pytest.raises(IndexError):
        _ = tensor[4]
    with pytest.raises(IndexError):
        _ = tensor[-5]


# ============================================================================
# StridedBufferView.copy_from_torch CPU fallback
# ============================================================================


@pytest.mark.skipif(sys.platform == "darwin", reason="Torch tests require CUDA")
def test_copy_from_torch_cpu_fallback():
    """copy_from_torch with CPU tensor falls back through numpy (3 lines)."""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed")

    device = helpers.get_device(DeviceType.cuda)
    tensor = Tensor.empty(device, dtype="float", shape=(4,))
    cpu_data = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    tensor.copy_from_torch(cpu_data)
    device.wait_for_idle()
    result = tensor.to_numpy()
    np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0, 4.0])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
