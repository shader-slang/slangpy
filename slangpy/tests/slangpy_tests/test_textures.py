# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import numpy as np

from slangpy import (
    DeviceType,
    Format,
    TextureType,
    TextureDesc,
    TextureUsage,
    Texture,
    ALL_MIPS,
    InstanceBuffer,
    Module,
)
from slangpy.types import NDBuffer
from slangpy.reflection import ScalarType
from slangpy.builtin.texture import SCALARTYPE_TO_TEXTURE_FORMAT
from slangpy.types.buffer import _slang_to_numpy
from slangpy.testing import helpers

from typing import Union


def load_test_module(device_type: DeviceType):
    device = helpers.get_device(device_type)
    return Module(device.load_module("test_textures.slang"))


# Generate random data for a texture with a given array size and mip count.


def make_rand_data(type: TextureType, array_length: int, mip_count: int):

    if type == TextureType.texture_cube:
        array_length *= 6
        type = TextureType.texture_2d

    levels = []
    for i in range(0, array_length):
        sz = 32
        mips = []
        for i in range(0, mip_count):
            if type in (TextureType.texture_1d, TextureType.texture_1d_array):
                mips.append(np.random.rand(sz, 4).astype(np.float32))
            elif type in (TextureType.texture_2d, TextureType.texture_2d_array):
                mips.append(np.random.rand(sz, sz, 4).astype(np.float32))
            elif type == TextureType.texture_3d:
                mips.append(np.random.rand(sz, sz, sz, 4).astype(np.float32))
            else:
                raise ValueError(f"Unsupported resource type: {type}")
            sz = int(sz / 2)
        levels.append(mips)
    return levels


# Generate dictionary of arguments for creating a texture.
def make_args(type: TextureType, array_length: int, mips: int):

    desc = TextureDesc()
    desc.format = Format.rgba32_float
    desc.usage = TextureUsage.shader_resource | TextureUsage.unordered_access
    desc.mip_count = mips
    desc.array_length = array_length
    desc.width = 32

    if array_length > 1:
        if type == TextureType.texture_1d:
            type = TextureType.texture_1d_array
        elif type == TextureType.texture_2d:
            type = TextureType.texture_2d_array
        elif type == TextureType.texture_cube:
            type = TextureType.texture_cube_array

    desc.type = type

    if type in (TextureType.texture_1d, TextureType.texture_1d_array):
        desc.width = 32
    elif type in (
        TextureType.texture_2d,
        TextureType.texture_2d_array,
        TextureType.texture_2d_ms,
        TextureType.texture_2d_ms_array,
        TextureType.texture_cube,
        TextureType.texture_cube_array,
    ):
        desc.width = 32
        desc.height = 32
    elif type in (TextureType.texture_3d,):
        desc.width = 32
        desc.height = 32
        desc.depth = 32
    else:
        raise ValueError(f"Unsupported resource type: {type}")

    return desc


@pytest.mark.parametrize(
    "type",
    [
        TextureType.texture_1d,
        TextureType.texture_2d,
        TextureType.texture_3d,
        TextureType.texture_cube,
    ],
)
def make_grid_data(type: TextureType, array_length: int = 1):
    if array_length == 1:
        if type == TextureType.texture_1d:
            data = np.zeros((32, 1), dtype=np.int32)
            for i in range(32):
                data[i, 0] = i
        elif type == TextureType.texture_2d:
            data = np.zeros((32, 32, 2), dtype=np.int32)
            for i in range(32):
                for j in range(32):
                    data[i, j] = [i, j]
        elif type == TextureType.texture_3d:
            data = np.zeros((32, 32, 32, 3), dtype=np.int32)
            for i in range(32):
                for j in range(32):
                    for k in range(32):
                        data[i, j, k] = [i, j, k]
        elif type == TextureType.texture_cube:
            # cube
            data = np.zeros((6, 32, 32, 3), dtype=np.int32)
            for i in range(6):
                for j in range(32):
                    for k in range(32):
                        data[i, j, k] = [i, j, k]
        else:
            raise ValueError("Invalid dims")
    else:
        if type == TextureType.texture_1d:
            data = np.zeros((array_length, 32, 2), dtype=np.int32)
            for ai in range(array_length):
                for i in range(32):
                    data[ai, i] = [ai, i]
        elif type == TextureType.texture_2d:
            data = np.zeros((array_length, 32, 32, 3), dtype=np.int32)
            for ai in range(array_length):
                for i in range(32):
                    for j in range(32):
                        data[ai, i, j] = [ai, i, j]
        elif type == TextureType.texture_3d:
            data = np.zeros((array_length, 32, 32, 32, 4), dtype=np.int32)
            for ai in range(array_length):
                for i in range(32):
                    for j in range(32):
                        for k in range(32):
                            data[ai, i, j, k] = [ai, i, j, k]
        elif type == TextureType.texture_cube:
            # cube
            data = np.zeros((array_length, 6, 32, 32, 4), dtype=np.int32)
            for ai in range(array_length):
                for i in range(6):
                    for j in range(32):
                        for k in range(32):
                            data[ai, i, j, k] = [ai, i, j, k]
        else:
            raise ValueError("Invalid dims")

    return data


@pytest.mark.parametrize(
    "type",
    [TextureType.texture_1d, TextureType.texture_2d, TextureType.texture_3d],
)
@pytest.mark.parametrize("slices", [1, 4])
@pytest.mark.parametrize("mips", [ALL_MIPS, 1, 4])
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_read_write_texture(device_type: DeviceType, slices: int, mips: int, type: TextureType):
    if device_type == DeviceType.cuda:
        pytest.skip("Limited texture support in CUDA backend")
    if device_type == DeviceType.metal:
        pytest.skip("Limited texture support in Metal backend")

    m = load_test_module(device_type)
    assert m is not None

    # No 3d texture arrays.
    if type == TextureType.texture_3d and slices > 1:
        return

    # populate a buffer of grid coordinates
    grid_coords_data = make_grid_data(type, slices)
    dims = len(grid_coords_data.shape) - 1
    grid_coords = InstanceBuffer(
        struct=getattr(m, f"int{dims}").as_struct(), shape=grid_coords_data.shape[0:-1]
    )
    grid_coords.copy_from_numpy(grid_coords_data)

    # Create texture and build random data
    src_tex = m.device.create_texture(make_args(type, slices, mips))
    dest_tex = m.device.create_texture(make_args(type, slices, mips))
    rand_data = make_rand_data(src_tex.type, src_tex.array_length, src_tex.mip_count)

    # Write random data to texture
    for slice_idx, slice_data in enumerate(rand_data):
        for mip_idx, mip_data in enumerate(slice_data):
            src_tex.copy_from_numpy(mip_data, layer=slice_idx, mip=mip_idx)

    array_nm = ""
    if slices > 1:
        array_nm = f"_array"

    func = getattr(m, f"copy_pixel_{type.name}{array_nm}")
    func(grid_coords, src_tex, dest_tex)

    # Read back data and compare (currently just messing with mip 0)
    for slice_idx, slice_data in enumerate(rand_data):
        data = dest_tex.to_numpy(layer=slice_idx, mip=0)
        assert np.allclose(data, rand_data[slice_idx][0])


@pytest.mark.parametrize(
    "type",
    [TextureType.texture_1d, TextureType.texture_2d, TextureType.texture_3d],
)
@pytest.mark.parametrize("slices", [1, 4])
@pytest.mark.parametrize("mips", [ALL_MIPS, 1, 4])
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_read_write_texture_with_resource_views(
    device_type: DeviceType, slices: int, mips: int, type: TextureType
):
    if device_type == DeviceType.cuda:
        pytest.skip("Limited texture support in CUDA backend")
    if device_type == DeviceType.metal:
        pytest.skip("Limited texture support in Metal backend")

    m = load_test_module(device_type)
    assert m is not None

    # No 3d texture arrays.
    if type == TextureType.texture_3d and slices > 1:
        return

    # populate a buffer of grid coordinates
    grid_coords_data = make_grid_data(type, slices)
    dims = len(grid_coords_data.shape) - 1
    grid_coords = InstanceBuffer(
        struct=getattr(m, f"int{dims}").as_struct(), shape=grid_coords_data.shape[0:-1]
    )
    grid_coords.copy_from_numpy(grid_coords_data)

    # Create texture and build random data
    src_tex = m.device.create_texture(make_args(type, slices, mips))
    dest_tex = m.device.create_texture(make_args(type, slices, mips))
    rand_data = make_rand_data(src_tex.type, src_tex.array_length, src_tex.mip_count)

    # Write random data to texture
    for slice_idx, slice_data in enumerate(rand_data):
        for mip_idx, mip_data in enumerate(slice_data):
            src_tex.copy_from_numpy(mip_data, layer=slice_idx, mip=mip_idx)

    array_nm = ""
    if slices > 1:
        array_nm = f"_array"

    for mip_idx in range(src_tex.mip_count):
        func = getattr(m, f"copy_pixel_{type.name}{array_nm}")
        func(
            grid_coords,
            src_tex.create_view(mip=mip_idx),
            dest_tex.create_view(mip=mip_idx),
        )

    # Read back data and compare (currently just messing with mip 0)
    for slice_idx, slice_data in enumerate(rand_data):
        for mip_idx, mip_data in enumerate(slice_data):
            data = dest_tex.to_numpy(layer=slice_idx, mip=mip_idx)
            assert np.allclose(data, mip_data)


@pytest.mark.parametrize(
    "type",
    [TextureType.texture_1d, TextureType.texture_2d, TextureType.texture_3d],
)
@pytest.mark.parametrize("slices", [1, 4])
@pytest.mark.parametrize("mips", [ALL_MIPS, 1])
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_copy_value(device_type: DeviceType, slices: int, mips: int, type: TextureType):
    if device_type == DeviceType.metal and type == TextureType.texture_1d and mips > 1:
        pytest.skip("1D textures with mip maps are not supported on Metal")

    m = load_test_module(device_type)
    assert m is not None

    # No 3d texture arrays.
    if type == TextureType.texture_3d and slices > 1:
        return

    # Create texture and build random data
    src_tex = m.device.create_texture(make_args(type, slices, mips))
    dest_tex = m.device.create_texture(make_args(type, slices, mips))
    rand_data = make_rand_data(src_tex.type, src_tex.array_length, src_tex.mip_count)

    # Write random data to texture
    for slice_idx, slice_data in enumerate(rand_data):
        for mip_idx, mip_data in enumerate(slice_data):
            src_tex.copy_from_numpy(mip_data, layer=slice_idx, mip=mip_idx)

    m.copy_value(src_tex, dest_tex)

    # Read back data and compare (currently just messing with mip 0)
    for slice_idx, slice_data in enumerate(rand_data):
        data = dest_tex.to_numpy(layer=slice_idx, mip=0)
        assert np.allclose(data, rand_data[slice_idx][0])


@pytest.mark.parametrize(
    "type",
    [TextureType.texture_1d, TextureType.texture_2d, TextureType.texture_3d],
)
@pytest.mark.parametrize("slices", [1, 4])
@pytest.mark.parametrize("mips", [ALL_MIPS, 1])
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_copy_mip_values_with_resource_views(
    device_type: DeviceType, slices: int, mips: int, type: TextureType
):
    if device_type == DeviceType.metal and type == TextureType.texture_1d and mips > 1:
        pytest.skip("1D textures with mip maps are not supported on Metal")

    m = load_test_module(device_type)
    assert m is not None

    # No 3d texture arrays.
    if type == TextureType.texture_3d and slices > 1:
        return

    # Create texture and build random data
    src_tex = m.device.create_texture(make_args(type, slices, mips))
    dest_tex = m.device.create_texture(make_args(type, slices, mips))
    rand_data = make_rand_data(src_tex.type, src_tex.array_length, src_tex.mip_count)

    # Write random data to texture
    for slice_idx, slice_data in enumerate(rand_data):
        for mip_idx, mip_data in enumerate(slice_data):
            src_tex.copy_from_numpy(mip_data, layer=slice_idx, mip=mip_idx)

    for mip_idx in range(src_tex.mip_count):
        m.copy_value(
            src_tex.create_view(mip=mip_idx, mip_count=1),
            dest_tex.create_view(mip=mip_idx, mip_count=1),
        )

    # Read back data and compare (currently just messing with mip 0)
    for slice_idx, slice_data in enumerate(rand_data):
        for mip_idx, mip_data in enumerate(slice_data):
            data = dest_tex.to_numpy(layer=slice_idx, mip=mip_idx)
            assert np.allclose(
                data, rand_data[slice_idx][mip_idx]
            ), f"Mismatch in slice {slice_idx}, mip {mip_idx}"


@pytest.mark.parametrize(
    "type",
    [TextureType.texture_1d, TextureType.texture_2d, TextureType.texture_3d],
)
@pytest.mark.parametrize("slices", [1, 4])
@pytest.mark.parametrize("mips", [ALL_MIPS, 1])
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_copy_mip_values_with_all_uav_resource_views(
    device_type: DeviceType, slices: int, mips: int, type: TextureType
):
    if device_type == DeviceType.metal and type == TextureType.texture_1d and mips > 1:
        pytest.skip("1D textures with mip maps are not supported on Metal")

    m = load_test_module(device_type)
    assert m is not None

    # No 3d texture arrays.
    if type == TextureType.texture_3d and slices > 1:
        return

    # Create texture and build random data
    src_tex = m.device.create_texture(make_args(type, slices, mips))
    dest_tex = m.device.create_texture(make_args(type, slices, mips))
    rand_data = make_rand_data(src_tex.type, src_tex.array_length, src_tex.mip_count)

    # Write random data to texture
    for slice_idx, slice_data in enumerate(rand_data):
        for mip_idx, mip_data in enumerate(slice_data):
            src_tex.copy_from_numpy(mip_data, layer=slice_idx, mip=mip_idx)

    for mip_idx in range(src_tex.mip_count):
        m.copy_value(
            src_tex.create_view(mip=mip_idx, mip_count=1),
            dest_tex.create_view(mip=mip_idx, mip_count=1),
        )

    # Read back data and compare (currently just messing with mip 0)
    for slice_idx, slice_data in enumerate(rand_data):
        for mip_idx, mip_data in enumerate(slice_data):
            data = dest_tex.to_numpy(layer=slice_idx, mip=mip_idx)
            assert np.allclose(
                data, rand_data[slice_idx][mip_idx]
            ), f"Mismatch in slice {slice_idx}, mip {mip_idx}"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("shape", [(2, 2), (8, 2), (16, 32), (4, 128)])
def test_texture_2d_shapes(device_type: DeviceType, shape: tuple[int, ...]):
    module = load_test_module(device_type)

    tex_data = np.random.random(shape + (4,)).astype(np.float32)
    tex = module.device.create_texture(
        type=TextureType.texture_2d,
        width=shape[1],
        height=shape[0],
        usage=TextureUsage.shader_resource | TextureUsage.unordered_access,
        format=Format.rgba32_float,
        data=tex_data,
    )

    copied = module.return_value(tex, _result="numpy")

    assert np.allclose(copied, tex_data)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("shape", [(2, 2, 2), (8, 2, 4), (16, 32, 8), (4, 128, 2)])
def test_texture_3d_shapes(device_type: DeviceType, shape: tuple[int, ...]):
    module = load_test_module(device_type)

    tex_data = np.random.random(shape + (4,)).astype(np.float32)
    tex = module.device.create_texture(
        type=TextureType.texture_3d,
        width=shape[2],
        height=shape[1],
        depth=shape[0],
        usage=TextureUsage.shader_resource | TextureUsage.unordered_access,
        format=Format.rgba32_float,
        data=tex_data,
    )

    copied = module.return_value(tex, _result="numpy")

    assert np.allclose(copied, tex_data)


def texture_return_value_impl(
    device_type: DeviceType,
    texel_name: str,
    dims: int,
    channels: int,
    return_type: Union[str, type],
):
    if device_type == DeviceType.cuda and texel_name == "half":
        pytest.skip("Issue with half type in CUDA backend")

    if texel_name in ("uint8_t", "int8_t") and device_type == DeviceType.d3d12:
        pytest.skip("8-bit types not supported by DXC")

    m = load_test_module(device_type)
    assert m is not None

    texel_dtype = m[texel_name].as_struct().struct
    assert isinstance(texel_dtype, ScalarType)

    shape = (64, 32, 16)[:dims]
    if texel_name == "uint":
        data = np.random.randint(255, size=shape + (channels,))
    else:
        data = np.random.random(shape + (channels,))
    np_dtype = _slang_to_numpy(texel_dtype)
    data = data.astype(np_dtype)

    dtype = texel_name if channels == 1 else f"{texel_name}{channels}"
    buffer = NDBuffer(m.device, dtype, shape=shape)
    buffer.copy_from_numpy(data)

    result = m.passthru.map(buffer.dtype)(buffer, _result=return_type)

    assert isinstance(result, Texture)
    if dims == 1:
        assert result.type == TextureType.texture_1d
        assert result.width == buffer.shape[dims - 1]
    elif dims == 2:
        assert result.type == TextureType.texture_2d
        assert result.width == buffer.shape[dims - 1]
        assert result.height == buffer.shape[dims - 2]
    elif dims == 3:
        assert result.type == TextureType.texture_3d
        assert result.width == buffer.shape[dims - 1]
        assert result.height == buffer.shape[dims - 2]
        assert result.depth == buffer.shape[dims - 3]

    expected_format = SCALARTYPE_TO_TEXTURE_FORMAT[texel_dtype.slang_scalar_type][channels - 1]
    assert expected_format is not None
    assert result.format == expected_format

    result_np = result.to_numpy()
    order = list(reversed(range(dims)))
    if channels > 1:
        order += [dims]
    # result_np = result_np.transpose(order)

    assert np.allclose(result_np, data.squeeze())


@pytest.mark.parametrize(
    "texel_name", ["uint8_t", "uint16_t", "int8_t", "int16_t", "float", "half", "uint"]
)
@pytest.mark.parametrize("dims", [1, 2, 3])
@pytest.mark.parametrize("channels", [1, 2, 4])
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_texture_return_value(device_type: DeviceType, texel_name: str, dims: int, channels: int):
    if device_type == DeviceType.metal:
        pytest.skip("Limited texture support in Metal backend")
    texture_return_value_impl(device_type, texel_name, dims, channels, Texture)


# This case checks for when the return type is the string "texture".
# This checks a subset of the "test_texture_return_value" parameters.
@pytest.mark.parametrize("texel_name", ["float"])
@pytest.mark.parametrize("dims", [1, 2, 3])
@pytest.mark.parametrize("channels", [4])
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_texture_return_value_str(
    device_type: DeviceType, texel_name: str, dims: int, channels: int
):
    texture_return_value_impl(device_type, texel_name, dims, channels, "texture")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
