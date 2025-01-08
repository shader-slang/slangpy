# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

import slangpy.tests.helpers as helpers
from slangpy import InstanceBuffer, Module
from slangpy.backend import DeviceType, Format, ResourceType, ResourceUsage


def load_test_module(device_type: DeviceType):
    device = helpers.get_device(device_type)
    return Module(device.load_module("test_textures.slang"))

# Generate random data for a texture with a given array size and mip count.


def make_rand_data(type: ResourceType, array_size: int, mip_count: int):

    if type == ResourceType.texture_cube:
        array_size *= 6
        type = ResourceType.texture_2d

    levels = []
    for i in range(0, array_size):
        sz = 32
        mips = []
        for i in range(0, mip_count):
            if type == ResourceType.texture_1d:
                mips.append(np.random.rand(sz, 4).astype(np.float32))
            elif type == ResourceType.texture_2d:
                mips.append(np.random.rand(sz, sz, 4).astype(np.float32))
            elif type == ResourceType.texture_3d:
                mips.append(np.random.rand(sz, sz, sz, 4).astype(np.float32))
            else:
                raise ValueError(f"Unsupported resource type: {type}")
            sz = int(sz / 2)
        levels.append(mips)
    return levels


# Generate dictionary of arguments for creating a texture.
def make_args(type: ResourceType, array_size: int, mips: int):
    args = {
        "format": Format.rgba32_float,
        "usage": ResourceUsage.shader_resource | ResourceUsage.unordered_access,
        "mip_count": mips,
        "array_size": array_size,
    }
    if type == ResourceType.texture_1d:
        args.update({"type": type, "width": 32})
    elif type == ResourceType.texture_2d:
        args.update({"type": type, "width": 32, "height": 32})
    elif type == ResourceType.texture_3d:
        args.update({"type": type, "width": 32, "height": 32, "depth": 32})
    elif type == ResourceType.texture_cube:
        args.update({"type": type, "width": 32, "height": 32})
    else:
        raise ValueError(f"Unsupported resource type: {type}")
    return args


@pytest.mark.parametrize(
    "type",
    [
        ResourceType.texture_1d,
        ResourceType.texture_2d,
        ResourceType.texture_3d,
        ResourceType.texture_cube,
    ],
)
def make_grid_data(type: ResourceType, array_size: int = 1):
    if array_size == 1:
        if type == ResourceType.texture_1d:
            data = np.zeros((32, 1), dtype=np.int32)
            for i in range(32):
                data[i, 0] = i
        elif type == ResourceType.texture_2d:
            data = np.zeros((32, 32, 2), dtype=np.int32)
            for i in range(32):
                for j in range(32):
                    data[i, j] = [i, j]
        elif type == ResourceType.texture_3d:
            data = np.zeros((32, 32, 32, 3), dtype=np.int32)
            for i in range(32):
                for j in range(32):
                    for k in range(32):
                        data[i, j, k] = [i, j, k]
        elif type == ResourceType.texture_cube:
            # cube
            data = np.zeros((6, 32, 32, 3), dtype=np.int32)
            for i in range(6):
                for j in range(32):
                    for k in range(32):
                        data[i, j, k] = [i, j, k]
        else:
            raise ValueError("Invalid dims")
    else:
        if type == ResourceType.texture_1d:
            data = np.zeros((array_size, 32, 2), dtype=np.int32)
            for ai in range(array_size):
                for i in range(32):
                    data[ai, i] = [ai, i]
        elif type == ResourceType.texture_2d:
            data = np.zeros((array_size, 32, 32, 3), dtype=np.int32)
            for ai in range(array_size):
                for i in range(32):
                    for j in range(32):
                        data[ai, i, j] = [ai, i, j]
        elif type == ResourceType.texture_3d:
            data = np.zeros((array_size, 32, 32, 32, 4), dtype=np.int32)
            for ai in range(array_size):
                for i in range(32):
                    for j in range(32):
                        for k in range(32):
                            data[ai, i, j, k] = [ai, i, j, k]
        elif type == ResourceType.texture_cube:
            # cube
            data = np.zeros((array_size, 6, 32, 32, 4), dtype=np.int32)
            for ai in range(array_size):
                for i in range(6):
                    for j in range(32):
                        for k in range(32):
                            data[ai, i, j, k] = [ai, i, j, k]
        else:
            raise ValueError("Invalid dims")

    return data


@pytest.mark.parametrize(
    "type",
    [
        ResourceType.texture_1d,
        ResourceType.texture_2d,
        ResourceType.texture_3d
    ],
)
@pytest.mark.parametrize("slices", [1, 4])
@pytest.mark.parametrize("mips", [0, 1, 4])
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_read_write_texture(
    device_type: DeviceType, slices: int, mips: int, type: ResourceType
):
    m = load_test_module(device_type)
    assert m is not None

    # No 3d texture arrays.
    if type == ResourceType.texture_3d and slices > 1:
        return

    if type == ResourceType.texture_1d and slices > 1:
        pytest.skip("Pending slang fix")

    # populate a buffer of grid coordinates
    grid_coords_data = make_grid_data(type, slices)
    dims = len(grid_coords_data.shape) - 1
    grid_coords = InstanceBuffer(struct=getattr(
        m, f"int{dims}").as_struct(), shape=grid_coords_data.shape[0:-1])
    grid_coords.from_numpy(grid_coords_data)

    # Create texture and build random data
    src_tex = m.device.create_texture(**make_args(type, slices, mips))
    dest_tex = m.device.create_texture(**make_args(type, slices, mips))
    rand_data = make_rand_data(src_tex.type, src_tex.array_size, src_tex.mip_count)

    # Write random data to texture
    for slice_idx, slice_data in enumerate(rand_data):
        for mip_idx, mip_data in enumerate(slice_data):
            src_tex.from_numpy(mip_data, array_slice=slice_idx, mip_level=mip_idx)

    array_nm = ""
    if slices > 1:
        array_nm = f"_array"

    func = getattr(m, f"copy_pixel_{type.name}{array_nm}")
    func(grid_coords, src_tex, dest_tex)

    # Read back data and compare (currently just messing with mip 0)
    for slice_idx, slice_data in enumerate(rand_data):
        data = dest_tex.to_numpy(array_slice=slice_idx, mip_level=0)
        assert np.allclose(data, rand_data[slice_idx][0])


@pytest.mark.parametrize(
    "type",
    [
        ResourceType.texture_1d,
        ResourceType.texture_2d,
        ResourceType.texture_3d
    ],
)
@pytest.mark.parametrize("slices", [1, 4])
@pytest.mark.parametrize("mips", [0, 1, 4])
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_read_write_texture_with_resource_views(
    device_type: DeviceType, slices: int, mips: int, type: ResourceType
):
    m = load_test_module(device_type)
    assert m is not None

    # No 3d texture arrays.
    if type == ResourceType.texture_3d and slices > 1:
        return
    if type == ResourceType.texture_3d and mips != 1:
        pytest.skip("Pending slang fix")

    if type == ResourceType.texture_1d and slices > 1:
        pytest.skip("Pending slang fix")

    # populate a buffer of grid coordinates
    grid_coords_data = make_grid_data(type, slices)
    dims = len(grid_coords_data.shape) - 1
    grid_coords = InstanceBuffer(struct=getattr(
        m, f"int{dims}").as_struct(), shape=grid_coords_data.shape[0:-1])
    grid_coords.from_numpy(grid_coords_data)

    # Create texture and build random data
    src_tex = m.device.create_texture(**make_args(type, slices, mips))
    dest_tex = m.device.create_texture(**make_args(type, slices, mips))
    rand_data = make_rand_data(src_tex.type, src_tex.array_size, src_tex.mip_count)

    # Write random data to texture
    for slice_idx, slice_data in enumerate(rand_data):
        for mip_idx, mip_data in enumerate(slice_data):
            src_tex.from_numpy(mip_data, array_slice=slice_idx, mip_level=mip_idx)

    array_nm = ""
    if slices > 1:
        array_nm = f"_array"

    for mip_idx in range(src_tex.mip_count):
        func = getattr(m, f"copy_pixel_{type.name}{array_nm}")
        func(grid_coords, src_tex.get_srv(mip_idx), dest_tex.get_uav(mip_idx))

    # Read back data and compare (currently just messing with mip 0)
    for slice_idx, slice_data in enumerate(rand_data):
        for mip_idx, mip_data in enumerate(slice_data):
            data = dest_tex.to_numpy(array_slice=slice_idx, mip_level=mip_idx)
            assert np.allclose(data, mip_data)


@pytest.mark.parametrize(
    "type",
    [
        ResourceType.texture_2d
    ],
)
@pytest.mark.parametrize("slices", [1])
@pytest.mark.parametrize("mips", [0])
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_read_write_texture_with_invalid_resource_views(
    device_type: DeviceType, slices: int, mips: int, type: ResourceType
):
    m = load_test_module(device_type)
    assert m is not None

    # No 3d texture arrays.
    if type == ResourceType.texture_3d and slices > 1:
        return

    if type == ResourceType.texture_1d and slices > 1:
        pytest.skip("Pending slang fix")

    # populate a buffer of grid coordinates
    grid_coords_data = make_grid_data(type, slices)
    dims = len(grid_coords_data.shape) - 1
    grid_coords = InstanceBuffer(struct=getattr(
        m, f"int{dims}").as_struct(), shape=grid_coords_data.shape[0:-1])
    grid_coords.from_numpy(grid_coords_data)

    # Create texture and build random data
    src_tex = m.device.create_texture(**make_args(type, slices, mips))
    dest_tex = m.device.create_texture(**make_args(type, slices, mips))

    array_nm = ""
    if slices > 1:
        array_nm = f"_array"

    with pytest.raises(ValueError):
        func = getattr(m, f"copy_pixel_{type.name}{array_nm}")
        func(grid_coords, src_tex.get_srv(0), dest_tex.get_srv(0))

    with pytest.raises(ValueError):
        func = getattr(m, f"copy_pixel_{type.name}{array_nm}")
        func(grid_coords, src_tex.get_uav(0), dest_tex.get_uav(0))


@pytest.mark.parametrize(
    "type",
    [
        ResourceType.texture_1d,
        ResourceType.texture_2d,
        ResourceType.texture_3d
    ],
)
@pytest.mark.parametrize("slices", [1, 4])
@pytest.mark.parametrize("mips", [1])
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_copy_value(
    device_type: DeviceType, slices: int, mips: int, type: ResourceType
):
    m = load_test_module(device_type)
    assert m is not None

    # No 3d texture arrays.
    if type == ResourceType.texture_3d and slices > 1:
        return

    if type == ResourceType.texture_1d and slices > 1:
        pytest.skip("Pending slang fix")

    # Create texture and build random data
    src_tex = m.device.create_texture(**make_args(type, slices, mips))
    dest_tex = m.device.create_texture(**make_args(type, slices, mips))
    rand_data = make_rand_data(src_tex.type, src_tex.array_size, src_tex.mip_count)

    # Write random data to texture
    for slice_idx, slice_data in enumerate(rand_data):
        for mip_idx, mip_data in enumerate(slice_data):
            src_tex.from_numpy(mip_data, array_slice=slice_idx, mip_level=mip_idx)

    m.copy_value(src_tex, dest_tex)

    # Read back data and compare (currently just messing with mip 0)
    for slice_idx, slice_data in enumerate(rand_data):
        data = dest_tex.to_numpy(array_slice=slice_idx, mip_level=0)
        assert np.allclose(data, rand_data[slice_idx][0])


@pytest.mark.parametrize(
    "type",
    [
        ResourceType.texture_1d,
        ResourceType.texture_2d,
        ResourceType.texture_3d
    ],
)
@pytest.mark.parametrize("slices", [1, 4])
@pytest.mark.parametrize("mips", [0, 1])
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_copy_mip_values_with_resource_views(
    device_type: DeviceType, slices: int, mips: int, type: ResourceType
):
    m = load_test_module(device_type)
    assert m is not None

    # No 3d texture arrays.
    if type == ResourceType.texture_3d and slices > 1:
        return

    if type == ResourceType.texture_3d and mips != 1:
        pytest.skip("Pending slang fix")

    if type == ResourceType.texture_1d and slices > 1:
        pytest.skip("Pending slang fix")

    # Create texture and build random data
    src_tex = m.device.create_texture(**make_args(type, slices, mips))
    dest_tex = m.device.create_texture(**make_args(type, slices, mips))
    rand_data = make_rand_data(src_tex.type, src_tex.array_size, src_tex.mip_count)

    # Write random data to texture
    for slice_idx, slice_data in enumerate(rand_data):
        for mip_idx, mip_data in enumerate(slice_data):
            src_tex.from_numpy(mip_data, array_slice=slice_idx, mip_level=mip_idx)

    for mip_idx in range(src_tex.mip_count):
        m.copy_value(src_tex.get_srv(mip_idx), dest_tex.get_uav(mip_idx))

    # Read back data and compare (currently just messing with mip 0)
    for slice_idx, slice_data in enumerate(rand_data):
        for mip_idx, mip_data in enumerate(slice_data):
            data = dest_tex.to_numpy(array_slice=slice_idx, mip_level=mip_idx)
            assert np.allclose(data, rand_data[slice_idx][mip_idx])


@pytest.mark.parametrize(
    "type",
    [
        ResourceType.texture_1d,
        ResourceType.texture_2d,
        ResourceType.texture_3d
    ],
)
@pytest.mark.parametrize("slices", [1, 4])
@pytest.mark.parametrize("mips", [0, 1])
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_copy_mip_values_with_all_uav_resource_views(
    device_type: DeviceType, slices: int, mips: int, type: ResourceType
):
    m = load_test_module(device_type)
    assert m is not None

    # No 3d texture arrays.
    if type == ResourceType.texture_3d and slices > 1:
        return

    if type == ResourceType.texture_3d and mips != 1:
        pytest.skip("Pending slang fix")

    if type == ResourceType.texture_1d and slices > 1:
        pytest.skip("Pending slang fix")

    # Create texture and build random data
    src_tex = m.device.create_texture(**make_args(type, slices, mips))
    dest_tex = m.device.create_texture(**make_args(type, slices, mips))
    rand_data = make_rand_data(src_tex.type, src_tex.array_size, src_tex.mip_count)

    # Write random data to texture
    for slice_idx, slice_data in enumerate(rand_data):
        for mip_idx, mip_data in enumerate(slice_data):
            src_tex.from_numpy(mip_data, array_slice=slice_idx, mip_level=mip_idx)

    for mip_idx in range(src_tex.mip_count):
        m.copy_value(src_tex.get_uav(mip_idx), dest_tex.get_uav(mip_idx))

    # Read back data and compare (currently just messing with mip 0)
    for slice_idx, slice_data in enumerate(rand_data):
        for mip_idx, mip_data in enumerate(slice_data):
            data = dest_tex.to_numpy(array_slice=slice_idx, mip_level=mip_idx)
            assert np.allclose(data, rand_data[slice_idx][mip_idx])


@pytest.mark.parametrize(
    "type",
    [
        ResourceType.texture_2d,
    ],
)
@pytest.mark.parametrize("slices", [1])
@pytest.mark.parametrize("mips", [1])
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_invalid_resource_view(
    device_type: DeviceType, slices: int, mips: int, type: ResourceType
):
    m = load_test_module(device_type)
    assert m is not None

    # No 3d texture arrays.
    if type == ResourceType.texture_3d and slices > 1:
        return

    if type == ResourceType.texture_1d and slices > 1:
        pytest.skip("Pending slang fix")

    # Create texture and build random data
    src_tex = m.device.create_texture(**make_args(type, slices, mips))
    dest_tex = m.device.create_texture(**make_args(type, slices, mips))

    with pytest.raises(ValueError):
        for mip_idx in range(mips):
            m.copy_value(src_tex.get_srv(mip_idx), dest_tex.get_srv(mip_idx))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
