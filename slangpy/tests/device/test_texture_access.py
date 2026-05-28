# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import numpy as np
import numpy.typing as npt

import slangpy as spy
from slangpy.testing import helpers


# Generate random data for a texture with a given array size and mip level count.
def make_rand_data(type: spy.TextureType, array_size: int, mip_count: int):

    layer_count = array_size
    if type in [spy.TextureType.texture_cube, spy.TextureType.texture_cube_array]:
        layer_count *= 6

    levels = []
    for i in range(0, layer_count):
        sz = 32
        mips = []
        for i in range(0, mip_count):
            if type in [spy.TextureType.texture_1d, spy.TextureType.texture_1d_array]:
                mips.append(np.random.rand(sz, 4).astype(np.float32))
            elif type in [
                spy.TextureType.texture_2d,
                spy.TextureType.texture_2d_array,
                spy.TextureType.texture_2d_ms,
                spy.TextureType.texture_2d_ms_array,
                spy.TextureType.texture_cube,
                spy.TextureType.texture_cube_array,
            ]:
                mips.append(np.random.rand(sz, sz, 4).astype(np.float32))
            elif type in [spy.TextureType.texture_3d]:
                mips.append(np.random.rand(sz, sz, sz, 4).astype(np.float32))
            else:
                raise ValueError(f"Unsupported resource type: {type}")
            sz = int(sz / 2)
        levels.append(mips)
    return levels


# Generate dictionary of arguments for creating a texture.
def make_args(
    type: spy.TextureType,
    array_length: int,
    mip_count: int,
    format: spy.Format = spy.Format.rgba32_float,
    usage: spy.TextureUsage = spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
):
    args = {
        "format": format,
        "usage": usage,
        "mip_count": mip_count,
        "array_length": array_length,
    }
    if type in [spy.TextureType.texture_1d, spy.TextureType.texture_1d_array]:
        args.update({"type": type, "width": 32})
    elif type in [
        spy.TextureType.texture_2d,
        spy.TextureType.texture_2d_array,
        spy.TextureType.texture_2d_ms,
        spy.TextureType.texture_2d_ms_array,
        spy.TextureType.texture_cube,
        spy.TextureType.texture_cube_array,
    ]:
        args.update({"type": type, "width": 32, "height": 32})
    elif type in [spy.TextureType.texture_3d]:
        args.update({"type": type, "width": 32, "height": 32, "depth": 32})
    else:
        raise ValueError(f"Unsupported resource type: {type}")
    return args


def create_test_textures(
    device: spy.Device,
    count: int,
    type: spy.TextureType,
    array_length: int,
    mips: int,
    format: spy.Format = spy.Format.rgba32_float,
    usage: spy.TextureUsage = spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
):
    # No 3d texture arrays.
    if type == spy.TextureType.texture_3d and array_length > 1:
        pytest.skip("3d texture arrays not supported")

    # No 1D textures with mips on Metal.
    if (
        device.desc.type == spy.DeviceType.metal
        and type == spy.TextureType.texture_1d
        and mips != 1
    ):
        pytest.skip("Metal does not support 1d texture with mips")

    # Adjust texture type to account for arrayness
    if array_length > 1:
        if type == spy.TextureType.texture_1d:
            type = spy.TextureType.texture_1d_array
        elif type == spy.TextureType.texture_2d:
            type = spy.TextureType.texture_2d_array
        elif type == spy.TextureType.texture_2d_ms:
            type = spy.TextureType.texture_2d_ms_array
        elif type == spy.TextureType.texture_cube:
            type = spy.TextureType.texture_cube_array

    # Skip non-2d compressed format
    formatinfo = spy.get_format_info(format)
    if formatinfo.is_compressed and type != spy.TextureType.texture_2d:
        pytest.skip("Compressed formats only supported for 2D textures")

    # Create textures and build random data
    return tuple(
        [device.create_texture(**make_args(type, array_length, mips)) for _ in range(count)]
    )


def create_test_texture(
    device: spy.Device,
    type: spy.TextureType,
    array_length: int,
    mips: int,
    format: spy.Format = spy.Format.rgba32_float,
    usage: spy.TextureUsage = spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
):
    return create_test_textures(device, 1, type, array_length, mips, format, usage)[0]


@pytest.mark.parametrize(
    "type",
    [
        spy.TextureType.texture_1d,
        spy.TextureType.texture_2d,
        spy.TextureType.texture_3d,
        spy.TextureType.texture_cube,
    ],
)
@pytest.mark.parametrize("array_length", [1, 4, 16])
@pytest.mark.parametrize("mips", [spy.ALL_MIPS, 1, 4])
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_read_write_texture(
    device_type: spy.DeviceType, array_length: int, mips: int, type: spy.TextureType
):
    device = helpers.get_device(device_type)
    assert device is not None

    # Create texture and build random data
    tex = create_test_texture(device, type, array_length, mips)
    rand_data = make_rand_data(tex.type, tex.array_length, tex.mip_count)

    # Write random data to texture
    for layer_idx, layer_data in enumerate(rand_data):
        for mip_idx, mip_data in enumerate(layer_data):
            tex.copy_from_numpy(mip_data, layer=layer_idx, mip=mip_idx)

    # Read back data and compare
    for layer_idx, layer_data in enumerate(rand_data):
        for mip_idx, mip_data in enumerate(layer_data):
            data = tex.to_numpy(layer=layer_idx, mip=mip_idx)
            assert np.allclose(data, mip_data)


@pytest.mark.parametrize(
    "type",
    [
        spy.TextureType.texture_1d,
        spy.TextureType.texture_2d,
        spy.TextureType.texture_3d,
        spy.TextureType.texture_cube,
    ],
)
@pytest.mark.parametrize("array_length", [1, 4, 16])
@pytest.mark.parametrize("mips", [spy.ALL_MIPS, 1, 4])
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_upload_texture_single_mip(
    device_type: spy.DeviceType, array_length: int, mips: int, type: spy.TextureType
):
    device = helpers.get_device(device_type)
    assert device is not None

    # Create texture and build random data
    tex = create_test_texture(device, type, array_length, mips)
    rand_data = make_rand_data(tex.type, tex.array_length, tex.mip_count)

    # Write random data to texture
    encoder = device.create_command_encoder()
    for layer_idx, layer_data in enumerate(rand_data):
        for mip_idx, mip_data in enumerate(layer_data):
            encoder.upload_texture_data(tex, layer=layer_idx, mip=mip_idx, data=mip_data)
    device.submit_command_buffer(encoder.finish())
    device.wait_for_idle()

    # Read back data and compare
    for layer_idx, layer_data in enumerate(rand_data):
        for mip_idx, mip_data in enumerate(layer_data):
            data = tex.to_numpy(layer=layer_idx, mip=mip_idx)
            assert np.allclose(data, mip_data)


@pytest.mark.parametrize(
    "type",
    [
        spy.TextureType.texture_1d,
        spy.TextureType.texture_2d,
        spy.TextureType.texture_3d,
        spy.TextureType.texture_cube,
    ],
)
@pytest.mark.parametrize("array_length", [1, 4, 16])
@pytest.mark.parametrize("mips", [spy.ALL_MIPS, 1, 4])
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_upload_whole_texture(
    device_type: spy.DeviceType, array_length: int, mips: int, type: spy.TextureType
):
    device = helpers.get_device(device_type)
    assert device is not None

    # Create texture and build random data
    tex = create_test_texture(device, type, array_length, mips)
    rand_data = make_rand_data(tex.type, tex.array_length, tex.mip_count)

    # Write random data to texture
    datas: list[npt.ArrayLike] = []
    for layer_idx, layer_data in enumerate(rand_data):
        for mip_idx, mip_data in enumerate(layer_data):
            datas.append(mip_data)

    encoder = device.create_command_encoder()
    encoder.upload_texture_data(tex, datas)
    device.submit_command_buffer(encoder.finish())
    device.wait_for_idle()

    # Read back data and compare
    for layer_idx, layer_data in enumerate(rand_data):
        for mip_idx, mip_data in enumerate(layer_data):
            data = tex.to_numpy(layer=layer_idx, mip=mip_idx)
            assert np.allclose(data, mip_data)


@pytest.mark.parametrize(
    "type",
    [
        spy.TextureType.texture_1d,
        spy.TextureType.texture_2d,
        spy.TextureType.texture_3d,
        spy.TextureType.texture_cube,
    ],
)
@pytest.mark.parametrize("array_length", [1, 4])
@pytest.mark.parametrize("mips", [1, 4])
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize(
    "format", [spy.Format.rgba32_float, spy.Format.rgba8_unorm, spy.Format.bc1_unorm]
)
def test_texture_layout(
    device_type: spy.DeviceType,
    array_length: int,
    mips: int,
    type: spy.TextureType,
    format: spy.Format,
):
    device = helpers.get_device(device_type)
    assert device is not None

    # Create texture + get format info
    tex = create_test_texture(device, type, array_length, mips, format)
    formatinfo = spy.get_format_info(tex.format)

    def alignUp(value: int, alignment: int):
        return (value + alignment - 1) & ~(alignment - 1)

    # Validate sizes using row_alignment of 1 (i.e. packed)
    w = tex.width
    h = tex.height
    d = tex.depth
    for mip in range(tex.mip_count):
        layout = tex.get_subresource_layout(mip, row_alignment=1)
        assert layout.size.x == w
        assert layout.size.y == h
        assert layout.size.z == d
        assert layout.col_pitch == formatinfo.bytes_per_block
        assert (
            layout.row_pitch
            == layout.col_pitch * alignUp(w, formatinfo.block_width) // formatinfo.block_width
        )
        assert (
            layout.slice_pitch
            == layout.row_pitch * alignUp(h, formatinfo.block_height) // formatinfo.block_height
        )
        w = max(1, w // 2)
        h = max(1, h // 2)
        d = max(1, d // 2)

    # Validate sizes using row_alignment of 256 (i.e. D3D12)
    w = tex.width
    h = tex.height
    d = tex.depth
    for mip in range(tex.mip_count):
        layout = tex.get_subresource_layout(mip, row_alignment=256)
        assert layout.size.x == w
        assert layout.size.y == h
        assert layout.size.z == d
        assert layout.col_pitch == formatinfo.bytes_per_block
        assert layout.row_pitch == alignUp(
            layout.col_pitch * alignUp(w, formatinfo.block_width) // formatinfo.block_width,
            256,
        )
        assert (
            layout.slice_pitch
            == layout.row_pitch * alignUp(h, formatinfo.block_height) // formatinfo.block_height
        )
        w = max(1, w // 2)
        h = max(1, h // 2)
        d = max(1, d // 2)


@pytest.mark.parametrize(
    "type",
    [
        spy.TextureType.texture_1d,
        spy.TextureType.texture_2d,
        spy.TextureType.texture_3d,
    ],
)
@pytest.mark.parametrize("array_length", [1, 4])
@pytest.mark.parametrize("mips", [spy.ALL_MIPS, 1, 4])
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_shader_read_write_texture(
    device_type: spy.DeviceType, array_length: int, mips: int, type: spy.TextureType
):
    if device_type == spy.DeviceType.cuda:
        pytest.skip("CUDA needs texture format attributes")

    device = helpers.get_device(device_type)
    assert device is not None

    # Create texture and build random data
    (src_tex, dst_tex) = create_test_textures(device, 2, type, array_length, mips)
    rand_data = make_rand_data(src_tex.type, src_tex.array_length, src_tex.mip_count)

    # Write random data to texture
    for layer_idx, slice_data in enumerate(rand_data):
        for mip_idx, mip_data in enumerate(slice_data):
            src_tex.copy_from_numpy(mip_data, layer=layer_idx, mip=mip_idx)

    for mip in range(src_tex.mip_count):
        dims = len(rand_data[0][0].shape) - 1
        if array_length == 1:

            COPY_TEXTURE_SHADER = f"""
        [shader("compute")]
        [numthreads(1, 1, 1)]
        void copy_color(
            uint{dims} tid: SV_DispatchThreadID,
            Texture{dims}D<float4> src,
            RWTexture{dims}D<float4> dest
        )
        {{
            dest[tid] = src[tid];
        }}
        """
            module = device.load_module_from_source(
                module_name=f"test_shader_read_write_texture_{array_length}_{dims}",
                source=COPY_TEXTURE_SHADER,
            )
            copy_kernel = device.create_compute_kernel(
                device.link_program([module], [module.entry_point("copy_color")])
            )

            src_view = src_tex.create_view(
                {"subresource_range": spy.SubresourceRange({"mip": mip})}
            )
            assert src_view.subresource_range.layer == 0
            assert src_view.subresource_range.layer_count == 1
            assert src_view.subresource_range.mip == mip
            assert src_view.subresource_range.mip_count == src_tex.mip_count - mip

            dst_view = dst_tex.create_view(
                {"subresource_range": spy.SubresourceRange({"mip": mip})}
            )
            assert dst_view.subresource_range.layer == 0
            assert dst_view.subresource_range.layer_count == 1
            assert dst_view.subresource_range.mip == mip
            assert dst_view.subresource_range.mip_count == dst_tex.mip_count - mip

            copy_kernel.dispatch(
                [src_tex.width, src_tex.height, src_tex.depth],
                src=src_view,
                dest=dst_view,
            )
        else:

            COPY_TEXTURE_SHADER = f"""
        [shader("compute")]
        [numthreads(1, 1, 1)]
        void copy_color(
            uint{dims} tid: SV_DispatchThreadID,
            Texture{dims}DArray<float4> src,
            RWTexture{dims}DArray<float4> dest
        )
        {{
            uint{dims+1} idx = uint{dims+1}(tid, 0);
            dest[idx] = src[idx];
        }}
        """
            module = device.load_module_from_source(
                module_name=f"test_shader_read_write_texture_{array_length}_{dims}",
                source=COPY_TEXTURE_SHADER,
            )
            copy_kernel = device.create_compute_kernel(
                device.link_program([module], [module.entry_point("copy_color")])
            )
            for i in range(0, array_length):
                desc = spy.TextureViewDesc(
                    {
                        "subresource_range": spy.SubresourceRange(
                            {"layer": i, "layer_count": 1, "mip": mip}
                        )
                    }
                )

                srv = src_tex.create_view(desc)
                assert srv.subresource_range.layer == i
                assert srv.subresource_range.layer_count == 1
                assert srv.subresource_range.mip == mip
                assert srv.subresource_range.mip_count == src_tex.mip_count - mip

                uav = dst_tex.create_view(desc)
                assert uav.subresource_range.layer == i
                assert uav.subresource_range.layer_count == 1
                assert uav.subresource_range.mip == mip
                assert uav.subresource_range.mip_count == dst_tex.mip_count - mip

                copy_kernel.dispatch(
                    [src_tex.width, src_tex.height, src_tex.depth], src=srv, dest=uav
                )

    # Read back data and compare
    for layer_idx, slice_data in enumerate(rand_data):
        for mip_idx, mip_data in enumerate(slice_data):
            data = dst_tex.to_numpy(layer=layer_idx, mip=mip_idx)
            assert np.allclose(data, mip_data)


@pytest.mark.parametrize(
    "type",
    [
        spy.TextureType.texture_2d,
    ],
)
@pytest.mark.parametrize("array_length", [1, 4])
@pytest.mark.parametrize("mips", [1, 4])
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_copy_texture(
    device_type: spy.DeviceType, array_length: int, mips: int, type: spy.TextureType
):
    device = helpers.get_device(device_type)
    assert device is not None

    # Create texture and build random data
    (src, dst) = create_test_textures(device, 2, type, array_length, mips)
    rand_data = make_rand_data(src.type, src.array_length, src.mip_count)

    # Write random data to texture
    for layer_idx, layer_data in enumerate(rand_data):
        for mip_idx, mip_data in enumerate(layer_data):
            src.copy_from_numpy(mip_data, layer=layer_idx, mip=mip_idx)

    # Copy src to dst
    encoder = device.create_command_encoder()
    idx = 0
    for layer_idx in range(array_length):
        for mip_idx in range(mips):
            encoder.copy_texture(
                dst,
                layer_idx,
                mip_idx,
                spy.uint3(0),
                src,
                layer_idx,
                mip_idx,
                spy.uint3(0),
            )
            idx += 1
    device.submit_command_buffer(encoder.finish())

    # Read back data and compare
    for layer_idx, layer_data in enumerate(rand_data):
        for mip_idx, mip_data in enumerate(layer_data):
            data = dst.to_numpy(layer=layer_idx, mip=mip_idx)
            assert np.allclose(data, mip_data)


@pytest.mark.parametrize(
    "type",
    [
        spy.TextureType.texture_2d,
    ],
)
@pytest.mark.parametrize("array_length", [1, 4])
@pytest.mark.parametrize("mips", [1, 4])
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_copy_texture_to_buffer_and_back(
    device_type: spy.DeviceType, array_length: int, mips: int, type: spy.TextureType
):
    device = helpers.get_device(device_type)
    assert device is not None

    # Create texture and build random data
    (src, dst) = create_test_textures(device, 2, type, array_length, mips)
    rand_data = make_rand_data(src.type, src.array_length, src.mip_count)

    # Write random data to texture
    for layer_idx, layer_data in enumerate(rand_data):
        for mip_idx, mip_data in enumerate(layer_data):
            src.copy_from_numpy(mip_data, layer=layer_idx, mip=mip_idx)

    # Copy src to dst
    encoder = device.create_command_encoder()
    for layer_idx in range(array_length):
        for mip_idx in range(mips):
            layout = src.get_subresource_layout(mip_idx)
            buffer = device.create_buffer(
                size=layout.size_in_bytes,
                usage=spy.BufferUsage.copy_source | spy.BufferUsage.copy_destination,
            )
            encoder.copy_texture_to_buffer(
                buffer,
                0,
                buffer.size,
                layout.row_pitch,
                src,
                layer_idx,
                mip_idx,
                spy.uint3(0),
            )
            encoder.copy_buffer_to_texture(
                dst,
                layer_idx,
                mip_idx,
                spy.uint3(0),
                buffer,
                0,
                buffer.size,
                layout.row_pitch,
            )
    device.submit_command_buffer(encoder.finish())
    device.wait_for_idle()

    # Read back data and compare
    idx = 0
    for layer_idx, layer_data in enumerate(rand_data):
        for mip_idx, mip_data in enumerate(layer_data):
            data = dst.to_numpy(layer=layer_idx, mip=mip_idx)
            assert np.allclose(data, mip_data)
            idx += 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
