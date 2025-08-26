# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import numpy as np

import slangpy as spy
from slangpy.testing import helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_bindless_texture(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    if not device.has_feature(spy.Feature.bindless):
        pytest.skip("Bindless not supported on this device.")

    module = device.load_module("test_bindless_texture.slang")
    program = device.link_program(
        modules=[module], entry_points=[module.entry_point("compute_main")]
    )
    kernel = device.create_compute_kernel(program)

    TEXTURE_COUNT = 8

    # create linear and point samplers
    sampler_linear = device.create_sampler()
    sampler_point = device.create_sampler(
        min_filter=spy.TextureFilteringMode.point,
        mag_filter=spy.TextureFilteringMode.point,
    )

    # create some 2x1 textures
    textures = []
    texture_views = []
    for i in range(TEXTURE_COUNT):
        texture = device.create_texture(
            width=2,
            height=1,
            format=spy.Format.r32_float,
            usage=spy.TextureUsage.shader_resource,
            data=np.array([i, i + 1], dtype=np.float32),
        )
        textures.append(texture)
        texture_views.append(texture.create_view())

    texture_info_layout = module.layout.get_type_layout(
        module.layout.find_type_by_name("StructuredBuffer<TextureInfo>")
    ).element_type_layout

    texture_infos_buffer = device.create_buffer(
        size=TEXTURE_COUNT * texture_info_layout.stride,
        usage=spy.BufferUsage.shader_resource,
    )

    results_buffer = device.create_buffer(
        size=TEXTURE_COUNT * 4,
        usage=spy.BufferUsage.unordered_access,
    )

    # fill texture infos for accessing bindless textures & samplers
    # even textures are sampled with linear sampler, odd textures with point sampler
    c = spy.BufferCursor(texture_info_layout, texture_infos_buffer, load_before_write=False)
    for i in range(TEXTURE_COUNT):
        c[i].texture = texture_views[i].descriptor_handle_ro
        c[i].sampler = [sampler_linear, sampler_point][i % 2].descriptor_handle
        c[i].uv = spy.float2(0.5)
    c.apply()

    kernel.dispatch(
        thread_count=[TEXTURE_COUNT, 1, 1],
        texture_infos=texture_infos_buffer,
        results=results_buffer,
    )

    # read back results
    results = results_buffer.to_numpy().view(np.float32)
    assert np.allclose(results, [0.5, 2.0, 2.5, 4.0, 4.5, 6.0, 6.5, 8.0])


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_bindless_buffer(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    if not device.has_feature(spy.Feature.bindless):
        pytest.skip("Bindless not supported on this device.")

    module = device.load_module("test_bindless_buffer.slang")
    program = device.link_program(
        modules=[module], entry_points=[module.entry_point("compute_main")]
    )
    kernel = device.create_compute_kernel(program)

    BUFFER_COUNT = 6

    # create some read-only buffers with different data
    ro_buffers = []
    for i in range(BUFFER_COUNT):
        buffer = device.create_buffer(
            size=4 * 4,  # 4 floats
            usage=spy.BufferUsage.shader_resource,
            data=np.array([i * 10, i * 10 + 1, i * 10 + 2, i * 10 + 3], dtype=np.float32),
        )
        ro_buffers.append(buffer)

    # create some read-write buffers (initially zeros)
    rw_buffers = []
    for i in range(BUFFER_COUNT):
        buffer = device.create_buffer(
            size=4 * 4,  # 4 floats
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            data=np.zeros(4, dtype=np.float32),
        )
        rw_buffers.append(buffer)

    buffer_info_layout = module.layout.get_type_layout(
        module.layout.find_type_by_name("StructuredBuffer<BufferInfo>")
    ).element_type_layout

    buffer_infos_buffer = device.create_buffer(
        size=BUFFER_COUNT * buffer_info_layout.stride,
        usage=spy.BufferUsage.shader_resource,
    )

    results_buffer = device.create_buffer(
        size=BUFFER_COUNT * 4,
        usage=spy.BufferUsage.unordered_access,
    )

    # fill buffer infos for accessing bindless buffers
    c = spy.BufferCursor(buffer_info_layout, buffer_infos_buffer, load_before_write=False)
    for i in range(BUFFER_COUNT):
        c[i].ro_buffer = ro_buffers[i].descriptor_handle_ro
        c[i].rw_buffer = rw_buffers[i].descriptor_handle_rw
        c[i].offset = i % 4  # access different elements in each buffer
    c.apply()

    kernel.dispatch(
        thread_count=[BUFFER_COUNT, 1, 1],
        buffer_infos=buffer_infos_buffer,
        results=results_buffer,
    )

    # read back results from the results buffer
    results = results_buffer.to_numpy().view(np.float32)
    expected_results = np.array(
        [
            0,  # buffer 0, offset 0: 0 * 10 + 0 = 0
            11,  # buffer 1, offset 1: 1 * 10 + 1 = 11
            22,  # buffer 2, offset 2: 2 * 10 + 2 = 22
            33,  # buffer 3, offset 3: 3 * 10 + 3 = 33
            40,  # buffer 4, offset 0: 4 * 10 + 0 = 40
            51,  # buffer 5, offset 1: 5 * 10 + 1 = 51
        ],
        dtype=np.float32,
    )
    assert np.allclose(results, expected_results)

    # read back and verify the RW buffers were written to correctly
    for i in range(BUFFER_COUNT):
        rw_data = rw_buffers[i].to_numpy().view(np.float32)
        offset = i % 4
        expected_value = (i * 10 + offset) + 100.0  # original value + 100
        assert np.isclose(
            rw_data[offset], expected_value
        ), f"RW buffer {i} at offset {offset}: expected {expected_value}, got {rw_data[offset]}"
        # Other elements should still be zero
        for j in range(4):
            if j != offset:
                assert np.isclose(
                    rw_data[j], 0.0
                ), f"RW buffer {i} at offset {j}: expected 0.0, got {rw_data[j]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
