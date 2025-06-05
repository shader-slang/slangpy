# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import sys
import slangpy as spy
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
import sglhelpers as helpers


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
