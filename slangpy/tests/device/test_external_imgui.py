# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest

import slangpy as spy
from slangpy.testing import helpers


imgui_bundle = pytest.importorskip("imgui_bundle")
imgui = imgui_bundle.imgui

from slangpy.ui.imgui_bundle import create_imgui_context, render_imgui_draw_data, sync_draw_data_textures


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_render_external_imgui_draw_data(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    if spy.Feature.rasterization not in device.features:
        pytest.skip("Device does not support rasterization")

    width = 160
    height = 120
    target = device.create_texture(
        format=spy.Format.rgba8_unorm,
        width=width,
        height=height,
        usage=spy.TextureUsage.render_target | spy.TextureUsage.shader_resource,
        label="external_imgui_target",
    )

    ui_context = spy.ui.Context(device)
    external_ctx = create_imgui_context(width, height)
    imgui.set_current_context(external_ctx)
    checker = np.zeros((24, 24, 4), dtype=np.uint8)
    checker[:] = (240, 30, 20, 255)
    checker_texture = device.create_texture(
        format=spy.Format.rgba8_unorm,
        width=checker.shape[1],
        height=checker.shape[0],
        usage=spy.TextureUsage.shader_resource,
        data=checker,
        label="external_imgui_checker_texture",
    )

    imgui.new_frame()
    imgui.set_next_window_pos((4, 4))
    imgui.set_next_window_size((140, 90))
    imgui.begin("external imgui")
    imgui.text("Hello from imgui_bundle")
    imgui.image(imgui.ImTextureRef(ui_context.texture_id(checker_texture)), (24, 24))
    imgui.end()
    imgui.render()
    draw_data = imgui.get_draw_data()
    synced_textures = sync_draw_data_textures(device, ui_context, draw_data)

    encoder = device.create_command_encoder()
    encoder.clear_texture_uint(target, clear_value=spy.uint4(0, 0, 0, 255))
    render_imgui_draw_data(ui_context, draw_data, target, encoder)
    device.submit_command_buffer(encoder.finish())

    pixels = target.to_numpy().reshape((height, width, 4))

    image_region = pixels[48:72, 12:36]
    assert np.mean(image_region[..., 0]) > 180
    assert np.mean(image_region[..., 1]) < 80
    assert np.mean(image_region[..., 2]) < 80

    assert np.sum(pixels[..., :3]) > 100000
    assert np.max(pixels[18:42, 12:120, :3]) > 180


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
