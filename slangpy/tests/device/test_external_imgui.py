# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest

import slangpy as spy
from slangpy.testing import helpers


imgui_bundle = pytest.importorskip("imgui_bundle")
imgui = imgui_bundle.imgui

from slangpy.ui.imgui_bundle import (
    begin_frame,
    create_imgui_context,
    handle_keyboard_event,
    handle_mouse_event,
    render_imgui_draw_data,
    sync_draw_data_textures,
)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_render_external_imgui_draw_data(device_type: spy.DeviceType):
    """Render imgui_bundle draw data through slangpy and verify visible output."""
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
    sync_draw_data_textures(device, ui_context, draw_data)

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


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_begin_frame_and_event_forwarding(device_type: spy.DeviceType):
    """Forward representative input events and confirm the render path still works."""
    device = helpers.get_device(device_type)
    if spy.Feature.rasterization not in device.features:
        pytest.skip("Device does not support rasterization")

    width, height = 160, 120
    target = device.create_texture(
        format=spy.Format.rgba8_unorm,
        width=width,
        height=height,
        usage=spy.TextureUsage.render_target | spy.TextureUsage.shader_resource,
        label="event_fwd_target",
    )

    ui_context = spy.ui.Context(device)
    external_ctx = create_imgui_context(width, height)
    imgui.set_current_context(external_ctx)

    # Mock event helpers — spy events are read-only C++ structs that cannot be
    # constructed with specific field values from Python, so we use duck-typed
    # stand-ins that satisfy the attribute/method interface used by the helpers.
    class _Vec2:
        def __init__(self, x: float = 0.0, y: float = 0.0):
            self.x, self.y = x, y

    class _KbEvent:
        def __init__(
            self,
            type: spy.KeyboardEventType,
            key: spy.KeyCode = spy.KeyCode.unknown,
            codepoint: int = 0,
        ):
            self.type = type
            self.key = key
            self.codepoint = codepoint

        def has_modifier(self, _: spy.KeyModifier) -> bool:
            return False

    class _MouseEvent:
        def __init__(
            self,
            type: spy.MouseEventType,
            pos: tuple[float, float] = (0.0, 0.0),
            button: spy.MouseButton = spy.MouseButton.left,
            scroll: tuple[float, float] = (0.0, 0.0),
        ):
            self.type = type
            self.pos = _Vec2(*pos)
            self.button = button
            self.scroll = _Vec2(*scroll)

        def has_modifier(self, _: spy.KeyModifier) -> bool:
            return False

    class _UnknownMouseButton:
        def __init__(self, value: int):
            self.value = value

    io = imgui.get_io()

    # Queue move and keyboard input first. In this binding, a queued mouse-button
    # press suppresses observable key-down state until a later frame.
    handle_mouse_event(_MouseEvent(spy.MouseEventType.move, pos=(50, 40)))
    kb_result = handle_keyboard_event(_KbEvent(spy.KeyboardEventType.key_press, key=spy.KeyCode.a))
    assert isinstance(kb_result, bool)
    handle_keyboard_event(_KbEvent(spy.KeyboardEventType.input, codepoint=ord("A")))

    # imgui_bundle applies queued input events during new_frame(), so validate the
    # forwarded state immediately after begin_frame() and before widget creation.
    begin_frame(width, height)
    assert (io.mouse_pos.x, io.mouse_pos.y) == (50.0, 40.0)
    assert isinstance(kb_result, bool)
    assert imgui.is_key_down(imgui.Key.a) is True
    imgui.render()

    # Queue button and scroll input for the next frame and validate the mouse state
    # before rendering any widgets.
    handle_mouse_event(_MouseEvent(spy.MouseEventType.button_down, button=spy.MouseButton.left))
    handle_mouse_event(_MouseEvent(spy.MouseEventType.button_down, button=_UnknownMouseButton(99)))
    handle_mouse_event(_MouseEvent(spy.MouseEventType.scroll, scroll=(0, 1)))
    begin_frame(width, height)
    assert bool(io.mouse_down[imgui.MouseButton_.left.value]) is True

    handle_mouse_event(_MouseEvent(spy.MouseEventType.button_up, button=spy.MouseButton.left))
    handle_keyboard_event(_KbEvent(spy.KeyboardEventType.key_release, key=spy.KeyCode.a))

    imgui.set_next_window_pos((4, 4))
    imgui.set_next_window_size((140, 90))
    imgui.begin("Event Test")
    imgui.text("Hello")
    imgui.end()
    imgui.render()

    draw_data = imgui.get_draw_data()
    sync_draw_data_textures(device, ui_context, draw_data)

    # Verify the rendering pipeline still works after input forwarding.
    encoder = device.create_command_encoder()
    encoder.clear_texture_uint(target, clear_value=spy.uint4(0, 0, 0, 255))
    render_imgui_draw_data(ui_context, draw_data, target, encoder)
    device.submit_command_buffer(encoder.finish())

    pixels = target.to_numpy().reshape((height, width, 4))
    assert np.sum(pixels[..., :3]) > 0


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_sync_draw_data_textures_reuses_and_releases(device_type: spy.DeviceType):
    """Reuse synchronized textures across frames and release them on destroy requests."""
    device = helpers.get_device(device_type)
    if spy.Feature.rasterization not in device.features:
        pytest.skip("Device does not support rasterization")

    width = 160
    height = 120

    ui_context = spy.ui.Context(device)
    external_ctx = create_imgui_context(width, height)
    imgui.set_current_context(external_ctx)

    imgui.new_frame()
    imgui.begin("Texture Lifecycle")
    imgui.text("Hello")
    imgui.end()
    imgui.render()

    draw_data = imgui.get_draw_data()
    draw_textures = list(draw_data.textures)
    assert len(draw_textures) == 1

    created_textures = sync_draw_data_textures(device, ui_context, draw_data)
    assert len(created_textures) == 1

    draw_texture = draw_textures[0]
    texture_id = draw_texture.get_tex_id()
    assert texture_id != 0
    assert ui_context.get_texture(texture_id) is not None

    created_textures = sync_draw_data_textures(device, ui_context, draw_data)
    assert created_textures == []
    assert draw_texture.get_tex_id() == texture_id
    assert ui_context.get_texture(texture_id) is not None

    class _FakeGpuTexture:
        def __init__(self):
            self.updated_pixels = None

        def copy_from_numpy(self, pixels: np.ndarray):
            self.updated_pixels = np.array(pixels, copy=True)

    class _FakeUiContext:
        def __init__(self, texture_id: int, texture: _FakeGpuTexture):
            self.texture_id_value = texture_id
            self.texture = texture

        def get_texture(self, candidate_texture_id: int):
            if candidate_texture_id == self.texture_id_value:
                return self.texture
            return None

        def release_texture(self, _: int) -> bool:
            return False

        def texture_id(self, _: object) -> int:
            return self.texture_id_value

    class _FakeDrawTexture:
        def __init__(self, texture_id: int):
            self.status = imgui.ImTextureStatus.want_updates
            self.height = 1
            self.width = 1
            self.bytes_per_pixel = 4
            self.unique_id = 123456
            self._texture_id = texture_id
            self.pixels = np.array([[[255, 0, 0, 255]]], dtype=np.uint8)

        def get_tex_id(self) -> int:
            return self._texture_id

        def get_pixels_array(self) -> np.ndarray:
            return self.pixels.reshape(-1)

        def set_status(self, status: imgui.ImTextureStatus) -> None:
            self.status = status

        def set_tex_id(self, texture_id: int) -> None:
            self._texture_id = texture_id

        def destroy_pixels(self) -> None:
            pass

    class _FakeDrawData:
        def __init__(self, texture: _FakeDrawTexture):
            self.textures = [texture]

    fake_texture = _FakeGpuTexture()
    fake_ui_context = _FakeUiContext(texture_id, fake_texture)
    fake_draw_texture = _FakeDrawTexture(texture_id)
    updated_textures = sync_draw_data_textures(
        device, fake_ui_context, _FakeDrawData(fake_draw_texture)
    )
    assert updated_textures == []
    assert fake_draw_texture.get_tex_id() == texture_id
    assert fake_ui_context.get_texture(texture_id) is not None
    assert fake_texture.updated_pixels is not None
    assert np.array_equal(
        fake_texture.updated_pixels[0, 0], np.array([255, 0, 0, 255], dtype=np.uint8)
    )
    assert fake_ui_context.release_texture(texture_id) is False

    draw_texture.set_status(imgui.ImTextureStatus.want_destroy)
    sync_draw_data_textures(device, ui_context, draw_data)

    assert draw_texture.get_tex_id() == 0
    assert ui_context.get_texture(texture_id) is None
    assert ui_context.release_texture(texture_id) is False

    class _MissingTextures:
        pass

    with pytest.raises(TypeError, match="draw_data must expose a 'textures' iterable"):
        sync_draw_data_textures(None, None, _MissingTextures())  # type: ignore[arg-type]

    class _BadTexture:
        status = imgui.ImTextureStatus.ok

    class _BadDrawData:
        textures = [_BadTexture()]

    with pytest.raises(
        TypeError, match="draw_data.textures elements must expose the imgui texture interface"
    ):
        sync_draw_data_textures(None, None, _BadDrawData())  # type: ignore[arg-type]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
