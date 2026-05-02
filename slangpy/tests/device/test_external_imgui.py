# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest

import slangpy as spy
import slangpy.ui.imgui_bundle as imgui_bundle_helpers
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
    texture_ref,
)


_FAKE_DRAW_VERT_DTYPE = np.dtype(
    [
        ("pos", np.float32, 2),
        ("uv", np.float32, 2),
        ("col", np.uint32),
    ]
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
        format=spy.Format.rgba8_unorm_srgb,
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
    imgui.image(texture_ref(checker_texture), (24, 24))
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

    # Mock event helpers - spy events are read-only C++ structs that cannot be
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

    created_textures = sync_draw_data_textures(device, ui_context, draw_data)
    assert created_textures == []
    assert draw_texture.get_tex_id() == texture_id

    class _FakeGpuTexture:
        def __init__(self):
            self.updated_pixels = None

        def copy_from_numpy(self, pixels: np.ndarray):
            self.updated_pixels = np.array(pixels, copy=True)

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
    fake_texture_id = imgui_bundle_helpers._register_texture(fake_texture)  # type: ignore[arg-type]
    fake_draw_texture = _FakeDrawTexture(fake_texture_id)
    updated_textures = sync_draw_data_textures(device, ui_context, _FakeDrawData(fake_draw_texture))
    assert updated_textures == []
    assert fake_draw_texture.get_tex_id() == fake_texture_id
    assert fake_texture.updated_pixels is not None
    assert np.array_equal(
        fake_texture.updated_pixels[0, 0], np.array([255, 0, 0, 255], dtype=np.uint8)
    )

    draw_texture.set_status(imgui.ImTextureStatus.want_destroy)
    sync_draw_data_textures(device, ui_context, draw_data)

    assert draw_texture.get_tex_id() == 0

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


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_render_external_imgui_draw_data_rejects_invalid_indices(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    if spy.Feature.rasterization not in device.features:
        pytest.skip("Device does not support rasterization")

    ui_context = spy.ui.Context(device)
    target = device.create_texture(
        format=spy.Format.rgba8_unorm,
        width=8,
        height=8,
        usage=spy.TextureUsage.render_target | spy.TextureUsage.shader_resource,
        label="invalid_external_imgui_target",
    )
    texture = device.create_texture(
        format=spy.Format.rgba8_unorm,
        width=1,
        height=1,
        usage=spy.TextureUsage.shader_resource,
        data=np.array([[[255, 255, 255, 255]]], dtype=np.uint8),
        label="invalid_external_imgui_texture",
    )

    class _Vec2:
        def __init__(self, x: float, y: float):
            self.x = x
            self.y = y

    class _ClipRect:
        def __init__(self, x: float, y: float, z: float, w: float):
            self.x = x
            self.y = y
            self.z = z
            self.w = w

    class _Buffer:
        def __init__(self, data: np.ndarray):
            self._data = np.ascontiguousarray(data)

        def size(self) -> int:
            return len(self._data)

        def data_address(self) -> int:
            return self._data.ctypes.data

    class _Command:
        def __init__(self, texture: spy.Texture):
            self.clip_rect = _ClipRect(0.0, 0.0, 8.0, 8.0)
            self.elem_count = 3
            self.idx_offset = 0
            self.vtx_offset = 0
            self.texture = texture

    class _CmdList:
        def __init__(self, texture: spy.Texture):
            vertices = np.zeros(1, dtype=_FAKE_DRAW_VERT_DTYPE)
            indices = np.array([0, 1, 0], dtype=np.uint32)
            self.vtx_buffer = _Buffer(vertices)
            self.idx_buffer = _Buffer(indices)
            self.cmd_buffer = [_Command(texture)]

    class _DrawData:
        def __init__(self, texture: spy.Texture):
            self.cmd_lists = [_CmdList(texture)]
            self.display_pos = _Vec2(0.0, 0.0)
            self.display_size = _Vec2(8.0, 8.0)
            self.framebuffer_scale = _Vec2(1.0, 1.0)

    encoder = device.create_command_encoder()
    with pytest.raises(RuntimeError, match="references out-of-bounds vertex"):
        ui_context.render_draw_data(_DrawData(texture), target, encoder)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
