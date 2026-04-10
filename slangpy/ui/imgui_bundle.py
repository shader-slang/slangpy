# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Helpers for integrating ``imgui_bundle`` with slangpy's UI rendering backend."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    import slangpy as spy

# ---------------------------------------------------------------------------
# Internal key mapping (lazy-initialised on first use)
# ---------------------------------------------------------------------------

_KEY_MAP: Optional[Dict[int, Any]] = None
_MOUSE_BUTTON_MAP: Optional[Dict[int, int]] = None
_TEXTURES_BY_ID: Dict[int, Any] = {}
_TEXTURE_IDS_BY_PY_ID: Dict[int, int] = {}
_NEXT_TEXTURE_ID = 1


def _register_texture(texture: spy.Texture) -> int:
    global _NEXT_TEXTURE_ID

    texture_py_id = id(texture)
    existing = _TEXTURE_IDS_BY_PY_ID.get(texture_py_id)
    if existing is not None:
        return existing

    texture_id = _NEXT_TEXTURE_ID
    _NEXT_TEXTURE_ID += 1
    _TEXTURES_BY_ID[texture_id] = texture
    _TEXTURE_IDS_BY_PY_ID[texture_py_id] = texture_id
    return texture_id


def _resolve_texture(texture_id: int) -> Optional[spy.Texture]:
    return _TEXTURES_BY_ID.get(texture_id)


def _release_texture(texture_id: int) -> bool:
    texture = _TEXTURES_BY_ID.pop(texture_id, None)
    if texture is None:
        return False
    _TEXTURE_IDS_BY_PY_ID.pop(id(texture), None)
    return True


class _ResolvedDrawCommand:
    __slots__ = ("clip_rect", "elem_count", "idx_offset", "vtx_offset", "texture")

    def __init__(self, cmd: Any, texture: spy.Texture):
        self.clip_rect = cmd.clip_rect
        self.elem_count = cmd.elem_count
        self.idx_offset = cmd.idx_offset
        self.vtx_offset = cmd.vtx_offset
        self.texture = texture


class _ResolvedDrawList:
    __slots__ = ("cmd_buffer", "vtx_buffer", "idx_buffer")

    def __init__(self, cmd_list: Any, cmd_buffer: List[_ResolvedDrawCommand]):
        self.cmd_buffer = cmd_buffer
        self.vtx_buffer = cmd_list.vtx_buffer
        self.idx_buffer = cmd_list.idx_buffer


class _ResolvedDrawData:
    __slots__ = ("cmd_lists", "display_pos", "display_size", "framebuffer_scale")

    def __init__(self, draw_data: Any, cmd_lists: List[_ResolvedDrawList]):
        self.cmd_lists = cmd_lists
        self.display_pos = draw_data.display_pos
        self.display_size = draw_data.display_size
        self.framebuffer_scale = draw_data.framebuffer_scale


def _resolve_draw_data(draw_data: Any) -> _ResolvedDrawData:
    cmd_lists = []
    for list_index, cmd_list in enumerate(draw_data.cmd_lists):
        resolved_commands = []
        for cmd_index, cmd in enumerate(cmd_list.cmd_buffer):
            texture_id = cmd.get_tex_id()
            texture = _resolve_texture(texture_id)
            if texture is None:
                raise ValueError(
                    f"Unknown imgui texture ID {texture_id} in draw_data.cmd_lists[{list_index}].cmd_buffer[{cmd_index}]"
                )
            resolved_commands.append(_ResolvedDrawCommand(cmd, texture))
        cmd_lists.append(_ResolvedDrawList(cmd_list, resolved_commands))
    return _ResolvedDrawData(draw_data, cmd_lists)


def _get_key_map() -> Dict[int, Any]:
    """Build the slangpy-to-ImGui key mapping used by keyboard forwarding."""
    global _KEY_MAP
    if _KEY_MAP is not None:
        return _KEY_MAP

    import slangpy as spy

    from imgui_bundle import imgui

    K = spy.KeyCode
    IK = imgui.Key

    # Names that differ between spy.KeyCode and imgui.Key.
    _REMAP: Dict[str, str] = {
        "key0": "_0",
        "key1": "_1",
        "key2": "_2",
        "key3": "_3",
        "key4": "_4",
        "key5": "_5",
        "key6": "_6",
        "key7": "_7",
        "key8": "_8",
        "key9": "_9",
        "left": "left_arrow",
        "right": "right_arrow",
        "up": "up_arrow",
        "down": "down_arrow",
        "left_control": "left_ctrl",
        "right_control": "right_ctrl",
        "keypad_del": "keypad_decimal",
    }

    _KEY_MAP = {}
    for name, key in K.__members__.items():
        imgui_name = _REMAP.get(name, name)
        if imgui_name in IK.__members__:
            _KEY_MAP[key.value] = IK[imgui_name]

    return _KEY_MAP


def _get_mouse_button_map() -> Dict[int, int]:
    """Map slangpy MouseButton values to imgui mouse button indices."""
    global _MOUSE_BUTTON_MAP
    if _MOUSE_BUTTON_MAP is not None:
        return _MOUSE_BUTTON_MAP

    import slangpy as spy

    from imgui_bundle import imgui

    _MOUSE_BUTTON_MAP = {
        spy.MouseButton.left.value: imgui.MouseButton_.left.value,
        spy.MouseButton.middle.value: imgui.MouseButton_.middle.value,
        spy.MouseButton.right.value: imgui.MouseButton_.right.value,
    }
    return _MOUSE_BUTTON_MAP


# ---------------------------------------------------------------------------
# Public event-forwarding helpers
# ---------------------------------------------------------------------------


def handle_keyboard_event(event: spy.KeyboardEvent) -> bool:
    """
    Forward a slangpy keyboard event to the current imgui context.

    :param event: The keyboard event from ``AppWindow.on_keyboard_event``.
    :return: ``True`` if ImGui wants to capture keyboard input (caller should
             skip its own keyboard handling).
    """
    import slangpy as spy

    from imgui_bundle import imgui

    io = imgui.get_io()

    io.add_key_event(imgui.Key.mod_shift, event.has_modifier(spy.KeyModifier.shift))
    io.add_key_event(imgui.Key.mod_ctrl, event.has_modifier(spy.KeyModifier.ctrl))
    io.add_key_event(imgui.Key.mod_alt, event.has_modifier(spy.KeyModifier.alt))

    if (
        event.type == spy.KeyboardEventType.key_press
        or event.type == spy.KeyboardEventType.key_release
    ):
        imgui_key = _get_key_map().get(event.key.value, imgui.Key.none)
        io.add_key_event(imgui_key, event.type == spy.KeyboardEventType.key_press)
    elif event.type == spy.KeyboardEventType.input:
        io.add_input_characters_utf8(chr(event.codepoint))

    return io.want_capture_keyboard


def handle_mouse_event(event: spy.MouseEvent) -> bool:
    """
    Forward a slangpy mouse event to the current imgui context.

    :param event: The mouse event from ``AppWindow.on_mouse_event``.
    :return: ``True`` if ImGui wants to capture mouse input (caller should
             skip its own mouse handling).
    """
    import slangpy as spy

    from imgui_bundle import imgui

    io = imgui.get_io()

    io.add_key_event(imgui.Key.mod_shift, event.has_modifier(spy.KeyModifier.shift))
    io.add_key_event(imgui.Key.mod_ctrl, event.has_modifier(spy.KeyModifier.ctrl))
    io.add_key_event(imgui.Key.mod_alt, event.has_modifier(spy.KeyModifier.alt))

    if event.type == spy.MouseEventType.button_down or event.type == spy.MouseEventType.button_up:
        btn = _get_mouse_button_map().get(event.button.value)
        if btn is not None:
            io.add_mouse_button_event(btn, event.type == spy.MouseEventType.button_down)
    elif event.type == spy.MouseEventType.move:
        io.add_mouse_pos_event(event.pos.x, event.pos.y)
    elif event.type == spy.MouseEventType.scroll:
        io.add_mouse_wheel_event(event.scroll.x, event.scroll.y)

    return io.want_capture_mouse


def begin_frame(width: int, height: int, delta_time: float = 1.0 / 60.0) -> None:
    """
    Begin a new imgui_bundle frame with the given display dimensions.

    Call this once per frame *before* issuing any ImGui widget calls and
    *after* forwarding input events.

    :param width: Display width in pixels.
    :param height: Display height in pixels.
    :param delta_time: Time elapsed since the previous frame, in seconds.
    """
    from imgui_bundle import imgui

    io = imgui.get_io()
    io.display_size = imgui.ImVec2(width, height)
    io.delta_time = delta_time
    imgui.new_frame()


def texture_ref(texture: spy.Texture) -> Any:
    """
    Create an ``imgui.ImTextureRef`` for a slangpy texture using helper-managed IDs.

    :param texture: Texture to expose to the external Dear ImGui binding.
    :return: ``imgui.ImTextureRef`` wrapping a helper-managed texture ID.
    """
    from imgui_bundle import imgui

    return imgui.ImTextureRef(_register_texture(texture))


# ---------------------------------------------------------------------------
# Texture and rendering helpers
# ---------------------------------------------------------------------------


def sync_draw_data_textures(
    device: spy.Device,
    ui_context: spy.ui.Context,
    draw_data: Any,
) -> List[spy.Texture]:
    """
    Upload font/image atlas textures referenced in *draw_data* to the GPU.

    Processes external Dear ImGui texture requests and keeps the helper-managed
    texture registry in sync with ``draw_data.textures``.

    New textures are uploaded once, incremental updates reuse the existing GPU
    texture, and ``want_destroy`` entries release the corresponding helper-side
    registration.

    :param device: GPU device used to create texture resources.
    :param ui_context: Unused compatibility parameter retained for existing callers.
    :param draw_data: The ``imgui.DrawData`` object returned by ``imgui.get_draw_data()``.
    :return: A list of textures created during this call.
    """
    import slangpy as spy

    from imgui_bundle import imgui

    del ui_context

    if not hasattr(draw_data, "textures"):
        raise TypeError("draw_data must expose a 'textures' iterable")

    draw_textures = draw_data.textures
    if not isinstance(draw_textures, Iterable):
        raise TypeError("draw_data.textures must be iterable")

    required_texture_attrs = (
        "get_tex_id",
        "status",
        "get_pixels_array",
        "height",
        "width",
        "bytes_per_pixel",
        "set_status",
        "set_tex_id",
        "destroy_pixels",
    )
    draw_textures = list(draw_textures)
    for i, tex in enumerate(draw_textures):
        missing = [name for name in required_texture_attrs if not hasattr(tex, name)]
        if missing:
            raise TypeError(
                "draw_data.textures elements must expose the imgui texture interface; "
                f"draw_data.textures[{i}] is missing required attributes: {', '.join(missing)}"
            )

    _BPP_TO_FORMAT = {
        1: spy.Format.r8_unorm,
        2: spy.Format.rg8_unorm,
        # 3 (RGB) is not supported; GPUs lack a native RGB8_UNORM texture format.
        4: spy.Format.rgba8_unorm,
    }

    font_tex = imgui.get_io().fonts.tex_data
    textures: List[spy.Texture] = []
    for idx, tex in enumerate(draw_textures):
        status = tex.status
        texture_id = tex.get_tex_id()

        if status == imgui.ImTextureStatus.ok:
            continue

        if status == imgui.ImTextureStatus.want_destroy:
            if texture_id:
                _release_texture(texture_id)
                tex.set_tex_id(0)
            tex.set_status(imgui.ImTextureStatus.destroyed)
            continue

        if status == imgui.ImTextureStatus.want_updates and texture_id:
            texture = _resolve_texture(texture_id)
            if texture is not None:
                bpp = tex.bytes_per_pixel
                if bpp not in _BPP_TO_FORMAT:
                    raise ValueError(
                        f"Unsupported bytes_per_pixel={bpp} for texture update at index {idx}"
                    )
                pixels = tex.get_pixels_array().reshape((tex.height, tex.width, bpp))
                texture.copy_from_numpy(pixels)
                tex.set_status(imgui.ImTextureStatus.ok)
                if font_tex is None or tex.unique_id != font_tex.unique_id:
                    tex.destroy_pixels()
                continue

        if status == imgui.ImTextureStatus.destroyed:
            continue

        bpp = tex.bytes_per_pixel
        if bpp not in _BPP_TO_FORMAT:
            raise ValueError(f"Unsupported bytes_per_pixel={bpp} for texture at index {idx}")
        pixels = tex.get_pixels_array().reshape((tex.height, tex.width, bpp))
        texture = device.create_texture(
            format=_BPP_TO_FORMAT[bpp],
            width=tex.width,
            height=tex.height,
            usage=spy.TextureUsage.shader_resource,
            data=pixels,
            label=f"imgui_bundle_texture_{idx}",
        )
        tex.set_tex_id(_register_texture(texture))
        tex.set_status(imgui.ImTextureStatus.ok)
        if font_tex is None or tex.unique_id != font_tex.unique_id:
            tex.destroy_pixels()
        textures.append(texture)
    return textures


def create_imgui_context(width: int, height: int) -> Any:
    """
    Create a standalone ``imgui_bundle`` context configured for offscreen rendering.

    :param width: Display width in pixels.
    :param height: Display height in pixels.
    :return: The newly created ImGui context (already set as current).
    """
    from imgui_bundle import imgui

    ctx = imgui.create_context()
    imgui.set_current_context(ctx)

    io = imgui.get_io()
    io.display_size = imgui.ImVec2(width, height)
    io.delta_time = 1.0 / 60.0
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures

    io.fonts.add_font_default()
    if io.fonts.tex_data is not None:
        # Accessing the pixel array forces Dear ImGui to build the font atlas.
        io.fonts.tex_data.get_pixels_array()
    return ctx


def render_imgui_draw_data(
    context: spy.ui.Context,
    draw_data: Any,
    texture: Union[spy.Texture, spy.TextureView],
    command_encoder: spy.CommandEncoder,
) -> None:
    """
    Render draw data produced by an external Dear ImGui Python binding.

    This helper currently supports bindings that expose draw-list buffers with
    ``data_address()`` methods, such as ``imgui_bundle``.

    :param context: The slangpy UI context used to access the shared renderer.
    :param draw_data: The ``imgui.DrawData`` object returned by ``imgui.get_draw_data()``.
    :param texture: Texture or texture view to render to.
    :param command_encoder: Command encoder to encode commands to.
    """
    resolved_draw_data = _resolve_draw_data(draw_data)
    context._render_marshaled_draw_data(resolved_draw_data, texture, command_encoder)
