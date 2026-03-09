# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Helpers for integrating ``imgui_bundle`` with slangpy's UI rendering backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Union

if TYPE_CHECKING:
    import slangpy as spy


def sync_draw_data_textures(
    device: spy.Device,
    ui_context: spy.ui.Context,
    draw_data: Any,
) -> List[spy.Texture]:
    """
    Upload font/image atlas textures referenced in *draw_data* to the GPU.

    Each texture in ``draw_data.textures`` is uploaded as an RGBA8 texture and
    registered with *ui_context* so that the renderer can resolve ``ImTextureID``
    values during rendering.

    :param device: GPU device used to create texture resources.
    :param ui_context: The slangpy UI context that owns texture ID mappings.
    :param draw_data: The ``imgui.DrawData`` object returned by ``imgui.get_draw_data()``.
    :return: A list of the created GPU textures (kept alive for the caller).
    """
    import slangpy as spy

    from imgui_bundle import imgui

    textures: List[spy.Texture] = []
    for idx, tex in enumerate(draw_data.textures):
        pixels = tex.get_pixels_array().reshape((tex.height, tex.width, tex.bytes_per_pixel))
        texture = device.create_texture(
            format=spy.Format.rgba8_unorm,
            width=tex.width,
            height=tex.height,
            usage=spy.TextureUsage.shader_resource,
            data=pixels,
            label=f"imgui_bundle_texture_{idx}",
        )
        tex.set_tex_id(ui_context.texture_id(texture))
        tex.set_status(imgui.ImTextureStatus.ok)
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

    :param context: The slangpy UI context that owns texture ID mappings.
    :param draw_data: The ``imgui.DrawData`` object returned by ``imgui.get_draw_data()``.
    :param texture: Texture or texture view to render to.
    :param command_encoder: Command encoder to encode commands to.
    """

    context._render_marshaled_draw_data(draw_data, texture, command_encoder)
