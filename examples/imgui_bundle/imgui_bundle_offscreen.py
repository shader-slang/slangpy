# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import slangpy as spy
from slangpy.ui.imgui_bundle import create_imgui_context, render_imgui_draw_data, sync_draw_data_textures
from imgui_bundle import imgui


OUTPUT_PATH = Path(__file__).with_name("imgui_bundle_offscreen.png")


def create_checker_texture(device: spy.Device) -> spy.Texture:
    size = 96
    cell = 12
    y, x = np.mgrid[:size, :size]
    on = ((x // cell) + (y // cell)) % 2 == 0
    data = np.where(on[..., None], [242, 184, 82, 255], [24, 30, 39, 255]).astype(np.uint8)

    return device.create_texture(
        format=spy.Format.rgba8_unorm,
        width=size,
        height=size,
        usage=spy.TextureUsage.shader_resource,
        data=data,
        label="checker_texture",
    )


def build_ui(preview_texture_id: int):
    imgui.set_next_window_pos((12, 12))
    imgui.set_next_window_size((700, 500))

    imgui.begin("imgui_bundle + slangpy")
    imgui.text("External Dear ImGui draw data rendered by slangpy")
    imgui.separator()

    if imgui.begin_tab_bar("main_tabs"):
        if imgui.begin_tab_item_simple("Table"):
            if imgui.begin_table("stats", 3, imgui.TableFlags_.borders | imgui.TableFlags_.row_bg):
                imgui.table_setup_column("Metric")
                imgui.table_setup_column("Value")
                imgui.table_setup_column("Notes")
                imgui.table_headers_row()
                rows = [
                    ("Backend", "imgui_bundle", "External binding"),
                    ("Renderer", "slangpy", "Native SGL backend"),
                    ("Mode", "Offscreen", "Saved to PNG"),
                ]
                for metric, value, notes in rows:
                    imgui.table_next_row()
                    imgui.table_next_column()
                    imgui.text(metric)
                    imgui.table_next_column()
                    imgui.text(value)
                    imgui.table_next_column()
                    imgui.text(notes)
                imgui.end_table()
            imgui.end_tab_item()

        if imgui.begin_tab_item_simple("Widgets"):
            values = np.sin(np.linspace(0.0, 4.0 * np.pi, 64)).astype(np.float32)
            imgui.text("PlotLines and child regions work without using spy.ui widgets.")
            imgui.plot_lines("Signal", values, graph_size=(0, 90))
            imgui.begin_child("log", (0, 120), imgui.ChildFlags_.borders)
            for idx in range(8):
                imgui.bullet_text(f"Log line {idx}: draw-data path stays in Python until render.")
            imgui.end_child()
            imgui.end_tab_item()

        if imgui.begin_tab_item_simple("Image"):
            imgui.text("SGL texture shown through ImTextureID")
            imgui.image(imgui.ImTextureRef(preview_texture_id), (192, 192))
            imgui.end_tab_item()

        imgui.end_tab_bar()

    imgui.end()


def main():
    width = 768
    height = 540

    device = spy.Device(enable_debug_layers=True)
    ui_context = spy.ui.Context(device)
    external_ctx = create_imgui_context(width, height)
    checker_texture = create_checker_texture(device)

    render_target = device.create_texture(
        format=spy.Format.rgba8_unorm,
        width=width,
        height=height,
        usage=spy.TextureUsage.render_target | spy.TextureUsage.shader_resource,
        label="imgui_bundle_offscreen_target",
    )

    imgui.set_current_context(external_ctx)
    imgui.new_frame()
    build_ui(ui_context.texture_id(checker_texture))
    imgui.render()
    draw_data = imgui.get_draw_data()
    synced_textures = sync_draw_data_textures(device, ui_context, draw_data)

    encoder = device.create_command_encoder()
    encoder.clear_texture_uint(render_target, clear_value=spy.uint4(16, 20, 26, 255))
    render_imgui_draw_data(ui_context, draw_data, render_target, encoder)
    device.submit_command_buffer(encoder.finish())

    render_target.to_bitmap().write_async(str(OUTPUT_PATH))
    device.wait()
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
