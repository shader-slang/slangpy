# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import slangpy as spy
import numpy as np
from pathlib import Path

EXAMPLE_DIR = Path(__file__).parent

device = spy.Device(
    enable_debug_layers=True,
    compiler_options={"include_paths": [EXAMPLE_DIR]},
)

vertices = np.array([-1, -1, 1, -1, 0, 1], dtype=np.float32)
indices = np.array([0, 1, 2], dtype=np.uint32)

vertex_buffer = device.create_buffer(
    usage=spy.BufferUsage.vertex_buffer | spy.BufferUsage.shader_resource,
    label="vertex_buffer",
    data=vertices,
)

index_buffer = device.create_buffer(
    usage=spy.BufferUsage.index_buffer | spy.BufferUsage.shader_resource,
    label="index_buffer",
    data=indices,
)

render_texture = device.create_texture(
    format=spy.Format.rgba32_float,
    width=1024,
    height=1024,
    usage=spy.TextureUsage.render_target,
    label="render_texture",
)

input_layout = device.create_input_layout(
    input_elements=[
        {
            "semantic_name": "POSITION",
            "semantic_index": 0,
            "format": spy.Format.rg32_float,
        }
    ],
    vertex_streams=[{"stride": 8}],
)

program = device.load_program("render_pipeline.slang", ["vertex_main", "fragment_main"])
pipeline = device.create_render_pipeline(
    program=program,
    input_layout=input_layout,
    targets=[{"format": spy.Format.rgba32_float}],
)

command_encoder = device.create_command_encoder()
with command_encoder.begin_render_pass(
    {"color_attachments": [{"view": render_texture.create_view({})}]}
) as pass_encoder:
    pass_encoder.bind_pipeline(pipeline)
    pass_encoder.set_render_state(
        {
            "viewports": [spy.Viewport.from_size(render_texture.width, render_texture.height)],
            "scissor_rects": [
                spy.ScissorRect.from_size(render_texture.width, render_texture.height)
            ],
            "vertex_buffers": [vertex_buffer],
            "index_buffer": index_buffer,
            "index_format": spy.IndexFormat.uint32,
        }
    )

    pass_encoder.draw({"vertex_count": 3})
device.submit_command_buffer(command_encoder.finish())

spy.tev.show(render_texture, "render_pipeline")
