# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import slangpy as spy
from pathlib import Path

EXAMPLE_DIR = Path(__file__).parent

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
COUNT = 100

device = spy.Device(compiler_options={"include_paths": [EXAMPLE_DIR]})

tex = device.create_texture(
    format=spy.Format.rgba32_float,
    width=IMAGE_WIDTH,
    height=IMAGE_HEIGHT,
    usage=spy.TextureUsage.unordered_access,
)

program = device.load_program("checkerboard.slang", ["compute_main"])
kernel = device.create_compute_kernel(program)

for i in range(COUNT):
    kernel.dispatch(
        thread_count=[tex.width, tex.height, 1],
        vars={"g_texture": tex, "g_checker_size": i + 1},
    )
    spy.tev.show_async(tex, f"test_{i:04d}")
