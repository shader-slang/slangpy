# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
import argparse
import numpy as np
import slangpy as spy
from pathlib import Path

EXAMPLE_DIR = Path(__file__).parent

# Build and parse command line
# fmt: off
parser = argparse.ArgumentParser(description="Slang-based BC7 - mode 6 compressor")
parser.add_argument("input_path", help="Path to the input texture.")
parser.add_argument("-o", "--output_path", help="Optional path to save the decoded BC7 texture.")
parser.add_argument("-s", "--opt_steps", type=int, default=100, help="Number of optimization (gradient descene) steps.")
parser.add_argument("-b", "--benchmark", action="store_true", help="Run in benchmark mode to measure processing time.")
parser.add_argument("-t", "--tev", action="store_true", help="Show images in tev image viewer.")
parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")
# fmt: on
# args = parser.parse_args()
args = parser.parse_args(
    [
        str(EXAMPLE_DIR.parent.parent / "data/test_images/monalisa.jpg"),
        "-o",
        "monalisa_bc7.jpg",
        "-t",
        "-b",
        "-v",
    ]
)

# Load input image
try:
    image = spy.Bitmap(args.input_path).convert(
        pixel_format=spy.Bitmap.PixelFormat.rgba,
        component_type=spy.Bitmap.ComponentType.float32,
        srgb_gamma=False,
    )
    w, h = image.width, image.height
    input = np.asarray(image)
except Exception as e:
    print(f"Failed to load input image from {args.input_path}:\n{e}")
    sys.exit(1)

# Create device
device = spy.Device(
    enable_debug_layers=args.verbose,
    compiler_options={"include_paths": [EXAMPLE_DIR]},
)

# Create input texture
input_tex = device.create_texture(
    format=spy.Format.rgba32_float,
    width=w,
    height=h,
    usage=spy.TextureUsage.shader_resource,
    data=input,
)

# Show input texture in tev
if args.tev:
    spy.tev.show_async(input_tex, name="tinybc-input")

# Create decoded texture
decoded_tex = device.create_texture(
    format=spy.Format.rgba32_float,
    width=w,
    height=h,
    usage=spy.TextureUsage.unordered_access,
)

# Load shader module
constants = f"export static const bool USE_ADAM = true;\nexport static const uint OPT_STEPS = {args.opt_steps};\n"
program = device.load_program("tinybc.slang", ["compute_main"], constants)
kernel = device.create_compute_kernel(program)

t = spy.Timer()

# When running in benchmark mode amortize overheads over many runs to measure more accurate GPU times
num_iters = 1000 if args.benchmark else 1

# Setup query pool to measure GPU time
queries = device.create_query_pool(spy.QueryType.timestamp, num_iters * 2)

# Compress!
command_encoder = device.create_command_encoder()
for i in range(num_iters):
    command_encoder.write_timestamp(queries, i * 2)
    kernel.dispatch(
        thread_count=[w, h, 1],
        vars={
            "input_tex": input_tex,
            "decoded_tex": decoded_tex,
            "lr": 0.1,
            "adam_beta_1": 0.9,
            "adam_beta_2": 0.999,
        },
        command_encoder=command_encoder,
    )
    command_encoder.write_timestamp(queries, i * 2 + 1)
device.submit_command_buffer(command_encoder.finish())

# Wait for GPU to finish and get timestamps
device.wait()

total_cpu_time_sec = t.elapsed_s()

times = np.asarray(queries.get_timestamp_results(0, num_iters * 2))
comp_time_sec = np.mean(times[1::2] - times[0::2])

# Calculate and print performance metrics
if args.benchmark:
    textures_per_sec = 1 / comp_time_sec
    giga_texels_per_sec = w * h * textures_per_sec / 1e9
    print(f"Benchmark:")
    print(f"- Number of optimization steps: {args.opt_steps}")
    print(f"- Compression time: {1e3 * comp_time_sec:.4g} ms")
    print(f"- Compression throughput: {giga_texels_per_sec:.4g} GTexels/s")
    print(f"- Total CPU time: {total_cpu_time_sec:.4g} s")

# Calculate and print PSNR
decoded = decoded_tex.to_numpy()
mse = np.mean((input - decoded) ** 2)
psnr = 20 * np.log10(1.0 / np.sqrt(mse))
print(f"PSNR: {psnr:.4g}")

# Show decoded texture in tev
if args.tev:
    spy.tev.show_async(decoded_tex, name="tinybc-decoded")

# Output decoded texture
if args.output_path:
    decoded_tex.to_bitmap().convert(
        pixel_format=spy.Bitmap.PixelFormat.rgb,
        component_type=spy.Bitmap.ComponentType.uint8,
        srgb_gamma=True,
    ).write_async(args.output_path)
