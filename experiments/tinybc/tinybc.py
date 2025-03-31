# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import sys
import sgl
import time
import pathlib
import argparse
import numpy as np
import slangpy as spy

from slangpy.experimental.gridarg import grid

# Get local directory
local_dir = pathlib.Path(__file__).parent.absolute()

# Create a slangpy device with the local folder for slang includes
device = spy.create_device(include_paths=[local_dir, ])

# Build and parse command line
parser = argparse.ArgumentParser(description="Slang-based BC7 - mode 6 compressor")
parser.add_argument("-i", "--input_path", help="Path to the input texture.",
                    default=local_dir/"sample.jpg")
parser.add_argument("-o", "--output_path", help="Optional path to save the decoded BC7 texture.")
parser.add_argument("-s", "--opt_steps", type=int, default=100,
                    help="Number of optimization (gradient descene) steps.")
parser.add_argument("-b", "--benchmark", action="store_true",
                    help="Run in benchmark mode to measure processing time.")
args = parser.parse_args()

# Load texture
try:
    loader = sgl.TextureLoader(device)
    input_tex = loader.load_texture(args.input_path, options={"load_as_normalized": True,
                                                              "load_as_srgb": False,
                                                              "extend_alpha": True, })
    w, h = input_tex.width, input_tex.height
    print(f"\nTexture dimensions: {w}x{h}")
except Exception as e:
    print(f"\nError loading the texture: {e}")
    sys.exit(1)

# Create output texture
decoded_tex = device.create_texture(
    width=w, height=h, mip_count=1,
    format=sgl.Format.rgba32_float,
    usage=sgl.TextureUsage.unordered_access)

# Load module and setup encoder kernel
encoder_fn = spy.Module.load_from_file(device, "tinybc.slang").encoder
encoder_fn = encoder_fn.constants({"kUseAdam": True, "kNumOptimizationSteps": args.opt_steps})
encoder_fn = encoder_fn.set(
    gInputTex=input_tex,
    gDecodedTex=decoded_tex,
    lr=0.1,
    adamBeta1=0.9,
    adamBeta2=0.999,
    textureDim=sgl.int2(w, h)
)

# When running in benchmark mode amortize overheads over many runs to measure more accurate GPU times
num_iters = 1000 if args.benchmark else 1

# Compress!
start_time = time.time()
for i in range(num_iters):
    # Compress input texture using BC7 mode 6, and output decompressed result
    encoder_fn(grid(shape=(w // 4, h // 4)))

# Calculate and print performance metrics
if args.benchmark:
    device.wait_for_idle()
    comp_time_in_sec = (time.time() - start_time) / num_iters
    textures_per_sec = 1 / comp_time_in_sec
    giga_texels_per_sec = w * h * textures_per_sec / 1E9
    print(f"\nBenchmark mode:")
    print(f"  - Number of optimization passes: {args.opt_steps}")
    print(
        f"  - Compression time: {1E3 * comp_time_in_sec:.4g} ms --> {giga_texels_per_sec:.4g} GTexels/s")

# Calculate and print PSNR
mse = np.mean((input_tex.to_numpy() / 255.0 - decoded_tex.to_numpy()) ** 2)
psnr = 20 * np.log10(1.0 / np.sqrt(mse))
print(f"\nPSNR: {psnr:.4g}")

# Output decoded texture
if args.output_path:
    img = decoded_tex.to_bitmap().convert(component_type=sgl.Bitmap.ComponentType.uint8, srgb_gamma=False)
    img.write(args.output_path)
