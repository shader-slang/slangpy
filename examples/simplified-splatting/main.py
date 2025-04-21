# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
import time
import logging

sys.path.insert(0, r"C:\users\tongg\appdata\local\packages\pythonsoftwarefoundation.python.3.13_qbz5n2kfra8p0\localcache\local-packages\python313\site-packages")
sys.path.insert(0, r"C:\Users\tongg\sgl\build\windows-vs2022\Release\python")

import slangpy as spy
import sgl
import pathlib
import imageio
import numpy as np
import signal
import sys
import types
from typing import Optional

print(f"sgl module location: {sgl.__file__}")
print(f"spy module location: {spy.__file__}")

# Set up signal handler for graceful exit
def signal_handler(sig: int, frame: Optional[types.FrameType]) -> None:
    print("\nCtrl+C detected. Exiting gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create an SGL device, which will handle setup and invocation of the Slang
# compiler for us. We give it both the slangpy PATH and the local include
# PATH so that it can find Slang shader files
device = sgl.Device(
    type=sgl.DeviceType.d3d12,
#    type=sgl.DeviceType.vulkan,
    enable_debug_layers=False, # This is not working for RHI yet
    compiler_options={
        "include_paths": [
            spy.SHADER_PATH,
            pathlib.Path(__file__).parent.absolute(),
        ],
    },
)


# Load our Slang module -- we'll take a look at this in just a moment
module = spy.Module.load_from_file(device, "simplediffsplatting2d.slang")

# Create a buffer to store Gaussian blobs. We're going to make a very small one,
# because right now this code is not very efficient, and will take a while to run.
# For now, we are going to create 200 blobs, and each blob will be comprised of 9
# floats:
#   blob center x and y (2 floats)
#   sigma (a 2x2 covariance matrix - 4 floats)
#   color (3 floats)
NUM_BLOBS = 200
FLOATS_PER_BLOB = 9
# SlangPy lets us create a Tensor and initialize it easily using numpy to generate
# random values. This Tensor includes storage for gradients, because we call .with_grads()
# on the created spy.Tensor.
blobs = spy.Tensor.numpy(device, np.random.rand(
    NUM_BLOBS * FLOATS_PER_BLOB).astype(np.float32)).with_grads()

# Load our target image from a file, using the imageio package,
# and store its width and height in W, H
image = imageio.imread("./jeep.jpg")
W = image.shape[0]
H = image.shape[1]

# Convert the image from RGB_u8 to RGBA_f32 -- we're going
# to be using texture values during derivative propagation,
# so we need to be dealing with floats here.
image = (image / 256.0).astype(np.float32)
image = np.concatenate([image, np.ones((W, H, 1), dtype=np.float32)], axis=-1)
input_image = device.create_texture(
    data=image,
    width=W,
    height=H,
    format=sgl.Format.rgba32_float,
    usage=sgl.TextureUsage.shader_resource)

# Create a per_pixel_loss Tensor to hold the calculated loss, and create gradient storage
per_pixel_loss = spy.Tensor.empty(device, dtype=module.float4, shape=(W, H))
per_pixel_loss = per_pixel_loss.with_grads()
# Set per-pixel loss' derivative to 1 (using a 1-line function in the slang file)
module.ones(per_pixel_loss.grad_in)

# Create storage for the Adam update moments
# The Adam optimization algorithm helps us update the inputs to the function being optimized
# in an efficient manner. It stores two "moments": the first is a moving average of the
# of the gradient of the loss function. The second is a moving average of the squares of these
# gradients. This allows us to "step" in the desired direction while maintaining momentum toward
# the goal
adam_first_moment = spy.Tensor.zeros_like(blobs)
adam_second_moment = spy.Tensor.zeros_like(blobs)

# Pre-allocate a texture to send data to tev occasionally.
current_render = device.create_texture(
    width=W,
    height=H,
    format=sgl.Format.rgba32_float,
    usage=sgl.TextureUsage.shader_resource | sgl.TextureUsage.unordered_access)

iterations = 2
total_time = 0
try:
    for iter in range(iterations):
        iter_start = time.time()
        
        # Time backward pass
        bwd_start = time.time()
        module.perPixelLoss.bwds(per_pixel_loss,
                                spy.grid(shape=(input_image.width, input_image.height)),
                                blobs, input_image)
        bwd_time = time.time() - bwd_start

        # Time Adam update
        adam_start = time.time()
        module.adamUpdate(blobs, blobs.grad_out, adam_first_moment, adam_second_moment)
        adam_time = time.time() - adam_start

        # Time rendering
        if iter % 1 == 0:
            # Time renderBlobsToTexture separately
            render_start = time.time()
            module.renderBlobsToTexture(current_render,
                                    blobs,
                                    spy.grid(shape=(input_image.width, input_image.height)))
            render_only_time = time.time() - render_start
            
            # Time tev.show_async separately
            tev_start = time.time()
            sgl.tev.show_async(current_render, name=f"optimization_{(iter // 1):03d}")
            tev_time = time.time() - tev_start
            
            # Total render time (both operations)
            render_time = render_only_time + tev_time
        
        # Calculate and log detailed timing
        iter_time = time.time() - iter_start
        total_time += iter_time
        avg_time = total_time / (iter + 1)
        logger.info(f"Iteration {iter}: {iter_time:.3f}s (avg: {avg_time:.3f}s)")
        logger.info(f"  - Backward: {bwd_time:.3f}s")
        logger.info(f"  - Adam: {adam_time:.3f}s")
        logger.info(f"  - Render: {render_time:.3f}s")
        logger.info(f"    * renderBlobsToTexture: {render_only_time:.3f}s")
        logger.info(f"    * tev.show_async: {tev_time:.3f}s")

except KeyboardInterrupt:
    print("\nCtrl+C detected. Exiting gracefully...")
    sys.exit(0)
