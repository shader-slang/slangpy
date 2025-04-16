# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from app import App
import slangpy as spy
import sgl
import pathlib
import numpy as np
import imageio
from tqdm import tqdm

# Load up an input image.
image = imageio.imread("./jeep.jpg")
W = image.shape[0]
H = image.shape[1]

# Create SGL windowed app and get the device
app = App("Diffusion Splatting 2D", W, H)
device = app.device

# 2D -> 1D dispatch-ID mapping utility to help us work around slangpy's 1D dispatch restriction


def calcCompressedDispatchIDs(x_max: int, y_max: int, wg_x: int, wg_y: int):
    local_x = np.arange(0, wg_x, dtype=np.uint32)
    local_y = np.arange(0, wg_y, dtype=np.uint32)
    local_xv, local_yv = np.meshgrid(local_x, local_y, indexing="ij")
    local_xyv = np.stack([local_xv, local_yv], axis=-1)
    local_xyv = np.tile(local_xyv.reshape(wg_x * wg_y, 2).astype(np.uint32),
                        ((x_max // wg_x) * (y_max // wg_y), 1))
    local_xyv = local_xyv.reshape((x_max * y_max, 2))

    group_x = np.arange(0, (x_max // wg_x), dtype=np.uint32)
    group_y = np.arange(0, (y_max // wg_y), dtype=np.uint32)
    group_xv, group_yv = np.meshgrid(group_x, group_y, indexing="ij")
    group_xyv = np.stack([group_xv, group_yv], axis=-1)
    group_xyv = np.tile(group_xyv[:, :, None, None, :], (1, 1, wg_y, wg_x, 1))
    group_xyv = group_xyv.reshape((x_max * y_max, 2)).astype(np.uint32)

    return ((group_xyv * np.array([wg_x, wg_y])[None, :] + local_xyv).astype(np.uint32))


# Load module
module = spy.Module.load_from_file(device, "diffsplatting2d.slang")

# Randomize the blobs buffer
NUM_BLOBS = 20480 * 2
FLOATS_PER_BLOB = 9
blobs = spy.Tensor.numpy(device, np.random.rand(
    NUM_BLOBS * FLOATS_PER_BLOB).astype(np.float32)).with_grads()

WORKGROUP_X, WORKGROUP_Y = 8, 4


assert (W % WORKGROUP_X == 0) and (H % WORKGROUP_Y == 0)

# Go from RGB_u8 -> RGBA_f32
image = (image / 256.0).astype(np.float32)
image = np.concatenate([image, np.ones((W, H, 1), dtype=np.float32)], axis=-1)
input_image = device.create_texture(
    data=image,
    width=W,
    height=H,
    format=sgl.Format.rgba32_float,
    usage=sgl.TextureUsage.shader_resource)

dispatch_ids = spy.NDBuffer(device, dtype=module.uint2, shape=(W, H))
dispatch_ids.copy_from_numpy(calcCompressedDispatchIDs(W, H, WORKGROUP_X, WORKGROUP_Y))

per_pixel_loss = spy.Tensor.empty(device, dtype=module.float4, shape=(W, H))
per_pixel_loss = per_pixel_loss.with_grads()
# Set per-pixel loss' derivative to 1 (using a 1-line function in the slang file)
module.ones(per_pixel_loss.grad_in)

adam_first_moment = spy.Tensor.zeros_like(blobs)
adam_second_moment = spy.Tensor.zeros_like(blobs)

# Pre-allocate a texture to send data to tev occasionally.
current_render = device.create_texture(
    width=W,
    height=H,
    format=sgl.Format.rgba32_float,
    usage=sgl.TextureUsage.shader_resource | sgl.TextureUsage.unordered_access)

iterations = 10000
for iter in tqdm(range(iterations)):
    if not app.process_events():
        exit(0)

    # Backprop the unit per-pixel loss with auto-diff.
    module.perPixelLoss.bwds(per_pixel_loss, dispatch_ids, blobs, input_image)

    # Update
    module.adamUpdate(blobs, blobs.grad_out, adam_first_moment, adam_second_moment)

    if iter % 10 == 0:
        module.renderBlobsToTexture(app.output, blobs, dispatch_ids)
        app.present()

# Keep window processing events until user closes it.
while app.process_events():
    app.present()
