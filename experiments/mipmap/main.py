# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from app import App
import slangpy as spy
import numpy as np
import sgl
import sys
import time
import math
from pathlib import Path

# Create the app and load the sample shader.
app = App(width=2048, height=512, title="Mipmap Example")
module = spy.Module.load_from_file(app.device, "mipmapping.slang")

# The purpose of this example is to render BRDF using input material and
# normal map textures.
#
# This can be expected to give correct results at full res, but when using
# lower res texture inputs (mipmap levels) the result will be somewhat
# incorrect, represented as per-pixel L2 loss values, which tell us how
# the inputs need to change to give the correct values.
#
# This example trains the normal and roughness maps so that the rendered
# output BRDF looks as closs as possible to the downsampled original.


# Generate a UV grid for the window, starting at (0, 0) for the top left pixel
window_w, window_h = app.window.width, app.window.height
windowUVs = module.pixelToUV(spy.grid((window_h, window_w)), sgl.int2(window_w, window_h))


# Simple function to read a texture from a file and exit if an error occurs.
def createTextureFromFile(device: sgl.Device, filepath: str):
    try:
        loader = sgl.TextureLoader(device)
        texture = loader.load_texture(Path(__file__).parent / filepath)
        return texture

    except Exception as e:
        print(f"\nError loading the texture: {e}")
        sys.exit(1)


def downsampleTensorFloat3(mip0: spy.Tensor) -> spy.Tensor:
    mip1_shape = (mip0.shape[0] // 2, mip0.shape[1] // 2)
    mip2_shape = (mip1_shape[0] // 2, mip1_shape[1] // 2)

    mip1: spy.Tensor = module.downSampleFloat3(mip0, spy.grid(mip1_shape), _result='tensor')
    mip2: spy.Tensor = module.downSampleFloat3(mip1, spy.grid(mip2_shape), _result='tensor')

    return mip2

def downsampleTensorFloat(mip0: spy.Tensor) -> spy.Tensor:
    mip1_shape = (mip0.shape[0] // 2, mip0.shape[1] // 2)
    mip2_shape = (mip1_shape[0] // 2, mip1_shape[1] // 2)

    mip1: spy.Tensor = module.downSampleFloat(mip0, spy.grid(mip1_shape), _result='tensor')
    mip2: spy.Tensor = module.downSampleFloat(mip1, spy.grid(mip2_shape), _result='tensor')

    return mip2


def getRandomDir():
    r = math.sqrt(np.random.rand())
    phi = np.random.rand() * math.pi * 2
    Lx = r * math.sin(phi)
    Ly = r * math.cos(phi)
    Lz = math.sqrt(max(1 - r ** 2, 0))
    return sgl.float3(Lx, Ly, Lz)


# Material, normal map and roughness textures used for this example.
albedo_map: spy.Tensor = module.toAlbedoMap(createTextureFromFile(
    app.device, "PavingStones070_2K.diffuse.jpg"), _result='tensor')
normal_map: spy.Tensor = module.toNormalMap(createTextureFromFile(
    app.device, "PavingStones070_2K.normal.jpg"), _result='tensor')
roughness_map: spy.Tensor = module.toRoughnessMap(createTextureFromFile(
    app.device, "PavingStones070_2K.roughness.jpg"), _result='tensor')

downsampled_albedo_map = downsampleTensorFloat3(albedo_map)
downsampled_normal_map = downsampleTensorFloat3(normal_map)
downsampled_roughness_map = downsampleTensorFloat(roughness_map)


# Tensors for training the normals and roughness.
# One option is to start from the downsampled normal and roughness maps.
#trained_normals_without_grads: spy.Tensor = downsampleTensorFloat3(normal_map)
#trained_roughness_without_grads: spy.Tensor = downsampleTensorFloat(roughness_map)

# Another option is to start with uniform normals, however this only works with an Adam optimizer.
trained_normals_without_grads = spy.Tensor.empty(app.device, shape=(512,512), dtype='float3')
module.baseNormal(_result=trained_normals_without_grads)
trained_roughness_without_grads = spy.Tensor.empty(app.device, shape=(512,512), dtype='float')
module.baseRoughness(_result=trained_roughness_without_grads)

trained_normals = trained_normals_without_grads.with_grads()
trained_roughness = trained_roughness_without_grads.with_grads()

# Tensor containing the training loss and its derivative to propagate backwards (set to 1).
training_loss = spy.Tensor.zeros(module.device, trained_normals.shape, module.float).with_grads()
training_loss.grad_in.copy_from_numpy(np.ones(training_loss.shape.as_tuple(), dtype=np.float32))

# m and v tensors for the Adam optimizer.
m_normal = spy.Tensor.zeros(module.device, trained_normals.shape, module.float3)
v_normal = spy.Tensor.zeros(module.device, trained_normals.shape, module.float3)

m_roughness = spy.Tensor.zeros(module.device, trained_roughness.shape, module.float)
v_roughness = spy.Tensor.zeros(module.device, trained_roughness.shape, module.float)

# Learning rate for training.
learning_rate = 0.001


# In case we want to start training when pressing the 'tab' key.
#start = False
#
#def on_keyboard_event(key: sgl.KeyboardEvent):
#    if key.type == sgl.KeyboardEventType.key_press and key.key == sgl.KeyCode.tab:
#        global start
#        start = True
#
#app.on_keyboard_event = on_keyboard_event


# Run the training.
iter = 0
while app.process_events():
    #if not start:
    #    continue
    # Generate random light and view dirs on the hemisphere.
    light_dir = getRandomDir()
    view_dir = getRandomDir()

    # Alternative: Smooth sweep of light direction.
    #t = math.sin(time.time()*2)*1
    #light_dir = sgl.math.normalize(sgl.float3(t, t, 1.0))
    #view_dir = sgl.float3(0, 0, 1)

    # Full res rendered output BRDF from full res inputs.
    rendered: spy.Tensor = module.renderFullRes(albedo_map, normal_map,
        roughness_map, light_dir, view_dir, _result='tensor')

    # Downsampled output (avg) from the full res inputs.
    downsampled = downsampleTensorFloat3(rendered)

    # Lower res BRDF from downsampled inputs (for comparison).
    lowres: spy.Tensor = module.renderFullRes(downsampled_albedo_map, downsampled_normal_map,
        downsampled_roughness_map, light_dir, view_dir, _result='tensor')

    # Take the function that calculates the loss, i.e. the difference between the downsampled output
    # and the output calculated with downsampled albedo/normals, and run it 'backwards'
    # This propagates the gradient of training_loss back to the gradients of trained_normals.
    module.calculateLoss.bwds(downsampled, downsampled_albedo_map,
        trained_normals, trained_roughness, light_dir, view_dir, _result=training_loss)
    # trained_normals.grad_out now tells us how trained_normals needs to change
    # so that training_loss changes by training_loss.grad_in

    # We want training_loss to go down, so we subtract a tiny bit of that gradient.
    # One option is to use gradient descent which is simple but can get stuck in local minima.
    #module.gradientDescentFloat3(trained_normals, trained_normals.grad, learning_rate)
    #module.gradientDescentFloat3(trained_roughness, trained_roughness.grad, learning_rate)
    # In the next iteration, the updated trained_normals now hopefully reduces the loss.

    # Another option is to use an Adam optimizer.
    module.adamFloat3(trained_normals, trained_normals.grad, m_normal, v_normal, learning_rate)
    module.adamFloat(trained_roughness, trained_roughness.grad, m_roughness, v_roughness, learning_rate)

    # This is just here for debugging purposes, to confirm the data looks correct.
    #iter += 1
    #if iter % 50 == 0:
    #    iter = 0
    #    resultArray = training_loss.to_numpy()
    #    loss = np.sum(resultArray) / resultArray.size
    #    print("Iteration: {}, Loss: {}".format(iter, loss))
    #    print("parameter {}".format(trained_normals.to_numpy()))

    # Render current progress.
    loss: spy.Tensor = module.renderLoss(
        downsampled, downsampled_albedo_map, trained_normals, trained_roughness, light_dir, view_dir, _result='tensor')
    result: spy.Tensor = module.renderFullRes(
        downsampled_albedo_map, trained_normals, trained_roughness, light_dir, view_dir, _result='tensor')

    # Compare the loss between the trained result and the lowres result.
    # Green pixels indicate a better result, and red pixels indicate a worse
    # result.
    loss_lowres: spy.Tensor = module.lossLowres(downsampled, lowres, _result='tensor')
    loss_diff: spy.Tensor = module.lossDiff(loss, loss_lowres, _result='tensor')

    module.showTrainingProgress(result, loss_diff, downsampled_normal_map,
        trained_normals_without_grads, trained_roughness_without_grads, windowUVs, _result=app.output)

    app.present()
