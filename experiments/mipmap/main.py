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
# This example trains the mipmaps so that the rendered output looks as close
# as possible to the downsampled original


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


# Tensor for training the normal map.
# One option is to start from the downsampled normal map.
#trained_normals_without_grads: spy.Tensor = downsampleTensor(normal_map)

# Another option is to start with uniform normals, however this only works with an Adam optimizer.
trained_normals_without_grads = spy.Tensor.empty(app.device, shape=(512,512), dtype='float3')
module.baseNormal(_result=trained_normals_without_grads)
trained_normals = trained_normals_without_grads.with_grads()

trained_roughness_without_grads = spy.Tensor.empty(app.device, shape=(512,512), dtype='float')
module.baseRoughness(_result=trained_roughness_without_grads)
trained_roughness = trained_roughness_without_grads.with_grads()

# Tensor containing the training loss and its derivative to propagate backwards (set to 1).
training_loss = spy.Tensor.zeros(module.device, trained_normals.shape, module.float).with_grads()
training_loss.grad_in.copy_from_numpy(np.ones(training_loss.shape.as_tuple(), dtype=np.float32))

# m and v tensors for the Adam optimizer.
m_normal = spy.Tensor.zeros(module.device, trained_normals.shape, module.float3)
v_normal = spy.Tensor.zeros(module.device, trained_normals.shape, module.float3)

m_roughness = spy.Tensor.zeros(module.device, trained_roughness.shape, module.float)
v_roughness = spy.Tensor.zeros(module.device, trained_roughness.shape, module.float)

# This learning rate seems to produce a reasonable result using gradient descent.
grad_learning_rate = 0.001
# This learning rate seems better for the adam optimizer.
adam_learning_rate = 0.1

# Run the training.
iter = 0
while app.process_events():
    # Generate random light and view dirs on the hemisphere
    light_dir = getRandomDir()
    view_dir = getRandomDir()

    # Alternative: Smooth sweep of light direction
    #t = math.sin(time.time()*2)*1
    #light_dir = sgl.math.normalize(sgl.float3(t, t, 1.0))

    # Full res rendered output BRDF from full res inputs (mode 1).
    rendered: spy.Tensor = module.renderFullRes(albedo_map, normal_map, roughness_map, light_dir, view_dir, _result='tensor')

    # Downsampled output (avg) from the full res inputs (mode 2).
    downsampled = downsampleTensorFloat3(rendered)

    # Take the function that calculates the loss, i.e. the difference between the downsampled output
    # and the output calculated with downsampled albedo/normals, and run it 'backwards'
    # This propagates the gradient of training_loss back to the gradients of trained_normals.
    module.calculateLoss.bwds(downsampled, downsampled_albedo_map,
                              trained_normals, trained_roughness, light_dir, view_dir, _result=training_loss)
    # trained_normals.grad_out now tells us how trained_normals needs to change
    # so that training_loss changes by training_loss.grad_in

    # We want training_loss to go down, so we subtract a tiny bit of that gradient.
    #module.gradientDescent(trained_normals, trained_normals.grad, grad_learning_rate)
    # In the next iteration, the updated trained_normals now hopefully reduces the loss

    # Another option is to use an Adam optimizer.
    module.adamFloat3(trained_normals, trained_normals.grad, m_normal, v_normal, adam_learning_rate)
    module.adamFloat(trained_roughness, trained_roughness.grad, m_roughness, v_roughness, adam_learning_rate)

    iter += 1

    if iter % 50 == 0:
        resultArray = training_loss.to_numpy()
        loss = np.sum(resultArray) / resultArray.size
        print("Iteration: {}, Loss: {}".format(iter, loss))
        print("parameter {}".format(trained_normals.to_numpy()))

    # Render current progress
    loss: spy.Tensor = module.renderLoss(
        downsampled, downsampled_albedo_map, trained_normals, trained_roughness, light_dir, view_dir, _result='tensor')
    result: spy.Tensor = module.renderFullRes(
        downsampled_albedo_map, trained_normals, trained_roughness, light_dir, view_dir, _result='tensor')

    # TODO: We could also throw in some more visualization modes.
    module.showTrainingProgress(result, loss, downsampled_normal_map,
                                trained_normals_without_grads, trained_roughness_without_grads, windowUVs, _result=app.output)

    app.present()
