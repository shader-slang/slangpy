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
app = App(width=1024, height=1024, title="Mipmap Example")
module = spy.Module.load_from_file(app.device, "mipmapping.slang")

# The purpose of this example is to render BRDF using input material and
# normal map textures.
#
# This can be expected to give correct results at full res, but when using
# lower res texture inputs (mipmap levels) the result will be somewhat
# incorrect, represented as per-pixel L2 loss values, which tell us how
# the inputs need to change to give the correct values.
#
# This app has 5 modes, set by the mode global variable in main.py:
# 1. Evaluate pixel color using full res textures. This simply reads full res
#    texture data and evaluates a simple BRDF (in this case phong).
# 2. Evaluate a lower res pixel color by using mode (1) to evaluate 4 adjacent
#    values and take the average. This is similar to using lower res mipmap
#    levels but should give a more accurate result.
# 3. Evaluate a lower res pixel by using mode (1) with lower res texture
#    inputs, ie, lower res mipmap levels. This should be less accurate
#    than mode 2.
# 4. Show the difference between mode (1) and (3), ie the per-pixel
#    L2 loss values between full res and low res inputs.
# 5. Train input normals until per-pixel loss values between this and (2)
#    reach 0.
# 6. Render the BRDF using the trained normals from (5). This should give a
#    result that more closely matches (2) than (3).
mode = 1


# Generate a UV grid for the window, starting at (0, 0) for the top left pixel
window_w, window_h = app.window.width, app.window.height
windowUVs = module.pixelToUV(spy.grid((window_h, window_w)), sgl.int2(window_w, window_h))


def on_keyboard_event(key: sgl.KeyboardEvent):
    """Cycle modes using the 'tab' key."""
    if key.type == sgl.KeyboardEventType.key_press and key.key == sgl.KeyCode.tab:
        global mode
        mode += 1
        if mode > 6:
            mode = 1
        if mode == 1:
            print("Mode 1: Full res inputs")
        elif mode == 2:
            print("Mode 2: Avg from full res inputs")
        elif mode == 3:
            print("Mode 3: Downsampled inputs")
        elif mode == 4:
            print("Mode 4: Loss between (2) and (3)")
        elif mode == 5:
            print("Mode 5: Trained inputs")
        elif mode == 6:
            print("Mode 6: Render BRDF from trained normals")


app.on_keyboard_event = on_keyboard_event


# Simple function to read a texture from a file and exit if an error occurs.
def createTextureFromFile(device: sgl.Device, filepath: str):
    try:
        loader = sgl.TextureLoader(device)
        texture = loader.load_texture(Path(__file__).parent / filepath)
        return texture

    except Exception as e:
        print(f"\nError loading the texture: {e}")
        sys.exit(1)


# Material and normal map textures used for this example.
albedo_map: spy.Tensor = module.toAlbedoMap(createTextureFromFile(
    app.device, "stonewall_2k_albedo.jpg"), _result='tensor')
normal_map: spy.Tensor = module.toNormalMap(createTextureFromFile(
    app.device, "stonewall_2k_normal.jpg"), _result='tensor')


# Tensor for training the normal map
trained_normals: spy.Tensor = module.downSample(
    normal_map, spy.grid((1024, 1024)), _result='tensor').with_grads(zero=True)
trained_normals: spy.Tensor = module.downSample(
    trained_normals, spy.grid((512, 512)), _result='tensor').with_grads(zero=True)

# Tensor containing the training loss and its derivative to propagate backwards (set to 1)
training_loss = spy.Tensor.zeros(module.device, trained_normals.shape, module.float).with_grads()
training_loss.grad_in.copy_from_numpy(np.ones(training_loss.shape.as_tuple(), dtype=np.float32))

# This learning rate is specifically for using the downsampled normals as the initial values.
learning_rate = 0.001
max_iter = 1500

# TODO: As an alternative we can start with uniform normals and train those, but this only seems to work for static light directions.
# trained_normals = spy.Tensor.empty(app.device, shape=(512,512), dtype='float3').with_grads(zero = True)
# module.baseNormal(_result=trained_normals).with_grads(zero = True)

# TODO: This learning rate is specifically for using the base normal as the initial values.
# learning_rate = 0.1

# Run the app, using the previously defined mode. Only mode 5 will perform
# actual training.
iter = 0
while app.process_events():
    t = math.sin(time.time()*2)*1
    light_dir = sgl.math.normalize(sgl.float3(t, t, 1.0))

    # Full res rendered output BRDF from full res inputs (mode 1).
    rendered: spy.Tensor = module.renderFullRes(albedo_map, normal_map, light_dir, _result='tensor')

    # Downsampled output (avg) from the full res inputs (mode 2).
    downsampled: spy.Tensor = module.downSample(rendered, spy.grid((1024, 1024)), _result='tensor')
    downsampled: spy.Tensor = module.downSample(downsampled, spy.grid((512, 512)), _result='tensor')

    # Downsampled inputs (mode 3).
    downsampled_albedo_map: spy.Tensor = module.downSample(
        albedo_map, spy.grid((1024, 1024)), _result='tensor')
    downsampled_albedo_map: spy.Tensor = module.downSample(
        downsampled_albedo_map, spy.grid((512, 512)), _result='tensor')
    downsampled_normal_map: spy.Tensor = module.downSample(
        normal_map, spy.grid((1024, 1024)), _result='tensor')
    downsampled_normal_map: spy.Tensor = module.downSample(
        downsampled_normal_map, spy.grid((512, 512)), _result='tensor')

    if mode == 1:
        # Render BRDF from full res inputs.
        module.showTensorFloat3(rendered, windowUVs, True, _result=app.output)
    elif mode == 2:
        # Render BRDF by taking average of neighboring full res inputs.
        module.showTensorFloat3(downsampled, windowUVs, True, _result=app.output)
    elif mode == 3:
        # Render BRDF using downsampled inputs.
        downres: spy.Tensor = module.renderFullRes(
            downsampled_albedo_map, downsampled_normal_map, light_dir, _result='tensor')
        module.showTensorFloat3(downres, windowUVs, True, _result=app.output)
    elif mode == 4:
        # Render L2 loss between (2) and (3).
        loss: spy.Tensor = module.renderLoss(
            downsampled, downsampled_albedo_map, downsampled_normal_map, light_dir, _result='tensor')
        module.showTensorFloat3(loss, windowUVs, False, _result=app.output)
    elif mode == 5:
        iter += 1
        if iter <= max_iter:
            # Take the function that calculates the loss, i.e. the difference between the downsampled output
            # and the output calculated with downsampled albedo/normals, and run it 'backwards'
            # This propagates the gradient of training_loss back to the gradients of trained_normals.
            module.calculateLoss.bwds(downsampled, downsampled_albedo_map,
                                      trained_normals, light_dir, _result=training_loss)
            # trained_normals.grad_out now tells us how trained_normals needs to change
            # so that training_loss changes by training_loss.grad_in

            # We want training_loss to go down, so we subtract a tiny bit of that gradient.
            module.gradientDescent(trained_normals, trained_normals.grad, learning_rate)
            # In the next iteration, the updated trained_normals now hopefully reduces the loss

            if iter % 50 == 0:
                resultArray = training_loss.to_numpy()
                loss = np.sum(resultArray) / resultArray.size
                print("Iteration: {}, Loss: {}".format(iter, loss))
                print("parameter {}".format(trained_normals.to_numpy()))

        # Render the loss, it should approach 0 if everything is working.
        loss: spy.Tensor = module.renderLoss(
            downsampled, downsampled_albedo_map, trained_normals, light_dir, _result='tensor')
        module.showTensorFloat3(loss, windowUVs, False, _result=app.output)
    elif mode == 6:
        # Render the result.
        result: spy.Tensor = module.renderFullRes(
            downsampled_albedo_map, trained_normals, light_dir, _result='tensor')
        module.showTensorFloat3(result, windowUVs, True, _result=app.output)

    app.present()
