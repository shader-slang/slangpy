# SPDX-License-Identifier: Apache-2.0

from app import App
import slangpy as spy
import numpy as np
import sgl
import sys
import time
import math
from slangpy.types import call_id
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
#    reach 0. This should give a rendered BRDF result from a low res
#    normal map that more closely resembles (2) than (3).
mode = 1

# Cycle modes using the 'tab' key.
def on_keyboard_event(key: sgl.KeyboardEvent):
    if key.type == sgl.KeyboardEventType.key_press and key.key == sgl.KeyCode.tab:
        global mode
        mode += 1
        if mode > 5:
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
albedo_map: spy.Tensor = module.toAlbedoMap(createTextureFromFile(app.device, "stonewall_2k_albedo.jpg"), _result='tensor')
normal_map: spy.Tensor = module.toNormalMap(createTextureFromFile(app.device, "stonewall_2k_normal.jpg"), _result='tensor')


# TODO: Can we get away with not creating the tensors in this way, and not declaring the sample points?
#w,h = 512,512 # TODO: Should this be 512x512 or 1024x1024?
#sample_points_array = np.random.randn(w*h, 2).astype(np.float32)
#i = 0
#for x in range(w):
#    for y in range(h):
#        sample_points_array[i] = [x, y]
#        i += 1
#sample_points = spy.Tensor.from_numpy(app.device, sample_points_array)

# Start with normals that are all uniform, but may want to use the downsampled normals instead.
#normal_array = np.random.randn(w*h, 3).astype(np.float32)
#for i in range(w*h):
#    normal_array[i] = [0, 0, 1]
#trained_normals = spy.Tensor.from_numpy(app.device, normal_array).with_grads(zero = True)

#forward_result = spy.Tensor.from_numpy(app.device, np.zeros((w*h, ), dtype=np.float32)).with_grads()

all_ones = np.ones((512*512, 1), dtype=np.float32)

trained_normals: spy.Tensor = module.downSample(normal_map, spy.grid((1024,1024)), _result='tensor').with_grads(zero = True)
trained_normals: spy.Tensor = module.downSample(trained_normals, spy.grid((512,512)), _result='tensor').with_grads(zero = True)

# Run the app, using the previously defined mode. Only mode 5 will perform
# actual training.
iter = 0
while app.process_events():
    t = math.sin(time.time()*2)*1
    light_dir = sgl.math.normalize(sgl.float3(t,t, 1.0))

    # Full res rendered output BRDF from full res inputs (mode 1).
    rendered: spy.Tensor = module.renderFullRes(albedo_map, normal_map, light_dir, _result='tensor')

    # Downsampled output (avg) from the full res inputs (mode 2).
    # TODO: Decide how much we want to downsample these.
    downsampled: spy.Tensor = module.downSample(rendered, spy.grid((1024,1024)), _result='tensor')
    downsampled: spy.Tensor = module.downSample(downsampled, spy.grid((512,512)), _result='tensor')

    # Downsampled inputs (mode 3).
    downsampled_albedo_map: spy.Tensor = module.downSample(albedo_map, spy.grid((1024,1024)), _result='tensor')
    downsampled_albedo_map: spy.Tensor = module.downSample(downsampled_albedo_map, spy.grid((512,512)), _result='tensor')
    downsampled_normal_map: spy.Tensor = module.downSample(normal_map, spy.grid((1024,1024)), _result='tensor')
    downsampled_normal_map: spy.Tensor = module.downSample(downsampled_normal_map, spy.grid((512,512)), _result='tensor')

    if mode == 1:
        # Render BRDF from full res inputs.
        module.showTensorFloat3(rendered, app.output, spy.grid(rendered.shape), sgl.int2(0,0))
    elif mode == 2:
        # Render BRDF by taking average of neighboring full res inputs.
        module.showTensorFloat3(downsampled, app.output, spy.grid(rendered.shape), sgl.int2(0,0))
    elif mode == 3:
        # Render BRDF using downsampled inputs.
        downres: spy.Tensor = module.renderFullRes(downsampled_albedo_map, downsampled_normal_map, light_dir, _result='tensor')
        module.showTensorFloat3(downres, app.output, spy.grid(rendered.shape), sgl.int2(0,0))
    elif mode == 4:
        # Render L2 loss between (2) and (3).
        loss: spy.Tensor = module.renderLoss(downsampled, downsampled_albedo_map, downsampled_normal_map, light_dir, _result='tensor')
        module.showTensorFloat3(loss, app.output, spy.grid(rendered.shape), sgl.int2(0,0))
    elif mode == 5:
        # TODO: Only train up to a maximum iter value.
        iter += 1
        if iter <= 1500:
            # Render BRDF from trained inputs to match (2).

            # Run the shader that calculates the loss (the difference between the downsampled output, and the output calculated with downsampled albedo and the broken normals).
            forward_result: spy.Tensor = module.forward(downsampled, downsampled_albedo_map, trained_normals, light_dir, _result='tensor').with_grads()
            forward_result.grad.storage.copy_from_numpy(all_ones)

            # Run said shader backwards to get the derivative of the normals with respect to the loss - aka how changing the normals would affect the loss.
            module.forward.bwds(downsampled, downsampled_albedo_map, trained_normals, light_dir, _result=forward_result)

            # Subtract a tiny bit of that gradient - i.e. if making normal.z greater makes the loss more, we want to make normal.z smaller.
            # TODO: Is this the correct way to do this?
            normal_array = trained_normals.to_numpy()
            normal_grads = trained_normals.grad.to_numpy()
            normal_array = normal_array - 0.0001 * normal_grads

            trained_normals.storage.copy_from_numpy(normal_array)
            trained_normals.grad.clear()

            if iter % 50 == 0:
                resultArray = forward_result.to_numpy()
                loss = np.linalg.norm(resultArray) / (512 * 512)
                print("Iteration: {}, Loss: {}".format(iter, loss))
                print("parameter {}".format(trained_normals.to_numpy()))

        # Render the result.
        # TODO: Hard to tell if it's correct from this, so for now I'm rendering the loss function instead to verify.
        #result: spy.Tensor = module.renderFullRes(downsampled_albedo_map, trained_normals, light_dir, _result='tensor')
        #module.showTensorFloat3(result, app.output, spy.grid(rendered.shape), sgl.int2(0,0))

        # TODO: Instead just render the loss, it should approach 0 if everything is working.
        loss: spy.Tensor = module.renderLoss(downsampled, downsampled_albedo_map, trained_normals, light_dir, _result='tensor')
        module.showTensorFloat3(loss, app.output, spy.grid(rendered.shape), sgl.int2(0,0))

    app.present()
