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
# 5. Train per-pixel loss values to reach the expected values from (4).
#    This is shown by rendering the learned loss values as white pixels,
#    which should match the values from mode (4).
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
    #downsampled: spy.Tensor = module.downSample(downsampled, spy.grid((512,512)), _result='tensor')

    # Downsampled inputs (mode 3).
    downsampled_albedo_map: spy.Tensor = module.downSample(albedo_map, spy.grid((1024,1024)), _result='tensor')
    #downsampled_albedo_map: spy.Tensor = module.downSample(downsampled_albedo_map, spy.grid((512,512)), _result='tensor')
    downsampled_normal_map: spy.Tensor = module.downSample(normal_map, spy.grid((1024,1024)), _result='tensor')
    #downsampled_normal_map: spy.Tensor = module.downSample(downsampled_normal_map, spy.grid((512,512)), _result='tensor')

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
        # Render L2 loss between (3) and (4).
        loss: spy.Tensor = module.renderLoss(downsampled, downsampled_albedo_map, downsampled_normal_map, light_dir, _result='tensor')
        module.showTensorFloat3(loss, app.output, spy.grid(rendered.shape), sgl.int2(0,0))
    elif mode == 5:
        # Render BRDF from trained inputs to more closely match (3).
        # TODO: Implement me!
        pass

    app.present()
