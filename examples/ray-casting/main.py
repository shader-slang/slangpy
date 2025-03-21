# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import slangpy as spy
import numpy as np
import sgl
from app import App
from slangpy.types import call_id
import math
import imageio
import os

app = App()
module = spy.Module.load_from_file(app.device, "raycasting.slang")
       
uniforms = {
    "_type":  "Uniforms",
    "screenSize": sgl.float2(app._window.width, app._window.height),
    "focalLength": 24.0,
    "frameHeight": 24.0,
}

radius = 15.0
alpha:float = 0.0
pi:float = 3.14159265359
beta:float = -pi / 4.0
def updateCamera():
    global alpha, beta, pi
    cameraDir = [-math.cos(alpha) * math.sin(beta), -math.cos(beta), -math.sin(alpha) * math.sin(beta)]
    betaUp = beta + pi*0.5
    cameraUp = [math.cos(alpha) * math.sin(betaUp), math.cos(betaUp), math.sin(alpha) * math.sin(betaUp)]
    cameraRight = np.cross(cameraDir, cameraUp).tolist()
    cameraPos = [-cameraDir[0]*radius, -cameraDir[1]*radius, -cameraDir[2]*radius]
    uniforms["cameraDir"] = cameraDir
    uniforms["cameraUp"] = cameraUp
    uniforms["cameraRight"] = cameraRight
    uniforms["cameraPosition"] = cameraPos
    return

lastMouseX = 0
lastMouseY = 0
isButtonDown = False
def mouseEvent(e : sgl.MouseEvent):
    global lastMouseX, lastMouseY, isButtonDown, alpha, beta
    if e.button == sgl.MouseButton.left:
        isButtonDown = e.is_button_down()
    if e.is_move():
        if isButtonDown:
            dx = e.pos.x - lastMouseX
            dy = e.pos.y - lastMouseY
            alpha += dx*0.01
            beta += dy*0.01
            updateCamera()
        lastMouseX = e.pos.x
        lastMouseY = e.pos.y
    return

app.on_mouse_event = mouseEvent
updateCamera()

device = app.device
pathname = os.path.realpath(__file__)
imageData = imageio.v3.imread("imageio:bricks.jpg")

# reshape imageData into 512x512x4, adding an alpha channel
if imageData.shape[2] == 3:
    imageData = np.concatenate([imageData, np.ones((512, 512, 1), dtype=np.uint8) * 255], axis=2)

tex = device.create_texture(width=512, height=512, format=sgl.Format.rgba8_unorm,
                            usage=sgl.ResourceUsage.shader_resource | sgl.ResourceUsage.unordered_access,
                            data=imageData)
samplerState = device.create_sampler()

while app.process_events():
    uniforms["screenSize"] = sgl.float2(app._window.width, app._window.height)
    module.raytraceScene(call_id(), uniforms, tex, samplerState, _result=app.output)
    app.present()
