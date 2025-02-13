# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from app import App
import slangpy as spy
import sgl
from slangpy.types import call_id

# This Camera class mimics the Camera struct from rasterizer2d.slang, setting
# the origin, scale, and window size that the Slang camera struct expects. The
# window size is retrieved from the app window to avoid redundant state.


class Camera:
    # Origin is at 0,0 with 1,1 scale.
    def __init__(self, app):
        self.o = sgl.float2(0.0, 0.0)
        self.scale = sgl.float2(1.0, 1.0)
        self.app = app

    # Return a dict with the class variables mapped to the names that the
    # Slang struct expects.
    def get_this(self):
        return {
            "o": self.o,
            "scale": self.scale,
            "frameDim": sgl.float2(self.app._window.width, self.app._window.height),
            "_type": "Camera"
        }


# Create app and load the rasterizer2d shader.
app = App()
rasterizer2d = spy.Module.load_from_file(app.device, "rasterizer2d.slang")

# Buffer of 3 vertex positions for the triangle.
vertices = [
    sgl.float2(-0.75,  0.75),
    sgl.float2(-0.75, -0.75),
    sgl.float2(0.75, -0.75)
]

# Setup the camera with the app's frame dimensions.
camera = Camera(app)

# Run the app.
while app.process_events():
    # Call the rasterize function in Slang, passing the camera and vertices
    # array. We also use call_id to pass the pixel coordinate within window
    # width and height, and set result to render the output color returned by
    # the rasterize function.
    rasterizer2d.rasterize(camera, vertices, call_id(), _result=app.output)
    app.present()
