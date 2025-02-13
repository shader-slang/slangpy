# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from app import App
import slangpy as spy
import sgl
from slangpy.types import call_id


# This Camera class mimics the Camera struct from brdf.slang, setting
# the origin, scale, and window size that the Slang camera struct expects. The
# window size is retrieved from the app window to avoid redundant state.
class Camera:
    # Origin is at 0,0 with 1,1 scale.
    def __init__(self, app: App):
        super().__init__()
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

# This Properties class mimics the Properties struct from brdf.slang,
# setting the baseColor, roughness, metallic, and specular values that the
# Properties struct expects.


class Properties:
    def __init__(self, b: sgl.float3, r: float, m: float, s: float):
        super().__init__()
        self.b = b
        self.r = r
        self.m = m
        self.s = s

    # Return a dict mapping the values to the Slang struct names.
    def get_this(self):
        return {
            "baseColor": self.b,
            "roughness": self.r,
            "metallic": self.m,
            "specular": self.s,
            "_type": "Properties"
        }


# Create app and load the brdf shader.
app = App()
brdf = spy.Module.load_from_file(app.device, "brdf.slang")

# Setup the camera with the app's frame dimensions.
camera = Camera(app)

# BRDF lighting parameters.
properties = Properties(sgl.float3(0.2, 0.0, 1.0), 0.4, 0.6, 1.0)

# Run the app.
while app.process_events():
    # Call the computeSphereBRDF function in Slang, passing the camera and BRDF
    # parameters. We also use call_id to pass the pixel coordinate within window
    # width and height, and set result to render the output color returned by
    # the computeSphereBRDF function.
    brdf.computeSphereBRDF(camera, properties, call_id(), _result=app.output)
    app.present()
