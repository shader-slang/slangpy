# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sgl
import slangpy as spy
import pathlib
import numpy as np

print("SlangPy textures example (https://slangpy.shader-slang.org/en/latest/textures.html)")
print("This example requires tev (https://github.com/Tom94/tev) to display results.")

# Create an SGL device with the local folder for slangpy includes
device = spy.create_device(include_paths=[
    pathlib.Path(__file__).parent.absolute(),
])

# Load module
module = spy.Module.load_from_file(device, "example.slang")

# Generate a random image
rand_image = np.random.rand(128*128*4).astype(np.float32)*0.25
tex = device.create_texture(width=128, height=128, format=sgl.Format.rgba32_float,
                            usage=sgl.TextureUsage.shader_resource | sgl.TextureUsage.unordered_access,
                            data=rand_image)

# Display it with tev
sgl.tev.show(tex, name='photo')

# Call the module's add function, passing:
# - a float4 constant that'll be broadcast to every pixel
# - the texture to an inout parameter
module.brighten(sgl.float4(0.5), tex)

# Show the result
sgl.tev.show(tex, name='brighter')
