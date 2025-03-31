# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sgl
import slangpy as spy
import pathlib

print("SlangPy nested types example (https://slangpy.shader-slang.org/en/latest/nested.html)")
print("This example requires tev (https://github.com/Tom94/tev) to display results.")

# Create an SGL device with the local folder for slangpy includes
device = spy.create_device(include_paths=[
    pathlib.Path(__file__).parent.absolute(),
])

# Load module
module = spy.Module.load_from_file(device, "example.slang")

# Create a texture to store the results
tex = device.create_texture(width=128, height=128, format=sgl.Format.rgba32_float,
                            usage=sgl.TextureUsage.shader_resource | sgl.TextureUsage.unordered_access)

# Tell slangpy that the src and dest types map to a float4
module.copy_vector(
    src={
        '_type': 'float4',
        'x': 1.0,
        'y': spy.rand_float(min=0, max=1, dim=1),
        'z': 0.0,
        'w': 1.0
    },
    dest=tex)

# Show the result
sgl.tev.show(tex, name='tex')
