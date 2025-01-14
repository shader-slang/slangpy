# SPDX-License-Identifier: Apache-2.0

import sgl
import slangpy as spy
import pathlib

# Create an SGL device with the local folder for slangpy includes
device = spy.create_device(include_paths=[
    pathlib.Path(__file__).parent.absolute(),
])

# Load module
module = spy.Module.load_from_file(device, "example.slang")

# Create a texture to store the results
tex = device.create_texture(width=128, height=128, format=sgl.Format.rgba32_float,
                            usage=sgl.ResourceUsage.shader_resource | sgl.ResourceUsage.unordered_access)

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
