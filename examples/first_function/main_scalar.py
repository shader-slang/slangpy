# SPDX-License-Identifier: Apache-2.0

import sgl
import slangpy as spy
import pathlib

# Create an SGL device with the slangpy+local include paths
device = sgl.Device(compiler_options={
    "include_paths": [
        spy.SHADER_PATH,
        pathlib.Path(__file__).parent.absolute(),
    ],
})

# Create a simple function
module = spy.Module.load_from_file(device, "example.slang")

# Call the function and print the result
result = module.add(1.0, 2.0)
print(result)

# SlangPy also supports named parameters
result = module.add(a=1.0, b=2.0)
print(result)
