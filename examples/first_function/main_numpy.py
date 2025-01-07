# SPDX-License-Identifier: Apache-2.0

import sgl
import slangpy as spy
import pathlib
import numpy as np

# Create an SGL device with the slangpy+local include paths
device = sgl.Device(compiler_options={
    "include_paths": [
        spy.SHADER_PATH,
        pathlib.Path(__file__).parent.absolute(),
    ],
})

# Create a simple function
module = spy.Module.load_from_file(device, "example.slang")

# Create a couple of buffers with 1,000,000 random floats in
a = np.random.rand(1000000).astype(np.float32)
b = np.random.rand(1000000).astype(np.float32)

# Call our function and ask for a numpy array back (the default would be a buffer)
result = module.add(a, b, _result='numpy')

# Print the first 10
print(result[:10])
