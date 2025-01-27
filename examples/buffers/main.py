# SPDX-License-Identifier: Apache-2.0

import sgl
import slangpy as spy
import pathlib
import numpy as np

# Create an SGL device with the local folder for slangpy includes
device = spy.create_device(include_paths=[
    pathlib.Path(__file__).parent.absolute(),
])

# Load module
module = spy.Module.load_from_file(device, "example.slang")

# Create a couple of 2D 16x16 buffers
image_1 = spy.NDBuffer(device, dtype=module.Pixel, shape=(16, 16))
image_2 = spy.NDBuffer(device, dtype=module.Pixel, shape=(16, 16))

# Use a cursor to fill the first buffer with readable structured data.
cursor_1 = image_1.cursor()
for x in range(16):
    for y in range(16):
        cursor_1[x+y*16].write({
            'r': (x+y)/32.0,
            'g': 0,
            'b': 0,
        })
cursor_1.apply()

# Use the fact that we know the buffers are just 16x16 grids of 3 floats
# to just populate the 2nd buffer straight from random numpy array
image_2.from_numpy(0.1*np.random.rand(16*16*3).astype(np.float32))

# Call the module's add function
result = module.add(image_1, image_2)

# Use a cursor to read and print pixels (would also be readable in the watch window)
result_cursor = result.cursor()
for x in range(16):
    for y in range(16):
        pixel = result_cursor[x+y*16].read()
        print(f"Pixel ({x},{y}): {pixel}")

# Or if installed, we can use tev to show the result (https://github.com/Tom94/tev)
tex = device.create_texture(data=result.to_numpy(), width=16,
                            height=16, format=sgl.Format.rgb32_float)
sgl.tev.show(tex)
