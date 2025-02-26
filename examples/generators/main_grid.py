# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import slangpy as spy
import pathlib
import numpy as np

print("SlangPy grid generator example (https://slangpy.shader-slang.org/en/latest/generator_grid.html)")

# Create an SGL device with the local folder for slangpy includes
device = spy.create_device(include_paths=[
    pathlib.Path(__file__).parent.absolute(),
])

# Load module
module = spy.Module.load_from_file(device, "ids.slang")

# Populate a 4x4 numpy array of int2s with call ids
res = module.myfunc(spy.grid(shape=(4, 4)), _result='numpy')

# [ [ [0,0], [1,0], [2,0], [3,0] ], [ [0,1], [1,1], [2,1], [3,1] ], ...
print(res)

# Populate a 4x4 numpy array of int2s with call ids
res = module.myfunc(spy.grid(shape=(4, 4), stride=(2, 2)), _result='numpy')

# [ [ [0,0], [2,0], [4,0], [6,0] ], [ [0,2], [2,2], [4,2], [6,2] ], ...
print(res)

# Don't fix the grid's shape, but specify an explicit stride and
# provide a pre-allocated numpy array to populate.
res = np.zeros((4, 4, 2), dtype=np.int32)
module.myfunc(spy.grid(shape=(-1, -1), stride=(4, 4)), _result=res)

# [ [ [0,0], [4,0], [8,0], [12,0] ], [ [0,4], [4,4], [8,4], [12,4] ], ...
print(res)
