# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import slangpy as spy
import pathlib
import numpy as np

print("SlangPy id generator examples (https://slangpy.shader-slang.org/en/latest/generator_ids.html)")

# Create an SGL device with the local folder for slangpy includes
device = spy.create_device(include_paths=[
    pathlib.Path(__file__).parent.absolute(),
])

# Load module
module = spy.Module.load_from_file(device, "ids.slang")

# Populate a 4x4 numpy array of int2s with call ids
res = np.zeros((4, 4, 2), dtype=np.int32)
module.myfunc(spy.call_id(), _result=res)

# [ [ [0,0], [1,0], [2,0], [3,0] ], [ [0,1], [1,1], [2,1], [3,1] ], ...
print(res)

# Do the same but with a function that takes an int[2] array as input
res = np.zeros((4, 4, 2), dtype=np.int32)
module.myfuncarray(spy.call_id(), _result=res)

# [ [ [0,0], [0,1], [0,2], [0,3] ], [ [1,0], [1,1], [1,2], [1,3] ], ...
print(res)

# Populate a 4x4 numpy array of int3s with hardware thread ids
res = np.zeros((4, 4, 3), dtype=np.int32)
module.myfunc3d(spy.thread_id(), _result=res)

# [ [ [0,0,0], [1,0,0], [2,0,0], [3,0,0] ], [ [4,0,0], [5,0,0], ...
print(res)
