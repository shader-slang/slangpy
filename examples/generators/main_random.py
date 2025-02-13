# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import slangpy as spy
import pathlib
import numpy as np

print("SlangPy random number generator examples (https://slangpy.shader-slang.org/en/latest/generator_random.html)")

# Create an SGL device with the local folder for slangpy includes
device = spy.create_device(include_paths=[
    pathlib.Path(__file__).parent.absolute(),
])

# Load module
module = spy.Module.load_from_file(device, "random.slang")

# Populate a 4x4 numpy array of int2s with random integer hashes
res = np.zeros((4, 4, 2), dtype=np.int32)
module.myfunc(spy.wang_hash(), _result=res)

# [-1062647446,-1219659480], [663891101,1738326990] ...
print(res)


# Populate a 4x4 numpy array of float2s with random values
res = np.zeros((4, 4, 2), dtype=np.float32)
module.myfuncfloat(spy.rand_float(min=0, max=10), _result=res)

# [3.0781631,3.6783838], [3.2699034, 4.611035] ...
print(res)
