# SPDX-License-Identifier: Apache-2.0

import slangpy as spy
import pathlib
import numpy as np

# Create an SGL device with the local folder for slangpy includes
device = spy.create_device(include_paths=[
    pathlib.Path(__file__).parent.absolute(),
])

# Load module
module = spy.Module.load_from_file(device, "example.slang")

# Create a tensor with attached grads from a numpy array
tensor = spy.Tensor.numpy(device, np.array([1, 2, 3, 4], dtype=np.float32)).with_grads()

# evaluate the polynomial
result = module.polynomial(a=2, b=8, c=-1, x=tensor)
print(result)
