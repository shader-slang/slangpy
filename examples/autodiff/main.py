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
# Note: We pass zero=True to initialize the grads to zero on allocation
x = spy.Tensor.numpy(device, np.array([1, 2, 3, 4], dtype=np.float32)).with_grads(zero=True)

# Evaluate the polynomial and ask for a tensor back
# Expecting result = 2x^2 + 8x - 1
result: spy.Tensor = module.polynomial(a=2, b=8, c=-1, x=x, _result='tensor')
print(result)

# Attach gradients to the result, and set them to 1 for the backward pass
result = result.with_grads()
result.grad.storage.from_numpy(np.array([1, 1, 1, 1], dtype=np.float32))

# Call the backwards version of module.polynomial
# This will read the grads from _result, and write the grads to x
# Expecting result = 4x + 8
module.polynomial.bwds(a=2, b=8, c=-1, x=x, _result=result)
print(x.grad.to_numpy())
