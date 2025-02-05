# SPDX-License-Identifier: Apache-2.0

import slangpy as spy
import pathlib
import torch

# Make sure pytorch is in cuda mode
torch.device('cuda')

# Create an SGL device with the local folder for slangpy includes
device = spy.create_device(include_paths=[
    pathlib.Path(__file__).parent.absolute(),
])

# Load torch wrapped module.
module = spy.TorchModule.load_from_file(device, "example.slang")

# Create a tensor
x = torch.tensor([1, 2, 3, 4], dtype=torch.float32, device='cuda', requires_grad=True)

# Evaluate the polynomial. Result will now default to a torch tensor.
# Expecting result = 2x^2 + 8x - 1
result = module.polynomial(a=2, b=8, c=-1, x=x)
print(result)

# Run backward pass on result, using result grad == 1
# to get the gradient with respect to x
result.backward(torch.ones_like(result))
print(x.grad)
