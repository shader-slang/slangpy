# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import slangpy as spy
import pathlib

print("SlangPy first-function example (https://slangpy.shader-slang.org/en/latest/firstfunctions.html)")

# Create an SGL device with the local folder for slangpy includes
device = spy.create_device(include_paths=[
    pathlib.Path(__file__).parent.absolute(),
])

# Load module
module = spy.Module.load_from_file(device, "example.slang")

# Call the function and print the result
result = module.add(1.0, 2.0)
print(result)

# SlangPy also supports named parameters
result = module.add(a=1.0, b=2.0)
print(result)
