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

tensor = spy.Tensor.zeros(device, module, (3, 4),
                          spy.reflectiontypes.scalar_names[spy.reflectiontypes.ScalarType.float32])
