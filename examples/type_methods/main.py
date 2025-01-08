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

# Load module
module = spy.Module.load_from_file(device, "example.slang")

# Create buffer of particles (.as_struct is used to make python typing happy!)
particles = spy.InstanceBuffer(
    struct=module.Particle.as_struct(),
    shape=(10,))

# Construct every particle with position of 0, and use slangpy's rand_float
# functionality to supply a different rand vector for each one.
particles.construct(
    p=sgl.float3(0),
    v=spy.rand_float(-1, 1, 3)
)

# Print all the particles by breaking them down into groups of 6 floats
print(particles.to_numpy().view(dtype=np.float32).reshape(-1, 6))

# Update the particles
particles.update(0.1)

# Print all the particles by breaking them down into groups of 6 floats
print(particles.to_numpy().view(dtype=np.float32).reshape(-1, 6))
