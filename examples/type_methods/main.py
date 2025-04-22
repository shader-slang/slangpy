# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import sgl
import slangpy as spy
import pathlib
import numpy as np

print("SlangPy type methods example (https://slangpy.shader-slang.org/en/latest/typemethods.html)")

# Create an SGL device with the local folder for slangpy includes
device = spy.create_device(include_paths=[
    pathlib.Path(__file__).parent.absolute(),
])

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
result_cursor = particles.buffer.cursor()

for i in range(particles.shape[0]):
    instance = result_cursor[i].read()
    print(f"{instance['position']} {instance['velocity']}")

# Update the particles
particles.update(0.1)

result_cursor = particles.buffer.cursor()
# Print all the particles by breaking them down into groups of 6 floats
for i in range(particles.shape[0]):
    instance = result_cursor[i].read()
    print(f"{instance['position']} {instance['velocity']}")
