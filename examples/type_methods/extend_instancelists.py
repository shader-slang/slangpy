# SPDX-License-Identifier: Apache-2.0
import sgl
import slangpy as spy
import pathlib
import numpy as np

# Create an SGL device with the local folder for slangpy includes
device = spy.create_device(include_paths=[
    pathlib.Path(__file__).parent.absolute(),
])

# Load module
module = spy.Module.load_from_file(device, "example.slang")

# Custom type that wraps an InstanceList of particles


class MyParticles(spy.InstanceList):

    def __init__(self, name: str, count: int):
        super().__init__(module.Particle.as_struct())
        self.name = name
        self.position = spy.NDBuffer(device, dtype=module.float3, shape=(count,))
        self.velocity = spy.NDBuffer(device, dtype=module.float3, shape=(count,))

    def print_particles(self):
        print(self.name)
        print(self.position.to_numpy().view(dtype=np.float32).reshape(-1, 3))
        print(self.velocity.to_numpy().view(dtype=np.float32).reshape(-1, 3))


# Create buffer of particles (.as_struct is used to make python typing happy!)
particles = MyParticles("particle buffer", 10)

# Construct every particle with position of 0, and use slangpy's rand_float
# functionality to supply a different rand vector for each one.
particles.construct(
    p=sgl.float3(0),
    v=spy.rand_float(-1, 1, 3)
)

# Print all the particles by breaking them down into groups of 6 floats
particles.print_particles()

# Update the particles
particles.update(0.1)

# Print all the particles by breaking them down into groups of 6 floats
particles.print_particles()
