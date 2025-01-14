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

# Add 2 identically shaped 2d float buffers
a = np.random.rand(10, 5).astype(np.float32)
b = np.random.rand(10, 5).astype(np.float32)
res = module.add_floats(a, b, _result='numpy')
print(f"A Shape:   {a.shape}")
print(f"B Shape:   {b.shape}")
print(f"Res Shape: {res.shape}")
print("")

# Add the same value to all of the elements of a 2d float buffer
a = np.random.rand(10, 5).astype(np.float32)
b = 10
res = module.add_floats(a, b, _result='numpy')
print(f"A Shape:   {a.shape}")
print(f"Res Shape: {res.shape}")
print("")


# Dimension 1 of A is broadcast
a = np.random.rand(10, 1).astype(np.float32)
b = np.random.rand(10, 5).astype(np.float32)
res = module.add_floats(a, b, _result='numpy')
print(f"A Shape:   {a.shape}")
print(f"B Shape:   {b.shape}")
print(f"Res Shape: {res.shape}")
print("")

# Dimension 0 of A is broadcast
a = np.random.rand(1, 5).astype(np.float32)
b = np.random.rand(10, 5).astype(np.float32)
res = module.add_floats(a, b, _result='numpy')
print(f"A Shape:   {a.shape}")
print(f"B Shape:   {b.shape}")
print(f"Res Shape: {res.shape}")
print("")

# Dimension 0 of A and 1 of B are broadcast
a = np.random.rand(1, 5).astype(np.float32)
b = np.random.rand(10, 1).astype(np.float32)
res = module.add_floats(a, b, _result='numpy')
print(f"A Shape:   {a.shape}")
print(f"B Shape:   {b.shape}")
print(f"Res Shape: {res.shape}")
print("")

# Add a float3 and an array of 3 floats!
a = sgl.float3(1, 2, 3)
b = np.random.rand(3).astype(np.float32)
res = module.add_floats(a, b, _result='numpy')
print(f"A Shape:   {a.shape}")
print(f"B Shape:   {b.shape}")
print(f"Res Shape: {res.shape}")
print("")

# Should get a shape mismatch error, as slangpy won't 'pad' dimensions
try:
    a = np.random.rand(3).astype(np.float32)
    b = np.random.rand(5, 3).astype(np.float32)
    res = module.add_floats(a, b, _result='numpy')
except ValueError as e:
    # print(e)
    pass

# Now using add_vectors(float3, float3), no shape mismatch error
# as a is treated as a single float3, and b is an array of 5 float3s,
# and SlangPy will auto-pad single values.
a = np.random.rand(3).astype(np.float32)
b = np.random.rand(5, 3).astype(np.float32)
res = module.add_vectors(a, b, _result='numpy')
print(f"A Shape:   {a.shape}")
print(f"B Shape:   {b.shape}")
print(f"Res Shape: {res.shape}")
print("")

# Create a sampler and texture
sampler = device.create_sampler()
tex = device.create_texture(width=32, height=32, format=sgl.Format.rgb32_float,
                            usage=sgl.ResourceUsage.shader_resource)
tex.from_numpy(np.random.rand(32, 32, 3).astype(np.float32))

# Sample the texture at a single UV coordinate. Results in 1 thread,
# as the uv coordinate input is a single float 2.
a = sgl.float2(0.5, 0.5)
res = module.sample_texture_at_uv(a, sampler, tex, _result='numpy')
print(f"A Shape: {a.shape}")
print(f"Res Shape: {res.shape}")

# Sample the texture at 20 UV coordinates. Results in 20 threads.
# Although the texture has shape [32,32,3] (32x32 pixels of float3s),
# in this case it acts as a single value, as it is being passed to
# a function that takes an [n,m,3] structure (a float3 texture). As a
# result, the texture is effectively *broadcast* to all threads.
a = np.random.rand(20, 2).astype(np.float32)
res = module.sample_texture_at_uv(a, sampler, tex, _result='numpy')
print(f"A Shape: {a.shape}")
print(f"Res Shape: {res.shape}")
