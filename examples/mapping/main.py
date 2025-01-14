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

# --------------------------------------------------------------
# Basic explicit mapping
# --------------------------------------------------------------

# Map all dimensions of a and b to the same dimensions of the result
# Exactly the same as default behaviour
a = np.random.rand(10, 3, 4)
b = np.random.rand(10, 3, 4)
result = module.add.map((0, 1, 2), (0, 1, 2))(a, b, _result='numpy')
assert np.allclose(result, a+b)

# The same using named parameters
a = np.random.rand(10, 3, 4)
b = np.random.rand(10, 3, 4)
result = module.add.map(a=(0, 1, 2), b=(0, 1, 2))(a=a, b=b, _result='numpy')
assert np.allclose(result, a+b)

# --------------------------------------------------------------
# Broadcasting by mapping lower dimensional arguments
# --------------------------------------------------------------

# Explicitly map lower dimensional value 'b'
a = np.random.rand(8, 8).astype(np.float32)
b = np.random.rand(8).astype(np.float32)

# This is equivalent to padding b with empty dimensions, as numpy would
# result[i,j] = a[i,j] + b[j]
result = module.add.map(a=(0, 1), b=(1,))(a=a, b=b, _result='numpy')
assert np.allclose(result, a+b)

# The same thing (didn't need to specify a as 1-to-1 mapping is default)
result = module.add.map(b=(1,))(a=a, b=b, _result='numpy')
assert np.allclose(result, a+b)

# --------------------------------------------------------------
# Mathematical outer product by mapping to different dimensions
# --------------------------------------------------------------

# Slang multiply function has signature multiply(float a, float b)
# a is mapped to dimension 0, giving kernel dimension [0] size 10
# b is mapped to dimension 1, giving kernel dimension [1] size 20
# overall kernel (and thus result) shape is (10,20)
# result[i,j] = a[i] * b[j]
a = np.random.rand(10).astype(np.float32)
b = np.random.rand(20).astype(np.float32)
result = module.multiply.map(a=(0,), b=(1,))(a=a, b=b, _result='numpy')
assert np.allclose(result, np.outer(a, b))

# --------------------------------------------------------------
# Mathematical transpose by re-ordering dimensions
# --------------------------------------------------------------

# Slang copy function has signature float copy(float val)
# and just returns the value you pass it.
# result[i,j] = a[j,i]
a = np.random.rand(10, 20).astype(np.float32)
result = module.copy.map(val=(1, 0))(val=a, _result='numpy')
assert np.allclose(result, a.T)

# --------------------------------------------------------------
# Resolve generic function by mapping arguments explicitly
# --------------------------------------------------------------

# Map argument types explicitly
src = np.random.rand(100).astype(np.float32)
dest = np.zeros_like(src)
module.copy_generic.map(src=(0,), dest=(0,))(
    src=src,
    dest=dest
)
assert np.allclose(src, dest)

# --------------------------------------------------------------
# Resolve generic function by mapping argument types
# --------------------------------------------------------------

src = np.random.rand(100).astype(np.float32)
dest = np.zeros_like(src)
module.copy_generic.map(src='float', dest='float')(
    src=src,
    dest=dest
)
assert np.allclose(src, dest)
