.. _tensors_python:

Tensors In Python
===================

Introduction
------------

The ``Tensor`` type is SlangPy's primary multidimensional container, providing functionality similar to NumPy arrays or PyTorch tensors. It represents an N-dimensional view of GPU memory with a specified element type and shape.

A ``Tensor`` consists of:

- **Storage**: An underlying ``Buffer`` that holds the actual GPU memory
- **Data type**: A ``SlangType`` describing the element type (primitives like ``float``, ``int``, or user-defined Slang structs)
- **Shape**: A tuple defining the size of each dimension
- **Strides**: A tuple defining the memory layout (defaults to row-major/contiguous)
- **Offset**: An element offset into the storage buffer (defaults to 0)
- **Gradients**: Optional gradient storage for automatic differentiation

Creating Tensors
----------------

The ``Tensor`` class provides several factory methods for creating tensors:

**empty** - Create an uninitialized tensor:

.. code-block:: python

    import slangpy as spy

    device = spy.create_device()
    module = spy.Module.load_from_file(device, "shader.slang")

    # Create a 1D tensor of floats
    tensor = spy.Tensor.empty(device, shape=(100,), dtype=float)

    # Create a 2D tensor of a custom struct type
    tensor = spy.Tensor.empty(device, shape=(64, 64), dtype=module.Pixel)

    # Specify custom buffer usage flags
    tensor = spy.Tensor.empty(
        device,
        shape=(256, 256),
        dtype="float4",
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
    )

**zeros** - Create a zero-initialized tensor:

.. code-block:: python

    # Create a 3D tensor initialized to zero
    tensor = spy.Tensor.zeros(device, shape=(32, 32, 32), dtype=float)

**from_numpy** - Create a tensor from a NumPy array:

.. code-block:: python

    import numpy as np

    # Create from numpy array, copying data to GPU
    data = np.random.rand(100, 100).astype(np.float32)
    tensor = spy.Tensor.from_numpy(device, data)

    # The tensor will have the same shape and dtype as the numpy array
    print(tensor.shape)  # Shape(100, 100)

**empty_like / zeros_like** - Create tensors matching another tensor:

.. code-block:: python

    original = spy.Tensor.empty(device, shape=(10, 20), dtype=float)

    # Create uninitialized tensor with same shape and dtype
    copy = spy.Tensor.empty_like(original)

    # Create zero-initialized tensor with same shape and dtype
    zeros = spy.Tensor.zeros_like(original)

**load_from_image** - Load an image file as a tensor:

.. code-block:: python

    # Load an image as a floating-point tensor
    texture = spy.Tensor.load_from_image(
        device,
        "image.png",
        flip_y=True,      # Flip vertically
        linearize=False,  # Apply sRGB to linear conversion
        scale=1.0,        # Scale values
        offset=0.0,       # Offset values
        grayscale=False   # Convert to grayscale
    )

    # Result will be float, float2, float3, or float4 depending on channels

Although it is not typically recommended, it is also possible to construct a tensor directly from an existing buffer through
the use of its constructor. This behaviour may be replaced with a factory method in the future:

.. code-block:: python

    # Create a buffer manually
    buffer = device.create_buffer(
        element_count=100,
        struct_size=4,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
    )

    # Wrap it in a Tensor
    tensor = spy.Tensor(
        storage=buffer,
        dtype=float,
        shape=(10, 10),
        strides=None,  # Use default row-major layout
        offset=0
    )

Tensor Properties
-----------------

Tensors expose several read-only properties:

.. code-block:: python

    tensor = spy.Tensor.empty(device, shape=(10, 20, 30), dtype=float)

    # Core properties
    print(tensor.device)         # Device this tensor is allocated on
    print(tensor.dtype)          # SlangType of elements
    print(tensor.shape)          # Shape(10, 20, 30)
    print(tensor.strides)        # Shape(600, 30, 1) - row-major strides
    print(tensor.offset)         # 0 - offset into storage buffer
    print(tensor.element_count)  # 6000 - total number of elements

    # Storage
    print(tensor.storage)        # Underlying Buffer object
    print(tensor.usage)          # BufferUsage flags
    print(tensor.memory_type)    # MemoryType (device_local, etc.)

    # Gradient storage (see Gradient Storage section)
    print(tensor.grad_in)        # Input gradient tensor (or None)
    print(tensor.grad_out)       # Output gradient tensor (or None)
    print(tensor.grad)           # Convenience property (returns grad_out or grad_in)

Data Transfer
-------------

**to_numpy** - Copy tensor data to a NumPy array:

.. code-block:: python

    tensor = spy.Tensor.zeros(device, shape=(10, 10), dtype=float)

    # Copy to numpy array with matching shape
    array = tensor.to_numpy()
    print(array.shape)  # (10, 10)
    print(array.dtype)  # np.float32

    # For struct types, returns raw bytes
    struct_tensor = spy.Tensor.empty(device, shape=(5,), dtype=module.MyStruct)
    array = struct_tensor.to_numpy()  # shape=(5, sizeof(MyStruct)), dtype=uint8

**to_torch** - Create a PyTorch tensor view (zero-copy when possible):

.. code-block:: python

    import torch

    tensor = spy.Tensor.zeros(device, shape=(100, 100), dtype=float)

    # Create torch tensor sharing the same GPU memory
    torch_tensor = tensor.to_torch()
    print(torch_tensor.shape)  # torch.Size([100, 100])
    print(torch_tensor.device)  # cuda:0 (or cpu depending on device type)

**copy_from_numpy** - Copy data from a NumPy array:

.. code-block:: python

    tensor = spy.Tensor.empty(device, shape=(100,), dtype=float)

    # Copy data from numpy array
    data = np.random.rand(100).astype(np.float32)
    tensor.copy_from_numpy(data)

    # Array shape must match or be broadcastable
    tensor2 = spy.Tensor.empty(device, shape=(10, 10), dtype=float)
    tensor2.copy_from_numpy(data)  # OK - can reshape 100 elements to 10x10

**copy_from_torch** - Copy data from a PyTorch tensor:

.. code-block:: python

    import torch

    tensor = spy.Tensor.empty(device, shape=(100,), dtype=float)

    # Copy from torch tensor
    torch_data = torch.randn(100, device='cuda')
    tensor.copy_from_torch(torch_data)

Tensor Operations
-----------------

Views and Reshaping
~~~~~~~~~~~~~~~~~~~

**view** - Create a new view with different shape/strides:

.. code-block:: python

    # Create a 1D tensor
    tensor = spy.Tensor.from_numpy(device, np.arange(12, dtype=np.float32))

    # Reshape to 2D view (shares same storage)
    view_2d = tensor.view(shape=(3, 4))
    print(view_2d.shape)  # Shape(3, 4)

    # Custom strides and offset
    view = tensor.view(shape=(6,), strides=(2,), offset=0)  # Every other element

**broadcast_to** - Broadcast tensor to a larger shape:

.. code-block:: python

    # Create a 1D tensor
    tensor = spy.Tensor.from_numpy(device, np.array([1, 2, 3], dtype=np.float32))

    # Broadcast to 2D (shape will be (5, 3), but storage is shared)
    broadcasted = tensor.broadcast_to(shape=(5, 3))
    print(broadcasted.shape)  # Shape(5, 3)

.. note::
   Broadcasting creates a view with modified strides but doesn't copy data.
   Broadcasted dimensions have stride 0.

Indexing
~~~~~~~~

The subscript operator provides a range of indexing capabilities, similar to those of numpy and PyTorch
for accessing or slicing up a tensor. Note that in all the cases below, the new tensor is simply a view
onto the original tensor's storage, and no data is copied.

.. code-block:: python

    tensor = spy.Tensor.zeros(device, shape=(10, 20, 30), dtype=float)

    # Single index - select first element along first dimension
    sub = tensor[0]       # Shape(20, 30)

    # Slice notation
    sub = tensor[1:5]     # Shape(4, 20, 30)
    sub = tensor[:, 10:]  # Shape(10, 10, 30)

    # Step/stride
    sub = tensor[::2]     # Shape(5, 20, 30) - every other element

Buffer Cursors
~~~~~~~~~~~~~~

Cursors provide a convenient way to read and write structured data:

.. code-block:: python

    # Create a tensor of custom structs
    tensor = spy.Tensor.empty(device, shape=(10,), dtype=module.Pixel)

    # Get a cursor for reading/writing
    cursor = tensor.cursor()

    # Write data
    for i in range(10):
        cursor[i].write({'r': i * 0.1, 'g': 0.5, 'b': 1.0})

    # Apply changes (uploads to GPU)
    cursor.apply()

    # Read data back
    cursor = tensor.cursor()
    for i in range(10):
        pixel = cursor[i].read()
        print(f"Pixel {i}: r={pixel['r']}, g={pixel['g']}, b={pixel['b']}")

    # Cursors can also read a range of elements
    cursor = tensor.cursor(start=2, count=5)  # Elements 2-6

Utility Operations
~~~~~~~~~~~~~~~~~~

**clear** - Zero out tensor contents:

.. code-block:: python

    tensor = spy.Tensor.empty(device, shape=(100, 100), dtype=float)

    # Clear with automatic command submission
    tensor.clear()

    # Or use an existing command encoder
    encoder = device.create_command_encoder()
    tensor.clear(command_encoder=encoder)
    # ... other commands ...
    device.submit_command_encoder(encoder)

**is_contiguous** - Check if tensor has contiguous memory layout:

.. code-block:: python

    tensor = spy.Tensor.empty(device, shape=(10, 20), dtype=float)
    print(tensor.is_contiguous())  # True1

    # After slicing, may not be contiguous
    sliced = tensor[::2]
    print(sliced.is_contiguous())  # False

**uniforms** - Get uniform buffer representation:

.. code-block:: python
1
    # Create uniform buffer from tensor metadata
    # Useful for passing tensor parameters to shaders
    uniforms = tensor.uniforms()

PyTorch Comparison
------------------

Many ``Tensor`` operations have PyTorch equivalents:

+----------------------------------+------------------------------------------+------------------------------------+
| SlangPy                          | PyTorch                                  | Notes                              |
+==================================+==========================================+====================================+
| ``Tensor.empty(device, ...)``    | ``torch.empty(..., device=device)``      | Both create uninitialized tensors  |
+----------------------------------+------------------------------------------+------------------------------------+
| ``Tensor.zeros(device, ...)``    | ``torch.zeros(..., device=device)``      | Both create zero-initialized       |
+----------------------------------+------------------------------------------+------------------------------------+
| ``Tensor.from_numpy(device, a)`` | ``torch.from_numpy(a).to(device)``       | SlangPy copies, PyTorch can share  |
+----------------------------------+------------------------------------------+------------------------------------+
| ``tensor.to_numpy()``            | ``tensor.cpu().numpy()``                 | Both copy to CPU                   |
+----------------------------------+------------------------------------------+------------------------------------+
| ``tensor.to_torch()``            | N/A                                      | Creates PyTorch view of SlangPy    |
+----------------------------------+------------------------------------------+------------------------------------+
| ``tensor.view(shape)``           | ``tensor.view(shape)``                   | Similar reshaping semantics        |
+----------------------------------+------------------------------------------+------------------------------------+
| ``tensor.broadcast_to(shape)``   | ``tensor.expand(shape)``                 | Both create broadcasted views      |
+----------------------------------+------------------------------------------+------------------------------------+
| ``tensor[idx]``                  | ``tensor[idx]``                          | Similar indexing syntax            |
+----------------------------------+------------------------------------------+------------------------------------+
| ``tensor.clear()``               | ``tensor.zero_()``                       | Both zero out contents             |
+----------------------------------+------------------------------------------+------------------------------------+
| ``tensor.with_grads()``          | ``tensor.requires_grad_(True)``          | Enable gradient tracking           |
+----------------------------------+------------------------------------------+------------------------------------+
| ``tensor.detach()``              | ``tensor.detach()``                      | Remove from gradient graph         |
+----------------------------------+------------------------------------------+------------------------------------+
| ``tensor.grad``                  | ``tensor.grad``                          | Access gradient tensor             |
+----------------------------------+------------------------------------------+------------------------------------+

The key differences to be aware of are:

- SlangPy tensors always live on the GPU device they were created on. There is no concept of a 'cpu' tensor in SlangPy.
- SlangPy supports arbitrary Slang struct types as elements, not just numeric types.

Vectorization in Kernel Calls
------------------------------

When passing tensors to Slang functions, SlangPy automatically vectorizes the call based on the tensor's shape and the function's parameter types. This is one of SlangPy's most powerful features.

Basic Vectorization
~~~~~~~~~~~~~~~~~~~~

The simplest case is passing a tensor where the element type matches the parameter type:

.. code-block:: python

    # Slang function
    # float square(float x) { return x * x; }

    # Python
    input = spy.Tensor.from_numpy(device, np.array([1, 2, 3, 4], dtype=np.float32))
    result = module.square(input)

    # SlangPy generates a kernel that:
    # - Dispatches 4 threads
    # - Thread i loads input[i], calls square(), writes to result[i]

Multi-dimensional tensors automatically infer the dispatch shape:

.. code-block:: python

    # 2D tensor of shape (10, 20)
    input = spy.Tensor.empty(device, shape=(10, 20), dtype=float)
    result = module.square(input)

    # SlangPy dispatches a 2D grid of (10, 20) threads

Mapping to Array Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensors can map to Slang array parameters, loading multiple elements per thread:

.. code-block:: python

    # Slang function
    # Particle sum_particles(Particle particles[5]) { ... }

    # Python - tensor of shape (10, 5)
    particles = spy.Tensor.empty(device, shape=(10, 5), dtype=module.Particle)
    result = module.sum_particles(particles)

    # SlangPy generates a kernel that:
    # - Dispatches 10 threads
    # - Thread i loads particles[i*5:(i+1)*5] as an array of 5 elements
    # - Calls sum_particles() with that array

This works because the trailing dimension (5) matches the array size, and the leading dimensions (10) determine the dispatch shape.

Mapping to Vector/Matrix Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensors of basic types can map to vector or matrix types:

.. code-block:: python

    # Slang function
    # float dot_product(float3 v) { return dot(v, v); }

    # Python - tensor of shape (100, 3)
    vectors = spy.Tensor.empty(device, shape=(100, 3), dtype=float)
    result = module.dot_product(vectors)

    # SlangPy generates a kernel that:
    # - Dispatches 100 threads
    # - Thread i loads vectors[i,:] as a float3
    # - Calls dot_product() with that vector

This also works for matrices:

.. code-block:: python

    # Slang function
    # float determinant(float2x2 m) { ... }

    # Python - tensor of shape (50, 2, 2)
    matrices = spy.Tensor.empty(device, shape=(50, 2, 2), dtype=float)
    result = module.determinant(matrices)

    # SlangPy dispatches 50 threads, each loading a 2x2 matrix

Mapping to Lower-Rank Tensors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensors can also map to lower-rank tensor parameters in Slang:

.. code-block:: python

    # Slang function
    # float sum_row(ITensor<float, 1> row) { ... }

    # Python - tensor of shape (10, 20)
    data = spy.Tensor.empty(device, shape=(10, 20), dtype=float)
    result = module.sum_row(data)

    # SlangPy generates a kernel that:
    # - Dispatches 10 threads
    # - Thread i receives a 1D tensor view of row i
    # - Calls sum_row() with that view

.. warning::
   While flexible, this mapping is less efficient than direct element access because it requires recalculating the layout
   of the lower rank tensor on the GPU for each thread.

Gradients
----------------

The Python ``Tensor`` type supports automatic differentiation by storaging an (optional) gradient tensor alongside the primal tensor.

Attaching Gradients
~~~~~~~~~~~~~~~~~~~

Use ``with_grads()`` to attach gradient tensors:

.. code-block:: python

    # Create a tensor and attach gradients
    x = spy.Tensor.from_numpy(device, np.array([1, 2, 3, 4], dtype=np.float32))
    x = x.with_grads(zero=True)

    # Now x has gradient storage
    print(x.grad_in)   # Gradient tensor (zero-initialized)
    print(x.grad_out)  # Gradient tensor (same as grad_in by default)
    print(x.grad)      # Convenience property (returns grad_out if available, else grad_in)

By default, ``with_grads()`` creates and zero-initializes a new gradient tensor with the same shape and dtype (derivative type) as the primal.

Accessing Gradients
~~~~~~~~~~~~~~~~~~~~

After running backward differentiation, gradients are stored in the attached tensors:

.. code-block:: python

    # Forward pass
    x = spy.Tensor.from_numpy(device, np.array([1, 2, 3, 4], dtype=np.float32))
    x = x.with_grads(zero=True)
    result = module.polynomial(a=2, b=8, c=-1, x=x)

    # Attach gradients to result and set to 1
    result = result.with_grads()
    result.grad.copy_from_numpy(np.ones(4, dtype=np.float32))

    # Backward pass
    module.polynomial.bwds(a=2, b=8, c=-1, x=x, _result=result)

    # Access gradients
    x_grad = x.grad.to_numpy()
    print(x_grad)  # Derivatives with respect to x

Separate Input/Output Gradients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For functions with ``inout`` parameters that both read and write, you may need separate input and output gradient buffers to avoid race conditions:

.. code-block:: python

    # Slang function with inout parameter
    # [Differentiable]
    # void modify_inplace(inout float x) { x = x * x; }

    # Create separate gradient buffers
    grad_in = spy.Tensor.zeros(device, shape=(100,), dtype=float)
    grad_out = spy.Tensor.zeros(device, shape=(100,), dtype=float)

    # Attach both
    x = spy.Tensor.from_numpy(device, data)
    x = x.with_grads(grad_in=grad_in, grad_out=grad_out, zero=True)

    # Forward pass
    module.modify_inplace(x)

    # Backward pass - reads from grad_in, writes to grad_out
    module.modify_inplace.bwds(x)

In most cases, the same buffer is used for both input and output gradients. Separate buffers are only needed when:

- The function has ``inout`` parameters
- You want to avoid accumulation hazards
- Debugging gradient flow

Detaching Gradients
~~~~~~~~~~~~~~~~~~~

Use ``detach()`` to create a view without gradient storage:

.. code-block:: python

    x = spy.Tensor.zeros(device, shape=(100,), dtype=float).with_grads()

    # Create detached view (no gradients)
    x_detached = x.detach()
    print(x_detached.grad_in)  # None
    print(x_detached.grad_out)  # None

    # Original still has gradients
    print(x.grad_in)  # <Tensor...>

This is useful for creating non-differentiable intermediate results.

.. note::
   SlangPy **always accumulates** gradients. Make sure to zero gradient buffers before backward passes, either by using ``zero=True`` in ``with_grads()`` or by calling ``tensor.grad.clear()``.

Summary
-------

The ``Tensor`` type provides:

- **Flexible creation** via factory methods (``empty``, ``zeros``, ``from_numpy``, etc.)
- **NumPy/PyTorch interop** with efficient data transfer
- **Views and reshaping** without copying data
- **Buffer cursors** for structured data access
- **Automatic vectorization** when calling Slang functions
- **Gradient storage** for automatic differentiation
- **Support for custom Slang types** beyond basic numeric types

For details on using tensors within Slang code, see :ref:`tensors_slang`. For information on differentiable tensors, see :ref:`tensors_differentiable`.
