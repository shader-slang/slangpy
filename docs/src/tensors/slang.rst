.. _tensors_slang:

Tensors In Slang
===================

Introduction
------------

SlangPy provides a comprehensive set of tensor types for use within Slang shaders. These types allow you to work with multi-dimensional arrays on the GPU with different access patterns and capabilities. All tensor types are templates parameterized by an element type ``T`` and number of dimensions ``D``.

Tensor Type Overview
--------------------

Commonly Used Types
~~~~~~~~~~~~~~~~~~~

The four most commonly used tensor types are:

- **Tensor<T, D>** - Read-only tensor for loading data
- **RWTensor<T, D>** - Read-write tensor for loading and storing data
- **DiffTensor<T, D>** - Read-only differentiable tensor (for forward/backward passes)
- **WDiffTensor<T, D>** - Write-only differentiable tensor (for forward/backward passes)

These cover the majority of use cases for compute kernels and differentiable programming.

Complete Type List
~~~~~~~~~~~~~~~~~~

For specialized scenarios, additional tensor types are available:

**Non-differentiable tensors:**

- ``Tensor<T, D>`` - Read-only
- ``WTensor<T, D>`` - Write-only
- ``RWTensor<T, D>`` - Read-write
- ``AtomicTensor<T, D>`` - Read-write with atomic operations (requires ``T : IAtomicAddable``)

**Differentiable tensors:**

- ``DiffTensor<T, D>`` - Read-only with writable atomic output gradients
- ``WDiffTensor<T, D>`` - Write-only with readable input gradients
- ``RWDiffTensor<T, D>`` - Read-write with both gradients

**Primal-only differentiable tensors** (store only primal values, no separate gradient buffers):

- ``PrimalTensor<T, D>`` - Read-only primal tensor
- ``WPrimalTensor<T, D>`` - Write-only primal tensor
- ``RWPrimalTensor<T, D>`` - Read-write primal tensor

.. note::
   **What are PrimalTensor types for?**

   ``PrimalTensor`` types exist primarily as an internal mechanism to allow passing tensors *without* gradients to ``IDiffTensor`` interfaces.

   When you write a function accepting ``IDiffTensor<T, D>`` and call it with a Python tensor that has no gradients attached, SlangPy binds a ``PrimalTensor`` instead of a ``DiffTensor``. This avoids errors from trying to bind null gradient buffers and eliminates unnecessary binding overhead.

   Once upcoming Slang auto-diff improvements land, the need for separate ``IDiffTensor`` interfaces will be removed, and ``PrimalTensor`` types will no longer be necessary. At that point, the only difference between ``DiffTensor`` and ``Tensor`` will be that the former has gradient storage and the latter does not.

Interface Types
---------------

For maximum flexibility when writing reusable functions, SlangPy provides interface types that abstract over the concrete tensor implementations. **Using interfaces is strongly recommended for function parameters** because:

1. The generated kernel can choose the most efficient storage type (e.g., ``PrimalTensor`` vs ``DiffTensor``)
2. The same function can work for both forward and backward differentiation passes
3. Code is more generic and reusable, and will be compatible with future tensor types added to SlangPy.

Available tensor interfaces:

- ``ITensor<T, D>`` - Read-only tensor interface
- ``IWTensor<T, D>`` - Write-only tensor interface
- ``IRWTensor<T, D>`` - Read-write tensor interface
- ``IDiffTensor<T, D>`` - Read-only differentiable tensor interface
- ``IWDiffTensor<T, D>`` - Write-only differentiable tensor interface
- ``IRWDiffTensor<T, D>`` - Read-write differentiable tensor interface

.. code-block:: slang

    // Good: Uses interface types - works with any compatible tensor implementation
    void process_data(int2 idx, ITensor<float, 2> input, IRWTensor<float, 2> output)
    {
        float value = input[idx];
        output[idx] = value * 2.0;
    }

    // Less flexible: Requires specific tensor types
    void process_data_concrete(int2 idx, Tensor<float, 2> input, RWTensor<float, 2> output)
    {
        float value = input[idx];
        output[idx] = value * 2.0;
    }

When SlangPy generates a kernel that calls a function accepting interface types, it automatically selects the appropriate concrete type based on the Python tensor's properties (read-only, writable, differentiable, etc.).

Currently, the main use-case for concrete Tensor types is when you need to directly access the gradient buffers for custom operations, such as manually written backwards passes, as these are only exposed by the concrete ``DiffTensor`` types.

Tensor Operations
-----------------

Accessing Elements
~~~~~~~~~~~~~~~~~~

Tensors support multiple ways to access elements:

**Subscript operator**:

.. code-block:: slang

    void example(int2 idx, ITensor<float, 2> input, IRWTensor<float, 2> output)
    {
        // Read using subscript with array indices
        int[2] arr_idx = {idx[0], idx[1]};
        float value = input[arr_idx];

        // Read using subscript with vector indices
        int2 vec_idx = int2(idx.x, idx.y);
        value = input[vec_idx];

        // Read using subscript with variadic indices
        value = input[idx.x, idx.y];

        // Write using subscript
        output[arr_idx] = value * 2.0;
        output[vec_idx] = value * 2.0;
        output[idx.x, idx.y] = value * 2.0;
    }

**load/store methods**:

.. code-block:: slang

    void example_loadstore(int2 idx, ITensor<float, 2> input, IRWTensor<float, 2> output)
    {
        // Load using array indices
        int[2] arr_idx = {idx[0], idx[1]};
        float value = input.load(arr_idx);

        // Load using vector indices
        int2 vec_idx = int2(idx.x, idx.y);
        value = input.load(vec_idx);

        // Load using variadic indices
        value = input.load(idx.x, idx.y);

        // Store
        output.store(arr_idx, value * 2.0);
        output.store(vec_idx, value * 2.0);
        output.store(idx.x, idx.y, value * 2.0);
    }

.. warning::
   **Index Convention Differences**

   As noted in :ref:`index_representation`, there is an important difference between array and vector indexing:

   - **Array indices**: ``int[2]`` - Follow tensor dimension order (e.g., ``[row, col]`` for 2D)
   - **Variadic indices**: Multiple integer arguments in order (e.g., ``(row, col)`` for 2D)
   - **Vector indices**: ``int2`` - Reverse order with x component indexing the rightmost dimension (e.g., ``(col, row)`` for 2D)

   .. code-block:: slang

       void index_demo(ITensor<float, 2> tensor)
       {
           // These access the SAME element at row=3, col=5:
           int[2] arr_idx = {3, 5};        // Array: [row, col]
           int2 vec_idx = int2(5, 3);      // Vector: (col, row) - note reversed order!

           float value1 = tensor.load(arr_idx);
           float value2 = tensor.load(vec_idx);  // Same as value1
       }

Tensor Properties
~~~~~~~~~~~~~~~~~

All tensors expose a ``shape`` property to query dimensions:

.. code-block:: slang

    void check_dimensions(ITensor<float, 3> tensor)
    {
        uint[3] dims = tensor.shape;
        uint width = dims[0];
        uint height = dims[1];
        uint depth = dims[2];

        // Use dimensions in computations
        if (width > 100 && height > 100) {
            // ...
        }
    }

Working with Structs
~~~~~~~~~~~~~~~~~~~~

Tensors can store any user-defined struct types, so can be used in place of a ``StructuredBuffer`` in classical GPU programming:

.. code-block:: slang

    struct Particle
    {
        float3 position;
        float3 velocity;
        float mass;
    };

    void update_particles(int idx, ITensor<Particle, 1> particles_in, IRWTensor<Particle, 1> particles_out)
    {
        // Load entire struct
        Particle p = particles_in[idx];

        // Update fields
        p.position += p.velocity * 0.016;  // 60 FPS timestep

        // Store back
        particles_out[idx] = p;
    }

Note that when a differentiable tensor type is used, the user-defined struct must implement both ``IDifferentiable`` and ``IAtomicAddable`` interfaces. If gradient accumulation is not needed (eg the input tensor is write-only, so gradients are read-only) the 2 ``atomicAdd`` functions required by ``IAtomicAddable`` can be left as no-ops but must be present.

Differentiable Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~

Differentiable tensor types (``DiffTensor``, ``WDiffTensor``, ``RWDiffTensor``) support automatic differentiation. Operations on these tensors can be differentiated by Slang's auto-diff system:

.. code-block:: slang

    [Differentiable]
    float compute_loss(int idx, IDiffTensor<float, 1> predictions, IDiffTensor<float, 1> targets)
    {
        float pred = predictions[idx];
        float target = targets[idx];

        // Mean squared error
        float diff = pred - target;
        return diff * diff;
    }

When this function is called with ``Tensor`` arguments from Python, SlangPy can automatically generate both forward and backward passes. See :ref:`autodiff` for more details on automatic differentiation and :ref:`tensors_differentiable` for using differentiable tensors.

Atomic Operations
~~~~~~~~~~~~~~~~~

The ``AtomicTensor`` type supports atomic operations, and is typically used by SlangPy for thread-safe accumulation of gradients:

.. code-block:: slang

    void accumulate_gradients(int idx, Tensor<float, 1> local_grads, AtomicTensor<float, 1> global_grads)
    {
        float local_grad = local_grads[idx];

        // Atomic add - safe for concurrent writes from multiple threads
        global_grads.add(idx, local_grad);
    }

Both ``DiffTensor`` and ``RWDiffTensor`` use ``AtomicTensor`` internally for gradient accumulation in backward passes.

Examples
---------------------

Element-wise Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Whilst it would be unusual to write an element-wise operation manually (since SlangPy automatically vectorizes function calls), this example shows a classical element-wise scaling operation:

.. code-block:: slang

    void scale_values(int2 idx, ITensor<float, 2> input, IRWTensor<float, 2> output, float scale)
    {
        output[idx] = input[idx] * scale;
    }

Called from Python with:

.. code-block:: python

    input = spy.Tensor.from_numpy(device, data)
    output = spy.Tensor.empty(device, shape=input.shape, dtype=float)
    module.scale_values(spy.grid(shape=input.shape), input, output, scale=2.0)

The most common reason to utilize tensor types in this way is when upgrading an old code base that already operates on the deprecated Slang ``TensorView`` and ``DiffTensorView`` types, which required explicit element-wise kernels.

Neighborhood Operations
~~~~~~~~~~~~~~~~~~~~~~~~

Accessing neighboring elements (e.g., convolution, blur) currently requires access to the full tensor:

.. code-block:: slang

    void blur_3x3(int2 idx, ITensor<float, 2> input, IRWTensor<float, 2> output)
    {
        float sum = 0.0;
        int count = 0;

        // 3x3 neighborhood
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int2 neighbor = idx + int2(dx, dy);

                // Check bounds
                if (neighbor.x >= 0 && neighbor.x < input.shape[0] &&
                    neighbor.y >= 0 && neighbor.y < input.shape[1]) {
                    sum += input[neighbor];
                    count++;
                }
            }
        }

        output[idx] = sum / float(count);
    }

Work is in progress to support this pattern with a tile abstraction to allow more efficient shared memory usage.

Reduction Operations
~~~~~~~~~~~~~~~~~~~~

Summing or finding max/min across a dimension:

.. code-block:: slang

    void sum_rows(int row, ITensor<float, 2> input, IRWTensor<float, 1> output)
    {
        uint width = input.shape[1];
        float sum = 0.0;

        for (uint col = 0; col < width; col++) {
            sum += input.load(row, col);
        }

        output[row] = sum;
    }

Called from Python:

.. code-block:: python

    input = spy.Tensor.from_numpy(device, data_2d)  # Shape (100, 200)
    output = spy.Tensor.empty(device, shape=(100,), dtype=float)
    module.sum_rows(spy.grid(shape=(100,)), input, output)

Differentiable Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Functions that work in both forward and backward passes:

.. code-block:: slang

    [Differentiable]
    void apply_activation(int idx, IDiffTensor<float, 1> input, IWDiffTensor<float, 1> output)
    {
        float x = input[idx];

        // ReLU activation
        output[idx] = max(0.0, x);
    }

Called from Python for forward pass:

.. code-block:: python

    input = spy.Tensor.from_numpy(device, data).with_grads()
    output = spy.Tensor.empty(device, shape=input.shape, dtype=float).with_grads()

    # Forward pass
    module.apply_activation(spy.grid(shape=input.shape), input, output)

    # ... compute loss and set output gradients ...

    # Backward pass
    module.apply_activation.bwds(spy.grid(shape=input.shape), input, output)

Generic slang function to take generic tensors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A normalization function that works for any floating-point tensor:

.. code-block:: slang

    void normalize_tensor<T: __BuiltInFloatingPointType>(int idx, ITensor<T, 1> input, IRWTensor<T, 1> output)
    {
        T value = input[idx];
        output[idx] = value / T(255.0);
    }

When called from Python, SlangPy will select the appropriate concrete tensor types based on the properties of the passed tensors. In this case, if ``Tensor`` of float16 were passed, SlangPy would generate a kernel using ``ITensor<float16, 1>`` and ``IRWTensor<float16, 1>``.

Internals
---------

Underlying Storage
~~~~~~~~~~~~~~~~~~

Tensors are implemented on top of GPU buffer resources:

- ``Tensor`` and read-only variants use ``StructuredBuffer<T>``, or ``ImmutablePtr<T>`` in CUDA
- ``RWTensor`` and read-write variants use ``RWStructuredBuffer<T>`` or ``Ptr<T>`` in CUDA
- ``DiffTensor`` types wrap both primal and gradient buffers
- ``AtomicTensor`` uses ``RWByteAddressBuffer`` or ``Ptr<T>`` in CUDA

Each tensor stores:

- Buffer reference (``_data``)
- Shape array (``_shape``)
- Stride array (``_strides``)
- Offset (``_offset``)

Memory Layout
~~~~~~~~~~~~~

Tensors use row-major layout by default (rightmost dimension has smallest stride).

For a 3D tensor of shape ``[D0, D1, D2]``, the strides are:

- ``stride[0] = D1 * D2``
- ``stride[1] = D2``
- ``stride[2] = 1``

The linear index for element ``[i, j, k]`` is computed as:

.. code-block:: slang

    int linear_idx = i * stride[0] + j * stride[1] + k * stride[2] + offset;

Summary
-------

SlangPy's tensor types provide:

- **Multiple access modes**: Read-only, write-only, read-write, atomic
- **Automatic differentiation**: Differentiable tensor variants for AD
- **Flexible interfaces**: Generic functions work with any tensor implementation
- **Multiple indexing styles**: Subscripts, load/store, variadic indices
- **Multi-dimensional support**: 1D, 2D, 3D, and higher-dimensional tensors
- **Struct element types**: Not limited to primitive types

For details on using tensors from Python, see :ref:`tensors_python`. For information on automatic differentiation with tensors, see :ref:`tensors_differentiable`.
