.. _tensors_differentiable:

Differentiable Tensors
======================

Introduction
------------

Differentiable tensors are a specialized category of SlangPy tensors designed for automatic differentiation (AD). They enable you to compute gradients of Slang functions with respect to tensor inputs, making them essential for machine learning, optimization, and physics simulation tasks.

While there are many ways to combine automatic differentiation with tensors, following a few key guidelines will ensure your code works correctly and efficiently.

Key Guidelines for Success
--------------------------

To avoid common pitfalls when working with differentiable tensors, follow these three rules:

1. **Avoid Read-Write Tensors** - Use either ``IDiffTensor`` (input/read-only) or ``IWDiffTensor`` (output/write-only), not ``IRWDiffTensor``. This clearly separates inputs from outputs and avoids the need for separate gradient buffers.

2. **Use Default Gradient Behavior** - Call ``with_grads()`` without arguments. Don't manually specify ``grad_in`` or ``grad_out`` unless you have a specific reason (like in-place operations).

3. **Zero Gradient Buffers** - Always zero out tensors whose gradients will be written to. Use ``zero=True`` when calling ``with_grads()``, or manually call ``tensor.grad.clear()``.

Following these guidelines will prevent gradient accumulation bugs, memory issues, and binding errors.

Basic Example: Polynomial Function
-----------------------------------

Let's start with a simple differentiable function that operates on tensors:

.. code-block:: slang

    // example.slang
    import "slangpy";

    [Differentiable]
    void polynomial(int idx, float a, float b, float c, IDiffTensor<float, 1> x, IWDiffTensor<float, 1> result)
    {
        float xval = x[idx];
        result[idx] = a * xval * xval + b * xval + c;
    }

From Python, we can evaluate this function and compute gradients:

.. code-block:: python

    import slangpy as spy
    import numpy as np

    device = spy.create_device()
    module = spy.Module.load_from_file(device, "example.slang")

    # Create input tensor with gradients (automatically zeroed)
    x = spy.Tensor.from_numpy(device, np.array([1, 2, 3, 4], dtype=np.float32))
    x = x.with_grads(zero=True)

    # Forward pass: evaluate polynomial
    # Result: 2*x^2 + 8*x - 1
    module.polynomial(idx=spy.grid(x.shape), a=2, b=8, c=-1, x=x, result=result)
    print(result.to_numpy())  # [9., 27., 53., 87.]

    # Attach gradients to result and set to 1
    result = result.with_grads()
    result.grad.copy_from_numpy(np.ones(4, dtype=np.float32))

    # Backward pass: compute gradients
    # Gradient: 4*x + 8
    module.polynomial.bwds(idx=spy.grid(x.shape), a=2, b=8, c=-1, x=x, result=result)
    print(x.grad.to_numpy())  # [12., 16., 20., 24.]

This example demonstrates the basic workflow:

1. Create input tensors with gradient storage
2. Run the forward pass
3. Attach gradients to outputs and initialize them
4. Run the backward pass with ``.bwds()``
5. Read accumulated gradients from input tensors

Why Use Interface Types?
~~~~~~~~~~~~~~~~~~~~~~~~

In the none-differentiable case, the interface types (``ITensor``, ``IWTensor``, ``IRWTensor``) are recommended for maximum flexibility (and in future, performance), but they are not critical. However for
SlangPy to correctly generate efficient backwards passes, use of interface types is essential.

Consider the following code:

.. code-block:: slang

    [Differentiable]
    void polynomial(int idx, float a, float b, float c, DiffTensor<float, 1> x, WDiffTensor<float, 1> result) { /*...*/ }

When faced with the following call from Python:

.. code-block:: python

    module.polynomial(idx=spy.grid(x.shape), a=2, b=8, c=-1, x=x, result=result)

SlangPy has no option but to generate a kernel that requires ``DiffTensor`` types to be bound. As these types have gradient buffers, it would be an error for either ``x`` or ``result`` to be a tensor without gradients attached - even though they aren't used! Furthermore, even if they do exist, expensive binding logic would need to be happen simply to bind the gradient buffers that are never accessed!

When we switch the example to use interface types instead:

.. code-block:: slang

    [Differentiable]
    void polynomial(int idx, float a, float b, float c, IDiffTensor<float, 1> x, IWDiffTensor<float, 1> result) { /*...*/ }

SlangPy can now generate a kernel that binds ``PrimalTensor`` and ``WPrimalTensor`` during the forward pass, and ``DiffTensor`` and ``WDiffTensor`` during the backward pass. This avoids errors when unused gradient buffers are missing, and avoids unnecessary binding overhead.

In future versions of SlangPy, we will aim to remove the need for ``IDiffTensor`` entirely and allow ``ITensor`` to be used directly in differentiable functions.

How Gradient Propagation Works
-------------------------------

Differentiable tensors achieve automatic differentiation by attaching custom derivative implementations to their ``load()`` and ``store()`` operations. When you call ``.bwds()``, Slang's auto-diff system:

1. Executes the backward pass using ``bwd_diff(function)``
2. For each ``load()`` in the forward pass, calls the corresponding ``_load_bwd()`` which accumulates output gradients
3. For each ``store()`` in the forward pass, calls the corresponding ``_store_bwd()`` which reads input gradients

Here's how ``DiffTensor`` implements gradient propagation (simplified from the actual implementation):

.. code-block:: slang

    public struct DiffTensor<T : IDifferentiable, let D : int>
    {
        public Tensor<T, D> _primal;                      // Forward values
        public AtomicTensor<T.Differential, D> _grad_out; // Gradient output

        // Forward pass: just load the primal value
        [Differentiable]
        [BackwardDerivative(_load_bwd_array)]
        public T load<I : __BuiltinIntegerType>(I idx[D])
        {
            return _primal.load(idx);
        }

        // Backward pass: accumulate gradient to grad_out
        void _load_bwd_array<I : __BuiltinIntegerType>(I idx[D], T.Differential grad)
        {
            _grad_out.add(idx, grad);  // Atomic accumulation
        }
    }

Similarly ``WDiffTensor`` implements a ``_store_bwd()`` that reads gradients from ``_grad_in`` during backpropagation, and ``RWDiffTensor``` implements both ``_load_bwd()`` and ``_store_bwd()``.

As the subscript operators (``operator[]``) are implemented in terms of ``load()`` and ``store()``, they automatically inherit the same gradient propagation behavior.

Using this mechanism, any operation that reads from a tensor in a forwards pass will accumulate gradients during the backwards pass, and any operation that writes to a tensor will read gradients during the backwards pass.

For more details on custom derivatives, see the slang documentation `here <https://shader-slang.com/slang/user-guide/autodiff.html>`_.

Why Avoid Read-Write Tensors?
------------------------------

The Problem with IRWDiffTensor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As mentioned earlier, during auto diff:

- A read-only (input) tensor will accumulate gradients during the backwards pass
- A write-only (output) tensor will read gradients during the backwards pass

If a tensor must both read gradients **and** accumulate them, using a single buffer would mean that the same memory is being read from and written to simultaneously during backpropagation. Unless there is a guarantee that every thread will read/write from 1 and only 1 unique element in the tensor, this would result in race conditions.

To solve this problem, the ``Tensor`` type in Python supports having separate input and output gradient buffers, and the ``RWDiffTensor`` has corresponding separate ``grad_in`` and ``grad_out`` buffers.

Given the following simple Slang function that reads, adjusts and then writes a tensor:

.. code-block:: slang

    [Differentiable]
    void process_inplace(int idx, IRWDiffTensor<float, 1> data)
    {
        float value = data[idx];  // Read
        data[idx] = value * 2.0;  // Write
    }

The Python side would need to be:

.. code-block:: python

    import slangpy as spy
    import numpy as np

    device = spy.create_device()
    module = spy.Module.load_from_file(device, "example.slang")

    # Create input tensor with gradients (ones for input, zeroed for output)
    x = spy.Tensor.from_numpy(device, np.array([1, 2, 3, 4], dtype=np.float32))
    x = x.with_grads(grad_in=spy.Tensor.ones(device, shape=(4,), dtype=float),
                     grad_out=spy.Tensor.zeros(device, shape=(4,), dtype=float))

    # Forward pass will populate the primals and ignore grads
    module.process_inplace(idx=spy.grid(x.shape), data=x)

    # Backwards pass will read primals, read grad_in and write grad_out
    module.process_inplace.bwds(idx=spy.grid(x.shape), data=x)

By explicitly allocating separate ``grad_in`` and ``grad_out`` buffers, we avoid race conditions and can use in place modifications, at the cost of some complexity.

Using Concrete Tensor Types
----------------------------

When to Use Concrete Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In most cases, you should use interface types (``IDiffTensor``, ``IWDiffTensor``). However, if you need direct access to the gradient buffers of your tensor, you will need to use the concrete ``DiffTensor`` types. The most common scenario is when implementing custom derivatives:

.. code-block:: slang

    // Custom activation function with explicit gradient handling
    [Differentiable]
    [BackwardDerivative(custom_activation_bwd)]
    float custom_activation(int idx, DiffTensor<float, 1> input)
    {
        float x = input.load(idx);

        // Custom activation: smooth step
        if (x <= 0.0) return 0.0;
        if (x >= 1.0) return 1.0;
        return x * x * (3.0 - 2.0 * x);  // Smoothstep
    }

    // Custom backward derivative
    void custom_activation_bwd(
        int idx,
        DiffTensor<float, 1> input,
        float dOutput)
    {
        float x = input._primal.load(idx);

        // Derivative of smoothstep
        float dInput;
        if (x <= 0.0 || x >= 1.0) {
            dInput = 0.0;
        } else {
            dInput = 6.0 * x * (1.0 - x);
        }

        // Manually accumulate gradient
        input._grad_out.add(idx, dInput * dOutput);
    }

.. warning::
   **Graphics Backend Limitation**

   When using concrete tensor types (``DiffTensor``, ``WDiffTensor``, etc.) rather than interfaces, you **must** pass tensors with gradients attached from Python.

   On D3D12 and Vulkan backends, binding a null descriptor is invalid. If you pass a tensor without gradients to a function expecting a concrete ``DiffTensor``, the shader will try to bind a null gradient buffer, causing a runtime error.

   .. code-block:: python

       # This will FAIL on D3D12/Vulkan if function uses DiffTensor<float,1>
       input = spy.Tensor.from_numpy(device, data)  # No gradients!
       module.function_with_concrete_type(input)

       # This will work
       input = spy.Tensor.from_numpy(device, data).with_grads()
       module.function_with_concrete_type(input)

   Interface types (``IDiffTensor``) avoid this issue because SlangPy can bind a ``PrimalTensor`` when no gradients are attached.

Practical Example: Neural Network Layer
----------------------------------------

Here's a complete example of a differentiable matrix-vector multiplication (like a neural network layer):

.. code-block:: slang

    [Differentiable]
    void linear_layer(
        int idx,
        IDiffTensor<float, 2> weights,    // Shape: [out_features, in_features]
        IDiffTensor<float, 1> bias,       // Shape: [out_features]
        IDiffTensor<float, 1> input,      // Shape: [in_features]
        IWDiffTensor<float, 1> output)    // Shape: [out_features]
    {
        uint in_features = weights.shape[1];

        // Compute weighted sum
        float sum = bias[idx];
        for (uint i = 0; i < in_features; i++) {
            sum += weights[idx, i] * input[i];
        }

        output[idx] = sum;
    }

Python usage:

.. code-block:: python

    # Create trainable parameters with gradients
    weights = spy.Tensor.from_numpy(device, weight_init).with_grads(zero=True)
    bias = spy.Tensor.from_numpy(device, bias_init).with_grads(zero=True)

    # Create input batch with gradients
    x = spy.Tensor.from_numpy(device, batch_data).with_grads(zero=True)

    # Allocate output
    y = spy.Tensor.empty(device, shape=(out_features,), dtype=float).with_grads()

    # Forward pass
    module.linear_layer(
        spy.grid(shape=(out_features,)),
        weights, bias, x, y
    )

    # Compute loss and set output gradients
    loss_gradients = compute_loss_gradients_somehow(y)
    y.grad.copy_from_numpy(loss_gradients)

    # Backward pass - accumulates gradients in weights, bias, and x
    module.linear_layer.bwds(
        spy.grid(shape=(out_features,)),
        weights, bias, x, y
    )

Summary
-------

Differentiable tensors enable automatic differentiation in SlangPy by:

- Attaching custom derivatives to ``load()`` and ``store()`` operations
- Maintaining separate primal (forward) and gradient (backward) storage
- Using atomic operations for gradient accumulation

**Best practices:**

1. Use interface types (``IDiffTensor``, ``IWDiffTensor``) for function parameters
2. Avoid read-write tensors (``IRWDiffTensor``) when possible
3. Use default ``with_grads()`` behavior unless you need in-place operations
4. Always zero gradient buffers before backward passes
5. Keep inputs and outputs separate for clarity and efficiency

**When to use what:**

- ``IDiffTensor`` - Reading tensor values (inputs, parameters)
- ``IWDiffTensor`` - Writing tensor values (outputs, results)
- ``IRWDiffTensor`` - Only for true in-place operations
- Concrete types (``DiffTensor``) - Only when implementing custom derivatives

For more information on tensors, see :ref:`tensors_python` and :ref:`tensors_slang`. For general automatic differentiation concepts, see the `Slang auto-diff documentation <https://shader-slang.com/slang/user-guide/autodiff.html>`_.
