Basic Auto-diff
===============

One of Slang's most powerful features is its auto-diff capabilities, documented in detail `here <https://shader-slang.com/slang/user-guide/autodiff.html>`_. SlangPy carries this feature over to Python, allowing you to easily calculate the derivative of a function.

A differentiable function
-------------------------

Let's start with a simple polynomial function:

.. code-block::

    [Differentiable]
    float polynomial(float a, float b, float c, float x) {
        return a * x * x + b * x + c;
    }

Note that it has the ``[Differentiable]`` attribute, which tells Slang to generate the backward propagation function.

The Tensor type
---------------

To store simple differentiable data, SlangPy utilizes the ``Tensor`` type. Here we'll initialize one from the data in a numpy array and use it to evaluate a polynomial.

.. code-block:: python

    # Create a tensor with attached grads from a numpy array
    # Note: We pass zero=True to initialize the grads to zero on allocation
    x = spy.Tensor.from_numpy(device, np.array([1, 2, 3, 4], dtype=np.float32)).with_grads()

    # Evaluate the polynomial
    # Expecting result = 2x^2 + 8x - 1
    result: spy.Tensor = module.polynomial(a=2, b=8, c=-1, x=x)
    print(result.to_numpy())

We could also have pre-allocated a tensor to fill in:

.. code-block:: python

    result = spy.Tensor.empty(device, shape=(4,), dtype=module.float)
    module.polynomial(a=2, b=8, c=-1, x=x, _result=result)

In both cases, we end up with a result tensor that contains the evaluated polynomial.

Backward pass
-------------

Now we'll attach gradients to the result and set them to 1, then run back propagation:

.. code-block:: python

    # Attach gradients to the result, and set them to 1 for the backward pass
    result = result.with_grads()
    result.grad.copy_from_numpy(np.array([1, 1, 1, 1], dtype=np.float32))

    # Call the backwards version of module.polynomial
    # This will read the grads from _result, and write the grads to x
    # Expecting result = 4x + 8
    module.polynomial.bwds(a=2, b=8, c=-1, x=x, _result=result)
    print(x.grad.to_numpy())

That's the lot! The call to ``bwds`` generates a kernel that calls ``bwds_diff(polynomial)`` in Slang, and automatically
deals with passing in/out the correct data.

It is worth noting that SlangPy currently **always accumulates** gradients, so you will need to ensure gradient buffers
are zeroed. In the demo above, we used ``zero=True`` when creating the tensor to do so.

If you're familiar with ML frameworks such as PyTorch, the big difference is that SlangPy is (by design) not a host side auto-grad system. It does not record an auto-grad graph, and instead requires you to explicitly call the backward function, providing the primals used in the forward call. However, SlangPy provides strong integration with PyTorch and all its auto-grad features.

Differentiable Tensors In Slang
-------------------------------

In the examples above, we relied on SlangPy's vectorized function calls to handle ``Tensor`` parameters automatically. In the above
example, SlangPy would have automatically generated a kernel in the forward pass that used:

- **x**: ``Tensor<float, 1>`` (read-only tensor)
- **result**: ``WTensor<float, 1>`` (write-only tensor)

In the backward pass, it would have used:

- **x**: ``DiffTensor<float, 1>`` (differentiable read-only tensor)
- **result**: ``WDiffTensor<float, 1>`` (differentiable write-only tensor)

Just like with regular tensors, there are scenarios where we may want to manipulate differentiable tensors directly within Slang code. The above `polynomial` function could be rewritten as follows:

.. code-block::

    [Differentiable]
    void polynomial(
        int2 idx, float a, float b, float c,
        IDiffTensor<float,1> x,
        IWDiffTensor<float,1> result)
    {
        float xval = x[idx];
        result[idx] = a * xval * xval + b * xval + c;
    }

And then called using a grid generator (see :ref:`generators_grid`):

.. code-block:: python

    # <allocate tensors and attach grads etc here>

    # Forward pass
    module.polynomial(
        spy.grid(shape=(16, 16)), a=2, b=8, c=-1,
        x=x,
        result=result)

    # Backward pass
    module.polynomial.bwds(
        spy.grid(shape=(16, 16)), a=2, b=8, c=-1,
        x=x,
        result=result)

For more details on the Differentiable tensors in Slang, see :ref:`tensors_differentiable`.

Summary
-------

Use of auto-diff in SlangPy requires:
- Marking your function as differentiable
- Using the ``Tensor`` type to store differentiable data
- Calling the ``bwds`` function to calculate gradients

SlangPy's tensor type currently only supports basic types for gradient accumulation due to the need for atomic accumulation. However, we intend to expand this to all struct types in the future.
