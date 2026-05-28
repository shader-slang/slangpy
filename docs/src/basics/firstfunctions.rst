Your First Function
===================

In this example, we'll initialize SlangPy, create a simple Slang function, and call it from Python.

You can find the complete code for this example `here <https://github.com/shader-slang/slangpy-samples/tree/main/examples/first_function/>`_.

First, let's define a simple Slang function to add two numbers together:

.. code-block::

    // example.slang

    // A simple function that adds two numbers together
    float add(float a, float b)
    {
        return a + b;
    }

Next, we'll create a Python script to initialize SlangPy, load the Slang module, and call the function:

.. code-block:: python

    ## main_scalar.py

    import slangpy as spy
    import pathlib

    # Create a SlangPy device; it will look in the local folder for any Slang includes
    device = spy.create_device(include_paths=[
            pathlib.Path(__file__).parent.absolute(),
    ])

    # Load the module
    module = spy.Module.load_from_file(device, "example.slang")

    # Call the function and print the result
    result = module.add(1.0, 2.0)
    print(result)

    # SlangPy also supports named parameters
    result = module.add(a=1.0, b=2.0)
    print(result)

Under the hood, the first time this function is invoked, SlangPy generates a compute kernel (and caches it in a temporary folder). This kernel handles loading scalar inputs from buffers, calling the ``add`` function, and writing the scalar result back to a buffer.

Automatic Vectorization
-----------------------

SlangPy’s **automatic vectorization** simplifies GPU programming by handling the tedious boilerplate code typically required to apply a function across entire containers such as buffers, tensors or textures. Instead of manually writing a new kernel for each task, such as adding all elements of two buffers, SlangPy automatically generates and launches the necessary compute kernel for you.

When you call a function with scalar inputs, like ``add(2.0, 3.0)``, SlangPy treats it as a single "scalar" call, generating a simple kernel to perform the operation once. However, when you pass in buffers or arrays, **automatic vectorization** kicks in. The library figures out how to map the function to the data, generates a compute kernel with the correct read/write logic and launches it with the appropriate number of threads.

This eliminates common, error-prone tasks:

- Writing a new kernel for each function and data type combination.
- Manually managing thread IDs and memory access for input and output buffers.
- Calculating and setting the correct number of threads for kernel launch.

This capability is a core feature of modern numerical computing libraries like SlangPy, NumPy, and PyTorch, allowing you to focus on the logic of your function rather than the low-level details of GPU kernel management.
For example, to add two buffers element-wise, you simply call the function with the buffers as arguments, and SlangPy handles the rest. This also extends to more complex scenarios, like adding a scalar value to every element of a multi-dimensional tensor, significantly reducing the chances of bugs and amount of boilerplate.

Explicit Thread Count
---------------------

Automatic vectorization infers the number of GPU threads from the shapes of the input arguments. However, there are cases where you want direct control over thread dispatch, for example, when writing a ``[CUDAKernel]`` function that manages its own indexing using ``cudaThreadIdx()`` rather than relying on SlangPy's vectorization.
For these kernels, SlangPy cannot infer a thread count from the inputs (since none of the arguments are being vectorized over). You can specify it explicitly using the special ``_thread_count`` keyword argument:

.. code-block::

    // example.slang

    // A kernel that squares each element, managing its own thread index.
    [CUDAKernel]
    [Differentiable]
    void square_kernel(uint count, DiffTensorView<float> input, DiffTensorView<float> output)
    {
        uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
        if (tid >= count)
            return;
        float x = input.load(tid);
        output.store(tid, x * x);
    }

.. code-block:: python

    ## main_cuda_kernel.py

    # ... initialization here ...

    import torch

    N = 5
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda")
    output = torch.zeros(N, device="cuda")

    # Dispatch exactly N threads — SlangPy cannot infer this from the inputs alone
    module.square_kernel(count=N, input=x, output=output, _thread_count=N)

``_thread_count`` is a special keyword (like ``_result``) consumed by SlangPy and never forwarded to the kernel itself. It is only valid when no arguments trigger automatic vectorization (i.e. call dimensionality is 0). Passing ``_thread_count`` alongside vectorized inputs will raise a ``ValueError``.

Differentiable ``[CUDAKernel]`` functions work the same way — pass ``_thread_count`` to the ``.bwds()`` call as well:

.. code-block:: python

    x_grad = torch.zeros(N, device="cuda")
    output_grad = torch.ones(N, device="cuda")

    module.square_kernel.bwds(
        count=N,
        input=diff_pair(x, x_grad),
        output=diff_pair(output, output_grad),
        _thread_count=N,
    )

A simple example
-----------------

With automatic vectorization, we can take the earlier Python script and scale it up to call the function with 2 large numpy arrays instead:

.. code-block:: python

    ## main_numpy.py

    # ... initialization here ...

    # Create a couple of buffers containing 1,000,000 random floats
    a = np.random.rand(1000000).astype(np.float32)
    b = np.random.rand(1000000).astype(np.float32)

    # Call our function and request a numpy array as the result (the default would be a buffer)
    result = module.add(a, b, _result='numpy')

    # Print the first 10 results
    print(result[:10])

SlangPy supports a wide range of data types and can handle arrays with arbitrary dimensions. This example demonstrates how a single Slang function can be called with both scalars and NumPy arrays. Beyond this, SlangPy also supports many more types such as buffers, textures, and tensors.
