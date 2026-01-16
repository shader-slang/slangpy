Tensors
=======

SlangPy provides a single multidimensional container type, named ``Tensor``, which behaves much like a NumPy array or Torch tensor.
``Tensor`` supports a variety of data types, including primitive types (e.g., float, int, vector) and user-defined Slang structs. Internally,
``Tensor`` will wrap either a ``StructuredBuffer`` or a ``Ptr<T>`` depending on platform.

Note: if updating from pre-0.41, see :ref:`tensorupdate` for migration instructions.

The ``Tensor`` type takes some basic storage and adds:

- **Data type**: A ``SlangType``, which can be a primitive type (e.g., float, vector) or a user-defined Slang struct.
- **Shape**: A tuple of integers describing the size of each dimension, similar to the shape of a NumPy array or Torch tensor.
- **Stride**: A tuple of integers describing the layout of each dimension. By default, ``Tensor`` uses a row-major layout, where the right-most dimension has the smallest stride.

.. warning::
   **Indexing Conventions**

   Tensors can be indexed using either array coordinates, which use the same convention as the stride (right-most dimension has the smallest stride), or vector coordinates, which follow a different convention where the x component has the smallest stride.
   This means the same tensor position requires different coordinate values: e.g. for a 2D tensor, array indexing uses `[row, col]` while vector indexing uses `(col, row)` for the same location.
   See :ref:`index_representation` for complete details on these differing index representation conventions.

Let's start with a simple Slang program that uses a custom type:

.. code-block::

    // Currently, to use custom types with SlangPy, they need to be explicitly imported.
    import "slangpy";

    // example.slang
    struct Pixel
    {
        float r;
        float g;
        float b;
    };

    // Add two pixels together
    Pixel add(Pixel a, Pixel b)
    {
        Pixel result;
        result.r = a.r + b.r;
        result.g = a.g + b.g;
        result.b = a.b + b.b;
        return result;
    }

*Note:* In many cases, a Slang module must import the ``slangpy`` module to resolve all types correctly during kernel generation. This is a known issue that we aim to address in the near future.

Initialization
--------------

Initialization follows the same steps as in the previous example:

.. code-block:: python

    import slangpy as spy
    import pathlib
    import numpy as np

    # Create a SlangPy device and use the local folder for Slang includes
    device = spy.create_device(include_paths=[
            pathlib.Path(__file__).parent.absolute(),
    ])

    # Load the module
    module = spy.Module.load_from_file(device, "example.slang")

Creating Tensors
----------------

We'll now create and initialize two tensors of type ``Pixel``. The first will use a tensor cursor for manual population, while the second will be populated using a NumPy array.

.. code-block:: python

    # Create two 2D tensors of size 16x16
    image_1 = spy.Tensor.empty(device, dtype=module.Pixel, shape=(16, 16))
    image_2 = spy.Tensor.empty(device, dtype=module.Pixel, shape=(16, 16))

    # Populate the first tensor using a cursor
    cursor_1 = image_1.cursor()
    for x in range(16):
        for y in range(16):
            cursor_1[x + y * 16].write({
                'r': (x + y) / 32.0,
                'g': 0,
                'b': 0,
            })
    cursor_1.apply()

    # Populate the second tensor directly from a NumPy array
    image_2.copy_from_numpy(0.1 * np.random.rand(16 * 16 * 3).astype(np.float32))

While using a cursor is more verbose, it offers powerful tools for reading and writing structured data. It even allows inspection of GPU tensor contents directly in the VSCode watch window.

For more details on the Python ``Tensor`` API, see :ref:`tensors_python`.

Calling the Function
--------------------

Once our data is ready, we can call the ``add`` function as usual:

.. code-block:: python

    # Call the module's add function
    result = module.add(image_1, image_2)

SlangPy understands that these tensors are effectively 2D arrays of ``Pixel``. It infers a 2D dispatch (16×16 threads in this case), where each thread reads one ``Pixel`` from each tensor, adds them together, and writes the result into a third tensor. By default, SlangPy automatically allocates and returns a new ``Tensor``.

Alternatively, we can pre-allocate the result tensor and pass it explicitly:

.. code-block:: python

    # Pre-allocate the result tensor
    result = spy.Tensor.empty(device, dtype=module.Pixel, shape=(16, 16))
    module.add(image_1, image_2, _result=result)

This approach is useful when inputs and outputs are pre-allocated upfront for efficiency.

Reading the Results
-------------------------------------

Finally, let's print the result and, if available, use `tev` to visualize it:

.. code-block:: python

    # Read and print pixel data using a cursor
    result_cursor = result.cursor()
    for x in range(16):
        for y in range(16):
            pixel = result_cursor[x + y * 16].read()
            print(f"Pixel ({x},{y}): {pixel}")

    # Display the result with tev (https://github.com/Tom94/tev)
    tex = device.create_texture(
        data=result.to_numpy(),
        width=16,
        height=16,
        format=spy.Format.rgb32_float
    )
    spy.tev.show(tex)

Tensors in Slang
----------------

In the above examples, we relied on SlangPy's vectorized function calls to handle ``Tensor`` parameters automatically. When calling `add`, the generated
kernel would have loaded and stored ``Pixel`` values from the input tensors based on the thread indices. However, there are scenarios where we may want to manipulate tensors directly within Slang code. SlangPy provides a range of Slang types to manage tensors, the most common of which are:

- ``Tensor<T, N>``: A read-only N-dimensional tensor of type T.
- ``RWTensor<T, N>``: A read-write N-dimensional tensor of type T.
- ``DiffTensor<T, N>``: A read-only differentiable N-dimensional tensor of type T.
- ``WDiffTensor<T, N>``: A write-only differentiable N-dimensional tensor of type T.

Each of these types also comes with a corresponding Slang interface (eg ``IRWTensor<T,N>``) that should be used when defining functions that accept tensor parameters wherever possible. For example, the `add` function could be rewritten as follows:

.. code-block::

    // example.slang
    import "slangpy";

    struct Pixel
    {
        float r;
        float g;
        float b;
    };

    // Add two pixels together
    void add(
        int2 index,
        ITensor<Pixel, 2> a,
        ITensor<Pixel, 2> b,
        IRWTensor<Pixel, 2> result)
    {
        // Load pixels from input tensors
        Pixel pixel_a = a[index];
        Pixel pixel_b = b[index];

        // Add the pixels
        Pixel pixel_result;
        pixel_result.r = pixel_a.r + pixel_b.r;
        pixel_result.g = pixel_a.g + pixel_b.g;
        pixel_result.b = pixel_a.b + pixel_b.b;

        // Store the result
        result[index] = pixel_result;
    }

And then called using a grid generator (see :ref:`generators_grid`):

.. code-block:: python

    # Call the module's add function using a grid generator
    result = spy.Tensor.empty(device, dtype=module.Pixel, shape=(16, 16))
    module.add(
        spy.grid(shape=(16, 16)),
        image_1,
        image_2,
        result
    )

For more details on the Python ``Tensor`` API, see :ref:`tensors_slang`.

Summary
-------

That's it! This tutorial demonstrated how to use ``Tensor`` to manipulate structured data in SlangPy. While we focused on basic tensor operations, there’s much more to explore, such as:

- Using ``InstanceLists`` to call type methods.
- Leveraging ``Tensor`` for differentiable data manipulation.
