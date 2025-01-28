Buffers
=======

SlangPy provides two key wrappers around classic structured buffers (represented in SGL as `Buffer` objects): ``NDBuffer`` and ``Tensor``.

The ``NDBuffer`` type takes a structured buffer with a defined stride and size and adds:

- **Data type**: A ``SlangType``, which can be a primitive type (e.g., float, vector) or a user-defined Slang struct.  
- **Shape**: A tuple of integers describing the size of each dimension, similar to the shape of a NumPy array or Torch tensor.

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

    # Create an SGL device with the local folder for slangpy includes
    device = spy.create_device(include_paths=[
            pathlib.Path(__file__).parent.absolute(),
    ])

    # Load the module
    module = spy.Module.load_from_file(device, "example.slang")

Creating Buffers
----------------

We'll now create and initialize two buffers of type ``Pixel``. The first will use a buffer cursor for manual population, while the second will be populated using a NumPy array.

.. code-block:: python

    # Create two 2D buffers of size 16x16
    image_1 = spy.NDBuffer(device, dtype=module.Pixel, shape=(16, 16))
    image_2 = spy.NDBuffer(device, dtype=module.Pixel, shape=(16, 16))

    # Populate the first buffer using a cursor
    cursor_1 = image_1.cursor()
    for x in range(16):
        for y in range(16):      
            cursor_1[x + y * 16].write({
                'r': (x + y) / 32.0,
                'g': 0,
                'b': 0,
            })
    cursor_1.apply()

    # Populate the second buffer directly from a NumPy array
    image_2.from_numpy(0.1 * np.random.rand(16 * 16 * 3).astype(np.float32))

While using a cursor is more verbose, it offers powerful tools for reading and writing structured data. It even allows inspection of GPU buffer contents directly in the VSCode watch window.

Calling the Function
--------------------

Once our data is ready, we can call the ``add`` function as usual:

.. code-block:: python

    # Call the module's add function
    result = module.add(image_1, image_2)

SlangPy understands that these buffers are effectively 2D arrays of ``Pixel``. It infers a 2D dispatch (16×16 threads in this case), where each thread reads one ``Pixel`` from each buffer, adds them together, and writes the result into a third buffer. By default, SlangPy automatically allocates and returns a new ``NDBuffer``.

Alternatively, we can pre-allocate the result buffer and pass it explicitly:

.. code-block:: python

    # Pre-allocate the result buffer
    result = spy.NDBuffer(device, dtype=module.Pixel, shape=(16, 16))
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
        format=sgl.Format.rgb32_float
    )
    sgl.tev.show(tex)

Summary
-------

That's it! This tutorial demonstrated how to use ``NDBuffer`` to manipulate structured data in SlangPy. While we focused on basic buffer operations, there’s much more to explore, such as:

- Using ``InstanceLists`` to call type methods.
- Leveraging ``Tensor`` for differentiable data manipulation.

