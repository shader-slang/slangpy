Your First Function
===================

In this example, we'll initialize SGL, create a simple Slang function, and call it from Python.

You can find the complete code for this example `here <https://github.com/shader-slang/slangpy/tree/main/examples/first_function/>`_.

First, let's define a simple Slang function to add two numbers together:

.. code-block::

    // example.slang

    // A simple function that adds two numbers together
    float add(float a, float b)
    {
        return a + b;
    }

Next, we'll create a Python script to initialize SGL, load the Slang module, and call the function:

.. code-block:: python

    ## main.py

    import slangpy as spy
    import pathlib

    # Create an SGL device with the local folder for slangpy includes
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

While this is a fun demonstration, dispatching a compute kernel just to add two numbers isn't particularly efficient! However, now that we have a functioning setup, we can scale it up and call the function with arrays instead:

.. code-block:: python

    ## main.py

    # ... initialization here ...

    # Create a couple of buffers with 1,000,000 random floats
    a = np.random.rand(1000000)
    b = np.random.rand(1000000)

    # Call our function and request a numpy array as the result (default would be a buffer)
    result = module.add(a, b, _result='numpy')

    # Print the first 10 results
    print(result[:10])

SlangPy supports a wide range of data types and can handle arrays with arbitrary dimensions. This example demonstrates how a single Slang function can be called with both scalars and NumPy arrays. Beyond this, SlangPy also supports many more types such as buffers, textures, and tensors.
