Your First Function
===================

In this simple example we're going to initial SGL, create a simple slang function, and 
call it from Python.

The code for this example can be found `here <https://github.com/shader-slang/slangpy/tree/main/examples/first_function/>`_.

First let's create a simple slang function to add 2 numbers together.

.. code-block::
    
    // example.slang

    // A simple function that adds two numbers together
    float add(float a, float b)
    {
        return a + b;
    }

Now create a simple Python script that starts up SGL, loads the slang module and 
calls the function.

.. code-block:: python

    ## main.py

    import sgl
    import slangpy as spy
    import pathlib

    # Create an SGL device with the slangpy+local include paths
    device = sgl.Device(compiler_options={
        "include_paths": [
            spy.SHADER_PATH,
            pathlib.Path(__file__).parent.absolute(),
        ],
    })

    # Load module
    module = spy.Module.load_from_file(device, "example.slang")

    # Call the function and print the result
    result = module.add(1.0,2.0)
    print(result)

    # SlangPy also supports named parameters
    result = module.add(a=1.0, b=2.0)
    print(result)

Under the hood, the first time the function is invoked SlangPy will generate a compute kernel 
(and save a copy to a .temp folder) that loads the scalar inputs from buffers, calls the 
``add`` function and writes the scalar result back to a buffer. 

This is fun, but obviously not particularly useful or efficient. Dispatching a compute kernel 
just to add 2 numbers together is a bit overkill! However, now that we can add 2 numbers, 
we can choose to call the function with arrays instead:

.. code-block:: python

    ## main.py

    #.... init here ....

    # Create a couple of buffers with 1,000,000 random floats in
    a = np.random.rand(1000000)
    b = np.random.rand(1000000)

    # Call our function and ask for a numpy array back (the default would be a buffer)
    result = module.add(a, b, _result='numpy')

    # Print the first 10
    print(result[:10])

SlangPy deals with many different types, and can handle arbitrary numbers of dimensions. This example shows how a single slang 
function can be written and called in a couple of simple ways - using scalars and numpy arrays, but SlangPy supports many other types such 
as buffers, textures and tensors. 
