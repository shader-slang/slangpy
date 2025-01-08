Buffers
=======

SlangPy supplies 2 important wrappers around a classic structured
buffer (represented in SGL as a Buffer object). These are the ``NDBuffer`` 
and ``NDDifferentiableBuffer``.

The ``NDBuffer`` types take a structured buffer that has a stride / size, and 
add to it:

* Element type - a ``SlangType``, which can be a primitive type such as float or vector, or a user defined Slang struct.
* Shape - a tuple of integers as with a classic numpy array or torch tensor.

Let's start with a simple slang program that uses a custom type:

.. code-block::
    
    // currently, to use custom types with slangpy it needs to be imported
    import "slangpy";

    // example.slang
    struct Pixel
    {
        float r;
        float g;
        float b;
    };

    // Add 2 pixels together
    Pixel add(Pixel a, Pixel b)
    {
        Pixel result;
        result.r = a.r + b.r;
        result.g = a.g + b.g;
        result.b = a.b + b.b;
        return result;
    }

Note that currently, in many situations a slang module needs to import the slangpy module for kernel 
generation to resolve all types correctly. This is an issue we aim to address in the near future.

Initialization is the same as the first function example:

.. code-block:: python

    import sgl
    import slangpy as spy
    import pathlib
    import numpy as np

    # Create an SGL device with the slangpy+local include paths
    device = sgl.Device(compiler_options={
        "include_paths": [
            spy.SHADER_PATH,
            pathlib.Path(__file__).parent.absolute(),
        ],
    })

    # Load module
    module = spy.Module.load_from_file(device, "example.slang")

Now we'll construct and initialize 2 buffers of type Pixel. We'll use a buffer cursor to 
populate the first, and a pure numpy array of floats to populate the 2nd.

.. code-block:: python

    # Create a couple of 2D 16x16 buffers
    image_1 = spy.NDBuffer(device, element_type=module.Pixel, shape=(16,16))
    image_2 = spy.NDBuffer(device, element_type=module.Pixel, shape=(16,16))

    # Use a cursor to fill the first buffer with readable structured data.
    cursor_1 = image_1.cursor()
    for x in range(16):
        for y in range(16):      
            cursor_1[x+y*16].write({
                'r': (x+y)/32.0,
                'g': 0,
                'b': 0,
            })
    cursor_1.apply()

    # Use the fact that we know the buffers are just 16x16 grids of 3 floats
    # to populate the 2nd buffer straight from random numpy array
    image_2.from_numpy(0.1*np.random.rand(16*16*3).astype(np.float32))

Whilst the cursor is definitely more wordy, it can be a very useful tool in both SGL and SlangPy,
as it allows you to both read and write structured data, and even view the contents of a GPU
buffer in the VSCode watch window.

Once the data is ready, we can call our function as normal:

.. code-block:: python

    # Call the module's add function
    result = module.add(image_1, image_2)

As SlangPy knows the 2 buffers are in effect 2D arrays of Pixels, it can infer that this is 
a `2D` dispatch (in this case of 16*16 threads), in which each thread will read a Pixel from 
each buffer, add them together and write the result to a 3rd buffer. In the absence of any 
override, slangpy automatically allocates and returns a new NDBuffer.

Alternatively, we can pre-allocate the buffer and pass it in:

.. code-block:: python

    # Alternative - pre-allocate the buffer
    result = spy.NDBuffer(device, element_type=module.Pixel, shape=(16,16))
    module.add(image_1, image_2, _result=result)

This can be very useful in scenarios in which you have all your inputs/outputs pre-allocated up 
front.

Finally, let's both print out the result and (if it's running) use tev to display the result:

.. code-block:: python

    # Use a cursor to read and print pixels (would also be readable in the watch window)
    result_cursor = result.cursor()
    for x in range(16):
        for y in range(16):
            pixel = result_cursor[x+y*16].read()
            print(f"Pixel ({x},{y}): {pixel}")

    # Or if installed, we can use tev to show the result (https://github.com/Tom94/tev)
    tex = device.create_texture(data=result.to_numpy(), width=16, height=16, format=sgl.Format.rgb32_float)
    sgl.tev.show(tex)

That's the lot! This tutorial demonstrated how to use NDBuffers to manipulate structured data in SlangPy. 
What it hasn't covered is the use of ``InstanceLists`` to actually call type methods, or the use 
of ``NDDifferentiableBuffer`` to store and manipulate differentiable data.