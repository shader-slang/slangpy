Brighten A Texture
==================

In this simple example we'll see the use of SlangPy to read/write a texture, along with 
the use of simple broadcasting and inout parameters. `tev <https://github.com/Tom94/tev>`_ is 
required to run the example and see results.

The full code for this example can be found `here <https://github.com/shader-slang/slangpy/tree/main/examples/textures>`_.

The Slang code is a very simple function that takes a constant and adds it to the value 
of an ``inout`` parameter.

.. code-block::
    
    // Add an amount to a given pixel
    void brighten(float4 amount, inout float4 pixel)
    {
        pixel += amount;
    }

We'll skip device initialization, and go straight to creating/showing a random texture:

.. code-block:: python

    #... device init + module load here ...

    # Generate a random image
    rand_image = np.random.rand(128*128*4).astype(np.float32)*0.25
    tex = device.create_texture(width=128, height=128, format=sgl.Format.rgba32_float,
                                usage=sgl.ResourceUsage.shader_resource | sgl.ResourceUsage.unordered_access,
                                data=rand_image)

    # Display it with tev
    sgl.tev.show(tex, name='photo')

Note the texture as created is both a shader resource and an unordered access resource so it 
can be read and written to in a shader.

We can now call the ``brighten`` function and show the result as usual:

.. code-block:: python

    # Call the module's add function, passing:
    # - a float4 constant that'll be broadcast to every pixel
    # - the texture to an inout parameter
    module.brighten(sgl.float4(0.5), tex)

    # Show the result
    sgl.tev.show(tex, name='brighter')

In this case SlangPy infers that this is a `2D` call, because it's passing a 2D texture of float4s
into a function that takes a float4. As the first parameter is a single float4, it gets broadcast
to every thread. Because the second parameter is an inout, SlangPy knows to both read and write
to the texture.

This very simple example shows manipulation of texture pixels and broadcasting. We could equally
have used the same function to add together 2 textures, or buffers, or a buffer and a texture, or 
a numpy array to a texture etc etc!


