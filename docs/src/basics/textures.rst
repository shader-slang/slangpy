Brighten A Texture
==================

In this example, we'll use SlangPy to read from and write to a texture, showcasing simple broadcasting and ``inout`` parameters. To view the results, you'll need `tev <https://github.com/Tom94/tev>`_.

You can find the complete code for this example `here <https://github.com/shader-slang/slangpy-samples/tree/main/examples/textures>`_.

Slang Code
----------

This Slang code defines a simple function that adds a value to an ``inout`` parameter:

.. code-block::

    // Add an amount to a given pixel
    void brighten(float4 amount, inout float4 pixel)
    {
        pixel += amount;
    }

Generating the Texture
----------------------

We'll skip the device initialization and module loading steps and go straight to generating and displaying a random texture:

.. code-block:: python

    # ... device initialization and module loading here ...

    # Generate a random image
    rand_image = np.random.rand(128 * 128 * 4).astype(np.float32) * 0.25
    tex = device.create_texture(
        width=128,
        height=128,
        format=spy.Format.rgba32_float,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        data=rand_image
    )

    # Display it with tev
    spy.tev.show(tex, name='photo')

*Note:* The texture is created with both ``shader_resource`` and ``unordered_access`` usage flags, enabling it to be both read from and written to in a shader.

Brightening the Texture
-----------------------

Next, we call the ``brighten`` function and display the updated texture:

.. code-block:: python

    # Call the module's brighten function, passing:
    # - a float4 constant broadcast to every pixel
    # - the texture as an inout parameter
    module.brighten(spy.float4(0.5), tex)

    # Display the result
    spy.tev.show(tex, name='brighter')

In this example:

- SlangPy infers a **2D dispatch** because a 2D texture of ``float4`` is passed into the function.
- The **first parameter** (a single ``float4``) is **broadcast** to every thread.
- The **second parameter** (marked ``inout``) allows both reading from and writing to the texture.

Summary
-------

In this example we've seen:

- **Textures:** How to read and write each pixel of a texture.
- **Broadcasting:** How a single scalar can be broadcast to every thread.

The same `brighten` function could also be used in many other ways, such as:

- Adding two textures together.
- Adding a buffer to a texture.
- Adding a texture to a buffer.

SlangPy's flexibility allows seamless integration between these types, making it easy to extend this example for more advanced scenarios.

Texture loading and color interpretation
----------------------------------------

``TextureLoader.Options.encoding`` selects ``TextureEncoding.automatic`` (the
default), ``linear``, or ``srgb``. Automatic follows bitmap gamma metadata.
Explicit linear or sRGB interpretation overrides that metadata without changing
the bitmap or re-encoding its pixels. Alpha remains linear.

.. code-block:: python

    loader = spy.TextureLoader(device)
    texture = loader.load_texture(
        "base_color.png",
        options={
            "encoding": spy.TextureEncoding.srgb,
            "y_handling": spy.YHandling.expand_to_rgba,
        },
    )

Channel expansion is independent of encoding. Y can be replicated into RGB with
alpha one; YA can be expanded while preserving alpha. sRGB sampling applies to
8-bit RGBA after expansion. Other component types, preserved R/RG layouts and
floating-point HDR inputs keep their existing formats. DDS textures retain their
authored formats.

This replaces ``load_as_srgb``: migrate ``True`` to ``TextureEncoding.automatic``
to preserve its metadata-dependent behavior, and ``False`` to
``TextureEncoding.linear``. Use ``TextureEncoding.srgb`` when the caller knows
that the stored color values are sRGB regardless of image metadata.
