Nested Types
============

SlangPy supports multiple ways of passing structured data to functions. The simplest approach is through Python dictionaries. This example demonstrates how structured data can be passed in SOA (Structure of Arrays) form and combined GPU side.

Passing a Dictionary
--------------------

Let's start with a straightforward function that copies the value of one `float4` into another:

.. code-block::

    void copy(float4 src, out float4 dest)
    {
        dest = src;
    }

We could directly use the ``copy`` function to copy a NumPy array into a texture, though this is 
relatively pointless since we can manually set the texture's data using SGL's ``from_numpy`` function.

.. code-block:: python

    # Create a texture to store results
    tex = device.create_texture(
        width=128,
        height=128,
        format=sgl.Format.rgba32_float,
        usage=sgl.ResourceUsage.shader_resource | sgl.ResourceUsage.unordered_access
    )

    # Copy source values into the texture
    module.copy(
        src=np.random.rand(128 * 128 * 4).reshape(128, 128, 4).astype(np.float32),
        dest=tex
    )

    # Display the result
    sgl.tev.show(tex, name='tex')

Instead, we can pass a **dictionary** as the source argument, specifying individual fields of
the source vector:

.. code-block:: python

    # Use dictionary nesting to copy structured source values into the texture
    module.copy(
        src={
            'x': 1.0,
            'y': spy.rand_float(min=0, max=1, dim=1),
            'z': 0.0,
            'w': 1.0
        },
        dest=tex
    )

Here `x`, `z` and `w` are set to constant values, while `y` is set to a random float between 0 and 1. However they could just as easily be Numpy arrays, NDBuffers or even (in this case) 1D textures!

This nesting approach works with any structured data, including multi-level custom structures. A common use case is storing data in **SOA form** (e.g., separate lists for particle positions and velocities) and combining them GPU-side into a single structure.

Explicit Types
--------------

Dictionaries are convenient but lack **type information**. In simple scenarios, SlangPy infers type information from the Slang function. However, type inference can break down when dealing with **generic functions**:

.. code-block::

    void copy_vector<let N : int>(vector<float, N> src, out vector<float, N> dest)
    {
        dest = src;
    }

Because the function explicitly specifies ``vector<float, N>`` as the argument type, SlangPy can map the ``Texture<float4>`` to ``vector<float, 4>``. However, dictionaries do not carry enough type information, leading to errors in the following code:

.. code-block:: python

    # This will cause an error
    module.copy_vector(
        src={
            'x': 1.0,
            'y': spy.rand_float(min=0, max=1, dim=1),
            'z': 0.0,
            'w': 1.0
        },
        dest=tex
    )

One workaround is explicitly requesting the specialized version of ``copy_vector`` from the module:

.. code-block:: python

    # Explicitly fetch the version of copy_vector with N=4
    copy_func = module.require_function('copy_vector<4>')

    # Call the specialized function
    copy_func(
        src={
            'x': 1.0,
            'y': spy.rand_float(min=0, max=1, dim=1),
            'z': 0.0,
            'w': 1.0
        },
        dest=tex
    )

While effective, this approach is not the most elegant and would generally be a last resort.

A more convenient solution for dictionaries is adding the ``_type`` field, explicitly specifying the structure's type:

.. code-block:: python

    # Explicitly declare the type using '_type'
    module.copy_vector(
        src={
            '_type': 'float4',
            'x': 1.0,
            'y': spy.rand_float(min=0, max=1, dim=1),
            'z': 0.0,
            'w': 1.0
        },
        dest=tex
    )

This approach avoids function specialization while keeping the dictionary structure clean and explicit.

If the function is made fully generic, even the texture argument will face ambiguity:

.. code-block::

    void copy_generic<T>(T src, out T dest)
    {
        dest = src;
    }

In this scenario, SlangPy has no way of knowing the concrete types of ``src`` and ``dest``. To resolve this, we can use the ``map`` method to explicitly define how the Python types should map to Slang types:

.. code-block:: python

    # Map argument types explicitly
    module.copy_generic.map(src='float4', dest='float4')(
        src={
            'x': 1.0,
            'y': spy.rand_float(min=0, max=1, dim=1),
            'z': 0.0,
            'w': 1.0
        },
        dest=tex
    )

The `map` method serves as SlangPy's primary mechanism for resolving type information in complex scenarios. It ensures accurate kernel generation and avoids runtime errors caused by ambiguous types.

Summary
-------

This example demonstrated:

- **Structured Data Passing:** Using dictionaries to represent structured arguments.
- **Type Resolution:** Handling generic functions with explicit mappings or ``_type`` fields.

The use of dictionraries to represent SOA data can be especially powerful when experimenting
with different ways to store data host side without worrying about how it affects shaders or kernel 
invokation.
