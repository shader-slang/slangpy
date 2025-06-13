Return Types
============

Slangpy can generate different types of container to hold the returned results of a Slang function. This is convenient for getting results in a preferred container type, such as a numpy array, texture, or tensor.

Let's start by reusing the example from :ref:`firstfunctions`.

Shader:

.. code-block::

    // example.slang

    // A simple function that adds two numbers together
    float add(float a, float b)
    {
        return a + b;
    }

In the original :ref:`firstfunctions` python example, we returned the result in a numpy array. Let's return it as a texture instead:

.. code-block:: python

    ## main.py

    # ... initialization here ...

    # Create a couple of buffers with 128x128 random floats
    a = np.random.rand(128, 128).astype(np.float32)
    b = np.random.rand(128, 128).astype(np.float32)

    # Call our function and ask for a texture back
    result = module.add(a, b, _result='texture')

    # Print the first 5x5 values
    print(result.to_numpy()[:5, :5])

    # Display the result using tev
    spy.tev.show(result, name='add random')

Here we use ``_result`` to specify that we want the result to be a texture.

The ``_result`` can be ``'numpy'``, ``'texture'``, or ``'tensor'``. You can also use ``_result`` to specify the type directly, like ``numpy.ndarray``, ``slangpy.Texture``, or ``slangpy.Tensor``. Or you can reuse an existing variable of one of those types by passing it directly.

You'll see more examples using ``_result`` in the rest of this documentation!
