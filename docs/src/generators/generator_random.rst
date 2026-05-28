Random Number Generators
========================

While there are many ways to generate random numbers on GPUs, SlangPy provides a couple of utilities to help you get started.

.. _generators_wanghash:

Wang Hash
---------

The Wang hash generator returns a 1D, 2D, or 3D integer hash value based on the current dispatch thread ID. As with the ID generator examples, we start with a simple Slang function that takes and returns an ``int2``:

.. code-block::

    int2 myfunc(int2 value) {
        return value;
    }

We can invoke this function and pass it the Wang hash generator as follows:

.. code-block:: python

    # Populate a 4x4 numpy array of int2s with random integer hashes
    res = np.zeros((4, 4, 2), dtype=np.int32)
    module.myfunc(spy.wang_hash(), _result=res)

    # Example output:
    # [-1062647446,-1219659480], [663891101,1738326990] ...
    print(res)

Each entry in the numpy array is now populated with a random integer hash. The ``wang_hash`` generator also supports an optional seed value.

As with the ID generator examples, we explicitly create the numpy array ``res``. This is necessary because the Wang hash generator does not define any inherent shape. Without a predefined 4x4 container, SlangPy would have no way to infer the intended dispatch size.

.. _generators_randfloat:

Random Float
------------

The random float generator builds on top of Wang hash to generate a random float within a given range. The following Slang function takes and returns a ``float2``:

.. code-block::

    float2 myfuncfloat(float2 value) {
        return value;
    }

We can invoke this function and pass it the random float generator as follows:

.. code-block:: python

    # Populate a 4x4 numpy array of float2s with random values
    res = np.zeros((4, 4, 2), dtype=np.float32)
    module.myfuncfloat(spy.rand_float(min=0, max=10), _result=res)

    # Example output:
    # [3.0781631,3.6783838], [3.2699034, 4.611035] ...
    print(res)

Just like the ``wang_hash`` generator, ``rand_float`` supports an optional seed value and does not define any inherent shape.
