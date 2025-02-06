.. _generators_grid: 

Grid Generator
==============

The grid generator is the first generator that actually affects the **shape** of the kernel it is passed to. In this 
sense, it can be thought of as a form of procedural buffer or tensor. When used in its simplest form, ``grid`` behaves
exactly like ``call_id``, with the addition of an explicit shape:

As with ``call_id``, we'll start with a simple Slang function that takes and returns an ``int2``:

.. code-block::

    int2 myfunc(int2 value) {
        return value;
    }

We can invoke this function and pass it the ``grid`` generator as follows:

.. code-block:: python 

    # Populate a 4x4 numpy array of int2s with call ids
    res = module.myfunc(spy.grid(shape=(4,4)), _result='numpy')

    # [ [ [0,0], [0,1], [0,2], [0,3] ], [ [1,0], [1,1], [1,2], [1,3] ], ...
    print(res)

The ``grid`` generator provides the grid coordinate of the current thread and the resulting numpy array 
is populated accordingly. In this case, because we specified the grid should have a shape of (4,4), the
resulting kernel (and thus output) is 4x4.

Additionally, ``grid`` supports a stride argument:

.. code-block:: python

    # Populate a 4x4 numpy array of int2s with call ids
    res = module.myfunc(spy.grid(shape=(4,4), stride=(2,2)), _result='numpy')

    # [ [ [0,0], [0,2] ], [ [2,0], [2,2] ]
    print(res)

The shaped nature of the grid generator can also be seen if we attempt to mix it with a result buffer of a missmatched shape:

..code-block:: python 

    # Fail to populate a 4x4 numpy array of int2s from an 8x8 grid
    res = np.zeros((4, 4, 2), dtype=np.int32)
    module.myfunc(spy.grid(shape=(8,8)), _result=res)

Here we've explicitly asked for an 8x8 grid, but the result buffer is only 4x4. This will raise an error, as the shapes don't match.

However, the grid generator allows for any dimensions of the shape to be set to ``-1`` (undefined), in which case SlangPy will attempt to infer the shape from the kernel. This can be very useful if another parameter is controlling the shape, but you still want to have a strided grid:

..code-block:: python 

    # Don't fix the grid's shape, but specify an explicit stride and
    # provide a pre-allocated numpy array to populate.
    res = np.zeros((4, 4, 2), dtype=np.int32)
    module.myfunc(spy.grid(shape=(-1,-1), stride=(4,4)), _result=res)

    # [ [ [0,0], [0,4], [0,8], [0,12] ], [ [4,0], [4,4], [4,8], [4,12] ], ...
    print(res)

In this case, the grid is allowed to be any size, but the stride is fixed to 4x4. The result buffer is pre-allocated to 4x4, and the grid is populated accordingly.