.. _generators_grid:

Grid Generator
==============

The grid generator is the first generator that directly influences the **shape** of the kernel it is passed to. In this sense, it functions as a procedural buffer or tensor. When used in its simplest form, ``grid`` behaves similarly to ``call_id`` but with the addition of an explicit shape parameter.

As with ``call_id``, we start with a simple Slang function that takes and returns an ``int2``:

.. code-block::

    int2 myfunc(int2 value) {
        return value;
    }

We can invoke this function and pass it the ``grid`` generator as follows:

.. code-block:: python

    # Populate a 4x4 numpy array of int2s with call IDs
    res = module.myfunc(spy.grid(shape=(4,4)), _result='numpy')

    # Example output:
    # [ [ [0,0], [1,0], [2,0], [3,0] ], [ [0,1], [1,1], [2,1], [3,1] ], ...
    print(res)

The ``grid`` generator provides the grid coordinate of the current thread, and the resulting numpy array is populated accordingly. Since we specified a shape of ``(4,4)``, the resulting kernel and output conform to this 4x4 structure.

.. warning::
   **Vector vs Array Dimension Ordering**

   As with the ``call_id`` generator, the convention used for grid coordinates depends on the parameter type that the ``grid`` generator is passed to.
   When passed to a vector parameter, the x component represents the smallest stride, the y component the next smallest stride, and so on.
   When passed to an array parameter, the right-most dimension has the smallest stride, the next dimension to the left has the next smallest stride, and so on.
   The x component of the vector is equivalent to the right-most dimension of the array, the y component the next dimension to the left, and so on.
   See :ref:`index_representation` for complete details on these differing index representation conventions.

Strides
-------

Additionally, ``grid`` supports a stride argument:

.. code-block:: python

    # Populate a 4x4 numpy array of int2s with call IDs using strides
    res = module.myfunc(spy.grid(shape=(4,4), stride=(2,2)), _result='numpy')

    # Example output:
    # [ [ [0,0], [2,0], [4,0], [6,0] ], [ [0,2], [2,2], [4,2], [6,2] ], ...
    print(res)

When both shape and stride are specified, the number of elements in the grid is determined by the shape, while the stride controls the spacing between elements. In this case, the grid is populated with a stride of ``(2,2)``.

Shape Mismatches
----------------

The shaped nature of the ``grid`` generator becomes evident when attempting to use it with a mismatched result buffer:

.. code-block:: python

    # Attempt to populate a 4x4 numpy array with an 8x8 grid
    res = np.zeros((4, 4, 2), dtype=np.int32)
    module.myfunc(spy.grid(shape=(8,8)), _result=res)  # This will raise an error

Here, we explicitly request an 8x8 grid, but the result buffer is only 4x4. Since the shapes do not match, this will trigger an error.

Undefined Dimensions
--------------------

The grid generator allows for any dimension of the shape to be set to ``-1`` (undefined), allowing SlangPy to infer the appropriate shape from the kernel. This is especially useful when another parameter controls the shape, but a specific stride is still needed:

.. code-block:: python

    # Allow the grid shape to be inferred while specifying a fixed stride
    res = np.zeros((4, 4, 2), dtype=np.int32)
    module.myfunc(spy.grid(shape=(-1,-1), stride=(4,4)), _result=res)

    # Example output:
    # [ [ [0,0], [4,0], [8,0], [12,0] ], [ [0,4], [4,4], [8,4], [12,4] ], ...
    print(res)

In this case, the grid size is inferred dynamically, but the stride remains fixed at ``(4,4)``. Since the result buffer is pre-allocated to ``(4,4)``, the grid is populated accordingly.
