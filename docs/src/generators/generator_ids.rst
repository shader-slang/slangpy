.. _id_generators:

Id Generators
=============

Id generators provide unique identifiers related to the current thread, which can be passed to a function. Currently available generators are ``call_id`` and ``thread_id``.

.. _generators_callid:

Call Id
-------

Similar to traditional compute kernels, SlangPy assigns a unique grid coordinate to each thread. However, while classic compute kernels operate within a fixed 3D grid, SlangPy supports arbitrary dimensionality. The ``call_id`` generator returns the grid coordinate of the current thread within SlangPy's execution model.

Consider the following simple Slang function, which takes and returns an ``int2``:

.. code-block::

    int2 myfunc(int2 value) {
        return value;
    }

We can invoke this function and pass it the ``call_id`` generator as follows:

.. code-block:: python

    # Populate a 4x4 numpy array of int2s with call IDs
    res = np.zeros((4,4,2), dtype=np.int32)
    module.myfunc(spy.call_id(), _result=res)

    # [ [ [0,0], [1,0], [2,0], [3,0] ], [ [0,1], [1,1], [2,1], [3,1] ], ...
    print(res)

The ``call_id`` generator provides the grid coordinate of the current thread. As a result, each entry in the numpy array is populated with its corresponding grid coordinate.

In this case, where the call id was passed to a vector type (``int2``), the x component represents the right-most dimension, the y component the next dimension to the left, and so on. As a result, the pixel on row 0, column 1 has been passed the vector value ``int2(1,0)``. This behaviour for vectors is consistent throughout SlangPy, and is designed to make
vector based indices map intuitively to how we think about coordinates within an image.

Alternatively, we could invoke the following function with the same arguments:

.. code-block::

    int2 myfuncarray(int[2] value) {
        return int2(value[0],value[1]);
    }

Now that the coordinates are represented as an array, they fall back to the standard (for ML) ordering in which the last dimension (in this case, D1) is the right most dimension. This means that the pixel on row 0, column 1 would be passed the array value ``[0,1]``. Consequentially, the output is transposed:

.. code-block:: python

    # Do the same but with a function that takes an int[2] array as input
    res = np.zeros((4, 4, 2), dtype=np.int32)
    module.myfuncarray(spy.call_id(), _result=res)

    # [ [ [0,0], [0,1], [0,2], [0,3] ], [ [1,0], [1,1], [1,2], [1,3] ], ...
    print(res)

When using ``call_id``, ensure that the parameter type matches the dimensionality of the dispatch. In this example, since the dispatch was a 2D kernel, the parameter was an ``int2``.

Note that we explicitly created the numpy array ``res``. This is necessary because the ``call_id`` generator does not define any inherent shape. Without a predefined 4x4 container, SlangPy would have no way to infer the intended dispatch size.

.. note::
   **Vector vs Array Dimension Ordering**

   See :ref:`index_representation` for complete details on the differing index representation conventions between vectors and arrays.

.. _generators_threadid:

Thread Id
---------

In some cases, it is useful to access the actual dispatch thread ID being executed. This can be achieved by using the ``thread_id`` generator:

.. code-block::

    int3 myfunc3d(int3 value) {
        return value;
    }

Passing ``thread_id`` returns the 3D dispatch thread ID for each call:

.. code-block:: python

    # Populate a 4x4 numpy array of int3s with hardware thread IDs
    res = np.zeros((4,4,3), dtype=np.int32)
    module.myfunc3d(spy.thread_id(), _result=res)

    #[ [ [0,0,0], [1,0,0], [2,0,0], [3,0,0] ], [ [4,0,0], [5,0,0], ...
    print(res)

The ``thread_id`` generator can be used with 1D, 2D, or 3D vectors.

.. warning::
   **Vector vs Array Dimension Ordering**

   As with the ``call_id`` generator, the convention used for thread coordinates depends on the parameter type that the ``thread_id`` generator is passed to.
   See :ref:`index_representation` for complete details on these differing index representation conventions.

Currently, SlangPy maps kernels to a 1D grid on the hardware, meaning that thread IDs will always have the form ``[X,0,0]``. This behavior may be subject to future modifications and user control.
