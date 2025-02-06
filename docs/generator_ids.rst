.. _id_generators:

Id Generators
=============

Id generators pass unique ids releated to the current thread to a function. Currently available are `call_id` and `thread_id`.

.. _generators_callid: 

Call Id
-------

As with classic compute kernels, SlangPy operates by assigning a unique grid coordinate to each thread. However where a classic compute kernel is always dispatched across a 3D grid, SlangPy supports any dimensionality. The ``call_id`` generator returns the SlangPy grid coordinate of the current thread.

If we start with the following simple Slang function that takes and returns an `int2`:

.. code-block:: slang

    int2 myfunc(int2 value) {
        return value;
    }

We can then invoke the function and pass it the ``call_id`` generator like so:

.. code-block:: python

    # Populate a 4x4 numpy array of int2s with call ids
    res = np.zeros((4,4,2), dtype=np.int32)
    module.myfunc(spy.call_id(), _result=res)

    #[ [ [0,0], [0,1], [0,2], [0,3] ], [ [1,0], [1,1], [1,2], [1,3] ], ... ]
    print(res)

The ``call_id`` generator will pass the grid coordinate of the current thread to the function. As a result, each 
entry in the numpy array is populated with its own grid coordinate.

When using ``call_id``, one must make sure the parameter type matches the dimensionality of the dispatch. In this example,
as the dispatch was a 2D kernel, the parameter was an int2.

Note that in this example we had to create the numpy array `res` ourselves. This is because the `call_id` generator does not define any shape itself. Without supplying a pre-created 4x4 container, SlangPy would have no way of inferring the fact that a 4x4 dispatch was desired.

.. _generators_threadid: 

Thread Id
---------

At times, it is still desirable to know the actual dispatch thread id being executed. This can be obtained by 
passing the ``thread_id`` generator instead, switching now to a 3D function:

.. code-block:: slang

    int3 myfunc3d(int3 value) {
        return value;
    }

Passing ``thread_id`` outputs the 3D dispatch thread id for each call:

.. code-block:: python

    # Populate a 4x4 numpy array of int3s with hardware thread ids
    res = np.zeros((4,4,3), dtype=np.int32)
    module.myfunc3d(spy.thread_id(), _result=res)

    #[ [ [0,0,0], [1,0,0], [2,0,0], [3,0,0] ], [ [4,0,0], [5,0,0], ... 
    print(res)

The ``thread_id`` generator supports being passed to 1D, 2D or 3D vectors.

Currently, SlangPy always maps kernels to a 1D grid on the hardware, so thread ids will always be of the form [X,0,0]. This is subject to change and user control in the future.
