Generators
==========

SlangPy provides a way to generate data dynamically within kernels, eliminating the need to supply it in a buffer or tensor. This is achieved using generators.

Code for all generator examples can be found `here <https://github.com/shader-slang/slangpy-samples/tree/main/examples/generators>`_

Generators can be passed to a Slang function in Python just like any other argument. When the kernel runs, the correct values are automatically passed to the corresponding parameter. For example, the following code demonstrates how to pass the `call_id` generator to a kernel:

.. code-block:: python

    res = np.zeros((4,4,2), dtype=np.int32)
    module.myfunc(spy.call_id(), _result=res)

The current generators available in SlangPy are:

- :ref:`Call Id <generators_callid>`
- :ref:`Thread Id <generators_threadid>`
- :ref:`Wang Hash <generators_wanghash>`
- :ref:`Rand float <generators_randfloat>`
- :ref:`Grid <generators_grid>`
