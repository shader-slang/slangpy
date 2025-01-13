Mapping
=======

In the previous broadcasting section, we saw how SlangPy applies broadcasting rules to automatically vectorize a function. Mapping gives more control over this process by allowing the user to explicitly specify the relationship between argument and kernel dimensions.

Consider the following simple call to an `add` function that adds 2 floats:

.. code-block:: python 

    a = np.random.rand(10,3,4)
    b = np.random.rand(10,3,4)
    result = mymodule.add(a,b, _result='numpy')

In this example:

- ``a`` and ``b`` are `arguments` to the ``add`` kernel, each with shape ``(10,3,4)``. 
- The kernel is dispatched with overall shape ``(10,3,4)``.
- Any given thread, ``[i,j,k]``, reads ``a[i,j,k]`` and ``b[i,j,k]`` and writes ``result[i,j,k]``. 

There is a simple 1-to-1 mapping of argument dimensions to kernel dimensions.

Re-mapping dimensions
---------------------

``map`` can be used to change how argument dimensions correspond to kernel dimensions. In the above example, we could have written:

.. code-block:: python 

    a = np.random.rand(10,3,4)
    b = np.random.rand(10,3,4)
    result = mymodule.add.map((0,1,2), (0,1,2))(a,b, _result='numpy')

The tuples passed to map specify how to map dimensions for each argument. In this case we're mapping dimension 0 to dimension 0, dimension 1 to dimension 1 and dimension 2 to dimension 2 for both a and b. This 1-to-1 mapping is the default behaviour. 

Mapping works with named parameters too, which can be a little clearer:

.. code-block:: python 

    # Assume the slang add function has signature add(float3 a, float3 b)
    a = np.random.rand(10,3,4)
    b = np.random.rand(10,3,4)
    result = mymodule.add.map(a=(0,1,2), b=(0,1,2))(a=a,b=b, _result='numpy')

----

**Mapping arguments with different dimensionalities**

As we've already seen, unlike Numpy, SlangPy by design doesn't auto-pad dimensions. When this behaviour is desirable, explicit mapping can be used to tell SlangPy exactly how to map the smaller inputs to those of the overall kernel:

.. code-block:: python 

    a = np.random.rand(8,8).astype(np.float32)
    b = np.random.rand(8).astype(np.float32)

    # Fails in SlangPy, as b is not auto-extended
    result = mymodule.add(a,b, _result='numpy')

    # Works, as we are explicilty mapping 
    # This is equivalent to padding b with empty dimensions, as numpy would
    # result[i,j] = a[i,j] + b[j]
    result = mymodule.add.map(a=(0,1), b=(1,))(a=a,b=b, _result='numpy')

    # The same thing (didn't need to specify a as 1-to-1 mapping is default)
    result = mymodule.add.map(b=(1,))(a=a,b=b, _result='numpy')

----

**Mapping arguments to different dimensions**

Another use case is performing some operation in which you wish to broadcast all the elements of one argument across the other. The simplest is the mathematical outer-product:

.. code-block:: python 

    # Assume the slang multiply function has signature multiply(float a, float b)
    # a is mapped to dimension 0, giving kernel dimension [0] size 10
    # b is mapped to dimension 1, giving kernel dimension [1] size 20
    # overall kernel (and thus result) shape is (10,20)
    # result[i,j] = a[i] * b[j]
    a = np.random.rand(10).astype(np.float32)
    b = np.random.rand(20).astype(np.float32)
    result = mymodule.multiply.map(a=(0,), b=(1,))(a=a,b=b, _result='numpy')

----

**Mapping to re-order dimensions**

Similarly, dimension indices can be adjusted to re-order the dimensions of an argument. A trivial example to transpose a matrix (replace rows with columns) would be:

.. code-block:: python 

    # Assume the slang copy function has signature float copy(float val)
    # and just returns the value you pass it.
    # result[i,j] = a[j,i]
    a = np.random.rand(10,20).astype(np.float32)
    result = mymodule.copy.map(val=(1,0))(val=a, _result='numpy')

----

**Mapping to resolve ambiguities**

Mapping can also be used to resolve ambiguities that would prevent SlangPy vectorizing normally. For example, consider the following generic function (from the `nested` section):

.. code-block::

    void copy_generic<T>(T src, out T dest)
    {
        dest = src;
    }

One way to resolve the ambiguities is to map dimensions as follows:

.. code-block:: python

    # Map argument types explicitly
    src = np.random.rand(100).astype(np.float32)
    dest = np.zeros_like(src)
    module.copy_generic.map(src=(0,), dest=(0,))(
        src=src,
        dest=dest
    )

Slangpy now knows:

- ``src`` and ``dest`` should map 1 dimension
- ``src`` and ``dest`` are both 1D arrays of ``float``

Thus it can infer that you want to pass ``float`` into ``copy_generic`` and generates the correct kernel.

Mapping types
-------------

Mapping can also be used to specify the type of the argument. Whilst this approach cannot be used 
to re-order dimensions, it can be a more readable way to resolve simple ambiguities. For example, we
could write the ``copy_generic`` call from above as follows:

.. code-block:: python

    # Map argument types explicitly
    src = np.random.rand(100)
    dest = np.zeros_like(src)
    module.copy_generic.map(src='float', dest='float')(
        src=src,
        dest=dest
    )

Where in the previous example SlangPy inferred type from dimensionality, it now knows:

- ``src`` and ``dest`` should map to ``float``
- ``src`` and ``dest`` are both 1D arrays of ``float``

Thus it can infer that you want a 1D kernel.

Summary
-------

This section has shown how to use the ``map`` function to fully control how arguments are mapped to kernel dimensions in SlangPy. This powerful functionality allows the vectorization of algorithms
that are more than simply running the same function on many elements in an array.


