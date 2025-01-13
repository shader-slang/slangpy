Mapping
=======

Mapping provides a way to explicitly control the relationship between argument dimensions and kernel dimensions in SlangPy. This extends the broadcasting rules discussed earlier, giving you more precise control over vectorization.

A Simple Example
----------------

Consider this call to an `add` function that adds two floats:

.. code-block:: python

    a = np.random.rand(10, 3, 4)
    b = np.random.rand(10, 3, 4)
    result = mymodule.add(a, b, _result='numpy')

In this case:

- ``a`` and ``b`` are arguments to the ``add`` kernel, each with the shape ``(10, 3, 4)``.
- The kernel is dispatched with an overall shape of ``(10, 3, 4)``.
- Each thread indexed by ``[i, j, k]`` processes ``a[i, j, k]`` and ``b[i, j, k]``, writing the result to ``result[i, j, k]``.

This represents a straightforward 1-to-1 mapping between argument dimensions and kernel dimensions.

Re-Mapping Dimensions
----------------------

The ``map`` function allows you to modify how argument dimensions correspond to kernel dimensions. For example, the earlier code could be rewritten as:

.. code-block:: python

    a = np.random.rand(10, 3, 4)
    b = np.random.rand(10, 3, 4)
    result = mymodule.add.map((0, 1, 2), (0, 1, 2))(a, b, _result='numpy')

Here, the tuples passed to ``map`` explicitly define the mapping: dimension 0 maps to 0, dimension 1 to 1, and dimension 2 to 2 for both ``a`` and ``b``. This is the default behavior in SlangPy.

Alternatively, you can use named parameters for clarity:

.. code-block:: python

    # Assuming the add function has the signature add(float3 a, float3 b)
    a = np.random.rand(10, 3, 4)
    b = np.random.rand(10, 3, 4)
    result = mymodule.add.map(a=(0, 1, 2), b=(0, 1, 2))(a=a, b=b, _result='numpy')

----

**Mapping arguments with different dimensionalities**

Unlike NumPy, SlangPy does not auto-pad dimensions by default. If this behavior is needed, ``map`` lets you explicitly define how smaller inputs are aligned with the kernel:

.. code-block:: python

    a = np.random.rand(8, 8).astype(np.float32)
    b = np.random.rand(8).astype(np.float32)

    # This will fail in SlangPy, as `b` is not automatically extended:
    result = mymodule.add(a, b, _result='numpy')

    # Use explicit mapping instead:
    # Equivalent to padding `b` as NumPy would
    result = mymodule.add.map(a=(0, 1), b=(1,))(a=a, b=b, _result='numpy')

    # Alternatively, you can omit mapping for a as it defaults to 1-to-1:
    result = mymodule.add.map(b=(1,))(a=a, b=b, _result='numpy')

----

**Mapping arguments to different dimensions**

Another use case is performing some operation in which you wish to broadcast all the elements of one argument across the other. The simplest is the mathematical outer-product:

.. code-block:: python

    # Assuming the multiply function has the signature multiply(float a, float b)
    a = np.random.rand(10).astype(np.float32)
    b = np.random.rand(20).astype(np.float32)

    # Map dimensions:
    # - a maps to dimension 0 (size 10)
    # - b maps to dimension 1 (size 20)
    # Resulting kernel and output shape: (10, 20)
    result = mymodule.multiply.map(a=(0,), b=(1,))(a=a, b=b, _result='numpy')

----

**Mapping to re-order dimensions**

Re-ordering argument dimensions is straightforward with ``map``. For example, to transpose a matrix:

.. code-block:: python

    # Assuming the copy function has the signature float copy(float val)
    a = np.random.rand(10, 20).astype(np.float32)

    # Swap rows and columns:
    result = mymodule.copy.map(val=(1, 0))(val=a, _result='numpy')

----

**Mapping to resolve ambiguities**

``map`` can resolve ambiguities that would otherwise prevent SlangPy from vectorizing. For example:

.. code-block:: python

    # A generic function from the 'nested' section:
    void copy_generic<T>(T src, out T dest) {
        dest = src;
    }

    # Explicitly map dimensions to remove ambiguity:
    src = np.random.rand(100).astype(np.float32)
    dest = np.zeros_like(src)
    result = module.copy_generic.map(src=(0,), dest=(0,))(src=src, dest=dest)

Slangpy now knows:

- ``src`` and ``dest`` should map 1 dimension
- ``src`` and ``dest`` are both 1D arrays of ``float``

Thus it can infer that you want to pass ``float`` into ``copy_generic`` and generates the correct kernel.

Mapping Types
-------------

``map`` can also define argument types directly, which may improve readability for simple cases:

.. code-block:: python

    src = np.random.rand(100)
    dest = np.zeros_like(src)

    # Map argument types explicitly:
    result = module.copy_generic.map(src='float', dest='float')(src=src, dest=dest)

Where in the previous example SlangPy inferred type from dimensionality, it now knows:

- ``src`` and ``dest`` should map to ``float``
- ``src`` and ``dest`` are both 1D arrays of ``float``

Thus it can infer that you want a 1D kernel.

Summary
-------

The ``map`` function in SlangPy provides powerful tools for customizing how arguments align with kernel dimensions. This capability allows you to:

- Precisely control dimension mappings for arguments, enabling efficient vectorization of complex operations.
- Handle cases where arguments have different dimensionalities by explicitly aligning dimensions, avoiding the need for auto-padding.
- Perform operations like broadcasting (e.g., outer products) and reordering dimensions (e.g., matrix transposition) with ease.
- Resolve ambiguities in generic functions, ensuring correct kernel generation and execution.

These features make ``map`` particularly useful for machine learning algorithms, where operations often involve multi-dimensional data with varying shapes and alignment requirements. By enabling fine-grained control over dimension mappings, SlangPy helps optimize operations like tensor manipulations, matrix multiplications, and custom kernels, which are foundational to modern ML workflows.
