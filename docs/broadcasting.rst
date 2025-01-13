Broadcasting
============

As we've seen so far in these tutorials, SlangPy's primary job is to take a function
that is designed to run on a single unit of data, and convert it to a `vector` function that 
runs on batches of data in parallel.

You can find the complete code for this example `here <https://github.com/shader-slang/slangpy/tree/main/examples/broadcasting/>`_.

So far, we've seen very simple examples of vectorizing in which all the parameters of a function 
are passed either equally sized buffers, or single values. For example:

.. code-block:: python

    # Adding all elements in 2 buffers of equal size
    mymodule.add(np.array([1, 2, 3]), np.array([4, 5, 6])

    # Adding a single value to every element of a buffer 
    mymodule.add(5, np.array([4, 5, 6])    

    # Adding all elements in 2 2D buffers of the same shape (2x2)
    mymodule.add(np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]))

The process of taking the arguments and inferring how to vectorize the function is known as 
auto-broadcasting. 

Before diving in, some terminology:

- `Dimensionality`: The number of dimensions of a value. For example, a 1D buffer has a `dimensionality` of 1, 
  a 2D buffer has a `dimensionality` of 2, a volume texture has a `dimensionality` of 3 etc.
- `Shape`: The size of each dimension of a value. For example, a 1D buffer of size 3 has a `shape` of (3,), a 32x32x32 volume texture has a shape of (32,32,32).

In effect, `dimensionality` is equal to the length of the `shape` tuple.

Note: For those new to broadcasting, a common point of confusion is that a `3D vector` does **not** have
a `dimensionality` of 3! Instead, it has a `dimensionality` of 1, and its `shape` is (3,).

Broadcasting with single floats
-------------------------------

Let's start with simple arrays of floats, as they're simple to reason about.

SlangPy's process is as follows:

- First, calculate the largest `dimensionality` of all arguments. This determines `dimensionality` of the kernel and its output.
- For each dimension, all the argument sizes must be compatible. Two sizes are compatible if they are equal or 1.
- If a dimension's size is 1, it is broadcast to match the size of the other arguments.

For example, consider the following cases of a function that takes 2 inputs A and B of given **shapes**, 
and generates an output of a given **shape**:

.. code-block:: python

    # For a function Out[x,y,z] = A[x,y,z] + B[x,y,z]

    # All dimensions match
    # Out[x,y,z] = A[x,y,z] + B[x,y,z]
    A       (10,3,4)
    B       (10,3,4)
    Out     (10,3,4)

    # A's first dimension is 1, so it is broadcast
    # Out[x,y,z] = A[x,0,z] + B[x,y,z]
    A       (10,1,4)
    B       (10,3,4)
    Out     (10,3,4)

    # Error as A and B's first dimensions are different sizes and not 1
    # Out[??,y,z] = A[??,y,z] + B[??,y,z]
    A       (10,3,4)
    B       (5,3,4)
    Out     Error

SlangPy will also support broadcasting a single value to all dimensions of the output. Programmatically,
a single value can be thought of as a value that isn't indexed - its dimensionality is 0, and its shape 
is (). 

Conceptually, broadcasting the same value to all dimensions is similar to adding dimensions of size 
1 to the value until it matches the output's dimensionality:

.. code-block:: python

    # For a function Out[x,y,z] = A[x,y,z] + B[x,y,z]

    # A single value is broadcast to all dimensions of the output
    # Out[x,y,z] = A + B[x,y,z]
    A       ()
    B       (10,3,4)
    Out     (10,3,4)

    # Conceptually the same as adding dimensions of size 1 to the 
    # value until it matches the output dimensionality.
    # Out[x,y,z] = A[0,0,0] + B[x,y,z]
    A       (1,1,1)
    B       (10,3,4)
    Out     (10,3,4)

Where SlangPy differs from Numpy and certain other ML packages is that it will by design **not**
automatically extend the dimensions of a value **unless** it is a single value. This is to prevent
accidental broadcasting of values that should be treated as errors. For example, consider the following

.. code-block:: python 

    Numpy would automatically extend A to (1,3,4), SlangPy does not
    A      (3,4)
    B      (10,3,4)
    Out    Error

Broadcasting with other types 
-----------------------------

Whilst Numpy and PyTorch operate only on simple data types such as float, int and bool, SlangPy 
functions can take any type of data as input - scalars, vectors, matrices, arrays, structs, buffers etc.
This makes the rules for broadcasting slightly more complex. Consider the following 2 functions:

.. code-block::

    float add_floats(float a, float b) { ... }

    float3 add_vectors(float3 a, float3 b) { ... }

How arguments are translated to the vectorized function depends on the function's signature. For example:

.. code-block:: python 

    a = float3(1,2,3);
    b = float3(4,5,6);

    # Each argument is treated as having shape (3,)
    # The kernel is invoked 3 times
    mymodule.add_floats(a,b)

    # Each argument is treated as having shape ()
    # The kernel is invoked once
    mymodule.add_vectors(a,b)

Here 2 vectors that have shape (3,) are passed to the 2 functions. In the first case, because
the parameters are genuine scalars, the output shape ends up also being (3,). However, in the second
case, the parameters themselves are vectors, so the output shape is ().

If we were to introduce an numpy array to the equation

.. code-block:: python

    a = float3(1,2,3);
    b = np.random.rand(10,3); # 10x3 random array

    # a has shape (3,), b has shape (10,3). 
    # As we don't auto-extend a, this is an error
    mymodule.add_floats(a,b)

    # a is treated as having shape (), b is treated as having shape (10,). 
    # The kernel is invoked 10 times and a is broadcast to all threads
    mymodule.add_vectors(a,b)

The general rule is that to calculate the dimensionality of an argument, SlangPy subtracts the dimensionality of the Slang parameter from the dimensionality of the Python input. So when a (10,3) buffer is passed to a function that takes a float3, the last dimension is consumed, leaving an 
argument shape of (10,).


Summary
-------

This tutorial gave an overview of how vectorizing and broadcasting work in SlangPy. If you're already familiar with Numpy, PyTorch or other ML frameworks it should be very familiar, with the only real extra complication being that of handling non-scalar types.

If you're new to broadcasting, this first read might have made your head spin a little. Don't worry! It's a topic that is **way** easier to learn in practice than in theory. The best way to get a feel for it is to start writing some SlangPy functions and see how the broadcasting rules work (or don't!) in practice. The `examples <https://github.com/shader-slang/slangpy/tree/main/examples/broadcasting/>`_ for this tutorial are a good place to start.


The next tutorial will cover use of the ``map`` function to be explicit about how arguments are mapped to the output allowing for more complex broadcasting rules.
