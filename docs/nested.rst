Nested Types 
============

SlangPy supports various ways of passing structured data to functions, but the simplest is through
the use of Python dictionaries. This example will show how a structure can be passed in SOA form, 
rather than as a single buffer, and combined kernel side. 

Let's start with the simplest function possible - one that copies the value of a float4 into 
another!

.. code-block::
    
    void copy(float4 src, out float4 dest)
    {
        dest = src;
    }

We could simply use the ``copy`` method to copy a numpy array into the texture, though that would be
relatively pointless given we could just set the texture's data manually:

.. code-block:: python
    
    # Create a texture to store the results
    tex = device.create_texture(width=128, height=128, format=sgl.Format.rgba32_float,
                                usage=sgl.ResourceUsage.shader_resource | sgl.ResourceUsage.unordered_access)

    # Use copy function to copy source values to the texture
    module.copy(
        src=np.random.rand(128*128*4).reshape(128,128,4).astype(np.float32), 
        dest=tex)

    # Show the result
    sgl.tev.show(tex, name='tex')

However, now that we have the copy function, we can instead pass a dictionary
as the source argument, allowing us to specify the data for individual components:

.. code-block:: python
    
    # Use copy function to copy source values to the texture
    module.copy(
        src={
            'x': 1.0,
            'y': spy.rand_float(min=0,max=1,dim=1),
            'z': 0.0,
            'w': 1.0
        }, 
        dest=tex)

This nesting functionality can be applied to any structured data, including multi-level custom 
structures. A common use case is to store data in SOA form (say separate lists of particle positions 
and velocities), but process them GPU side as a single structure.

Explicit types 
--------------

The one downside of dictionaries is that they carry no type information. Where possible, such as 
in the previous example, SlangPy will attempt to infer type information from the Slang code. However,
if the Slang function is made more generic, this inference is no longer possible:

.. code-block::
    
    void copy_vector<let N : int>(vector<float, N> src, out vector<float, N> dest)
    {
        dest = src;
    }

As the generic function explicitly specifies the arguments must be vectors, SlangPy is able to 
match the Texture<float4> to the vector<float,4> type. However, the dictionary carries no type information
so the following code would cause an error:

.. code-block:: python
    
    # Tell slangpy that the src and dest types map to a float4
    module.copy_vector(
        src={
            'x': 1.0,
            'y': spy.rand_float(min=0,max=1,dim=1),
            'z': 0.0,
            'w': 1.0
        }, 
        dest=tex)

The simplest, though least elegant fix would be to explicitly request the specialized
version of copy_vector from the loaded module: 

.. code-block:: python
    
    # Explicitly search for the version of copy_generic we want
    copy_func = module.require_function('copy_vector<4>')

    # Call it as normal
    copy_func(
        src={
            'x': 1.0,
            'y': spy.rand_float(min=0,max=1,dim=1),
            'z': 0.0,
            'w': 1.0
        }, 
        dest=tex)

Generally this isn't recommended, but it's good to have in your back pocket as a last resort!

The 2nd approach, specific to dictionaries, is to add the ``_type`` field to the dictionary, which 
tells SlangPy exactly what struct the dictionary represents: 

.. code-block:: python
    
    module.copy_vector(
        src={
            '_type': 'float4',
            'x': 1.0,
            'y': spy.rand_float(min=0, max=1, dim=1),
            'z': 0.0,
            'w': 1.0
        },
        dest=tex)

If we were to make the function fully generic however, even the texture argument would have trouble.
SlangPy has no way of knowing what types the generic constraints should be solved with:

.. code-block::
    
    void copy_generic<T>(T src, out T dest)
    {
        dest = src;
    }

In this situation, we can use the ``map`` method to tell SlangPy exactly what types both 
arguments correspond to.

.. code-block:: python
        
    # Tell slangpy that the src and dest types map to a float4
    module.copy_generic.map(src='float4',dest='float4')(
        src={
            'x': 1.0,
            'y': spy.rand_float(min=0,max=1,dim=1),
            'z': 0.0,
            'w': 1.0
        }, 
        dest=tex)

Argument mapping will be covered in more detail in later tutorials, but is SlangPy's
key mechanism for resolving type information. 
