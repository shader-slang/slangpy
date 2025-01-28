Type Methods
============

So far, we've seen how SlangPy understands both basic and user-defined Slang types and can vectorize across global Slang functions. In addition to this, SlangPy can also call methods of Slang types, whether mutable or immutable. This is achieved using two key classes:

- ``InstanceBuffer``: Represents a list of instances of a Slang type stored in a single ``NDBuffer``.  
- ``InstanceList``: Represents a list of instances of a Slang type stored in SOA (Structure of Arrays) form, where separate fields can be stored in separate buffers.

Instance Buffers
----------------

First, let's define a simple ``Particle`` class that can be constructed and updated:

.. code-block::

    import "slangpy";

    struct Particle
    {
        float3 position;
        float3 velocity;

        __init(float3 p, float3 v)
        {
            position = p;
            velocity = v;
        }

        [mutating]
        void update(float dt)
        {
            position += velocity * dt;
        }
    };

*Note:* Importing ``slangpy`` is necessary because in this example we use it to pass random floating-point values to the constructor. This requirement will be addressed in a future update.

Creating and Initializing Particles
-----------------------------------

Now, let's create a buffer of 10 particles and call their constructor:

.. code-block:: python

    # ... device/module initialization here ...

    # Create a buffer of particles (.as_struct ensures proper Python typing)
    particles = spy.InstanceBuffer(
        struct=module.Particle.as_struct(),
        shape=(10,)
    )

    # Construct particles with a position of (0, 0, 0) and random velocities
    particles.construct(
        p=sgl.float3(0),
        v=spy.rand_float(-1, 1, 3)
    )

    # Print particle data as groups of 6 floats (position + velocity)
    print(particles.to_numpy().view(dtype=np.float32).reshape(-1, 6))

SlangPy infers the types of the constructor arguments and broadcasts them appropriately:

- ``p`` (position) is broadcast to all 10 particles.
- ``v`` (velocity) generates a random 3D vector for each particle.

The ``this`` parameter is implicitly passed as a **1D buffer of 10 particles**, resulting in a **1D dispatch** of 10 threads.

Updating Particles
------------------

We can now update every particle:

.. code-block:: python

    # Update particle positions with a time delta of 0.1
    particles.update(0.1)

    # Print updated particle data
    print(particles.to_numpy().view(dtype=np.float32).reshape(-1, 6))

The ``update`` method is marked ``[mutating]`` in Slang, signaling to SlangPy that the ``this`` parameter is ``inout``. This ensures the modified particles are written back to the buffer after the method completes.

*Note:* Just like ``NDBuffer``, an ``InstanceBuffer`` can be passed as a parameter to a function, and SlangPy will automatically generate the correct kernel for reading from it.

Instance Lists
--------------

``InstanceList`` functions similarly to ``InstanceBuffer`` but can store individual fields in separate buffers (Structure of Arrays):

.. code-block:: python

    # Create an InstanceList for particles
    particles = spy.InstanceList(
        struct=module.Particle.as_struct(),
        data={
            "position": spy.NDBuffer(device, dtype=module.float3, shape=(10,)),
            "velocity": spy.NDBuffer(device, dtype=module.float3, shape=(10,))
        }
    )

The calls to ``construct`` and ``update`` will remain identical, however SlangPy will now generate a kernel that reads from and writes to the separate ``position`` and ``velocity`` buffers.

*Note:* Similar to ``InstanceBuffer``, an ``InstanceList`` can also be passed as a parameter, and SlangPy will handle kernel generation accordingly.

Inheriting InstanceList
-----------------------

``InstanceList`` is able to distingish between Slang fields and Python attributes, and thus supports Python-side inheritance, enabling you to extend its functionality with custom attribues and methods.

.. code-block:: python

    # Custom type wrapping an InstanceList of particles
    class MyParticles(spy.InstanceList):

        def __init__(self, name: str, count: int):
            super().__init__(module.Particle.as_struct())
            self.name = name
            self.position = spy.NDBuffer(device, dtype=module.float3, shape=(count,))
            self.velocity = spy.NDBuffer(device, dtype=module.float3, shape=(count,))

        def print_particles(self):
            print(self.name)
            print(self.position.to_numpy().view(dtype=np.float32).reshape(-1, 3))
            print(self.velocity.to_numpy().view(dtype=np.float32).reshape(-1, 3))

Here we've wrapped up the previous example in a simple class and added a Python only ``name`` field to assist with debugging. The ``construct`` and ``update`` methods will be inherited and an instance of ``MyParticles`` can be passed as a parameter to any function expecting a ``Particle``.

If you prefer the simplified ``InstanceBuffer``, it can also be inherited in the same way. In this case, Slang fields are ignored, and all attributes are assumed to be Python-only.

Summary
-------

This example demonstrated:

- Using `InstanceBuffer` and `InstanceList` for managing Slang type instances.
- Calling methods on Slang types, both mutable (`[mutating]`) and immutable.
- Broadcasting parameters across instances.
- Inheriting and extending `InstanceList` in Python.

``InstanceBuffer`` especially is a very lightweight wrapper, and can be used interchangeably with normal 
``NDBuffers``. As a result, favouring the use of an ``InstanceBuffer`` unless you have a good reason 
not to is generally recommended.