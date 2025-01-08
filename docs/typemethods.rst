Type Methods and Instance Lists
===============================

Thus far we've seen how SlangPy can understand both basic and user defined Slang types, and 
vectorise across global functions written in Slang. However it can also understand and 
call methods of Slang types, both mutable and immutable. This process starts with the use of 
the ``InstanceList`` and ``InstanceBuffer`` classes:

* ``InstanceBuffer``: Represents a list of instances of a Slang type stored in a single NDBuffer.
* ``InstanceList``: Represents a list of instances of a Slang type stored in SOA form, where separate fields can be stored in separate buffers.

Instance buffers
----------------

First we'll write a simple particle class that can be constructed and updated.

.. code-block::
    
    import "randfloatarg";

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

Note importing "randfloatarg" was necessary as we'll use slangpy to pass random floating point values
to the constructor. This is an issue we intend to resolve in the near future.

We can now create a buffer of 10 particles and call their constructor.

.. code-block:: python

    # ... device/module init code here ...

    # Create buffer of particles (.as_struct is used to make python typing happy!)
    particles = spy.InstanceBuffer(
        struct=module.Particle.as_struct(), 
        shape=(10,))

    # Construct every particle with position of 0, and use slangpy's rand_float 
    # functionality to supply a different rand vector for each one.
    particles.construct(
        p = sgl.float3(0),
        v = spy.rand_float(-1,1,3)
    )

    # Print all the particles by breaking them down into groups of 6 floats
    print(particles.to_numpy().view(dtype=np.float32).reshape(-1,6))

Just as with global functions, SlangPy can infer the types of the constructor arguments and deals
with broadcasting them to the correct size. In this case, the position of 0 is broadcast to every thread 
and SlangPy generates a random 3D vector for each velocity.

Even though neither parameter is multi dimensional, the ``this`` parameter is also passed in implicitly,
which is a 1D buffer of 10 particles, resulting in a 1D dispatch of 10 threads.

In exactly the same way, we can now update every particle:

.. code-block:: python

    # Update the particles
    particles.update(0.1)

    # Print all the particles by breaking them down into groups of 6 floats
    print(particles.to_numpy().view(dtype=np.float32).reshape(-1,6))

Note that the ``update`` method is marked as mutating in Slang, which tells SlangPy to treat the 'this'
parameter as inout, and thus copy the modified particles back to the buffer after ``Particle.update`` is called.

Additionally, just as with the ``NDBuffer``, an ``InstanceBuffer`` can be passed as a parameter to 
a global function, and SlangPy will automatically generate the correct kernel to read from it.

Instance Lists 
--------------

Instance lists are very similar to buffers, however they act as a class that has individual fields 
stored in separate buffers. They can also be inherited from in Python, allowing the user to mix and
match Python side code/fields and Slang side data.

Aside from tweaking the prints, the only change to our earlier code is to replace 
``InstanceBuffer`` with ``InstanceList`` and initialize it with a buffer for positions, and 
a buffer for velocities:

.. code-block:: python

    # Create buffer of particles (.as_struct is used to make python typing happy!)
    particles = spy.InstanceList(
        struct=module.Particle.as_struct(), 
        data={
            "position": spy.NDBuffer(device, element_type=module.float3, shape=(10,)),
            "velocity": spy.NDBuffer(device, element_type=module.float3, shape=(10,)),
        })

SlangPy will now automatically generate different kernels that reads from the position and velocity 
buffers, call a particle method and (optionally) write the new position/velocity back.

As with the ``InstanceBuffer``, the ``InstanceList`` can be passed as a parameter to a global function,
and SlangPy will automatically generate the correct kernel to read from it.

Inheriting Instance List 
------------------------

As the instance list is aware of its Slang structure, it is able to differentiate between 
Slang fields and Python fields. This allows the user to inherit from the instance list and
add their own fields/methods:

.. code-block:: python

    # Custom type that wraps an InstanceList of particles
    class MyParticles(spy.InstanceList):

        def __init__(self, name: str, count: int):
            super().__init__(module.Particle.as_struct())
            self.name = name
            self.position = spy.NDBuffer(device, element_type=module.float3, shape=(count,))
            self.velocity = spy.NDBuffer(device, element_type=module.float3, shape=(count,))

        def print_particles(self):
            print(self.name)
            print(self.position.to_numpy().view(dtype=np.float32).reshape(-1,3))
            print(self.velocity.to_numpy().view(dtype=np.float32).reshape(-1,3))

Here the majority of the earlier code has been cleanly wrapped in a Python class, which has 
an additional 'name' field to assist with debugging. ``construct`` and ``update`` are added 
by the base class, and can be called as usual.

Note that if the simplified ``InstanceBuffer`` is preferable, it can also be inherited from and 
will support the same general functionality. In this case, Slang fields are ignored and all 
attributes are assumed to be Python only.

Summary
-------

This example demonstrated the use of instance lists and buffers to allow the user to call 
methods on types.

Whilst it is beyond the scope of this tutorial, custom implementations of 
an InstanceList are also possible by implementing the ``IThis`` protocol - namely providing
``get_this`` and ``update_this`` functions. 