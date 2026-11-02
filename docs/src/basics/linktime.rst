Link-Time Specialization
========================

Slang can leave a constant or a type undefined in a module and have it supplied when the program is linked. SlangPy
exposes this for the functions it calls, which makes it the way to specialize a shader per call site without
editing its source or recompiling the module for every variant. Because the values are known when the kernel is
compiled, the downstream compiler can fold them and remove branches that can never run.

Link-time constants
-------------------

Declare the constants with ``extern``. They can live in their own module, imported by the code that uses them:

.. code-block::

    // settings.slang
    module settings;

    public extern static const int SAMPLE_COUNT;
    public extern static const bool USE_FAST_PATH;

.. code-block::

    // shading.slang
    import settings;

    float shade(float x)
    {
        float sum = 0.0;
        for (int i = 0; i < SAMPLE_COUNT; i++)
            sum += USE_FAST_PATH ? x : x * x;
        return sum;
    }

Then give the values with ``constants()`` on the function. It returns a new function object, so several
specializations of the same function can be kept side by side:

.. code-block:: python

    module = spy.Module.load_from_file(device, "shading.slang")

    shade_fast = module.shade.constants({"SAMPLE_COUNT": 4, "USE_FAST_PATH": True})
    shade_exact = module.shade.constants({"SAMPLE_COUNT": 4, "USE_FAST_PATH": False})

    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    print(shade_fast(x, _result="numpy"))   # [ 4.  8. 12.]
    print(shade_exact(x, _result="numpy"))  # [ 4. 16. 36.]

``constants()`` takes a dictionary from constant name to a Python ``bool``, ``int`` or ``float``, or a SlangPy vector
such as ``spy.float3``. SlangPy adds a matching ``export static const`` declaration to the kernel it generates for the
call, and Slang links it against the ``extern`` declaration. Every constant the called code uses must be given a value:
with a missing one, the call fails and Slang reports ``unresolved external symbol``.

Each distinct set of values is compiled once, as a kernel of its own, and reused afterwards. The kernel is keyed on its
complete generated source, which depends on the call's arguments as well as the constants, and lists the constants in
the dictionary's order: calls with the same arguments share one kernel only if they build the dictionary the same way.
Changing a value therefore costs a compile, which suits options that select a code path (a feature switch, a sample
count, a quality level) better than data that changes on every call; pass the latter as a normal argument.

Linking a module of constants
-----------------------------

The values can also come from a Slang module with ``export`` declarations, given in the ``link`` list when the
module is loaded. Every function of the loaded module then uses those values:

.. code-block:: python

    values = device.load_module_from_source(
        "values",
        """
        export static const int SAMPLE_COUNT = 8;
        export static const bool USE_FAST_PATH = true;
        """,
    )
    module = spy.Module.load_from_file(device, "shading.slang", link=[values])
    print(module.shade(2.0))  # 16.0

Link-time types
---------------

An ``extern struct`` that conforms to an interface works the same way for types. The implementation can be added to
the generated kernel with ``prelude()``:

.. code-block::

    // values.slang
    interface IVal
    {
        static int get();
    }

    extern struct VAL : IVal;

    int get_value()
    {
        return VAL::get();
    }

.. code-block:: python

    module = spy.Module.load_from_file(device, "values.slang")

    val_10 = """
    struct Val10 : IVal
    {
        static int get() { return 10; }
    }
    export struct VAL : IVal = Val10;
    """
    print(module.get_value.prelude(val_10)())  # 10
