.. _call_group_shape:

Call Group Shape
================

By default, SlangPy assigns one thread per element of the dispatch and numbers those threads linearly across the entire dispatch. ``call_group_shape()`` lets you partition a functional-API compute dispatch into rectangular tiles — called *call groups*. For an explicit group shape whose dimensions have a product greater than one, each tile is executed as a single GPU thread group. This controls how the linear dispatch thread IDs described on the :ref:`Id Generators <id_generators>` page are assigned across those tiles.

.. note::
   This page describes functional-API compute calls. Ray-tracing dispatches do not use this GPU thread-group model.

A Simple Example
----------------

Consider a 1D function that records the call-group ID and the thread's position within that group:

.. code-block::

    // myshader.slang
    import "slangpy";

    uint2 record_group(uint x) {
        CallShapeInfo group_id = CallShapeInfo::get_call_group_id();
        CallShapeInfo local_id = CallShapeInfo::get_call_group_thread_id();
        return uint2(group_id.shape[0], local_id.shape[0]);
    }

The Slang snippets on this page are cumulative. Add each later declaration to the same ``myshader.slang`` file before running its Python snippet.

The :ref:`grid generator <generators_grid>` supplies the one-dimensional call shape. Without an explicit call-group shape, ``call_group_id`` is the absolute call ID and ``call_group_thread_id`` is zero:

.. code-block:: python

    import slangpy as spy
    from slangpy.slangpy import Shape

    device = spy.create_device()
    module = spy.Module.load_from_file(device, "myshader.slang")

    default_result = module.record_group(spy.grid((8,)), _result="numpy")
    print(default_result.tolist())
    # [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0]]

Now suppose you want the 8 threads arranged into groups of 2. You can express this with ``call_group_shape``:

.. code-block:: python

    grouped_result = module.record_group.call_group_shape(Shape((2,)))(
        spy.grid((8,)), _result="numpy"
    )
    print(grouped_result.tolist())
    # [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1], [3, 0], [3, 1]]

This produces four GPU thread groups of two threads each. The ungrouped call still uses SlangPy's default 32-thread GPU group, but threads beyond the eight-element call shape exit before the user function runs. Call-group metadata treats each in-bounds call as its own group in that default case.

Accessing Group Information in Slang
-------------------------------------

Your Slang code can query where each thread sits relative to its group, with or without an explicit call-group shape. SlangPy provides three static methods on ``CallShapeInfo`` for this (available after ``import "slangpy"``):

- ``CallShapeInfo::get_call_id()`` — the thread's absolute grid coordinate
- ``CallShapeInfo::get_call_group_id()`` — which group this thread belongs to
- ``CallShapeInfo::get_call_group_thread_id()`` — the thread's position *within* its group

Each returns a ``CallShapeInfo`` value with two members: ``.dimensionality`` (the number of dimensions) and ``.shape`` (an array of indices). The indices follow ML convention: ``.shape[0]`` is the outermost dimension (row/Y), ``.shape[1]`` is the next (column/X), and so on.

The three values are related by a simple formula for each dimension ``i``:

.. code-block::

    call_id[i]  ==  call_group_id[i] * group_shape[i]  +  call_group_thread_id[i]

Here, ``group_shape`` means the effective call-group shape after SlangPy pads any omitted leading dimensions with ones; it is not a member of ``CallShapeInfo``.

Here is a 2D function that reports its group coordinates and flattened local thread index:

.. code-block::

    import "slangpy";

    uint3 report_ids(uint2 grid_cell) {
        CallShapeInfo group_id = CallShapeInfo::get_call_group_id();
        CallShapeInfo local_id = CallShapeInfo::get_call_group_thread_id();

        // Return (call_group_id.y, call_group_id.x, local thread index within group)
        return uint3(group_id.shape[0],
                     group_id.shape[1],
                     local_id.shape[0] * 4 + local_id.shape[1]);  // flattened local index for a 4-wide group
    }

Calling this over an ``(8, 8)`` grid with ``(2, 4)`` call groups:

.. code-block:: python

    result = module.report_ids.call_group_shape(Shape((2, 4)))(spy.grid((8, 8)), _result="numpy")

produces an ``(8, 8, 3)`` result where each entry is ``[group_row, group_col, local_index]``. For example:

.. code-block:: python

    result[0, 0].tolist()  # [0, 0, 0]
    result[1, 3].tolist()  # [0, 0, 7]
    result[2, 4].tolist()  # [1, 1, 0]

Without ``call_group_shape``, ``call_id`` and ``call_group_id`` are equal and ``call_group_thread_id`` is always ``0`` in every dimension.

Effect on Thread IDs
--------------------

Internally, each call group is dispatched as a single 1D GPU thread group whose size is the *product* of the call group dimensions. For ``Shape((2, 4))`` that is 8 threads. The GPU does not receive a 2D thread group — all the 2D structure is handled by SlangPy's kernel wrapper.

This grouping changes the flattened dispatch ID supplied when a ``spy.thread_id()`` generator is bound to a Slang integer or vector parameter. Without call groups, its shader-side ``x`` component increases linearly across the whole dispatch. With call groups, threads within the same group receive contiguous IDs — threads in later groups receive higher IDs.

**Without call groups** — 32×64 dispatch, linear thread IDs:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20

   * - Position
     - ``call_id``
     - ``thread_id.x``
   * - [0, 0]
     - [0, 0]
     - 0
   * - [0, 1]
     - [0, 1]
     - 1
   * - [1, 0]
     - [1, 0]
     - 64
   * - [1, 1]
     - [1, 1]
     - 65

**With** ``call_group_shape = (4, 8)`` — threads in each 4×8 tile have contiguous IDs:

.. list-table::
   :header-rows: 1
   :widths: 12 16 20 28 16

   * - Position
     - ``call_id``
     - ``call_group_id``
     - ``call_group_thread_id``
     - ``thread_id.x``
   * - [0, 0]
     - [0, 0]
     - [0, 0]
     - [0, 0]
     - 0
   * - [0, 1]
     - [0, 1]
     - [0, 0]
     - [0, 1]
     - 1
   * - [1, 0]
     - [1, 0]
     - [0, 0]
     - [1, 0]
     - 8
   * - [1, 1]
     - [1, 1]
     - [0, 0]
     - [1, 1]
     - 9
   * - [0, 8]
     - [0, 8]
     - [0, 1]
     - [0, 0]
     - 32
   * - [4, 0]
     - [4, 0]
     - [1, 0]
     - [0, 0]
     - 256

Notice that ``call_id`` is the same in both tables — it always reflects the thread's absolute grid coordinate regardless of grouping. Only the flattened ``thread_id.x`` assignment changes.

Using Groupshared Memory
------------------------

All threads in a call group are dispatched in the same GPU thread group and therefore share the same on-chip scratchpad (``groupshared`` / LDS). To index into a per-group ``groupshared`` array, use the *flattened* ``call_group_thread_id`` — not the shader-side value bound from ``spy.thread_id()``. The latter is a global index that increases across the entire dispatch, so its ``x`` component is out of range for any group after the first. ``call_group_thread_id``, by contrast, gives a per-dimension position within the group: dimension ``i`` runs from ``0`` to ``group_shape[i] - 1``, and the flattened value ``local_row * TILE_W + local_col`` spans ``0`` to ``group_size - 1``:

.. code-block::

    import "slangpy";

    #define TILE_H 4
    #define TILE_W 8

    groupshared float tile[TILE_H * TILE_W];  // 32 slots, one per thread in the group

    float tile_sum(uint2 grid_cell) {
        CallShapeInfo local = CallShapeInfo::get_call_group_thread_id();
        int local_row = local.shape[0];
        int local_col = local.shape[1];
        int local_idx = local_row * TILE_W + local_col;  // flatten to 1D, range [0, 31]

        // Write this thread's value into shared memory and sync
        tile[local_idx] = 1.0f;
        GroupMemoryBarrierWithGroupSync();

        // Every thread reads the whole tile and sums it
        float total = 0.0f;
        [ForceUnroll]
        for (int i = 0; i < TILE_H * TILE_W; ++i)
            total += tile[i];

        return total;  // 32.0 for a fully populated group
    }

The Python call-group shape must match the tile dimensions compiled into the Slang code:

.. code-block:: python

    TILE_H, TILE_W = 4, 8

    result = module.tile_sum.call_group_shape(Shape((TILE_H, TILE_W)))(
        spy.grid((8, 16)), _result="numpy"
    )

    print(result[0, 0])  # 32.0

Misaligned Dispatch Sizes
--------------------------

If the call shape is not an exact multiple of the group shape in every dimension, SlangPy rounds up the GPU dispatch to the nearest aligned size. The extra threads exit before any user code runs and never write to an automatically created output, whose shape remains the original logical call shape:

.. code-block::

    spy.grid((9,)),  call_group_shape = (4,)
    → dispatch rounds up to 12 threads
    → threads at positions 9, 10, 11 exit early

For functions without group-wide synchronization, the discarded threads usually only add wasted work. A function that uses ``GroupMemoryBarrierWithGroupSync()`` has a stricter requirement: every hardware thread in the group must reach the barrier. SlangPy's early return makes barrier participation non-uniform in a *boundary group* that straddles the call-shape edge. The behavior is undefined and may include incorrect results or a GPU hang; unwritten ``groupshared`` elements also have undefined values.

For example, a call over ``(20, 16)`` with ``call_group_shape = (16, 16)`` creates a boundary group covering rows 16–31. Only 64 of its 256 threads enter the user function; the other 192 exit first. A group-wide barrier reached by only those 64 threads is invalid, and the reduction has no defined result.

For functions that contain group-wide barriers, make the logical call shape an exact multiple of the group shape in every dimension. For the ``tile_sum`` example above, pad both dimensions and then slice the result:

.. code-block:: python

    TILE_H, TILE_W = 4, 8
    H, W = 6, 10

    H_padded = ((H + TILE_H - 1) // TILE_H) * TILE_H  # 6 -> 8
    W_padded = ((W + TILE_W - 1) // TILE_W) * TILE_W  # 10 -> 16

    padded_result = module.tile_sum.call_group_shape(Shape((TILE_H, TILE_W)))(
        spy.grid((H_padded, W_padded)), _result="numpy"
    )

    result = padded_result[:H, :W]

For a real function with input data, pad every input that contributes to the call shape and initialize the padded elements to suitable neutral values. If the function receives the original extent so it can mask reads or writes, all padded calls must still follow uniform control flow through every group-wide barrier; only mask output writes after the final barrier.

Constraints
-----------

- **All group dimensions must be ≥ 1.** Zero or negative values raise an error.
- **SlangPy rejects a total group size greater than 1024.** Because generated groups are physically one-dimensional, the product must also fit both ``device.info.limits.max_compute_threads_per_group`` and ``device.info.limits.max_compute_thread_group_size.x`` for the active device.
- **The group cannot have more dimensions than the call shape.** A 3D group shape on a 2D call is an error.
- **A group with fewer dimensions than the call shape is padded with leading 1s.** For a 2D call, ``Shape((8,))`` is treated as ``Shape((1, 8))``. ``Shape(())`` disables explicit grouping. A compatible all-ones shape also has no call-group effect, but the usual dimensionality check still applies; for example, ``Shape((1, 1))`` is invalid for a 1D call.
