.. _call_group_shape:

Call Group Shape
================

By default, SlangPy assigns one thread per element of the dispatch and numbers those threads linearly across the entire dispatch. ``call_group_shape()`` lets you partition the dispatch into rectangular tiles — called *call groups* — where each tile is executed as a single GPU thread group. This controls how the linear hardware thread IDs described on the :ref:`Id Generators <id_generators>` page are assigned across those tiles.

A Simple Example
----------------

Consider a 1D kernel that records each thread's position:

.. code-block::

    // myshader.slang
    import "slangpy";

    uint record(uint x) {
        return x;
    }

Calling this without ``call_group_shape`` dispatches all 8 threads in a single flat group of 32 (the default hardware group size):

.. code-block:: python

    import slangpy as spy
    from slangpy.slangpy import Shape

    device = spy.create_device()
    module = spy.Module.load_from_file(device, "myshader.slang")

    result = module.record(spy.grid((8,)), _result="numpy")

Now suppose you want the 8 threads arranged into groups of 2. You can express this with ``call_group_shape``:

.. code-block:: python

    result = module.record.call_group_shape(Shape((2,)))(spy.grid((8,)), _result="numpy")

This produces 4 call groups of 2 threads each. The output is identical — ``call_group_shape`` changes how threads are organised on the GPU, not what each thread computes.

Accessing Group Information in Slang
-------------------------------------

Once call groups are enabled, your Slang code can query where each thread sits relative to its group. SlangPy provides three static methods on ``CallShapeInfo`` for this (available after ``import "slangpy"``):

- ``CallShapeInfo::get_call_id()`` — the thread's absolute grid coordinate
- ``CallShapeInfo::get_call_group_id()`` — which group this thread belongs to
- ``CallShapeInfo::get_call_group_thread_id()`` — the thread's position *within* its group

Each returns a ``CallShapeInfo`` value with two members: ``.dimensionality`` (the number of dimensions) and ``.shape`` (an array of indices). The indices follow ML convention: ``.shape[0]`` is the outermost dimension (row/Y), ``.shape[1]`` is the next (column/X), and so on.

The three values are related by a simple formula for each dimension ``i``:

.. code-block::

    call_id[i]  ==  call_group_id[i] * group_shape[i]  +  call_group_thread_id[i]

Here is a 2D kernel that reports all three for each thread:

.. code-block::

    import "slangpy";

    uint3 report_ids(uint2 grid_cell) {
        CallShapeInfo call_id  = CallShapeInfo::get_call_id();
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

produces an ``(8, 8, 3)`` result where each entry is ``[group_row, group_col, local_index]``.

Without ``call_group_shape``, ``call_id`` and ``call_group_id`` are equal and ``call_group_thread_id`` is always ``0`` in every dimension.

Effect on Thread IDs
--------------------

Internally, each call group is dispatched as a single 1D GPU thread group whose size is the *product* of the call group dimensions. For ``Shape((2, 4))`` that is 8 threads. The GPU does not receive a 2D thread group — all the 2D structure is handled by SlangPy's kernel wrapper.

This grouping changes how the raw hardware thread index (``spy.thread_id()``) is assigned. Without call groups, ``spy.thread_id()`` increases linearly across the whole dispatch. With call groups, threads within the same group receive contiguous hardware thread IDs — threads in later groups receive higher IDs.

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

Notice that ``call_id`` is the same in both tables — it always reflects the thread's absolute grid coordinate regardless of grouping. Only the underlying hardware thread index changes.

Using Groupshared Memory
------------------------

All threads in a call group are dispatched in the same GPU thread group and therefore share the same on-chip scratchpad (``groupshared`` / LDS). To index into a per-group ``groupshared`` array, use the *flattened* ``call_group_thread_id`` — not ``spy.thread_id()``. The value from ``spy.thread_id().x`` is a global index that increases across the entire dispatch; it will be out of range for any group after the first. ``call_group_thread_id``, by contrast, gives a per-dimension position within the group: dimension ``i`` runs from ``0`` to ``group_shape[i] - 1``, and the flattened value ``local_row * TILE_W + local_col`` spans ``0`` to ``group_size - 1``:

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

Misaligned Dispatch Sizes
--------------------------

If the dispatch size is not an exact multiple of the group shape in every dimension, SlangPy rounds up to the nearest aligned size. The extra threads exit before any user code runs and never write to the output, which is always sized to the original (unpadded) dispatch:

.. code-block::

    spy.grid((9,)),  call_group_shape = (4,)
    → dispatch rounds up to 12 threads
    → threads at positions 9, 10, 11 exit early

This is fine for simple kernels, but it matters for ``groupshared`` memory. Because the early exit happens before user code, out-of-bounds threads never reach ``GroupMemoryBarrierWithGroupSync()`` or write to ``groupshared`` memory. In a *boundary group* — one that straddles the edge of the dispatch — only the in-bounds threads participate. A kernel that assumes every ``groupshared`` slot is written will silently produce wrong results for those groups.

For example, a dispatch over ``(20, 16)`` with ``call_group_shape = (16, 16)`` creates a boundary group covering rows 16–31. Only 64 of its 256 threads are in-bounds; the other 192 exit early. A reduction that sums all 256 ``groupshared`` slots will read 64 rather than 256 for those rows.

The fix is to ensure the dispatch size is an exact multiple of the group shape in every dimension. Pad the dispatch, then slice the result:

.. code-block:: python

    import math
    from slangpy.slangpy import Shape

    TILE_H, TILE_W = 16, 16
    H, W = 20, 16

    H_padded = math.ceil(H / TILE_H) * TILE_H   # 20 → 32
    buffer_output = spy.Tensor.empty(device, dtype=module.float, shape=(H_padded, W))

    module.main.call_group_shape(Shape((TILE_H, TILE_W)))(
        spy.grid((H_padded, W)), _result=buffer_output
    )

    result = buffer_output.to_numpy()[:H, :W]

Constraints
-----------

- **All group dimensions must be ≥ 1.** Zero or negative values raise an error.
- **The total group size (product of all dimensions) must not exceed 1024** — the maximum thread group size enforced by most GPU APIs.
- **The group cannot have more dimensions than the dispatch size.** A 3D group shape on a 2D dispatch is an error.
- **A group with fewer dimensions than the dispatch size is padded with leading 1s.** For a 2D dispatch, ``Shape((8,))`` is treated as ``Shape((1, 8))``. ``Shape(())`` and ``Shape((1, 1))`` are both treated as "no call groups" (equivalent to calling without ``call_group_shape``).
