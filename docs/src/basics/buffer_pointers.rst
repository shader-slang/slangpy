Buffer pointers
===============

``BufferPtr`` and ``RWBufferPtr`` are non-owning locations inside GPU buffers.
They support typed indexing, byte access and pointer arithmetic on D3D12, Vulkan,
CUDA and Metal. The shader types are available through ``import bufferptr;``;
this standalone module also works with low-level SGL kernels.

For example, save this as ``read_pointer.slang``:

.. code-block:: slang

    import bufferptr;

    uint read_value(BufferPtr<> src, uint index)
    {
        return src.as<uint>()[index];
    }

Then call it from Python:

.. code-block:: python

    import numpy as np
    import slangpy as spy

    device = spy.create_device()
    module = spy.Module.load_from_file(device, "read_pointer.slang")
    buffer = device.create_buffer(
        usage=spy.BufferUsage.shader_resource,
        data=np.array([10, 20, 30, 40], dtype=np.uint32),
    )
    ptr = spy.to_buffer_ptr(buffer, byte_offset=8)
    assert module.read_value(ptr, 0) == 30

Use ``spy.to_rw_buffer_ptr`` with a writable buffer for ``RWBufferPtr<>`` shader
parameters. Native C++ uses ``sgl::to_buffer_ptr`` and ``sgl::to_rw_buffer_ptr``
from ``sgl/device/buffer_ptr.h``. Both native cursor types and Python functional
calls accept these values. Host values bind to the corresponding untyped,
unaligned shader specialization; use ``as<T>()`` and
``with_load_alignment<N>()`` inside the shader to obtain other views.

Ownership and synchronization
-----------------------------

A pointer contains eight bytes and retains neither its buffer nor its device.
Keep every referenced allocation alive until GPU completion, including buffers
reached through pointer tables. Allocation replacement invalidates old pointers.
The caller must arrange resource states and barriers; binding pointer bits does
not declare resource usage. Subscripts read and write values rather than native
references. Updating one member can read and write the complete element, so
concurrent updates to different members are not independent or atomic.

Encoding and nulls
------------------

Vulkan, CUDA and Metal use native device pointers. D3D12 packs a descriptor index
into the low 32 bits and a byte offset into the high 32 bits. Encodings are
backend-specific and must not be serialized as portable asset data. The all-ones
encoding is null; zero-filled storage is not a portable null pointer because
D3D12 descriptor zero is valid. Default construction and shader ``nullptr`` use
the null sentinel. ``is_valid`` only checks that sentinel, not lifetime or bounds.

D3D12 read/write access modes use different SRV/UAV descriptors. Obtain both
from the host factories instead of casting between access modes in the shader.
Ordering is only meaningful inside one allocation and, on D3D12, one descriptor
view. Compare compatible pointers or ``nullptr``; use ``bits()`` explicitly for
encoded comparisons. Slang's implicit bool conversion can still accept some
unintended bool-convertible operands (`Slang #14268
<https://github.com/shader-slang/slang/issues/14268>`_).

Alignment and data layout
-------------------------

The host ``alignment`` argument checks the initial address against 4, 8 or 16
bytes; it does not change shader layout or access operands. Shader
``load_alignment`` defaults to zero for plain loads/stores. Values 4, 8 or 16
promise alignment at each effective address, including stores. Arithmetic retains
this promise; it does not verify it. Explicit ``load_aligned`` / ``store_aligned``
operations override it for that access.

Typed stepping includes element padding and follows the backend layout. Producers
and consumers must agree on field offsets as well as stride. Plain access still
requires natural type alignment and a size divisible by four. Shader operations
do not validate bounds, alignment, allocation identity or arithmetic overflow.

D3D12 raw access can disagree with structured-buffer/reflected layouts for nested
records with implicit padding (`Slang #14311
<https://github.com/shader-slang/slang/issues/14311>`_). Double stores also have a
known constant-lowering defect (`Slang #14259
<https://github.com/shader-slang/slang/issues/14259>`_). These cases are covered by
strict compiler regression tests; the API does not recursively reject affected
aggregate layouts or double fields.

Metal requirements
------------------

The initial Metal implementation requires ``spy.Feature.residency_set``. The host
factories reject devices without it because reflected direct-pointer tracking
cannot discover arbitrary resources referenced through GPU pointer tables.
Residency sets do not retain the Python/native buffer owner or provide barriers;
the ownership and synchronization rules above still apply. CPU and WebGPU are
outside this API's supported backend set.

Validation
----------

``slangpy/tests/device/test_buffer_ptr.py`` covers host construction, layout,
cursor writes and functional calls. ``test_buffer_ptr_extended.py`` exercises
arithmetic, alignment, casts, pointer tables, linked traversal, aggregate layouts
and compiler regressions. Both run in ordinary PR CI using the platform's default
backends, including Metal on macOS. Optional emitted-code checks use the configured
Slang CLI (or ``SLANGC``); emitting MSL does not establish Apple compilation or GPU
correctness.
