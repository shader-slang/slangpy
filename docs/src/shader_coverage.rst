Shader coverage (experimental)
==============================

Shader coverage records line, function-entry, and branch-arm execution counts
for compute programs. Enable instrumentation when creating the device or a Slang
session, then collect all compiled program generations through the device:

.. code-block:: python

    import json
    import slangpy as spy

    device = spy.Device(
        type=spy.DeviceType.vulkan,
        compiler_options={"coverage": spy.ShaderCoverageOptions()},
    )
    # Load shaders and dispatch normal SlangPy calls or explicit compute kernels.
    # ...
    snapshot = device.shader_coverage.snapshot(reset=True)
    device.close()

    for program in snapshot.programs:
        manifest = json.loads(program.manifest)
        for entry in manifest["entries"]:
            hits = program.counters[entry["counter"]]
            print(program.generation_id, entry["file"], entry["line"], hits)

Configuration
-------------

``SlangCompilerOptions.coverage`` defaults to ``None`` (disabled). Set it to
``ShaderCoverageOptions`` or a dictionary with ``lines``, ``functions``,
``branches``, ``counter_width``, and ``boolean`` fields. The three instrumentation
kinds default to enabled; at least one must be enabled. Options are copied into the
Slang session, so changing the original Python options object does not change
an existing session.

Set ``boolean=True`` to record hit/miss instead of execution counts:

.. code-block:: python

    options = spy.ShaderCoverageOptions(boolean=True, counter_width=32)
    device = spy.Device(type=spy.DeviceType.vulkan, compiler_options={"coverage": options})

Boolean instrumentation stores ``1`` without atomic increments, avoiding atomic
contention at the cost of execution frequencies. Slots remain 32 or 64 bits;
this is not a packed bitset. Repeated execution leaves a hit at ``1`` until reset.
Snapshots keep integer counters and manifests identify entries with
``mode: "boolean"``. Counting remains the default. Actual performance depends
on the shader and device.

Counters default to 64 bits. Counting uses 64-bit buffer atomics, and an
unsupported device raises an error when linking the program. For MoltenVK or
another device without those atomics, explicitly choose
``ShaderCoverageOptions(counter_width=32)``. There is no automatic fallback.
64-bit boolean slots require ``int64`` support but do not require 64-bit atomics.
32-bit GPU execution counters can wrap; widening them during readback cannot
recover lost counts. Host snapshot integers preserve the full GPU counter width.

``device.shader_coverage.capabilities`` reports ``supported``,
``counter_widths`` for counting, ``boolean_counter_widths`` for hit/miss, and an
explanatory ``reason`` when unavailable. Linking additionally checks compiler
metadata and the supported program layout.
The initial implementation targets Vulkan and CUDA, with one compute entry
point per linked program. Windows users should select Vulkan or CUDA explicitly;
D3D12 coverage is not implemented. CUDA and Vulkan execution, including 64-bit counters, have been validated on a
Windows RTX 4090.

Collection and lifetime
-----------------------

``snapshot()`` blocks until the readback completes and returns cumulative counts
since the previous reset. ``snapshot(reset=True)`` copies and clears all
registered counter buffers in one queue submission. ``reset()`` clears those
buffers without returning their counts. These operations use the default device
queue; custom CUDA streams are rejected while coverage programs are registered.

Already submitted work precedes the capture boundary. Recorded but unsubmitted
commands do not contribute until submitted, including commands recorded before
a reset or hot reload. RHI command buffers are single-use; record fresh commands
for repeated work. Internal coverage submissions do not invoke user submission
callbacks or trigger hot reload.

The collector retains generations after program destruction and hot reload,
including generations that have not executed yet. A reset clears counts but does
not remove generations. Retention is bounded at 4096 generations or 256 MiB of
counter buffers plus manifest text; exceeding either limit fails registration
with an error. Capture and create a new device to continue. Snapshots contain
host-owned data and remain readable after device close. Python fields are
read-only, and returned containers are copies.

Snapshot fields are ``collection_id`` (device collection timeline),
``capture_id``, ``interval_id``, ``reset_after``, and ``programs``. Each program
record has ``generation_id``, ``label``, ``manifest``, ``counter_width``, and
``counters``. Cumulative captures from the same interval overlap and must not be
summed. A failed readback after a submitted reset may leave counters cleared;
capture failure raises an exception rather than returning a partial result.

Attribution and reporting
-------------------------

Branch IDs and counter indices are local to each program generation. Keep
``(collection_id, generation_id)`` as their namespace. Different macro expansions
can have identical file, line, and column metadata, so those coordinates are not
a safe branch deduplication key. A compiler metadata extension for automatic
source-branch deduplication is a possible follow-up.

Snapshots currently expose raw manifests and counts. They do not freeze source
text or provide packaged HTML, LCOV, JSON persistence, or offline merging.
Source revision tracking and those reporting APIs are follow-up work. Coverage
describes instrumented sites in compiled programs, not uncompiled project code.

Image-processing example
------------------------

``examples/shader_coverage`` demonstrates normal SlangPy functional calls, cumulative
captures, and a final capture with reset. It generates a noisy image, runs an
edge-preserving denoiser and tone mapper, then adds HDR, transparency, black pixels,
and a denoising bypass to exercise missed branches. Each output is checked against
NumPy and GPU execution with instrumentation disabled.

From the repository root after building:

.. code-block:: console

    python -m examples.shader_coverage.shader_coverage --device vulkan --output-dir shader-coverage-report

Use ``--counter-width 32`` on MoltenVK, or ``--device cuda`` to test CUDA. Open
``shader-coverage-report/index.html`` for image pairs and before/after source
coverage. It needs no extra packages, image downloads, or Slang source checkout.
Add ``--slang-source /path/to/slang`` to generate LCOV files and annotated
coverage pages with Slang's existing converter and HTML renderer. The landing
page links to the ordinary-input and expanded-input reports alongside the
input/output images. Add ``--open`` to open it in a browser. Without the Slang
checkout, the page shows images, totals, and raw-capture links.

The entry point ``shader_coverage.py`` demonstrates instrumentation configuration
and explicit snapshot/reset boundaries. ``image_processing.py`` implements the
same workload for ordinary and instrumented devices. ``report.py`` handles
example-specific exports and rendering after device close, independently of the
GPU workload. These helpers do not define a reusable reporting or wrapper API.

Testing on Windows and Linux
----------------------------

Build SlangPy normally after updating submodules. This draft pins RHI to the
head of `slang-rhi PR #739 <https://github.com/shader-slang/slang-rhi/pull/739>`_
and uses the repository's released Slang dependency; local source overrides are
not required. Run the same test files for either backend:

.. code-block:: console

    python -m pytest slangpy/tests/device/test_shader_coverage.py slangpy/tests/device/test_shader_coverage_collector.py --device-types vulkan -v
    python -m pytest slangpy/tests/device/test_shader_coverage.py slangpy/tests/device/test_shader_coverage_collector.py --device-types cuda -v

Most cases explicitly use 32-bit counters so they also run on MoltenVK. The
64-bit boundary test seeds counters near ``2**32`` and checks GPU increments and
collector readback beyond that value; it skips only if the device lacks the
required atomics. Confirm that this test passes on supported hardware.
