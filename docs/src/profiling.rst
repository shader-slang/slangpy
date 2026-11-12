.. _sec-profiling:

CPU and GPU profiling
=====================

SlangPy includes a bounded instrumentation profiler for CPU work, command-buffer GPU work,
and functional SlangPy dispatches. It is intended for continuously running shader and
hot-reload applications: live history defaults to 120 frames and 100,000 zones, and producer
queues never block when full.

Basic use
---------

Activate a profiler around code whose automatic functional dispatches should be measured.
The first ``Profiler`` created on a thread becomes that thread's default current profiler. Later
profilers do not replace it; use the ``Profiler`` context manager to select one temporarily. The
free-standing ``profile_zone`` and ``profile_function`` contexts use this current profiler, while
``profile_frame`` establishes its frame boundaries. These profiling contexts never change the
current profiler and become no-ops when none is current, so instrumented functions do not need to
receive or look up a profiler. They also become no-ops while the current profiler is disabled, so
no zones are recorded until it is enabled again. Supplying a command encoder to a zone or function
adds GPU timestamps. A frame establishes both the frame timeline and the explicit boundaries used
by live frame statistics. Zones outside a frame remain available in traces but do not contribute to
frame statistics.

Default activation is thread-local. An automatically installed profiler must be destroyed on its
constructing thread after temporary profiler contexts on that thread have exited. Its destructor
removes the default entry when it is the only remaining stack entry. Worker threads can select an
existing profiler temporarily with ``with profiler:``. Profiler destruction must not race device
submission, command-buffer discard, or device close operations associated with recorded GPU zones.

.. code-block:: python

    profiler = spy.Profiler()

    with spy.profile_frame("frame"):
        command_encoder = device.create_command_encoder()
        with spy.profile_zone("simulation", command_encoder):
            module.simulate(_append_to=command_encoder)
        device.submit_command_buffer(command_encoder.finish())

    # Poll submitted timestamp queries once per application frame.
    profiler.tick()

``tick()`` does not deliberately wait for profiled submissions or unresolved timestamp queries.
On CUDA, refreshing the mapping between CPU and GPU timestamp clocks synchronizes queued GPU work.
Calibration is cached per device queue and refreshed at most once every two seconds when resolved
timestamp data actually needs conversion.

``profile_function`` derives its zone name and profiling-site metadata from its Python caller:

.. code-block:: python

    def render(command_encoder: spy.CommandEncoder) -> None:
        with spy.profile_function(command_encoder):
            module.render(_append_to=command_encoder)

The profiler UI must be rendered inside an active ImGui frame. It shows cached, frame-aligned
numeric statistics; rendering it does not poll GPU work or copy an active capture. Detailed
event timelines are exported to Perfetto rather than rendered in the embedded window.

The headline shows **FPS** and **CPU frame** (elapsed time inside ``profile_frame``, including
any waits). The Mean/Latest/P95 selector beside the checkboxes applies to CPU frame time
and the table's times and call counts; Mean is the default. FPS always uses
``1000 / mean frame interval in milliseconds``, regardless of the selector. The interval is
the time between consecutive frame starts.
This measures instrumented application cadence, not display presentation rate.
For example, image acquisition, presentation, and profiler polling outside a frame scope still
contribute to the interval before the next frame starts. Record one frame scope per application
iteration for this measurement to reflect application FPS.

Select a zone in the table to plot its CPU or GPU duration per frame along with its descendants.
Each frame is one stacked column, with colored bands for the selected subtree. Parents contribute
only the duration remaining after accounting for their children, avoiding double-counting
inclusive times. The stack expands to fit children when a parent is unmeasured or smaller than
its descendants, so GPU work remains visible beneath CPU-only parents. The graph always shows
individual frames from oldest to newest. CPU values are elapsed wall time, and GPU values are
directly measured durations. Both are inclusive: parent and child rows are not additive and do not
represent utilization. Hover a row for complete, unavailable, incomplete, and absent GPU counts.
Missing measurements contribute no band and are indicated by gray markers below the column;
measured descendants remain visible. Absent occurrences are valid zero samples. Hover a band
for its zone's inclusive measurement and the displayed remainder. Because CPU hierarchy does
not guarantee GPU containment, the GPU stack represents grouped measurements rather than total
GPU frame time or measured exclusive time.

The headline and history graph use the newest CPU-ended frames, even while GPU work is pending.
In GPU mode, a frame with pending measurements in the selected subtree is a gap until those
results arrive. The zone table keeps CPU and GPU columns aligned over completed frames, so it
can lag behind the headline and graph. Diagnostics show the latest CPU frame, completed table
frame and age, and the table's lag in frames. Pausing the display freezes both snapshots,
while the displayed age continues to increase.

.. code-block:: python

    spy.ui.render_profiler_window(profiler)

See ``examples/pathtracer/pathtracer.py`` and ``examples/pathtracer/pathtracer.cpp`` for interactive integrations.

Bounded captures and GPU completion
-----------------------------------

Captures have a 256 MiB default memory limit. A capture stops automatically before exceeding
its limit and reports ``truncated=True`` with the ``memory_limit`` stop reason. Producer-ring
overflow and GPU-query exhaustion drop complete measurements instead of blocking. The trace
and frame statistics expose these counters through their ``diagnostics`` property:

* ``diagnostics.producer_drop_count`` counts events dropped because a per-thread event queue or zone stack
  was full.
* ``diagnostics.thread_event_queue_high_water_mark`` is the maximum number of unread events observed in any
  per-thread producer queue.
* ``diagnostics.hierarchy_loss_count`` counts statistics nodes promoted to roots because their parent event
  was unavailable when the worker finalized orphaned events.
* ``diagnostics.gpu_query_exhaustion_count`` counts GPU zones that retained CPU timing but could not allocate
  a GPU timestamp-query pair.

``stop_capture`` performs one ``tick`` and flushes CPU producer queues, but it does not deliberately
wait for submitted GPU work or unresolved timestamp queries. The tick can still synchronize during
a backend timestamp-calibration refresh as described above. To require every submitted GPU
measurement in a capture, use this order:

.. code-block:: python

    device.wait()
    profiler.tick()
    trace = profiler.stop_capture()

Otherwise unresolved GPU zones are omitted and reported by ``diagnostics.pending_gpu_zone_count``.

Capture membership follows the CPU recording start of each zone. GPU work recorded before
``start_capture()`` is excluded even if submitted or executed during the capture. Work recorded
during capture is eligible when its GPU queries resolve before capture stops. Calibrated GPU
timestamps determine placement on the trace, not capture membership; approximate clock alignment
can place a captured GPU event slightly outside the capture's CPU-clock boundaries.

GPU timestamp calibration is cached independently for each device queue and refreshed at most
once every two seconds when resolved timestamp data actually needs conversion. GPU durations are
calculated directly from timestamp differences; calibration only aligns their start times with
the CPU timeline.

Frame statistics
----------------

``frame_stats_snapshot()`` returns immutable statistics for one global frame stream. At most one
frame can be open or closing at a time, and zones from any CPU thread are attached to that frame.
The frame name remains trace metadata rather than selecting a separate statistics history. The
snapshot retains at most ``ProfilerDesc.frame_stats_window_size`` completed frames.
Frames with allocated GPU timestamps remain pending until those timestamps resolve; a bounded
fallback reports missing GPU data rather than allowing an abandoned command buffer to stall the
history indefinitely.

Use ``frame_stats_snapshot(include_pending=True)`` for the newest CPU-ended frames without
waiting for GPU completion. This window is independently bounded by ``frame_stats_window_size``.
GPU results are matched by frame and zone identity; later snapshots update the correct frame,
while previously returned snapshots remain unchanged. Unresolved measurements have GPU status
``pending``. Both snapshot modes are published by the collector; reading them does not poll GPU
queries or wait for the device. Continue calling ``tick()`` to collect GPU results.

The two modes can have different frame windows and entry lists. Match samples using
``sample_frame_index`` and entries using hierarchical site-ID paths, rather than matching array
positions. The default snapshot remains the appropriate view for paired CPU/GPU comparisons.

Repeated occurrences of the same hierarchical zone path are summed within each frame. A zone
that does not run contributes zero CPU time and zero calls to that frame, so parent and child
rows always describe the same frame window. Per-call summaries retain exact count, total,
minimum, maximum, mean, and standard deviation without retaining every call duration.

Per-path frame time is the sum of inclusive occurrence durations at that path. Parent and child
rows are therefore not additive. Summed GPU duration can also exceed elapsed frame time when GPU
work overlaps. GPU frame statistics omit pending, unavailable, and incomplete measurements.
Coverage is derived from the entry-aligned GPU-status matrix rather than duplicated on each entry.

``ProfilerFrameStats`` exposes retained history as zero-copy, read-only NumPy arrays.
``sample_frame_index`` and ``sample_frame_time_ms`` have shape ``(sample_count,)``. The call-count,
CPU-time, GPU-time, and GPU-status arrays have row-major shape ``(sample_count, entry_count)``.
GPU status distinguishes absent zones, zones without requested GPU timing, complete measurements,
pending measurements, and incomplete requested measurements. Call-frequency summaries likewise
derive from ``sample_call_count`` rather than being stored redundantly on each entry.

``frame_time``, ``latest_frame_ms``, and ``sample_frame_time_ms`` retain their CPU elapsed scope
meaning. ``sample_frame_start_ns`` contains the CPU start timestamps, while
``sample_frame_interval_ms`` contains the preceding start-to-start interval for each retained
frame. Both are read-only arrays with shape ``(sample_count,)``. ``ProfilerFrame.interval_ns``
exposes the same interval in traces. The first frame after creation or an enable/disable
transition has interval zero (unavailable); ``frame_interval`` excludes these zeros from its
duration distribution. An interval remains valid when its preceding frame leaves the rolling
window. Clearing statistics clears the retained samples but preserves the next frame's cadence
relative to the last recorded frame.

.. code-block:: python

    stats = profiler.frame_stats_snapshot(include_pending=True)
    print(stats.frame_time.mean_ms, stats.frame_time.p95_ms)
    if stats.frame_interval.count:
        print("FPS:", 1000.0 / stats.frame_interval.mean_ms)
    for entry in stats.entries:
        print(entry.name, entry.cpu_time_per_frame.mean_ms)
    print(stats.sample_cpu_time_ms.shape)

    profiler.clear_frame_stats()

Queries and export
------------------

Queries run natively over immutable 4,096-zone column chunks. Chunking bounds individual
allocations and lets queries and trace export stream through large captures without first
flattening every zone. Names match exactly, and frame and timestamp ranges are half-open.
Column properties are zero-copy, read-only NumPy arrays; retaining an array also retains its
native chunk.

Profiler-site filenames, compact function names, and labels are interned once in process-lifetime
storage. Native compiler signatures are shortened to qualified function names for display, while
their full signatures remain internal identity keys. Site snapshots therefore copy stable string
views rather than owning duplicate metadata strings.

Python profiling contexts additionally cache their resolved native site IDs in a bounded cache.
The cache distinguishes the caller's code object, exact bytecode call expression, and requested
name. Repeated scopes therefore avoid rebuilding filename and qualified-function
strings or entering the process-global site registry. Calls with identical filename, line,
function, and name metadata resolve to the same native profiling site.

.. code-block:: python

    selection = trace.query_zones(
        name="simulation",
        timeline_type=spy.ProfilerTimelineType.gpu,
        frame_begin=10,
        frame_end=20,
    )
    print(selection.statistics().p95_ms)
    print(profiler.frame_stats_snapshot())
    trace.write_to_json("capture.json")

The frame-statistics string contains diagnostic counters followed by the global frame stream's
indented CPU/GPU timing hierarchy, which is suitable for terminal logs and coding agents.

The JSON writer streams Chrome trace-event JSON suitable for Perfetto without flattening the
chunks first. Profiler timestamps are nanoseconds from a monotonic steady clock and have no
wall-clock meaning. GPU timestamps are calibrated into that clock domain.

CPU sampling is not part of this profiler version. It is planned as a separate sampling track
that will reuse the trace, query, and export model.
