# Refactor the SlangPy Instrumentation Profiler

This ExecPlan is a living document. The sections Progress, Surprises and Discoveries, Decision Log, and Outcomes and Retrospective must be kept up to date as work proceeds.

This plan follows `.agents/PLANS.md` from the repository root.

## Purpose / Big Picture

SlangPy needs an application-wide instrumentation profiler that is useful while an application is running and can also produce offline traces for Perfetto or Chrome trace viewers. After this refactor, users can wrap CPU and GPU work in profiler zones, inspect realtime per-frame CPU/GPU timings in an ImGui window, and optionally capture a trace snapshot to Chrome JSON or Python-readable flat records for analysis.

This plan intentionally designs the profiler API from scratch. The current profiler implementation is work-in-progress code, so there is no backwards-compatibility requirement for existing temporary Python names, old test assumptions, or stale generated doc symbols. Existing examples and tests should be updated to the new API instead of preserving compatibility shims.

Sampling profiling is not part of this work. This is only an instrumentation profiler: timings come from explicit zones and frames inserted by SlangPy, C++ macros, or Python context managers.

## Progress

- [x] (2026-06-03) Inspected `src/sgl/device/profiler.h`, `src/sgl/device/profiler.cpp`, `src/slangpy_ext/device/profiler.cpp`, `tests/sgl/device/test_profiler.cpp`, and `slangpy/tests/slangpy_tests/test_profiler.py`.
- [x] (2026-06-03) Confirmed the current GPU path is split between a stub `GpuContextData::allocate_timestamp_query()` path and an older ActiveGpuRecording path that still exists but is commented out at the public callsites.
- [x] (2026-06-03) Chose v1 scope: trace capture, Python trace records, realtime frame statistics, and a basic ImGui profiler window.
- [x] (2026-06-03) Chose v1 defaults: frame statistics enabled by default, trace capture disabled by default.
- [x] (2026-06-03) Chose v1 GPU policy: do not support discarded/unsubmitted command encoders beyond ignoring them safely.
- [x] (2026-06-03) Revised this plan to remove backwards-compatibility API requirements and expand the initial profiler requirements into implementation-level guidance.
- [ ] Implement the new public C++ profiler types and Python bindings.
- [ ] Replace the temporary profiler internals with a single queue, worker, GPU recording, stats, and trace pipeline.
- [ ] Add the ImGui profiler window and update examples.
- [ ] Build, run profiler tests, run C++ tests, and run pre-commit.

## Surprises and Discoveries

- Observation: `GpuContextData::allocate_timestamp_query()` currently always returns `INVALID_QUERY_INDEX`, so the visible `begin_zone()` / `end_zone()` code cannot emit usable GPU timestamps through that path.
  Evidence: `src/sgl/device/profiler.cpp` defines the method as a stub, while `Profiler::begin_zone()` and `Profiler::end_zone()` call it before queuing temporary GPU events.

- Observation: A more complete GPU recording design already exists in `src/sgl/device/profiler.cpp` with `ActiveGpuRecording`, query blocks, submitted/discarded callbacks, pending submits, and GPU timestamp conversion, but the callsites are commented out.
  Evidence: `ProfilerImpl::begin_gpu_zone()` and `ProfilerImpl::end_gpu_zone()` exist, while the calls from `Profiler::begin_zone()` / `Profiler::end_zone()` are commented.

- Observation: The command layer already has enough submission identity for profiler GPU batching.
  Evidence: `CommandEncoder::recording_id()`, `CommandBuffer::recording_id()`, `CommandBuffer::_notify_submitted()`, `Device::_register_command_recording_submitted_callback()`, and `Device::_register_command_recording_discarded_callback()` are already present.

- Observation: Current Python tests assert temporary APIs and behavior that should not constrain the refactor.
  Evidence: `slangpy/tests/slangpy_tests/test_profiler.py` imports temporary profiler types and the plan now explicitly allows replacing those APIs.

- Observation: `src/slangpy_ext/py_doc.h` contains stale profiler symbols from an older API.
  Evidence: `py_doc.h` mentions `ProfilerTraceEvent`, `ProfilerFrameRecord`, `render_overlay`, and compatibility-style scope classes not present in the current public header.

## Decision Log

- Decision: Treat the profiler API as new API and remove backwards-compatibility requirements.
  Rationale: The current implementation is explicitly work-in-progress and preserving temporary API names would make the final design worse.
  Date/Author: 2026-06-03 / Codex

- Decision: Keep this as an instrumentation profiler only.
  Rationale: The user explicitly said sampling may be added later but is out of scope for this refactor.
  Date/Author: 2026-06-03 / Codex

- Decision: Enable frame statistics by default and leave trace capture disabled by default.
  Rationale: Realtime shader performance measurement should work immediately, while full trace retention has extra memory and processing cost.
  Date/Author: 2026-06-03 / Codex

- Decision: Resolve GPU query readiness and query results from `Profiler::tick()` on the caller thread in v1.
  Rationale: Timestamp query resolve/result code may not be thread safe yet, but CPU queue processing and aggregation can still run on a worker.
  Date/Author: 2026-06-03 / Codex

- Decision: Ignore discarded/unsubmitted GPU command recordings simply.
  Rationale: The intended common case is submitted command buffers; v1 should not spend complexity proving safe query reuse for abandoned encoders.
  Date/Author: 2026-06-03 / Codex

- Decision: Correlate CPU and GPU zones with a worker-assigned logical `event_id`.
  Rationale: Realtime frame statistics need to show CPU and GPU time for the same logical zone, but minting and wiring that ID on the hot path adds avoidable work. The worker can assign it while processing the ordered CPU/GPU queue events from one `begin_zone()` call.
  Date/Author: 2026-06-03 / Codex

- Decision: Store processed trace/stat data with compact integer IDs instead of raw pointers.
  Rationale: Source locations, names, timelines, zone children, and roots are repeated heavily; 32-bit IDs reduce storage and prepare the design for further compacting without bit-packing in v1.
  Date/Author: 2026-06-03 / Codex

- Decision: Use zone terminology for public timed records instead of introducing "span".
  Rationale: Users already think in profiler zones; CPU and GPU measurements can be represented as concrete trace zones with different `zone_id` values and the same logical `event_id`.
  Date/Author: 2026-06-03 / Codex

- Decision: Store the minimal durable data per trace zone and derive richer fields for UI/export snapshots.
  Rationale: Per-zone records dominate profiler memory. Depth, display path, parent event ID, timeline type, duration, parent links, and formatted labels can be derived from IDs, dictionaries, timestamps, timeline roots, and child lists when needed.
  Date/Author: 2026-06-03 / Codex

- Decision: Store child lists rather than parent IDs in durable trace zones.
  Rationale: The profiler UI and trace export need to traverse hierarchies frequently; compact child ranges make traversal direct while still avoiding pointer-heavy nested objects.
  Date/Author: 2026-06-03 / Codex

- Decision: Treat hot-path minimalism as the primary implementation constraint.
  Rationale: The profiler must remain usable in debug builds and while measuring shader workloads; all work except timestamp query allocation/writes and compact queue writes should move to the worker or `tick()` resolver.
  Date/Author: 2026-06-03 / Codex

## Outcomes and Retrospective

No implementation has been completed yet. This revision turns the initial design notes into a decision-complete implementation plan and removes the earlier compatibility requirement.

## Context and Orientation

The profiler lives primarily in `src/sgl/device/profiler.h` and `src/sgl/device/profiler.cpp`. It is exposed to Python by `src/slangpy_ext/device/profiler.cpp`. Current Python behavior is tested in `slangpy/tests/slangpy_tests/test_profiler.py`; current C++ behavior is tested in `tests/sgl/device/test_profiler.cpp`.

A profiler zone is a timed scope. A CPU zone records host timestamps from `sgl::Timer::now()`. A GPU zone records timestamp queries into a `CommandEncoder`; those queries become readable only after the command buffer is submitted and the GPU work completes. A profiler frame is an application frame used to group zones for realtime statistics.

The existing implementation already has useful pieces: per-thread queues through `ThreadData`, static source-location macros, string/source interning, command recording IDs, and device callbacks for submitted or discarded command buffers. It also has temporary or broken pieces: GPU timestamp allocation is stubbed in one path, the older GPU path is disconnected, trace storage currently stores pointer-heavy nested `ProfilerZone` trees, and `tick()` currently performs too much work synchronously.

The new design should split the profiler into four conceptual layers:

1. Hot-path recording. This runs inside zone begin/end calls and is the most important performance-sensitive path. It must stay small, non-blocking, and acceptable in debug builds.
2. Main-thread GPU query resolution. This runs from `Profiler::tick()` because query result access may not be thread safe yet.
3. Worker-thread processing. This drains queues, resolves stacks, builds frame statistics, and appends optional trace data.
4. Presentation/export. This exposes flat Python records, writes Chrome JSON, and renders a realtime ImGui window.

## Public Interfaces

Replace the current temporary profiler API with a clean API. Do not add compatibility aliases for temporary names such as old Python `ProfilerZoneScope` or `ProfilerFrameScope`; update examples and tests to the new API.

`ProfilerDesc` should contain:

- `bool frame_stats_enabled = true`
- `bool trace_enabled_on_start = false`
- `uint32_t stats_window_size = 120`
- `uint32_t gpu_query_pool_size = 64 * 1024`
- `uint32_t gpu_query_block_size = 256`
- `bool auto_zones_enabled = true`
- `bool debug_groups_enabled = false`

`Profiler` should expose:

- `enabled`, `set_enabled`
- `frame_stats_enabled`, `set_frame_stats_enabled`
- `auto_zones_enabled`, `set_auto_zones_enabled`
- `debug_groups_enabled`, `set_debug_groups_enabled`
- `start_trace(bool clear = true)`
- `stop_trace()`
- `clear_trace()`
- `trace_snapshot()`
- `stats_snapshot()`
- `tick()`
- `flush()`
- `begin_zone(...)` / `end_zone(...)` for C++ internals and macros
- `begin_frame(...)` / `end_frame(...)` for C++ internals and macros

Python should expose the same concepts using new names:

- `Profiler.zone(name: str | None = None, command_encoder: CommandEncoder | None = None, flags: ProfilerZoneFlags = ProfilerZoneFlags.none)` as a context manager.
- `Profiler.frame(name: str | None = None)` as a context manager.
- `Profiler.start_trace(clear: bool = True)`, `stop_trace()`, `clear_trace()`, `trace_snapshot()`, `stats_snapshot()`, `tick()`, and `flush()`.

Do not expose backwards-compatible `ProfilerZoneScope` or `ProfilerFrameScope` names. Update `examples/pathtracer/pathtracer.py` to use `with self.profiler.frame("frame"):` and `with self.profiler.zone("path_tracer", command_encoder):`.

`ProfilerTrace` should expose flat data, not nested trees:

- timelines: ID, type, name, thread ID, device ID, queue.
- source locations: ID, file, line, original function, display function.
- names: ID and display string.
- frames: ID, name ID, source ID, start/end timestamps.
- zones: zone ID, logical event ID, child-list range, timeline ID, frame ID, source ID, name ID, start timestamp, and end timestamp.
- child indices: a compact `uint32_t` array referenced by each zone's child-list range, plus root zone ranges on timelines or frames as needed.

`ProfilerTraceZone` should be the durable per-zone record and should stay deliberately small. Do not store duration, depth, display path, source strings, name strings, parent event ID, parent zone ID, timeline type, resolved state, or raw pointers in each zone. Derive duration from timestamps, derive depth/path/parent information while traversing from timeline/frame roots through child lists, derive CPU/GPU type from the timeline, and keep strings in source/name dictionaries.

`ProfilerTrace::write_to_json(path)` remains, but it should generate Chrome trace JSON from flat zones and child-index arrays. It should include metadata timeline events and enough derived args for analysis, including source file, line, display function, event ID, parent event ID when known during traversal, frame ID, raw timestamps, and whether the zone is CPU or GPU.

`ProfilerStatsSnapshot` should expose a hierarchical tree for the rolling frame-stat window. Each node should include display name, source ID, name ID, path, depth, CPU timing stats, GPU timing stats, sample counts, and unresolved GPU sample count. Timing stats include last, min, max, average, and standard deviation. The stats key should be hierarchical parent path plus source-location ID plus name ID so same-named zones from different callsites do not merge accidentally.

Add `sgl::ui::render_profiler_window(Profiler*, const ProfilerWindowDesc& = {})` and expose it as `slangpy.ui.render_profiler_window(profiler, ...)`. The UI should show the frame-stat hierarchy with CPU/GPU columns and a configurable rolling window size. It should also show pending GPU samples because GPU results arrive asynchronously.

## Plan of Work

First, simplify the profiler implementation around a single model. Remove the temporary GPU event queue path and delete `GpuContextData` unless a small part of it is reused under a new name. Reconnect the older `ActiveGpuRecording` idea, but reshape it to the new event/zone-record model. The final code should not have two separate GPU recording systems.

Define compact internal IDs. Source locations, names, timelines, frames, logical events, trace zones, child indices, root lists, and stats nodes should use integer IDs. Use `uint32_t` for dictionary IDs, zone IDs, timeline IDs, frame IDs, child-list offsets/counts, root-list offsets/counts, and stats-tree references where practical. Use `uint64_t` for worker-assigned logical event IDs if wraparound matters. Raw `ProfilerSourceLocation*` and `const char*` are acceptable only in hot-path event payloads because macros and interned strings naturally provide stable pointers. The worker must translate those pointers to compact IDs.

Keep durable per-zone storage minimal. A processed trace zone should be close to this shape: `uint32_t zone_id`, `uint64_t event_id`, `uint32_t first_child_index`, `uint32_t child_count`, `uint32_t timeline_id`, `uint32_t frame_id`, `uint32_t source_id`, `uint32_t name_id`, `uint64_t start_timestamp_ns`, and `uint64_t end_timestamp_ns`. Use an invalid ID sentinel for absent frame/name/source values. Store children in a separate compact `uint32_t` zone-ID array. Store timeline/frame roots the same way. Add a field only if it cannot be derived cheaply and is needed by more than one consumer.

Format verbose pretty function names in the worker. Preserve the original function string in the source dictionary, but derive a shorter display function for trace names and UI display. The implementation does not need an elaborate demangler in v1; it should at least strip noisy compiler signature prefixes/suffixes enough that macro-generated C++ function names are readable.

Redesign hot-path events. `begin_zone` should:

- return false immediately when profiling is disabled.
- read a CPU timestamp.
- read the current global frame ID atomically.
- copy dynamic names only when requested by flags, otherwise store stable name pointers.
- enqueue a small CPU begin event to the current thread's queue.
- if a command encoder is supplied and GPU profiling is supported, acquire or reuse a query pair, write the begin timestamp query, and enqueue a small GPU begin event immediately after the CPU begin event in the same thread queue.

`end_zone` should:

- read a CPU timestamp.
- enqueue a small CPU end event to the same thread queue.
- if a command encoder is supplied and the scope owns a GPU query pair, write the matching GPU end timestamp query and enqueue a small GPU end event immediately after the CPU end event.
- pop a debug group only if this scope pushed one.

Make the RAII zone scope store an opaque token returned by `begin_zone`, not just `Profiler*`. The token should contain only what `end_zone` needs on the hot path, such as whether the scope is active, whether it pushed a debug group, the command recording ID, and the GPU query pair/end query index if present. It should not require a hot-path logical event ID.

Use per-thread queues for CPU and GPU zone events. The hot path should not allocate trace zones from a trace slab, build zone trees, assign source/name dictionary IDs, assign logical event IDs, update stats, or append trace data. The queue payload should be compact and fixed-size where possible. Keep queue writes non-blocking as much as `moodycamel::ConcurrentQueue` allows. If allocation or enqueue fails, drop the event or disable that scope gracefully rather than blocking the application.

Keep GPU hot-path work to the unavoidable minimum. Writing timestamp queries must happen while encoding commands, and query indices must be known before those writes. The fast path should therefore use a per-thread/per-command-recording cached query block cursor so begin/end only reserve indices, write timestamps, and enqueue compact GPU events. Slow-path query block acquisition may lock, but it should happen only when the cached block is exhausted or a new command recording is first seen.

Add a worker thread owned by `ProfilerImpl`. The worker should sleep on a condition variable, wake on queued work or `flush()`, drain all known thread queues, and process resolved GPU batches delivered by `tick()`. The worker owns the expensive work: CPU stack reconstruction, source/name dictionary creation, frame-stat tree updates, optional trace append, and pruning history windows. `tick()` may wake the worker but should not do stack processing except for GPU query result access required on the caller thread.

Let the worker correlate CPU and GPU events from queue order. A GPU begin event from `begin_zone()` must appear immediately after the corresponding CPU begin event in the same producer queue, and a GPU end event must appear immediately after the corresponding CPU end event. The worker assigns the logical `event_id` when it processes the CPU begin event, then attaches the following GPU begin/end events to that logical zone while maintaining a separate GPU stack per command recording. This keeps the hot path from minting or looking up correlation IDs.

Keep trace capture optional and separate from stats. When trace capture is disabled, do not allocate trace zone storage and do not retain full trace history. Stats should still update when `frame_stats_enabled` is true. When trace capture starts, create or clear trace storage according to `start_trace(clear)`. Only events whose begin event occurs while trace capture is active should enter the trace. `trace_snapshot()` should call `tick()` and `flush()` so the returned snapshot includes all data visible before the snapshot request.

Use one application-wide frame model. `begin_frame` opens a global active frame ID, records a frame begin event, and publishes that ID atomically so subsequent zones can attach to it. `end_frame` closes that frame. Nested or overlapping frames should assert in debug builds and be ignored or closed conservatively in release builds. The stats UI is frame based, so ambiguous frame ownership should not be accepted silently.

Handle GPU zones per command recording in the worker. All GPU zones can be assumed to be captured from the same thread in v1, but command encoders may be interleaved on that thread. Therefore, maintain a separate worker-side GPU zone stack per command recording ID, not one global GPU stack. The hot path owns only the query-block cursor needed to reserve query indices and write timestamps. The worker owns queued GPU zone records and the stack of open GPU zone indices for each command recording.

Allocate GPU timestamp queries from a large pool per device/queue combination. Use descriptor sizes from `ProfilerDesc`. Allocate fixed-size query blocks to active command recordings. A GPU zone consumes two queries, begin and end. When a query block is exhausted, acquire another block from the same device/queue pool. Submitted query blocks may be recycled after query results have been read. Discarded/unsubmitted blocks may be abandoned for the lifetime of the profiler in v1.

Use submitted command recording callbacks as ownership handoff, not as resolve work. `Device::_register_command_recording_submitted_callback()` should move the hot-path query allocation state for that command recording into a pending submit list with its `submit_id`, recording ID, device, queue, and query blocks. The callback should not query result readiness or call `get_result()`. `Device::_register_command_recording_discarded_callback()` should remove active query allocation state and abandon associated query blocks. It does not need to prove query reuse safety.

Resolve GPU timestamps from `Profiler::tick()`. `tick()` should check pending submit IDs with `Device::is_submit_finished()`. For finished submits, capture or use a timestamp calibration for the device/queue, read query results, convert GPU timestamps into the same CPU nanosecond domain used by CPU zones, create raw GPU timestamp-result batches keyed by command recording ID plus query pool/index, recycle submitted query blocks, and pass those batches to the worker. The worker then joins resolved timestamps to the GPU zones it assembled from queued begin/end events. If timestamp query or calibration features are missing, GPU zones are not recorded and CPU zones still work.

Build frame statistics from CPU and GPU zones. CPU zones update stats when the CPU end event is processed. GPU zones update stats when `tick()` resolves queries, potentially one or more frames later. The `event_id` ties the CPU zone and GPU zone for the same logical instrumented scope together. Stats should keep a rolling window of configurable size and compute last, min, max, average, and standard deviation for CPU and GPU independently. The UI should make unresolved GPU samples visible instead of treating them as zero.

Keep trace data flat but child-traversable. The trace can still be exported as nested-looking Chrome events by traversing child ID ranges and timestamps, but storage should be arrays of records plus compact child-index/root-index arrays. Avoid raw `ProfilerZone*`, recursive object ownership, and per-zone `std::vector` allocations in permanent trace storage. Do not over-optimize with bit packing in v1; choose clear compact records that can be tightened later.

Split files lightly. Keep the public API in `src/sgl/device/profiler.h`. Keep the core runtime in `src/sgl/device/profiler.cpp` unless it becomes unwieldy. Move Chrome JSON writing and trace-specific helpers to `src/sgl/device/profiler_trace.cpp` if that makes the core readable. Put the ImGui window in `src/sgl/ui/profiler.h` and `src/sgl/ui/profiler.cpp` so UI code is separate from profiler core. Update `src/sgl/CMakeLists.txt` and `src/slangpy_ext/CMakeLists.txt` for any new files.

Refresh Python bindings in `src/slangpy_ext/device/profiler.cpp` and `src/slangpy_ext/ui/ui.cpp`. Bind the new flat trace records and stats snapshots. Bind Python context managers through small native helper types or nanobind lambdas; because there is no compatibility requirement, choose the simplest new Python API and update all tests/examples accordingly.

Refresh generated docs after the API stabilizes. Run the `slangpy_pydoc` target if the environment supports it, or explicitly update `src/slangpy_ext/py_doc.h` if the normal generated-doc flow is unavailable. The final tree should not keep stale profiler doc symbols.

## Milestones

Milestone 1: establish the new API and CPU-only processing path. At the end of this milestone, `Profiler.zone()`, `Profiler.frame()`, C++ zone/frame macros, trace start/stop/snapshot, and CPU frame stats work without GPU timestamps. Run the build and CPU profiler tests. Acceptance is a Python test that starts a trace, records nested CPU zones inside a frame, snapshots flat trace zones, writes JSON, and reads non-empty stats.

Milestone 2: add worker-thread processing. At the end of this milestone, hot-path zone calls only enqueue compact events, and `flush()` waits for the worker to process visible events. Acceptance is a stress test that records zones from multiple CPU threads, calls `flush()`, and verifies child-list traversal and stats without doing stack assembly on the caller thread.

Milestone 3: add GPU query recording and main-thread resolve. At the end of this milestone, manual GPU zones and SlangPy auto zones reserve query indices and write begin/end timestamp queries on the hot path, but GPU stack assembly and CPU/GPU correlation happen in the worker from queued events. Submitted recordings move to pending submits, `tick()` resolves completed GPU results, and stats/trace show matching CPU and GPU zones by event ID. Acceptance is a GPU test that submits command buffers, waits for submit completion, calls `tick()` and `flush()`, and finds both CPU and GPU samples for the same zone path.

Milestone 4: add ImGui UI and update examples. At the end of this milestone, `slangpy.ui.render_profiler_window(profiler)` draws a hierarchical frame-stat table with CPU/GPU timing columns and pending GPU counts. The path tracer example uses the new API. Acceptance is a build plus a smoke run or code-level test that the UI function is callable between UI begin/end.

Milestone 5: cleanup, docs, and validation. At the end of this milestone, temporary GPU code, compatibility API names, stale tests, and stale doc symbols are gone. Acceptance is the full validation command list passing or documented with concrete blockers.

## Concrete Steps

Work from the repository root:

    cd C:\projects\slangpy

Before editing implementation files, check the current worktree:

    git status --short

Edit `src/sgl/device/profiler.h` to define the final public structs, snapshots, and profiler methods. Remove temporary API surface that no longer matches this plan.

Edit `src/sgl/device/profiler.cpp` to replace the current temporary internals with the unified model described above. Prefer additive implementation within the file until tests pass, then remove obsolete types and commented paths.

Add `src/sgl/ui/profiler.h` and `src/sgl/ui/profiler.cpp` for the ImGui profiler window. Update `src/sgl/CMakeLists.txt`.

Edit `src/slangpy_ext/device/profiler.cpp` to bind the new profiler API, trace records, and stats records. Edit `src/slangpy_ext/ui/ui.cpp` to bind the profiler UI function. Update `src/slangpy_ext/CMakeLists.txt`.

Update `src/slangpy_ext/utils/slangpy.cpp` so SlangPy automatic zones use the new profiler begin/end token and continue to create CPU/GPU zones around functional dispatches when `auto_zones_enabled` is true.

Update `tests/sgl/device/test_profiler.cpp` and `slangpy/tests/slangpy_tests/test_profiler.py` to test the new API. Remove expectations for compatibility names or default trace capture.

Update `examples/pathtracer/pathtracer.py` and `examples/pathtracer/pathtracer.cpp` to use the new API and call `tick()` once per frame.

Build and test:

    cmake --build --preset windows-msvc-debug
    pytest slangpy/tests/slangpy_tests/test_profiler.py -v
    python tools/ci.py unit-test-cpp
    pre-commit run --all-files

If `pre-commit` modifies files, rerun it until it reports success.

## Validation and Acceptance

The refactor is accepted when these behaviors are demonstrated:

- Constructing `Profiler()` records realtime stats by default but does not retain trace zones until `start_trace()` is called.
- `start_trace(clear=True)`, `stop_trace()`, `clear_trace()`, and `trace_snapshot()` work from both C++ and Python.
- A CPU-only trace snapshot exposes flat timelines, sources, names, frames, zones, and child-index arrays that can traverse nested zones and write valid Chrome JSON.
- C++ macros and Python `profiler.zone()` / `profiler.frame()` context managers record nested zones under the current application frame.
- Frame stats show hierarchical CPU timings with last, min, max, average, standard deviation, and sample count over the configured rolling window.
- GPU zones written to submitted command buffers produce GPU trace zones after the submit finishes and `tick()` runs.
- CPU and GPU stats for the same logical zone correlate through `event_id`.
- Interleaved command encoders on the same thread maintain independent GPU zone stacks.
- Query blocks roll over when many GPU zones are recorded.
- Discarded or unsubmitted GPU command recordings do not crash and do not add GPU zones to stats or trace.
- The ImGui profiler window renders a hierarchy with CPU and GPU columns and indicates unresolved GPU samples.
- Stale backwards-compatibility Python names are not exposed, and tests/examples use only the new API.

## Idempotence and Recovery

The implementation should be safe to build and test repeatedly. `start_trace(clear=True)` should reset trace storage deterministically. `clear_trace()` should not affect rolling stats. `flush()` should be safe to call even when there is no pending work.

If GPU timestamp features are unsupported on a device, GPU profiling should be disabled for that device and CPU profiling should continue. Tests should skip GPU-specific assertions when `Feature::timestamp_query` or `Feature::timestamp_calibration` is missing.

If generated doc refresh is unavailable because `pybind11_mkdoc` or clang Python support is missing, record the exact error in the ExecPlan and either update `py_doc.h` manually or leave a clearly documented follow-up.

## Interfaces and Dependencies

Use existing dependencies only. Keep `moodycamel::ConcurrentQueue` for per-thread queues. Use existing `Device`, `CommandEncoder`, `CommandBuffer`, `QueryPool`, and command recording callback APIs. Use existing Dear ImGui integration in `src/sgl/ui`.

Do not introduce new third-party libraries. Do not add sampling profiler hooks. Do not require command encoders to be submitted for CPU profiling to work.

The final design depends on these repository APIs:

- `sgl::Timer::now()` for CPU nanosecond timestamps.
- `CommandEncoder::write_timestamp(QueryPool*, uint32_t)` for GPU timestamp writes.
- `Device::get_timestamp_calibration(CommandQueueType)` for CPU/GPU time correlation.
- `Device::is_submit_finished(uint64_t)` for non-blocking submit completion checks.
- `Device::_register_command_recording_submitted_callback()` and `_register_command_recording_discarded_callback()` for GPU recording ownership transitions.

## Artifacts and Notes

This revision expands the original compact plan with the full initial requirement set: optional trace capture, Python-readable traces, Chrome JSON export, realtime frame stats, async GPU timing, per-thread low-overhead queues, worker-thread processing, per-command-encoder GPU stacks, per-device/queue query pools, main-thread query resolution, separate trace/stat storage, CPU/GPU event correlation, compact ID-based storage, minimal per-zone records, formatted function names, and separate UI files. It also removes the earlier compatibility requirement for Python scope class names.
