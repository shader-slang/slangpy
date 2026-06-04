# Refactor the SlangPy Instrumentation Profiler

This ExecPlan is a living document. The sections Progress, Surprises and Discoveries, Decision Log, and Outcomes and Retrospective must be kept up to date as work proceeds.

This plan follows `.agents/PLANS.md` from the repository root.

## Purpose / Big Picture

SlangPy needs an application-wide instrumentation profiler that is useful while an application is running and can also produce offline traces for Perfetto or Chrome trace viewers. After this refactor, C++ code can wrap CPU and GPU work in profiler zones, SlangPy automatic dispatch zones can attach to the current profiler, users can inspect realtime per-frame CPU/GPU timings in an ImGui window, and trace snapshots can be exported to Chrome JSON or Python-readable flat records for analysis.

This plan intentionally designs the profiler API from scratch. The current profiler implementation is work-in-progress code, so there is no backwards-compatibility requirement for existing temporary Python names, old test assumptions, or stale generated doc symbols. Existing examples and tests should be updated to the new API instead of preserving compatibility shims.

Sampling profiling is not part of this work. This is only an instrumentation profiler: timings come from explicit zones and frames inserted by SlangPy automatic dispatch instrumentation or C++ macros/tokens. Python no longer exposes manual zone/frame context managers in this design pass.

## Progress

- [x] (2026-06-03) Inspected `src/sgl/device/profiler.h`, `src/sgl/device/profiler.cpp`, `src/slangpy_ext/device/profiler.cpp`, `tests/sgl/device/test_profiler.cpp`, and `slangpy/tests/slangpy_tests/test_profiler.py`.
- [x] (2026-06-03) Confirmed the current GPU path is split between a stub `GpuContextData::allocate_timestamp_query()` path and an older ActiveGpuRecording path that still exists but is commented out at the public callsites.
- [x] (2026-06-03) Chose v1 scope: trace capture, Python trace records, realtime frame statistics, and a basic ImGui profiler window.
- [x] (2026-06-03) Chose v1 defaults: frame statistics enabled by default, trace capture disabled by default.
- [x] (2026-06-03) Chose v1 GPU policy: do not support discarded/unsubmitted command encoders beyond ignoring them safely.
- [x] (2026-06-03) Revised this plan to remove backwards-compatibility API requirements and expand the initial profiler requirements into implementation-level guidance.
- [x] (2026-06-03) Reviewed the plan and resolved clarifying questions: start with the bool-returning hot path API, handle trace start/stop in the worker, disable GPU profiling with a warning on query exhaustion, ignore and warn on overlapping frames, allow Milestone 1 to process synchronously before the worker migration, and let explicit debug-group flags take precedence. The bool-returning API was later superseded by by-value tokens.
- [x] (2026-06-03) Added a simplicity and speed constraint: do not over-engineer the profiler; keep the implementation as simple as possible while preserving the fast hot path.
- [x] (2026-06-03) Reviewed this plan with correctness, completeness, clarity, efficiency, and performance passes, then incorporated required fixes for clock-domain conversion, trace epochs, worker teardown, hard GPU query accounting, batched query reads, milestone sequencing, and API wording contradictions.
- [x] (2026-06-03) Implemented the new public C++ profiler descriptor, flat trace records, stats snapshots, trace controls, and Python bindings.
- [x] (2026-06-03) Replaced the temporary profiler internals with a unified event pipeline using per-thread producer queues, synchronous `tick()`/`flush()` processing, GPU recording, hard query budgets, stats, and flat trace storage.
- [x] (2026-06-03) Added the ImGui profiler window and updated the path tracer Python example to use `Profiler.frame()` / `Profiler.zone()` plus a per-frame `tick()`.
- [x] (2026-06-03) Refreshed generated stubs and `py_doc.h`; stale public names `ProfilerZoneScope`, `ProfilerFrameScope`, and `render_overlay` no longer appear in `slangpy`, `src/slangpy_ext`, `tests`, or `examples`.
- [x] (2026-06-03) Built, ran the profiler Python tests, ran C++ unit tests, ran the stale-name grep, and ran pre-commit successfully.
- [x] (2026-06-03) Audited the profiler hot path and moved GPU source/name dictionary resolution out of `begin_zone()` and into completed-query resolution, removing the avoidable `data_mutex` lock from successful GPU zone begins.
- [x] (2026-06-03) Reworked GPU zones to flow through the per-thread event queue. `Profiler::begin_zone()` now queues the CPU begin event, optionally allocates a timestamp query pair, writes the begin timestamp, and queues a compact GPU begin event; GPU zone records, IDs, nesting, stats links, and frame association are built later while draining that thread's events.
- [x] (2026-06-03) Made frame association thread-local. Zone events no longer carry `event_id` or `frame_id`; the event processor assigns event IDs and applies only the active frame for the same producing thread.
- [x] (2026-06-03) Added the worker-thread event processor for per-thread profiler queues. `tick()` and `flush()` can still help drain events, and `tick()` remains responsible for checking GPU submit readiness and reading timestamp query results.
- [x] (2026-06-04) Replaced the bool-returning zone/frame begin APIs with by-value `ProfilerZoneToken` and `ProfilerFrameToken` payloads. The C++ RAII guards now store tokens, and the zone token keeps only the end-query write state instead of using `ActiveGpuRecording::query_stack`.
- [x] (2026-06-04) Collapsed GPU begin/end queue events into one GPU-zone metadata event queued from `Profiler::begin_zone()` after the begin timestamp is written. `Profiler::end_zone()` writes the reserved end timestamp and then queues the CPU end event.
- [x] (2026-06-04) Removed the Python `Profiler.zone()` / `Profiler.frame()` context-manager bindings and updated `examples/pathtracer/pathtracer.py` plus Python profiler tests to avoid manual Python zone/frame scopes.
- [x] (2026-06-04) Validated the token refactor with `pre-commit run --all-files`, `cmake --build --preset windows-msvc-debug`, `python tools/ci.py unit-test-cpp`, and `pytest slangpy/tests/slangpy_tests/test_profiler.py -v`.

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

- Observation: GPU timestamp conversion cannot assume `TimestampCalibration::cpu_timestamp` is already comparable to `sgl::Timer::now()`.
  Evidence: `src/sgl/device/types.h` exposes an explicit `CpuTimestampDomain` and CPU frequency for calibration, while `src/sgl/core/timer.cpp` currently returns nanoseconds from `std::chrono::high_resolution_clock`.

- Observation: The older GPU recording path reads timestamp query results per zone and can grow query pools when blocks are exhausted.
  Evidence: `src/sgl/device/profiler.cpp` uses `QueryPool::get_result()` for individual begin/end indices in the existing resolver path and creates additional query pools in the older query-block acquisition path, while `src/sgl/device/query.cpp` already exposes `QueryPool::get_results()` for batched block reads.

- Observation: Current hot-path queue writes treat enqueue failure as fatal instead of reporting a soft profiler-scope failure.
  Evidence: `ThreadData::queue_event()` in `src/sgl/device/profiler.cpp` calls `ConcurrentQueue::enqueue()` and terminates through `std::abort()` if the queue cannot allocate/enqueue.

- Observation: Reading an entire fixed-size GPU query block can include unwritten query slots and prevent GPU zones from resolving.
  Evidence: The first Python profiler test run after reconnecting GPU recording produced GPU timelines but no GPU zones. Tracking `used_query_count` per query block and reading only written timestamps fixed D3D12, Vulkan, and CUDA auto-zone traces.

- Observation: `pre-commit` needs write access to the user's pre-commit cache outside the workspace sandbox.
  Evidence: The first sandboxed `pre-commit run --all-files` failed with `sqlite3.OperationalError: attempt to write a readonly database` for `C:\Users\Simon Kallweit\.cache\pre-commit`; rerunning with approved escalation succeeded.

- Observation: Returning tokens lets GPU query state stay small while the worker still receives one GPU metadata event.
  Evidence: The 2026-06-04 implementation removed `begin_gpu_zone`, `end_gpu_zone`, `OpenGpuQuery`, and `ActiveGpuRecording::query_stack`; `Profiler::begin_zone()` queues a single `gpu_zone` event containing the reserved begin/end query indices, while `ProfilerZoneToken` keeps only the encoder, query pool, end query index, and debug-group state needed by `Profiler::end_zone()`.

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

- Decision: Convert GPU timestamps from raw query ticks using an explicit profiler-owned CPU/GPU anchor.
  Rationale: `TimestampCalibration::cpu_timestamp` may be in a CPU domain that is not `sgl::Timer::now()`. The implementation should use raw `QueryPool::get_results()` ticks and either use the existing midpoint anchor around `Device::get_timestamp_calibration()` or a helper that proves and converts the reported CPU domain to the `Timer::now()` domain.
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

- Decision: Do not over-engineer the profiler; keep the implementation as simple as possible while staying fast.
  Rationale: This code runs inside application instrumentation and shader-performance measurement workflows. Extra abstraction, clever generalization, multi-stage pipelines, or premature compaction can make the profiler harder to maintain and slower on the path it is supposed to measure.
  Date/Author: 2026-06-03 / User

- Decision: Supersede the earlier bool-returning `begin_zone()` / `begin_frame()` decision with by-value token APIs.
  Rationale: Tokens make begin/end pairing explicit, let end use thread data and the reserved GPU end-query state captured by begin, remove the per-recording GPU query stack, and reduce worker traffic to one GPU metadata event per GPU zone while preserving paired query allocation.
  Date/Author: 2026-06-04 / User/Codex

- Decision: Make `ProfilerZoneToken` and `ProfilerFrameToken` dumb by-value data payloads.
  Rationale: These tokens are part of the hot path. Only `profiler` is default-initialized to null so inactive tokens can be checked cheaply; all other fields are intentionally plain uninitialized payload and are filled only for active tokens. `end_zone()` and `end_frame()` take tokens by value to avoid move-only machinery and reset work.
  Date/Author: 2026-06-04 / User/Codex

- Decision: Require correct begin/end pairing while the command encoder is open; exceptions from invalid encoder use are hard failures.
  Rationale: Recovering from a token ended after encoder finish or from a leaked token would require extra completion markers, mutable records, or sparse query-read bookkeeping. This profiler system is new, and hard failure keeps incorrect instrumentation visible and the hot path simple.
  Date/Author: 2026-06-04 / User/Codex

- Decision: Remove manual Python zone/frame context-manager bindings.
  Rationale: The profiler scope guards are C++ implementation details and this new system has no backwards-compatibility requirement. Python should keep profiler configuration, snapshots, current-profiler stack management, UI rendering, and automatic SlangPy dispatch zones, without exposing manual `Profiler.zone()` / `Profiler.frame()` wrappers.
  Date/Author: 2026-06-04 / User/Codex

- Decision: Handle trace start, stop, and clear in the worker instead of sampling trace-enabled state in every zone begin event.
  Rationale: Trace state should not add a hot-path flag or epoch write to every instrumentation event. Trace control methods can enqueue worker control commands with timestamps, and the worker can decide which begin events fall inside active capture windows.
  Date/Author: 2026-06-03 / User

- Decision: Use worker-owned trace epochs and timestamp windows for deterministic trace clearing.
  Rationale: Per-thread event queues can deliver older zone events after a later `clear_trace()` command. A clear or `start_trace(clear=true)` must start a new capture epoch, discard stored trace records, and reject zones whose begin timestamps fall before the new epoch start even if those zone events are processed later.
  Date/Author: 2026-06-03 / Codex

- Decision: When GPU query allocation is exhausted, emit a warning and disable GPU profiling for that device/queue.
  Rationale: Blocking, reallocating unpredictably, or silently dropping partial GPU zones would make profiler behavior harder to reason about. CPU profiling should continue.
  Date/Author: 2026-06-03 / User

- Decision: Treat `gpu_query_pool_size` as a hard per-device/queue query budget and read completed queries in blocks.
  Rationale: The profiler should not allocate unbounded query pools while measuring performance. Batched `QueryPool::get_results()` calls avoid one RHI call per begin/end timestamp and make query-block recycling predictable.
  Date/Author: 2026-06-03 / Codex

- Decision: Ignore nested or overlapping frame begins and emit a warning.
  Rationale: The profiler has one application-wide frame model. Ignoring ambiguous frame begins preserves the current frame and makes the problem visible without corrupting frame statistics.
  Date/Author: 2026-06-03 / User

- Decision: Prove the final CPU/GPU event payload and correlation model synchronously before moving processing to the worker.
  Rationale: This lets the public API, flat trace shape, CPU stats, GPU query ownership, and CPU/GPU `event_id` correlation become testable before adding worker-thread lifecycle and synchronization complexity.
  Date/Author: 2026-06-03 / User

- Decision: `ProfilerZoneFlags::debug_group` takes precedence over `Profiler::debug_groups_enabled()`.
  Rationale: An explicit per-zone flag should force debug-group behavior for that scope. The profiler-wide setting controls whether automatic SlangPy zones request debug groups by adding that flag.
  Date/Author: 2026-06-03 / User

- Decision: Let the worker poll queues instead of notifying it from the hot path.
  Rationale: Waking a worker on every zone begin/end would add measurable overhead to the instrumentation path. The hot path should use direct queue writes without `previous_count` tracking or condition-variable notifications. Queueing events does not return a soft failure; memory exhaustion during enqueue terminates the process. The worker drains all queues in a pass and sleeps briefly, about 1ms, only if every queue contained no new work.
  Date/Author: 2026-06-03 / Codex

- Decision: Keep CPU event processing synchronous in this implementation pass and leave worker-thread migration as the remaining follow-up.
  Rationale: Milestones 1 and 2 explicitly allow proving the API, flat trace shape, CPU stats, GPU query ownership, and CPU/GPU correlation synchronously before introducing worker lifecycle and barrier complexity. The completed implementation keeps the hot path queue-based and validates the final payload shape first.
  Date/Author: 2026-06-03 / Codex

- Decision: Mint the logical CPU/GPU `event_id` in `begin_zone()` for this synchronous pass.
  Rationale: Active GPU recordings need a stable correlation ID before command-buffer submission callbacks hand query batches to the resolver. This adds one atomic increment to the hot path and should be revisited during the worker migration if strict worker-assigned IDs remain required.
  Date/Author: 2026-06-03 / Codex

- Decision: Defer GPU source/name ID resolution until query-batch processing.
  Rationale: GPU zone begin already has stable source-location and name pointers. Resolving compact dictionary IDs inside `begin_zone()` took `data_mutex` on every successful GPU zone begin; doing it during completed-query processing preserves trace/stat output while keeping the measured path leaner.
  Date/Author: 2026-06-03 / Codex

- Decision: Route GPU zone metadata through the per-thread event stream.
  Rationale: The hot path should only allocate timestamp query indices, write timestamps, and enqueue compact events. CPU/GPU correlation, source/name IDs, nesting, stats-node links, and frame IDs are derived later from the ordered events for the producing thread.
  Date/Author: 2026-06-03 / User/Codex

- Decision: Treat frames as thread-local producer state.
  Rationale: A frame opened on one thread must not implicitly classify zones produced by other threads. The producer only tracks a lightweight per-thread active-frame guard for RAII correctness; durable frame IDs are assigned by the event processor.
  Date/Author: 2026-06-03 / User/Codex

## Outcomes and Retrospective

Implemented the new profiler API and runtime foundation. `ProfilerDesc` now exposes the planned defaults and validation. `ProfilerTrace` exposes flat timelines, sources, names, frames, zones, child indices, and root indices. `ProfilerStats` exposes rolling CPU/GPU timing nodes. C++ macros use internal RAII guards backed by simple by-value zone/frame tokens. Python no longer exposes manual zone/frame context managers; it keeps profiler configuration, snapshots, current-profiler stack helpers, UI rendering, and automatic SlangPy dispatch zones.

The old stub GPU timestamp path and pointer-heavy trace tree were replaced by per-thread CPU event queues, synchronous processing in `tick()` / `flush()`, submitted/discarded command recording callbacks, per-device/queue hard query budgets, block-level raw query reads with explicit CPU/GPU timestamp anchors, and flat stat/trace storage. Discarded GPU recordings are ignored safely, submitted GPU zones resolve after `tick()`, and SlangPy automatic zones cache their interned profiler name on `NativeCallData`.

Added `sgl::ui::render_profiler_window()` / `slangpy.ui.render_profiler_window()` and updated `examples/pathtracer/pathtracer.py` to keep the profiler current while relying on automatic SlangPy dispatch zones instead of manual Python zone/frame context managers.

Validated with:

    cmake --build --preset windows-msvc-debug
    cmake --build --preset windows-msvc-debug --target slangpy_stub
    cmake --build --preset windows-msvc-debug --target slangpy_ui_stub
    cmake --build --preset windows-msvc-debug --target slangpy_pydoc
    pytest slangpy/tests/slangpy_tests/test_profiler.py -v
    python tools/ci.py unit-test-cpp
    rg "ProfilerZoneScope|ProfilerFrameScope|render_overlay" slangpy src/slangpy_ext tests examples
    pre-commit run --all-files

The worker migration is now in place for event processing. `flush()` waits for queued CPU/GPU zone/frame events to be processed, while unfinished GPU submissions are still resolved later from `tick()` once their timestamp queries are ready.

Follow-up hot-path audit: warmed CPU-only zone/frame begin/end paths remain limited to timestamps, TLS lookup, per-thread frame guard checks, token state, and queue writes after per-thread setup. GPU zone begin now performs CPU event enqueue, optional query-pair allocation, begin timestamp write, one GPU metadata event enqueue, and compact token population for the end query. GPU zone end writes the reserved end timestamp and queues the CPU end event. The producer path no longer assigns event/frame IDs, builds GPU zone records, queues separate GPU begin/end events, or uses a per-recording GPU query stack. It can still take `gpu_mutex` on command-recording cache misses or query-block rollover, and dynamic names with `ProfilerZoneFlags::copy_name` still use the global string interning mutex by design.

## Context and Orientation

The profiler lives primarily in `src/sgl/device/profiler.h` and `src/sgl/device/profiler.cpp`. It is exposed to Python by `src/slangpy_ext/device/profiler.cpp`. Current Python behavior is tested in `slangpy/tests/slangpy_tests/test_profiler.py`; current C++ behavior is tested in `tests/sgl/device/test_profiler.cpp`.

A profiler zone is a timed scope. A CPU zone records host timestamps from `sgl::Timer::now()`. A GPU zone records timestamp queries into a `CommandEncoder`; those queries become readable only after the command buffer is submitted and the GPU work completes. A profiler frame is an application frame used to group zones for realtime statistics.

The existing implementation already has useful pieces: per-thread queues through `ThreadData`, static source-location macros, string/source interning, command recording IDs, and device callbacks for submitted or discarded command buffers. It also has temporary or broken pieces: GPU timestamp allocation is stubbed in one path, the older GPU path is disconnected, trace storage currently stores pointer-heavy nested `ProfilerZone` trees, and `tick()` currently performs too much work synchronously.

The new design should split the profiler into four conceptual layers:

1. Hot-path recording. This runs inside zone begin/end calls and is the most important performance-sensitive path. It must stay small, non-blocking, and acceptable in debug builds.
2. Main-thread GPU query resolution. This runs from `Profiler::tick()` because query result access may not be thread safe yet.
3. Worker-thread processing. This drains queues, resolves stacks, builds frame statistics, and appends optional trace data.
4. Presentation/export. This exposes flat Python records, writes Chrome JSON, and renders a realtime ImGui window.

This split is a way to keep responsibilities understandable, not an invitation to build a framework. Prefer straightforward structs, vectors, maps, and one worker thread. Add an abstraction only when it removes real duplication or protects the hot path. Avoid multi-worker schedulers, generic task systems, intricate bit-packing, and speculative extension points in v1.

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

Descriptor validation should be explicit and tested. `stats_window_size` is the number of completed application frames retained for rolling statistics and must be at least 1. `gpu_query_pool_size` and `gpu_query_block_size` must be positive even numbers, and the block size must be less than or equal to the pool size. Invalid descriptors should fail during profiler construction rather than producing partially initialized profiling state.

`Profiler` should expose:

- `enabled`, `set_enabled`
- `frame_stats_enabled`, `set_frame_stats_enabled`
- `auto_zones_enabled`, `set_auto_zones_enabled`
- `debug_groups_enabled`, `set_debug_groups_enabled`
- `stats_window_size`, `set_stats_window_size`
- `start_trace(bool clear = true)`
- `stop_trace()`
- `clear_trace()`
- `trace_snapshot()`
- `stats_snapshot()`
- `tick()`
- `flush()`
- `begin_zone(...) -> ProfilerZoneToken` / `end_zone(ProfilerZoneToken)` for C++ internals and macros
- `begin_frame(...) -> ProfilerFrameToken` / `end_frame(ProfilerFrameToken)` for C++ internals and macros

Python should expose profiler configuration, application-wide current-profiler stack helpers, snapshots, trace controls, `tick()`, and `flush()`. It should not expose manual Python zone/frame context managers in this design pass; SlangPy automatic dispatch zones continue to use the current profiler when `auto_zones_enabled` is true.

`ProfilerZoneFlags` should expose `none`, `debug_group`, and `copy_name`. `debug_group` forces push/pop debug-group behavior for that scope. `copy_name` means the profiler owns a stable copy or interned version of the supplied name before the caller's string can die. Python `str` names should always be made safe by the binding, either by passing `copy_name` or by interning/copying in the context-manager helper.

Do not expose backwards-compatible Python names such as `ProfilerZoneScope` or `ProfilerFrameScope`, and do not expose manual Python `Profiler.zone()` / `Profiler.frame()` context managers. The C++ RAII helper may keep an internal implementation name if that is the simplest path, but generated docs, Python stubs, Python tests, and Python examples should not show manual Python zone/frame scope APIs. Update `examples/pathtracer/pathtracer.py` to keep `with self.profiler:` around dispatch work so automatic SlangPy zones see the current profiler.

`ProfilerTrace` should expose flat data, not nested trees:

- timelines: ID, type, name, thread ID, device ID, queue.
- source locations: ID, file, line, original function, display function.
- names: ID and display string.
- frames: ID, name ID, source ID, start/end timestamps.
- zones: zone ID, logical event ID, child-list range, timeline ID, frame ID, source ID, name ID, start timestamp, and end timestamp.
- child indices: a compact `uint32_t` zone-ID array referenced by each zone's child-list range.
- root indices: compact `uint32_t` zone-ID arrays referenced by each timeline and frame through `first_root_index` plus `root_count`.

`ProfilerTraceZone` should be the durable per-zone record and should stay deliberately small. Do not store duration, depth, display path, source strings, name strings, parent event ID, parent zone ID, timeline type, resolved state, or raw pointers in each zone. Derive duration from timestamps, derive depth/path/parent information while traversing from timeline/frame roots through child lists, derive CPU/GPU type from the timeline, and keep strings in source/name dictionaries. Use `std::numeric_limits<uint32_t>::max()` as the invalid sentinel for 32-bit IDs and `std::numeric_limits<uint64_t>::max()` as the invalid sentinel for missing logical event IDs.

`ProfilerTrace::write_to_json(path)` remains, but it should generate Chrome trace JSON from flat zones and child-index arrays. It should include metadata timeline events and enough derived args for analysis, including source file, line, display function, event ID, parent event ID when known during traversal, frame ID, raw timestamps, and whether the zone is CPU or GPU.

`ProfilerStatsSnapshot` should expose a hierarchical tree for the rolling frame-stat window. Each node should include display name, source ID, name ID, path, depth, CPU timing stats, GPU timing stats, sample counts, and unresolved GPU sample count. Timing stats include last, min, max, average, and standard deviation. The update-time stats key should be parent stats-node ID plus source-location ID plus name ID so same-named zones from different callsites do not merge accidentally without building path strings on every sample. Derive display paths only when producing snapshots or UI rows.

Add `sgl::ui::render_profiler_window(Profiler*, const ProfilerWindowDesc& = {})` and expose it as `slangpy.ui.render_profiler_window(profiler, ...)`. `ProfilerWindowDesc` should be small in v1: `std::string title = "Profiler"`, `bool show_controls = true`, `bool show_cpu = true`, `bool show_gpu = true`, and `bool show_pending_gpu = true`. The UI should show the frame-stat hierarchy with CPU/GPU columns and, when controls are visible, allow the rolling window size to be edited through `Profiler::set_stats_window_size()`. It should also show pending GPU samples because GPU results arrive asynchronously. The UI should use the latest processed stats snapshot or a cached versioned stats view; it should not force a full trace flush or rebuild every display path string every frame. Use ImGui clipping or equivalent row filtering for large trees.

## Plan of Work

Work in increments that keep the profiler testable. Milestone 1 should establish the new public API, CPU trace shape, and CPU stats without deleting the reusable older GPU recording machinery. The GPU milestone should then remove the temporary stub `GpuContextData` event path and reconnect the older `ActiveGpuRecording` idea after reshaping it to the new event/zone-record model. The final code should not have two separate GPU recording systems. Keep the code boring and direct: prefer a small number of plainly named data structures over layered abstractions, and measure complexity against hot-path cost and maintainability.

Define compact internal IDs. Source locations, names, timelines, frames, trace zones, child indices, root lists, and stats nodes should use `uint32_t` IDs and offsets/counts. Logical events should use `uint64_t` event IDs. Use `std::numeric_limits<uint32_t>::max()` for absent 32-bit IDs and `std::numeric_limits<uint64_t>::max()` for absent event IDs. Raw `ProfilerSourceLocation*` and `const char*` are acceptable only in hot-path event payloads because macros and interned strings naturally provide stable pointers. The synchronous processor in early milestones and the worker in the final design must translate those pointers to compact IDs.

Keep durable per-zone storage minimal. A processed trace zone should be close to this shape: `uint32_t zone_id`, `uint64_t event_id`, `uint32_t first_child_index`, `uint32_t child_count`, `uint32_t timeline_id`, `uint32_t frame_id`, `uint32_t source_id`, `uint32_t name_id`, `uint64_t start_timestamp_ns`, and `uint64_t end_timestamp_ns`. Use an invalid ID sentinel for absent frame/name/source values. Store children in a separate compact `uint32_t` zone-ID array. Store timeline/frame roots the same way. Add a field only if it cannot be derived cheaply and is needed by more than one consumer.

Format verbose pretty function names in the worker. Preserve the original function string in the source dictionary, but derive a shorter display function for trace names and UI display. The implementation does not need an elaborate demangler in v1; it should at least strip noisy compiler signature prefixes/suffixes enough that macro-generated C++ function names are readable.

Define timeline identity explicitly. CPU timelines are per producer thread and use the thread ID in `ProfilerTimelineInfo`. GPU timelines are per device plus queue, not per command recording. Command recording ID is a GPU stack and query-ownership key, while the final GPU zones for recordings on the same device/queue appear on that device/queue timeline.

Redesign hot-path events. `begin_zone` should:

- return an inactive `ProfilerZoneToken` immediately when profiling is disabled.
- read a CPU timestamp.
- copy dynamic names only when requested by flags, otherwise store stable name pointers.
- enqueue a small CPU begin event to the current thread's queue.
- if a command encoder is supplied and GPU profiling is supported, acquire or reuse a query pair, write the begin timestamp query, enqueue one GPU metadata event with both query indices, and store only the end-query write state in the returned token.

`end_zone` should:

- read a CPU timestamp.
- if the token has GPU query state, write the matching GPU end timestamp query. Failure to write this query is a hard failure because the encoder must still be open.
- enqueue a small CPU end event to the same thread queue.
- pop a debug group only if this scope pushed one.

Use simple by-value tokens as the begin/end contract. The RAII zone scope should store `ProfilerZoneToken`; the RAII frame scope should store `ProfilerFrameToken`. Only the `profiler` field is initialized for inactive tokens; all other fields are plain payload and must be filled before returning an active token. The implementation should only pop a debug group when the token records that this scope successfully pushed one. `debug_groups_enabled` controls whether automatic SlangPy zones add `ProfilerZoneFlags::debug_group`.

Use per-thread queues for CPU and GPU zone events. The hot path should not allocate trace zones from a trace slab, build zone trees, assign source/name dictionary IDs, assign logical event IDs, update stats, append trace data, or sample trace-enabled state. The queue payload should be compact and fixed-size where possible. Queuing events does not return a soft failure; if `moodycamel::ConcurrentQueue::enqueue()` fails due to memory exhaustion, terminate the process. If GPU query allocation fails for a scope, continue CPU-only unless the failure is query-pool exhaustion, which disables GPU profiling for that device/queue after a warning. The worker polls queues and sleeps briefly when a full pass contains no new events.

Keep GPU hot-path work to the unavoidable minimum. Writing timestamp queries must happen while encoding commands, and query indices must be known before those writes. The fast path should therefore use a per-thread/per-command-recording cached query block cursor so begin/end only reserve indices, write timestamps, and enqueue compact GPU events. Slow-path query block acquisition may lock, but it should happen only when the cached block is exhausted or a new command recording is first seen. Record the first thread ID that uses a command recording for GPU profiling; if the same recording ID is later seen from another thread in v1, emit a debug assertion or warning and record that scope as CPU-only rather than racing the per-thread cursor. If the configured query budget cannot provide another block, call `log_warn_once` or an equivalent one-time warning path and disable GPU profiling for that device/queue while leaving CPU profiling enabled.

Add a worker thread owned by `ProfilerImpl` only after the synchronous CPU/GPU event model is tested. The worker should poll all known thread queues, process resolved GPU batches delivered by `tick()`, and sleep briefly, about 1ms, only after a full pass found no new events. The worker owns the expensive work: CPU stack reconstruction, source/name dictionary creation, frame-stat tree updates, optional trace append, and pruning history windows. `tick()` may deliver resolved GPU work and `flush()` may wait on a barrier, but producer threads should not notify the worker from zone/frame begin/end.

Define shutdown before adding the worker. `ProfilerImpl` teardown should disable recording, unregister command-recording callbacks from each retained `Device`, clear thread-local GPU caches for the profiler, wake the worker, stop and join the worker, then destroy queues and trace/stat storage. The profiler must outlive instrumented producer threads in v1; if producer threads can still be entering profiler scopes during destruction, that remains outside v1 unless a retained activation model is added.

Define `flush()` precisely. `flush()` should call `tick()` once to deliver any already completed GPU submits, enqueue a worker barrier, wake the worker, and wait until all CPU events, trace-control commands, and resolved GPU batches visible before that barrier have been processed. It must not wait for unfinished GPU submissions. `trace_snapshot()` should call `tick()` and `flush()` before copying the trace. `stats_snapshot()` may return the latest processed stats without forcing a full flush; callers that need synchronous freshness can call `flush()` first.

Let the worker correlate CPU and GPU events from queue order. A GPU metadata event from `begin_zone()` must appear immediately after the corresponding CPU begin event in the same producer queue. The worker assigns the logical `event_id` when it processes the CPU begin event, attaches the GPU query metadata to the currently open CPU zone, and later closes the CPU zone when it processes the CPU end event. Correct instrumentation must write the end timestamp before the CPU end event is queued. This keeps the hot path from minting or looking up correlation IDs and avoids a worker-side GPU begin/end stack for normal token-backed scopes.

Keep trace capture optional and separate from stats. When trace capture is disabled, do not allocate trace zone storage and do not retain full trace history. Stats should still update when `frame_stats_enabled` is true. `start_trace(clear)`, `stop_trace()`, and `clear_trace()` should enqueue worker control commands with `Timer::now()` timestamps; the worker owns trace-active state, capture epochs, and active capture windows. `start_trace(clear=true)` and `clear_trace()` increment the capture epoch, clear stored trace data, and reject later-processed zone events whose begin timestamp is before the new epoch start. A zone enters trace storage only when its begin timestamp is greater than or equal to an active start timestamp, less than the corresponding stop timestamp if one exists, and in the current capture epoch. `trace_snapshot()` should call `tick()` and `flush()` so the returned snapshot includes all trace data visible before the snapshot request.

Use one application-wide frame model. `begin_frame` reserves a new frame ID synchronously, opens a global active frame ID, records a frame begin event, and publishes that ID atomically so subsequent zones can attach to it. The synchronous processor or worker later builds the frame record. `end_frame` closes that frame. Nested or overlapping frame begins are ignored with a warning in all builds rather than closing the existing frame. The stats UI is frame based, so ambiguous frame ownership should not be accepted silently.

Handle GPU zones per command recording in the worker. All GPU zones can be assumed to be captured from the same thread in v1, but command encoders may be interleaved on that thread. Therefore, maintain a separate worker-side GPU zone stack per command recording ID, not one global GPU stack. The hot path owns only the query-block cursor needed to reserve query indices and write timestamps. The worker owns queued GPU zone records and the stack of open GPU zone indices for each command recording.

Allocate GPU timestamp queries from a hard query budget per device/queue combination. Use descriptor sizes from `ProfilerDesc`; `gpu_query_pool_size` is the total cap across free, active, pending, and abandoned blocks for that device/queue. Allocate fixed-size query blocks to active command recordings. A GPU zone consumes two queries, begin and end. When a query block is exhausted, acquire another block from the same device/queue pool. Reset blocks when they are created or recycled, not during the begin/end hot path. Submitted query blocks may be recycled after query results have been read. Discarded/unsubmitted blocks may be abandoned for the lifetime of the profiler in v1. If abandoned and pending blocks exhaust the configured pool, emit a warning once, flip a cheap disabled flag for that device/queue, and skip further GPU profiling attempts for that device/queue until the profiler is destroyed or recreated.

Use submitted command recording callbacks as ownership handoff, not as resolve work. `Device::_register_command_recording_submitted_callback()` should move the hot-path query allocation state for that command recording into a pending submit list with its `submit_id`, recording ID, device, queue, and query blocks. The callback should not query result readiness or call `get_result()`. `Device::_register_command_recording_discarded_callback()` should remove active query allocation state and abandon associated query blocks. It does not need to prove query reuse safety.

Resolve GPU timestamps from `Profiler::tick()`. `tick()` should check pending submit IDs with `Device::is_submit_finished()`. For finished submits, capture a profiler-owned timestamp anchor by reading `Timer::now()` immediately before and after `Device::get_timestamp_calibration()` and using the midpoint as the CPU nanosecond timestamp corresponding to the returned GPU timestamp, unless a dedicated helper proves and converts the reported `TimestampCalibration::cpu_domain` to the `Timer::now()` domain. Read raw query ticks with block-level `QueryPool::get_results()` calls into reusable scratch buffers, convert those ticks with the anchor, create raw GPU timestamp-result batches keyed by command recording ID plus query pool/index, recycle submitted query blocks, and pass those batches to the synchronous processor or worker. Do not use `QueryPool::get_timestamp_result()` for profiler correlation because it produces seconds in the device timestamp-frequency domain without CPU-domain anchoring. The processor then joins resolved timestamps to the GPU zones it assembled from queued begin/end events. If timestamp query or calibration features are missing, GPU zones are not recorded and CPU zones still work.

Build frame statistics from CPU and GPU zones. CPU zones update stats when the CPU end event is processed. GPU zones update stats when `tick()` resolves queries, potentially one or more frames later. The `event_id` ties the CPU zone and GPU zone for the same logical instrumented scope together. Stats should keep samples for the newest `stats_window_size` completed application frames and compute last, min, max, average, and standard deviation for CPU and GPU independently. If the same stats node is sampled multiple times in a frame, preserve sample count rather than pretending it was a single call. The UI should make unresolved GPU samples visible instead of treating them as zero.

Keep trace data flat but child-traversable. The trace can still be exported as nested-looking Chrome events by traversing child ID ranges and timestamps, but storage should be arrays of records plus compact child-index/root-index arrays. Avoid raw `ProfilerZone*`, recursive object ownership, and per-zone `std::vector` allocations in permanent trace storage. Do not over-optimize with bit packing, custom allocators beyond what is already needed, or generalized compression schemes in v1; choose clear compact records that can be tightened later if profiling data proves they need it.

Split files lightly. Keep the public API in `src/sgl/device/profiler.h`. Keep the core runtime in `src/sgl/device/profiler.cpp` unless it becomes unwieldy. Move Chrome JSON writing and trace-specific helpers to `src/sgl/device/profiler_trace.cpp` if that makes the core readable. Put the ImGui window in `src/sgl/ui/profiler.h` and `src/sgl/ui/profiler.cpp` so UI code is separate from profiler core. Update `src/sgl/CMakeLists.txt` and `src/slangpy_ext/CMakeLists.txt` for any new files.

Refresh Python bindings in `src/slangpy_ext/device/profiler.cpp` and `src/slangpy_ext/ui/ui.cpp`. Bind the new flat trace records and stats snapshots. Bind Python context managers through small native helper types or nanobind lambdas; because there is no compatibility requirement, choose the simplest new Python API and update all tests/examples accordingly.

Refresh generated docs after the API stabilizes. Run the `slangpy_pydoc` target if the environment supports it, or explicitly update `src/slangpy_ext/py_doc.h` if the normal generated-doc flow is unavailable. The final tree should not keep stale profiler doc symbols.

## Milestones

Milestone 1: establish the new API and CPU-only processing path. At the end of this milestone, C++ zone/frame macros, explicit C++ tokens, trace start/stop/snapshot, and CPU frame stats work without GPU timestamps. This milestone may process CPU queues synchronously in `tick()` and `flush()`. Leave GPU cleanup and worker migration out of this milestone unless a small adapter is required to keep the build passing. Acceptance is a C++ test that starts a trace, records nested CPU zones inside a frame, snapshots flat trace zones, writes JSON, reads non-empty stats, validates `ProfilerDesc` defaults and invalid descriptors, and verifies that a copied trace snapshot remains readable after `clear_trace()`.

Milestone 2: add GPU query recording and main-thread resolve while processing remains synchronous. At the end of this milestone, manual GPU zones and SlangPy auto zones reserve query indices and write begin/end timestamp queries on the hot path, submitted recordings move to pending submits, `tick()` resolves completed GPU results using batched raw query reads and the explicit timestamp anchor, and stats/trace show matching CPU and GPU zones by event ID. Acceptance includes a happy-path GPU test, an interleaved-command-encoder test, a query-block rollover test, a query-exhaustion test that disables only GPU profiling for the affected device/queue, and a discarded/unsubmitted recording test that does not add GPU zones or crash.

Milestone 3: add worker-thread processing. At the end of this milestone, hot-path zone calls only enqueue compact events, expensive CPU/GPU stack assembly and stat/trace updates happen in the worker, and `flush()` waits only for work visible before its barrier rather than for unfinished GPU submissions. Acceptance is a stress test that records zones from multiple CPU threads, calls `flush()`, verifies child-list traversal and stats without doing stack assembly on the caller thread, verifies CPU/GPU event correlation still works after the worker migration, and destroys a profiler after registering device callbacks without leaking callbacks or racing the worker.

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

Update `src/slangpy_ext/utils/slangpy.cpp` so SlangPy automatic zones continue to use the C++ RAII-scope model, now backed by profiler tokens, and continue to create CPU/GPU zones around functional dispatches when `auto_zones_enabled` is true. Adjust flags or type names as needed. Cache the interned automatic zone name in `NativeCallData` or equivalent setup-time state so each dispatch does not hit the profiler string registry.

Update `tests/sgl/device/test_profiler.cpp` and `slangpy/tests/slangpy_tests/test_profiler.py` to test the new API. Remove expectations for compatibility names or default trace capture.

Update `examples/pathtracer/pathtracer.py` to keep the profiler current with `with self.profiler:` while relying on automatic SlangPy zones, and call `tick()` once per frame. Check `examples/pathtracer/pathtracer.cpp` for required profiler API renames, but avoid churn if it already calls `tick()` once per frame.

Build and test:

    cmake --build --preset windows-msvc-debug
    cmake --build --preset windows-msvc-debug --target slangpy_stub
    cmake --build --preset windows-msvc-debug --target slangpy_ui_stub
    cmake --build --preset windows-msvc-debug --target slangpy_pydoc
    pytest slangpy/tests/slangpy_tests/test_profiler.py -v
    python tools/ci.py unit-test-cpp
    rg "ProfilerZoneScope|ProfilerFrameScope|render_overlay" slangpy src/slangpy_ext tests examples
    pre-commit run --all-files

The `rg` command should report no stale Python-visible names in generated docs, stubs, Python tests, or examples; `rg` may exit with code 1 when there are no matches. It is acceptable for internal C++ RAII helper names to remain under `src/sgl` if they are not exposed. If `pre-commit` modifies files, rerun it until it reports success.

## Validation and Acceptance

The refactor is accepted when these behaviors are demonstrated:

- Constructing `Profiler()` records realtime stats by default but does not retain trace zones until `start_trace()` is called.
- `ProfilerDesc` defaults are visible through C++ and Python, valid descriptors configure the runtime, and invalid `stats_window_size`, `gpu_query_pool_size`, or `gpu_query_block_size` values fail during construction.
- `start_trace(clear=True)`, `stop_trace()`, `clear_trace()`, and `trace_snapshot()` work from both C++ and Python.
- `start_trace(clear=True)` and `clear_trace()` deterministically start a new capture epoch: zones that began before the clear timestamp do not appear in later snapshots even if their events are processed later.
- Returned trace and stats snapshots are stable copies. They remain readable, and trace JSON export still works, after the profiler is cleared, restarted, or destroyed.
- A CPU-only trace snapshot exposes flat timelines, sources, names, frames, zones, and child-index arrays that can traverse nested zones and write valid Chrome JSON.
- C++ macros and explicit C++ profiler tokens record nested zones under the current application frame.
- Frame stats show hierarchical CPU timings with last, min, max, average, standard deviation, and sample count over the configured rolling window.
- GPU zones written to submitted command buffers produce GPU trace zones after the submit finishes and `tick()` runs, using batched raw query reads rather than per-zone result calls.
- CPU and GPU stats for the same logical zone correlate through `event_id`.
- Interleaved command encoders on the same thread maintain independent GPU zone stacks.
- A command recording used for GPU profiling from more than one thread is rejected or degraded to CPU-only with a debug diagnostic rather than racing per-thread GPU query cursors.
- Query blocks roll over when many GPU zones are recorded.
- Exhausting GPU timestamp query capacity emits a warning and disables GPU profiling for the affected device/queue while CPU profiling continues.
- GPU timestamp query capacity is a hard per-device/queue cap across free, active, pending, and abandoned query blocks.
- Discarded or unsubmitted GPU command recordings do not crash and do not add GPU zones to stats or trace.
- Nested or overlapping frame begins are ignored with a warning and do not corrupt the active frame.
- `flush()` processes events visible before its barrier and does not wait for unfinished GPU submissions.
- Destroying a profiler after device callback registration unregisters callbacks, stops the worker, and does not leave stale current-profiler or thread-local GPU cache state.
- The ImGui profiler window renders a hierarchy with CPU and GPU columns and indicates unresolved GPU samples.
- Generated docs and Python stubs are refreshed or manually updated, stale backwards-compatibility Python names are not exposed, and tests/examples use only the new API.

## Idempotence and Recovery

The implementation should be safe to build and test repeatedly. `start_trace(clear=True)` should reset trace storage deterministically by starting a new capture epoch. `clear_trace()` should not affect rolling stats. `flush()` should be safe to call even when there is no pending work and should not wait for unfinished GPU submissions.

Profiler destruction in v1 assumes that producer threads are no longer entering profiler scopes. If that assumption is violated, record the limitation in Outcomes and Retrospective rather than attempting a partial retained activation model.

If GPU timestamp features are unsupported on a device, GPU profiling should be disabled for that device and CPU profiling should continue. Tests should skip GPU-specific assertions when `Feature::timestamp_query` or `Feature::timestamp_calibration` is missing.

If generated doc refresh is unavailable because `pybind11_mkdoc` or clang Python support is missing, record the exact error in the ExecPlan and update `py_doc.h` and generated `.pyi` surfaces manually enough that stale profiler symbols are gone. A follow-up is acceptable only for restoring the normal generated-doc workflow, not for leaving obsolete profiler API names in the tree.

## Interfaces and Dependencies

Use existing dependencies only. Keep `moodycamel::ConcurrentQueue` for per-thread queues. Use existing `Device`, `CommandEncoder`, `CommandBuffer`, `QueryPool`, and command recording callback APIs. Use existing Dear ImGui integration in `src/sgl/ui`. Use the existing logging helpers such as `log_warn_once` for one-time profiler warnings. Do not add infrastructure that is not directly required by this profiler refactor.

Do not introduce new third-party libraries. Do not add sampling profiler hooks. Do not require command encoders to be submitted for CPU profiling to work.

The final design depends on these repository APIs:

- `sgl::Timer::now()` for CPU nanosecond timestamps.
- `CommandEncoder::write_timestamp(QueryPool*, uint32_t)` for GPU timestamp writes.
- `Device::get_timestamp_calibration(CommandQueueType)` for CPU/GPU time correlation.
- `Device::is_submit_finished(uint64_t)` for non-blocking submit completion checks.
- `Device::_register_command_recording_submitted_callback()` and `_register_command_recording_discarded_callback()` for GPU recording ownership transitions.

## Artifacts and Notes

This revision expands the original compact plan with the full initial requirement set: optional trace capture, Python-readable traces, Chrome JSON export, realtime frame stats, async GPU timing, per-thread low-overhead queues, worker-thread processing, per-command-recording GPU stacks, per-device/queue query pools, main-thread query resolution, separate trace/stat storage, CPU/GPU event correlation, compact ID-based storage, minimal per-zone records, formatted function names, and separate UI files. It also removes the earlier compatibility requirement for Python scope class names. The 2026-06-03 review update makes clock-domain conversion, trace epochs, worker teardown, hard query accounting, batched query reads, stale-name validation, and milestone ordering explicit.
