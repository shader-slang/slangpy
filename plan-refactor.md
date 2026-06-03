# Profiler Refactor Plan

## Summary

Refactor the instrumentation profiler into an application-wide profiler with optional trace capture, realtime frame statistics, and a basic ImGui profiler window.

V1 decisions:
- `Profiler()` defaults to frame statistics on and trace capture off.
- `begin_frame`/`end_frame` define one application-wide active frame.
- GPU query resolution happens only from `Profiler::tick()` on the caller thread.
- Discarded/unsubmitted GPU recordings are ignored simply: they must not enter traces/stats, but v1 does not prove safe query reuse.
- Python trace access is flat span data, not nested object trees.
- Python gets both ergonomic `profiler.zone()` / `profiler.frame()` context managers and compatibility `ProfilerZoneScope` / `ProfilerFrameScope`.

## Public API Changes

- Extend `ProfilerDesc` with defaults:
  - `frame_stats_enabled = true`
  - `trace_enabled_on_start = false`
  - `stats_window_size = 120`
  - `gpu_query_pool_size = 64 * 1024`
  - `gpu_query_block_size = 256`
- Add trace control:
  - `Profiler::start_trace(bool clear = true)`
  - `Profiler::stop_trace()`
  - `Profiler::clear_trace()`
  - `Profiler::trace_snapshot()`
- Add processing/snapshot APIs:
  - `Profiler::tick()` resolves GPU work on the caller thread and wakes the worker.
  - `Profiler::flush()` waits until queued CPU/resolved-GPU events already visible to the profiler have been processed.
  - `Profiler::stats_snapshot()` returns the current frame-stat tree.
- Replace pointer-heavy trace storage with flat public records:
  - `ProfilerTraceTimeline`
  - `ProfilerTraceSource`
  - `ProfilerTraceName`
  - `ProfilerTraceFrame`
  - `ProfilerTraceSpan`
- `ProfilerTraceSpan` stores `span_id`, `event_id`, `parent_span_id`, `parent_event_id`, `timeline_id`, `frame_id`, `source_id`, `name_id`, `start_timestamp_ns`, `end_timestamp_ns`, and depth.
- Keep `ProfilerTrace::write_to_json(path)`; generate Chrome JSON from flat spans.
- Add stats records with CPU/GPU `last/min/max/avg/stddev/sample_count/pending_count`, keyed by hierarchical path plus source ID plus name ID.
- Add `sgl::ui::render_profiler_window(Profiler*, const ProfilerWindowDesc& = {})`, exposed as `slangpy.ui.render_profiler_window(profiler, ...)`.

## Implementation Plan

- First create and maintain an ExecPlan file following `.agents/PLANS.md`; this refactor is large enough that progress, discoveries, and decisions should be recorded during implementation.
- Clean up `src/sgl/device/profiler.cpp` by removing the split temporary GPU paths:
  - Delete the stub `GpuContextData::allocate_timestamp_query()` path.
  - Re-enable a single ActiveGpuRecording-style GPU path.
  - Remove queued `begin_gpu_zone` / `end_gpu_zone` events unless they are part of the final design.
- Redesign hot-path events:
  - `begin_zone` mints a unique `event_id`, captures `frame_id`, timestamp, source pointer, stable name pointer, and optional GPU encoder.
  - CPU events go into per-thread `moodycamel::ConcurrentQueue`.
  - GPU zones are recorded immediately into per-command-recording state with the same `event_id`.
  - Final processed spans store 32-bit source/name/timeline IDs, not raw pointers.
- Add a worker-owned processor:
  - Drain all thread queues.
  - Maintain per-thread CPU zone stacks.
  - Maintain source/name dictionaries and formatted display names.
  - Build frame-stat trees and optional trace spans.
  - Process resolved GPU span batches delivered by `tick()`.
- Keep trace and stats separate:
  - Stats accumulation runs when `frame_stats_enabled` is true.
  - Trace slab/storage is allocated and appended only for zones whose begin event occurred while trace capture was active.
  - `trace_snapshot()` calls `tick()` and `flush()` before copying current trace records.
- GPU handling:
  - Use one query-pool allocator per device/queue.
  - Allocate query blocks per active command recording.
  - Maintain a separate GPU zone stack per command recording to support interleaved encoders on one thread.
  - Submitted callbacks only move completed recordings to a pending list with `submit_id`; they do not query timestamps.
  - `tick()` checks pending submits, captures timestamp calibration, reads query results, converts GPU timestamps to CPU nanoseconds, releases submitted query blocks, and hands resolved spans to the worker.
  - Discarded callbacks erase active recordings and abandon their query blocks for profiler lifetime.
- Frame stats:
  - `begin_frame` opens one global active frame ID; nested/overlapping frames assert in debug and are ignored in release.
  - Zones record the active frame ID at begin time.
  - CPU stats update when CPU zones close.
  - GPU stats update asynchronously when matching `event_id` GPU spans resolve.
  - UI shows pending/unresolved GPU sample counts for recent frames.
- File organization:
  - Keep public types in `src/sgl/device/profiler.h`.
  - Keep core runtime, worker, GPU recording, and stats in `src/sgl/device/profiler.cpp`.
  - Move Chrome JSON writing and trace helpers to `src/sgl/device/profiler_trace.cpp`.
  - Put ImGui rendering in `src/sgl/ui/profiler.h` and `src/sgl/ui/profiler.cpp`.
  - Update `src/sgl/CMakeLists.txt`, `src/slangpy_ext/CMakeLists.txt`, `src/slangpy_ext/device/profiler.cpp`, and `src/slangpy_ext/ui/ui.cpp`.

## Tests And Acceptance

- C++ tests in `tests/sgl/device/test_profiler.cpp`:
  - Current-profiler stack behavior remains unchanged.
  - Trace is empty by default, captures only after `start_trace()`, and stops after `stop_trace()`.
  - CPU nested zones produce flat spans with correct parent IDs.
  - Frame stats report CPU min/max/avg/stddev over a rolling window.
  - GPU zones resolve after `device.wait_for_submit()` plus `profiler.tick()`.
  - Interleaved command encoders keep separate GPU stacks.
  - Query block rollover records all zones.
  - Discarded/unsubmitted GPU work does not crash and does not block later submitted zones.
- Python tests in `slangpy/tests/slangpy_tests/test_profiler.py`:
  - `profiler.zone()` / `profiler.frame()` work as context managers.
  - `ProfilerZoneScope` / `ProfilerFrameScope` compatibility names work.
  - `ProfilerTrace` exposes timelines, sources, names, frames, and spans.
  - SlangPy auto GPU zones appear in trace only when trace capture is started.
  - `stats_snapshot()` exposes CPU/GPU sample counts and timing fields.
- Update the path tracer example to call `profiler.tick()` once per frame and `slangpy.ui.render_profiler_window(profiler)` between UI begin/end.
- Validation commands:
  - `cmake --build --preset windows-msvc-debug`
  - `pytest slangpy/tests/slangpy_tests/test_profiler.py -v`
  - `python tools/ci.py unit-test-cpp`
  - `pre-commit run --all-files`; rerun if it modifies files.

## Assumptions

- Sampling profiling is out of scope.
- Only graphics queue GPU profiling is required for v1.
- Timestamp query and timestamp calibration features are required for GPU timings; unsupported devices record CPU zones only.
- A command encoder is not used concurrently from multiple threads.
- GPU query result calls remain main-thread/caller-thread work through `tick()`.
- Discarded GPU query ranges may be abandoned instead of recycled.
- No new external dependencies are introduced.
