# Profiler Event Recording V1

This ExecPlan is a living document. The sections Progress, Surprises and Discoveries, Decision Log, and Outcomes and Retrospective must be kept up to date as work proceeds.

This plan follows `.agents/PLANS.md` from the repository root.

## Purpose / Big Picture

SlangPy currently has profiler scopes and Python bindings, but the scopes do not produce a queryable recording. After this change, users can record CPU and GPU profiler zones, inspect a stable trace snapshot for debugging, and poll lightweight per-frame statistics in real time from an ImGui window or terminal reporter without blocking the application frame loop. GPU statistics are allowed to appear a few frames late because timestamp queries complete asynchronously.

The visible outcome is that code using `SGL_PROFILER_ZONE`, `Profiler::begin_zone()`, or the Python `Profiler` bindings can produce completed CPU spans, resolved GPU spans where supported, dropped/unresolved counters when profiling cannot collect GPU data, and live per-zone frame statistics that can be refreshed every frame without waiting for GPU completion.

## Progress

- [x] (2026-06-01) Drafted the initial profiler recording architecture.
- [x] (2026-06-01) Added realtime per-frame statistics as a first-class design goal.
- [ ] Convert the current profiler shell into CPU event recording with snapshots.
- [ ] Add worker-owned timeline assembly and live statistics publishing.
- [ ] Add GPU timestamp query rings and delayed GPU span resolution.
- [ ] Expose snapshot and live statistics APIs to Python.
- [ ] Add C++ and Python tests, then build, test, and run pre-commit.

## Surprises and Discoveries

- Observation: Realtime UI/terminal reporting should not call a full trace snapshot every frame.
  Evidence: Full snapshots copy raw timeline vectors and can grow with retained trace history, while an ImGui window needs a small, bounded per-frame summary.

- Observation: GPU timing data may naturally lag CPU frame closure.
  Evidence: GPU timestamp queries are only useful after command submission finishes and query results are ready, so live stats need an unresolved or delayed state rather than blocking the caller.

## Decision Log

- Decision: Keep Tracy as a reference only and do not add it as a dependency.
  Rationale: The project needs a small native profiler integrated with existing `Profiler`, `CommandEncoder`, `CommandBuffer`, `Device`, and Python binding APIs.
  Date/Author: 2026-06-01 / Codex

- Decision: Treat realtime per-frame stats as a separate worker-produced artifact named `LiveStatsStore`, not as a view rebuilt from `ProfilerSnapshot`.
  Rationale: Polling stats every frame must be cheap, bounded, and non-blocking, while full trace snapshots are for inspection and can copy larger history.
  Date/Author: 2026-06-01 / Codex

- Decision: Publish live stats through double buffering or an atomically swapped immutable snapshot.
  Rationale: Readers such as ImGui and terminal reporting should copy the latest completed stats without taking worker locks or waiting for GPU query resolution.
  Date/Author: 2026-06-01 / Codex

## Outcomes and Retrospective

Not started. Fill this in after each major milestone and at completion with what works, what remains, and any changes from this plan.

## Context and Orientation

The profiler public API currently lives in `src/sgl/device/profiler.h` and `src/sgl/device/profiler.cpp`. The current implementation tracks enabled state, auto-zone state, debug-group state, source-location interning, and thread-local current profiler scopes. `Profiler::begin_zone()` only checks whether profiling is enabled, and `Profiler::end_zone()` does not record data yet.

Python bindings live in `src/slangpy_ext/device/profiler.cpp`. Existing Python tests are in `slangpy/tests/slangpy_tests/test_profiler.py`, and native C++ tests are in `tests/sgl/device/test_profiler.cpp`.

GPU command recording passes through `src/sgl/device/command.h` and `src/sgl/device/command.cpp`. `CommandEncoder::finish()` creates a `CommandBuffer`. `Device::submit_command_buffers()` in `src/sgl/device/device.cpp` submits one or more command buffers and returns the global fence value used as the submit id. Timestamp query support is exposed by `QueryPool` in `src/sgl/device/query.h` and `src/sgl/device/query.cpp`, plus `Device::get_timestamp_calibration()`.

A profiler "zone" is a named scope with a begin and end. A "span" is the completed interval derived from a zone. A "timeline" is a lane of spans, such as one CPU thread or one GPU queue. A "frame" is an application-defined interval from `begin_frame()` to `end_frame()`. "Live stats" are compact per-zone aggregates for recent complete frames, intended for realtime display.

## Interfaces and Dependencies

Do not add Tracy or any new third-party dependency.

Extend `ProfilerDesc` in `src/sgl/device/profiler.h` with bounded settings:

- `event_block_size_bytes`, default 64 KiB.
- `trace_history_frame_count`, default large enough for debugging but bounded.
- `live_stats_frame_count`, default 120 frames.
- `gpu_timestamp_query_count`, default sized for typical nested frame profiling.
- `gpu_frame_settle_latency`, default a small number of frames such as 3.

Add read-only public structs in `src/sgl/device/profiler.h` and bind them to Python:

- `ProfilerTraceEvent`: span id, logical event id, parent ids, timeline id, source name/path, frame id, start time, end time, duration, depth, flags, resolved state.
- `ProfilerTimeline`: keep the existing struct and add fields only if needed for stable display.
- `ProfilerStats`: total event counts, dropped CPU/GPU counts, unresolved GPU counts, worker backlog, block rollover counts.
- `ProfilerSnapshot`: copied vectors of timelines, trace events, frame records, and stats for offline inspection.
- `ProfilerLiveZoneStats`: path/name, depth, CPU last/avg/min/max/count for the published window, GPU last/avg/min/max/count, unresolved GPU sample count, dropped sample count.
- `ProfilerLiveFrameStats`: published frame id, newest CPU frame id, newest GPU-resolved frame id, latency in frames, frame CPU duration, frame GPU duration if resolved, per-zone stats, and global counters.

Add public methods:

- `void Profiler::flush()`: publish the current thread partial block and wait only for the worker to process currently published CPU/event blocks. It must not wait for GPU completion.
- `ProfilerSnapshot Profiler::snapshot() const`: return a stable copied trace snapshot for inspection.
- `ProfilerStats Profiler::stats() const`: return counters.
- `ProfilerLiveFrameStats Profiler::live_stats() const`: return the latest worker-published live stats without blocking on the worker or GPU.
- Optional convenience aliases can be added later, such as `latest_frame_stats()`, if tests show the API reads better.

All Python function arguments added as part of this work must have type annotations where Python code is touched.

## Plan of Work

First implement CPU-only recording. Add an internal shared `ProfilerState` owned by `Profiler`. Each profiler and producer thread gets a `ThreadRecorder` with fixed-size `EventBlock`s. Zone begin and end write compact records only: record kind, CPU tick, source pointer, logical event id, flags, and optional GPU query references. The hot path must not allocate after a recorder has its current block, and overflow uses drop-and-count rather than blocking.

Generate logical event ids without a global atomic per zone. Use a collision-free representation, either a tuple of `producer_id` and `local_counter` or a bit-packed integer with documented bit widths and overflow behavior. The worker derives nesting, parent ids, depth, paths, frame membership, and durations.

Add a worker thread that consumes published blocks and owns `TimelineStore`. `TimelineStore` stores CPU timelines per thread, GPU timelines per profiler-local device/queue pair, append-only span storage, frame ranges, and source-location aggregates. Keep retained trace history bounded by `trace_history_frame_count`.

Add `LiveStatsStore` owned by the same worker. This store aggregates recent complete or settled frames into a small display-oriented data model. The worker publishes live stats by swapping an immutable snapshot pointer or flipping a double buffer. `Profiler::live_stats()` only copies the most recently published buffer and must not wait for the worker, GPU fences, or query readiness.

Define live frame publication rules explicitly. CPU stats for frame N become publishable once `end_frame()` for frame N has been processed. GPU stats for frame N are incorporated when submitted GPU spans for that frame have finished and query results are ready. If GPU data is still unresolved after `gpu_frame_settle_latency` newer frames, publish frame N with CPU stats, any resolved GPU stats, and unresolved GPU counters. Later publications may update rolling GPU aggregates when late GPU spans resolve, but the API must expose the frame ids and latency so UI code can label data as delayed.

Add GPU timestamp recording after CPU recording works. Maintain one hidden timestamp `QueryPool` ring per `Device` inside the profiler because `QueryPool` is device-owned. Allocate query pairs non-blockingly. If the ring lock or free pair is unavailable, record the CPU span, skip GPU timestamps, and increment `dropped_gpu_zone_count` and `dropped_gpu_query_count`.

Store completed GPU query pairs on `CommandEncoder`, move them to `CommandBuffer` in `finish()`, then attach the submit id returned from `Device::submit_command_buffers()` after a successful submit. The plan must handle batched submits by attaching the same submit id to all command buffers in the batch. If submit throws, if a command buffer is destroyed without submit, or if the profiler is destroyed with pending query pairs, recycle or mark the pairs dropped without blocking.

The worker polls pending submitted GPU spans. Once `Device::is_submit_finished(submit_id)` is true and both query results are ready, it reads timestamps, emits GPU timeline spans, updates live stats, and recycles the query pair. Query slots must be reset before reuse according to the current `QueryPool` API.

Add an internal profiler clock matching RHI timestamp calibration domains where possible. If calibration is unavailable or the CPU clock domain cannot be matched, keep GPU spans monotonic within their GPU timeline and mark CPU/GPU alignment as unresolved. Gate GPU timestamp recording on device support for timestamp queries and record a clear unsupported counter when unavailable.

Expand `ProfilerZoneFlags::auto_` as CPU always, GPU when an encoder is provided, and debug group only when enabled. Explicit `gpu` without an encoder records CPU if requested and increments a dropped-GPU counter. Keep debug group behavior callable even when timestamp recording is dropped.

## Concrete Steps

Work from the repository root:

    cd C:\src\slangpy

Add or update native profiler definitions in:

    src/sgl/device/profiler.h
    src/sgl/device/profiler.cpp

Wire GPU query ownership through:

    src/sgl/device/command.h
    src/sgl/device/command.cpp
    src/sgl/device/device.cpp

Update Python bindings and docs generated by the normal binding workflow:

    src/slangpy_ext/device/profiler.cpp

Update tests:

    tests/sgl/device/test_profiler.cpp
    slangpy/tests/slangpy_tests/test_profiler.py

Build before tests. CMake builds and tests must run outside the sandbox according to the repository instructions.

    cmake --build --preset windows-msvc-debug
    python tools/ci.py unit-test-cpp
    pytest slangpy/tests/slangpy_tests/test_profiler.py -v
    pre-commit run --all-files

If `pre-commit` modifies files, inspect the changes, then rerun `pre-commit run --all-files`.

## Validation and Acceptance

CPU recording acceptance: a test with nested zones returns a snapshot containing parent/child spans with stable names, non-negative durations, correct depth, and frame membership. A disabled profiler records no spans and reports no unexpected drops.

Block rollover acceptance: a tiny `event_block_size_bytes` forces rollover, preserves completed events, and increments rollover counters. If blocks are exhausted, the hot path drops and counts events rather than blocking.

Threading acceptance: zones recorded from multiple threads appear on separate CPU timelines. `flush()` publishes the calling thread partial block and returns after worker processing of currently published blocks without waiting for GPU completion.

Live stats acceptance: after several frames with repeated zones, `live_stats()` returns a bounded per-zone aggregate with the latest published frame id and rolling CPU stats. Calling `live_stats()` repeatedly from a frame loop must not block on GPU completion or worker backlog. Tests should verify that it returns the previous published value if the worker has not published a newer one yet.

GPU acceptance: on devices with timestamp query support, a GPU zone recorded with an encoder appears as a resolved GPU span after submit, wait, and flush. Before the GPU result is ready, live stats report CPU data plus unresolved GPU counts instead of blocking. With a tiny query ring, GPU spans are dropped and counted while CPU spans are preserved.

Unsupported backend acceptance: if timestamp queries or timestamp calibration are unavailable, CPU recording and live CPU stats still work, GPU timing is marked unsupported or unresolved, and no GPU query API call crashes.

Python acceptance: Python can create a `Profiler`, record manual begin/end zones, call `flush()`, `snapshot()`, `stats()`, and `live_stats()`, and inspect read-only public structs. If Python context-manager zone scopes are added, tests cover them.

## Idempotence and Recovery

All implementation steps should be additive until tests pass. Re-running tests and pre-commit should be safe. Query pairs must be recycled or dropped on command buffer destruction, failed submit, profiler shutdown, and device close. Profiler destruction should stop the worker thread without waiting for arbitrary GPU completion.

If GPU recording destabilizes a backend, keep CPU recording and live CPU stats enabled, gate GPU timestamp use behind feature checks, and record unsupported/dropped counters for that backend.

## Artifacts and Notes

Reference only:

- Tracy repository: https://github.com/wolfpld/tracy
- Tracy client source: https://raw.githubusercontent.com/wolfpld/tracy/master/public/client/TracyProfiler.cpp

Do not require these references to implement the plan. The local design above is the source of truth.
