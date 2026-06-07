# Split GPU Query Resolving From GPU Zone Resolving

This ExecPlan is a living document. The sections Progress, Surprises and Discoveries, Decision Log, and Outcomes and Retrospective must be kept up to date as work proceeds.

This plan follows `.agents/PLANS.md` from the repository root.

## Purpose / Big Picture

The append-only query page profiler still resolves GPU query readiness and GPU zone hierarchy together in `Profiler::tick()`. This keeps the main thread involved in trace/stats work and requires root-zone polling state. The goal is to make query slots the only bridge between the main-thread RHI query pump and the worker-side zone resolver. `tick()` should only resolve timestamp query slots and enqueue compact query-result events. The worker thread should own GPU zone hierarchy, trace emission, and stats accounting.

## Progress

- [x] (2026-06-06) Inspected current profiler structs, command encoder IDs, query page allocation, and existing GPU profiler tests.
- [x] (2026-06-06) Introduce GPU query tickets/slots and publish closed slots from `end_zone()`.
- [x] (2026-06-06) Move query status/result polling into a zone-unaware query pump.
- [x] (2026-06-06) Move GPU hierarchy and stats/trace resolution to worker-side event processing.
- [x] (2026-06-06) Remove now-unused root polling, retry/backoff, and profiler device-close callback state.
- [x] (2026-06-06) Update tests for tick-before-submit drop and per-command-stream GPU hierarchy.
- [x] (2026-06-06) Build, run C++ and Python profiler tests, and run pre-commit.

## Surprises and Discoveries

- Observation: Public command recording IDs and submit/discard callbacks are still exposed and tested in Python.
  Evidence: `slangpy/tests/slangpy_tests/test_command_buffer.py` verifies `recording_id` and command recording callbacks.
- Observation: The profiler no longer needs command recording lifecycle callbacks.
  Evidence: `src/sgl/utils/profiler.cpp` currently uses command recording IDs only implicitly through command encoders and does not register submit/discard callbacks after the append-only page refactor.

## Decision Log

- Decision: Keep the public command recording API for this pass.
  Rationale: The user explicitly chose to keep it. The profiler may use `recording_id` as an opaque command stream id, but must not depend on submit/discard callbacks.
  Date/Author: 2026-06-06 / Codex.
- Decision: Drop invalid query slots immediately when `tick()` sees them.
  Rationale: This matches the new contract that profiled command encoders must be submitted before the next profiler tick if their GPU zones should be retained.
  Date/Author: 2026-06-06 / Codex.

## Outcomes and Retrospective

Implemented the split resolver model. `Profiler::tick()` now pumps closed GPU query slots and enqueues `gpu_query_result` events. Worker-side per-thread event processing owns GPU stacks, pending zone records, orphan query results, trace emission, and stats accounting. Invalid query slots are dropped immediately.

Validation passed on 2026-06-06:

    cmake --build --preset windows-msvc-debug
    python tools/ci.py unit-test-cpp
    pytest slangpy/tests/slangpy_tests/test_profiler.py -v
    pre-commit run --all-files

## Context and Orientation

`src/sgl/utils/profiler.cpp` currently has append-only GPU query pages and TLS chunk allocation, but pending GPU zones are still globally stored and root-polled by `process_pending_gpu_zones()`. Query status polling, timestamp reads, timestamp calibration, hierarchy completion, trace emission, and stats updates are mixed in that path.

The new implementation should keep append-only page allocation but add stable query slots. `begin_zone()` writes the begin timestamp and queues CPU/GPU metadata. `end_zone()` writes the end timestamp and publishes the slot to a query pump. The query pump enqueues `gpu_query_result` events to the owner thread queue. Worker event processing matches those results to GPU zone records and completes trace/stats work.

## Validation and Acceptance

The implementation is accepted when submitted GPU zones still appear, discarded encoders and tick-before-submit encoders do not emit GPU zones, pending stats return to zero after ready/drop, small query pages cross append-only pages, and nested zones on different command encoders resolve as separate GPU roots.

Run:

    cmake --build --preset windows-msvc-debug
    python tools/ci.py unit-test-cpp
    pytest slangpy/tests/slangpy_tests/test_profiler.py -v
    pre-commit run --all-files

If pre-commit changes files, rerun the build and relevant tests.
