# Append-Only GPU Profiler Query Pages

This ExecPlan is a living document. The sections Progress, Surprises and Discoveries, Decision Log, and Outcomes and Retrospective must be kept up to date as work proceeds.

This plan follows `.agents/PLANS.md` from the repository root.

## Purpose / Big Picture

The GPU profiler currently tracks timestamp query ownership by command recording ID and listens for command recording submit/discard callbacks. This keeps discarded command recordings correct, but it adds a lot of state and work around every GPU profiler zone. After this change, GPU timestamp queries are allocated append-only from per-device query pages. Discarded command recordings simply leave invalid query holes that the profiler drops later. Users should still see CPU and submitted GPU zones in profiler traces, while the profiler hot path does less bookkeeping.

## Progress

- [x] (2026-06-06) Read `.agents/PLANS.md`, current `QueryPool`, current GPU profiler implementation, and existing C++ profiler tests.
- [x] (2026-06-06) Add non-throwing query result status support.
- [x] (2026-06-06) Replace recording-ID GPU tracking with append-only per-context pages.
- [x] (2026-06-06) Update/add C++ tests for query status, page crossing, discard holes, and device close behavior.
- [x] (2026-06-06) Build, run C++ and Python profiler tests, and run pre-commit.

## Surprises and Discoveries

- Observation: Existing GPU profiler C++ tests already cover submitted zones, multiple command buffers, discarded recordings, query-block crossing, and device-close cleanup.
  Evidence: `tests/sgl/device/test_profiler.cpp` contains GPU tests beginning at `TEST_CASE_GPU("profiler records gpu zones in trace json")`.
- Observation: The RHI already distinguishes invalid query ranges from pending query ranges.
  Evidence: `external/slang-rhi/include/slang-rhi.h` documents `IQueryPool::isResultReady()` returning `SLANG_FAIL` for no valid submitted result and false readiness for pending submitted work.
- Observation: The resolver's exponential backoff means tests should wait for eventual GPU trace resolution instead of assuming the first tick after `wait_for_submit()` polls again.
  Evidence: The initial C++ test run resolved CPU zones but missed the GPU zone when the ready poll happened before the retry timestamp.

## Decision Log

- Decision: Keep `QueryPool::is_result_ready()` throwing on invalid results and add a new `QueryPool::result_status()` API for the profiler.
  Rationale: This preserves current public behavior while giving the profiler a non-throwing invalid/pending/ready distinction.
  Date/Author: 2026-06-06 / Codex.
- Decision: Do not reuse timestamp query slots during a profiler lifetime.
  Rationale: Reusing slots safely would require knowing whether old finished-but-unsubmitted command buffers can still submit, which would reintroduce the lifetime tracking this refactor removes.
  Date/Author: 2026-06-06 / Codex.

## Outcomes and Retrospective

Implemented append-only GPU timestamp query pages per device/queue profiler context. The profiler no longer tracks command recording IDs, submit callbacks, discard callbacks, active recording maps, or separate GPU-zone events. Begin-zone events now carry the allocated query pair directly, and end-zone writes only the matching end timestamp. Discarded recordings become invalid query holes that the resolver retries with backoff and drops after the retention window.

Validation passed on 2026-06-06:

    cmake --build --preset windows-msvc-debug
    python tools/ci.py unit-test-cpp
    pytest slangpy/tests/slangpy_tests/test_profiler.py -v
    pre-commit run --all-files

## Context and Orientation

`src/sgl/device/query.h` and `src/sgl/device/query.cpp` wrap RHI query pools. Existing `is_result_ready()` throws through `SLANG_RHI_CALL` when a query range has no submitted result. The profiler needs a non-throwing result status so discarded command recordings can be ignored later.

`src/sgl/utils/profiler.cpp` owns CPU trace data, rolling stats, and GPU timestamp resolution. The current GPU path allocates query blocks per command recording, records GPU metadata keyed by `CommandRecordingID`, and receives submit/discard events from `Device`. The new design removes recording IDs from profiler query ownership. Each device/queue pair has append-only query pool pages. Each thread gets a small cached chunk from a page for fast query-pair allocation.

## Plan of Work

First add `enum class QueryResultStatus { invalid, pending, ready }` to `src/sgl/device/query.h` and implement `QueryPool::result_status()` in `src/sgl/device/query.cpp`. It calls `m_rhi_query_pool->isResultReady()`, returns `ready` if the call succeeds and reports ready, returns `pending` if the call succeeds and reports not ready, returns `invalid` for `SLANG_FAIL`, and otherwise throws through the normal RHI error path.

Next rewrite the profiler GPU internals. Remove the separate `gpu_zone` profiler event and put a `ProfilerGpuEvent` directly on the begin-zone event. Replace per-recording query block types with append-only page/chunk types. `GpuContext` keeps pages, current page allocation offset, a disabled flag, and the GPU timeline id. Thread-local cache stores the current profiler, context, query pool, next query index, and end query index. The fast path checks that cache and bumps by two indices. The slow path locks `gpu_mutex`, creates a new page when the current page is full, hands out a chunk of `gpu_query_block_size` queries, and disables GPU profiling after 16 pages.

CPU event processing builds GPU zone records as zones close. Child relationships are stored as indices into a stable vector of all pending GPU records. When a GPU zone closes with no parent GPU zone, enqueue its index as a root pending GPU zone. The resolver only processes root indices; it recursively resolves children once the root query pair is ready.

Finally update tests. Keep the existing user-visible behavior tests, change the query-block crossing test to use small query pool pages so it exercises append-only page creation, and add a fresh-query status test. The discarded-recording test should prove discarded GPU zones do not appear and do not leave pending stats.

## Concrete Steps

Run all commands from `C:\projects\slangpy`.

Build before testing:

    cmake --build --preset windows-msvc-debug

Run C++ tests:

    python tools/ci.py unit-test-cpp

Run Python profiler tests:

    pytest slangpy/tests/slangpy_tests/test_profiler.py -v

Run formatting/checks:

    pre-commit run --all-files

If pre-commit changes files, rerun the build and relevant tests.

## Validation and Acceptance

The implementation is accepted when submitted GPU profiler zones still appear in trace JSON, discarded command recordings do not emit GPU trace zones, no profiler stats node remains permanently pending after discarded work, small query pages resolve zones across page boundaries, and a fresh timestamp query reports `invalid` before submission.

The expected test outcome is that `python tools/ci.py unit-test-cpp` and `pytest slangpy/tests/slangpy_tests/test_profiler.py -v` pass. `pre-commit run --all-files` must also pass after any automatic edits are incorporated.

## Idempotence and Recovery

The refactor is source-only and can be retried by rebuilding and rerunning the same tests. Query allocation is append-only at runtime; profiler destruction releases pages normally through `ref<QueryPool>`. If a GPU backend does not support timestamp query or timestamp calibration, existing tests skip those GPU cases.

## Artifacts and Notes

No external artifacts are required.

## Interfaces and Dependencies

The new C++ support API is:

    enum class QueryResultStatus : uint32_t {
        invalid,
        pending,
        ready,
    };

    QueryResultStatus QueryPool::result_status(uint32_t index, uint32_t count);

No Python binding is required for `QueryResultStatus`.
