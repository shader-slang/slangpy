# Profiler Simplification Plan

This plan follows `.agents/PLANS.md` from the repository root. It is a living document; Progress,
Surprises and Discoveries, Decision Log, and Outcomes and Retrospective are kept current as the
implementation proceeds.

## Progress

- [x] (2026-06-22) Made profiler activation explicit and thread-local. `Profiler()` no longer becomes
  current on construction; `ProfilerScope`, `with profiler:`, and `with profiler.frame(...)` activate
  only the current thread.
- [x] (2026-06-22) Added Python `Profiler.frame(name: str = "frame")` as the public frame context and
  kept raw Python frame tokens unbound.
- [x] (2026-06-22) Replaced frame-window stats storage with per-node fixed sample rings. Stats are
  recorded for zones even when no frame is active.
- [x] (2026-06-22) Added worker-published cached `ProfilerStats` snapshots. `stats_snapshot()` is now a
  cached read and does not tick, flush, or rebuild live state.
- [x] (2026-06-22) Simplified trace windows so CPU/GPU zones and frames are appended only when both
  begin and end timestamps are inside the trace window.
- [x] (2026-06-22) Replaced GPU query pages/block allocation with one fixed query pool per device/queue
  context, per-use tickets, a context free list, and a small TLS slot cache.
- [x] (2026-06-22) Moved command recording submitted/discarded handling to lightweight callback events
  drained by `Profiler.tick()`.
- [x] (2026-06-22) Updated C++ and Python profiler tests for explicit activation, cached stats,
  trace-window behavior, and bounded GPU query exhaustion.
- [x] (2026-06-22) Verified with CMake build, targeted Python profiler tests, C++ unit tests, and Python
  typing check.

## Surprises and Discoveries

- Observation: `stats_snapshot()` previously called `tick()` and rebuilt stats while holding the live
  profiler data lock.
  Evidence: the old implementation called `Profiler::stats_snapshot()` -> `tick()` ->
  `ProfilerImpl::stats_snapshot()`.
- Observation: Close-time trace emission means tests must not expect zones to appear while they are
  still open.
  Evidence: the C++ dynamic-name trace test initially failed until the scoped zones were closed before
  `trace_snapshot()`.
- Observation: `python tools/ci.py unit-test-cpp` once raised after doctest printed success, but rerunning
  the test binary and then the wrapper both exited successfully.
  Evidence: direct `build\windows-msvc\Debug\sgl_tests.exe` and the subsequent wrapper run reported
  195 passed, 0 failed.

## Decision Log

- Decision: Remove public `frame_stats_enabled` and `gpu_query_block_size` rather than keeping them as
  inert compatibility knobs.
  Rationale: the profiler API is new, and the plan explicitly prefers deleting surface area and avoiding
  implementation branches.
  Date/Author: 2026-06-22 / Codex.
- Decision: Make `stats_window_size` a sample-ring capacity per stats node.
  Rationale: this satisfies sample-window semantics directly and avoids unbounded sample vectors plus
  snapshot-time history filtering.
  Date/Author: 2026-06-22 / Codex.
- Decision: Keep GPU query lifecycle reconciliation as a bounded scan of each fixed context pool.
  Rationale: it is simple, bounded by `gpu_query_pool_size / 2`, and the plan says to avoid a recording
  index until measurement proves it necessary.
  Date/Author: 2026-06-22 / Codex.
- Decision: Treat GPU query exhaustion as a per-zone GPU timing skip while preserving CPU zones.
  Rationale: this keeps the fast path non-blocking and bounded while maintaining reliable CPU profiling.
  Date/Author: 2026-06-22 / Codex.

## Outcomes and Retrospective

Implemented the simplification described by this plan. The profiler now has fewer public settings,
explicit thread-local activation, cached stats snapshots, simpler trace-window rules, and bounded
best-effort GPU timestamp storage. The main remaining follow-up is performance measurement of the
bounded lifecycle scan under unusually large query pools; no per-recording index was added without
evidence that it is needed.

## Goal
Make the profiler implementation smaller and easier to reason about, while preserving the two hard requirements:

- Fast-path zone recording stays cheap.
- A worker thread performs the heavy trace/stats/hierarchy work.

The profiler API is new and not fixed. Prefer deleting surface area and choosing simpler semantics over preserving early API behavior.

Success criteria:

- Fewer profiler-specific states, maps, and lifetime special cases than the current implementation.
- No new hot-path global locks or per-zone heap allocations.
- `stats_snapshot()` and profiler UI are cheap cached reads.
- GPU profiling is bounded and best-effort; CPU profiling remains reliable.

## Simplified Architecture

- Producer threads only do fast-path work: check cheap atomics, get TLS thread data, write CPU timestamps, optionally write GPU timestamp/debug commands, and enqueue compact events.
- `Profiler.tick()` does only main-thread GPU maintenance: drain closed GPU slots, drain command recording lifecycle events, poll submitted query results, enqueue GPU result/drop events, reset queries, and recycle slots.
- The worker thread owns all trace construction, GPU hierarchy completion, rolling stats aggregation, and published snapshots.
- `trace_snapshot()` remains the explicit synchronous export path and may flush. `stats_snapshot()` is intentionally not a flush point.

## Public API Simplifications

- Make profiler activation explicit and thread-local. Constructing `Profiler()` should not automatically push it as current, and the current-profiler stack should not be application-wide.
- Keep a simple activation context for Python and C++:
  - Python: `with profiler:`
  - Python frame helper: `with profiler.frame(name: str = "frame"):`
  - C++: `ProfilerScope` and `SGL_PROFILE_FRAME()`
- `with profiler.frame(...)` activates the profiler for the current thread while the frame is open. Same-thread nested frames on the same profiler fail clearly.
- Do not expose raw Python `begin_frame()`/`end_frame()` tokens.
- Keep `ProfilerDesc` object-only. Do not add dict construction.
- Remove or de-emphasize settings that create implementation branches unless they are still needed:
  - Prefer no `frame_stats_enabled`; stats are always maintained when profiling is enabled.
  - Prefer one GPU query pool size setting; avoid public `gpu_query_block_size` unless implementation proves it is needed.

## Stats Semantics

- Rolling stats are sample-window based, not frame-window based. `stats_window_size` means the last N samples for each stats node.
- Frames are trace/UI markers and convenient scopes, but stats do not require an active frame.
- Worker stats nodes are keyed by hierarchical zone path plus source/name identity.
- Each stats node stores fixed-capacity CPU/GPU sample rings. No unbounded `StatsSample` vectors and no snapshot-time filtering over all history.
- The worker publishes the latest immutable `ProfilerStats` snapshot at stable points, such as after processing a batch of events or after completing GPU results.
- `stats_snapshot()` returns the latest published snapshot under a small lock/ref swap. It must not call `tick()`, flush queues, or rebuild summaries.

## Trace Semantics

- Do not add trace-generation fields to every event.
- `start_trace(clear=True)` clears trace data and starts accepting newly completed trace records.
- `stop_trace()` records a stop timestamp. The worker keeps zones/frames whose begin and end timestamps are both inside the active trace window. Zones still open at stop are dropped from trace output when they eventually close.
- `clear_trace()` drops current trace records and invalidates trace IDs for currently open zones/frames. Those scopes may still close for stack correctness and stats, but they do not reappear in the cleared trace.
- This intentionally avoids synthetic close events, timestamp clamping, and per-event generation bookkeeping.

## GPU Query Simplifications

- Replace pages plus ad-hoc growth with one fixed-size query slot pool per device/queue context. Each slot owns one timestamp query pair.
- Maintain a context free list and a tiny TLS cache of free slot pointers. Refill TLS from the free list on the slow path.
- Give every slot allocation a fresh ticket. Recycled slot storage must never let a stale GPU result apply to a newer zone.
- If no free slot is available, skip GPU timing for that zone and keep the CPU zone. Do not block, allocate unbounded pages, or disable the whole context.
- Command recording callbacks should enqueue lightweight lifecycle events only. `tick()` drains those events and applies them to the bounded in-flight slot set.
- Avoid a per-recording slot map unless profiling shows the bounded scan in `tick()` is too expensive. A fixed pool makes a simple scan acceptable and much easier to reason about.
- Reset and recycle a slot only after its GPU result/drop event has been enqueued with the matching ticket.
- Keep reset/polling simple first. Do not add custom range coalescing, batching layers, or calibration throttling unless the straightforward bounded implementation is too slow.

## Lifetime Simplifications

- Removing application-wide automatic activation should eliminate most current-profiler lifetime hazards.
- TLS caches must validate a non-reused profiler/context generation before use and clear themselves when the owner changes.
- Do not dereference cached `ThreadData`, GPU context, or profiler pointers until the owner/generation check passes.

## Implementation Steps

1. Shrink the public API and tests first:
   - Make current-profiler activation explicit and thread-local.
   - Add `Profiler.frame(name: str = "frame")`.
   - Remove tests that require construction-time activation or cross-thread current-profiler propagation.
2. Simplify stats:
   - Replace frame-window stats storage with per-node fixed sample rings.
   - Add worker-published cached `ProfilerStats`.
   - Make `stats_snapshot()` a pure cached read.
3. Simplify trace windows:
   - Remove per-event generation plans.
   - Implement the simple active-window inclusion/drop rules above.
4. Simplify GPU query storage:
   - Replace query pages/page caps with fixed context pools and free lists.
   - Add per-use tickets and best-effort GPU timing drop on exhaustion.
5. Simplify command lifecycle handling:
   - Use callback event queues plus bounded `tick()` reconciliation.
   - Only introduce a recording-id index if a measured profile shows `tick()` cost is unacceptable.
6. Update bindings, stubs, docs, and UI expectations:
   - `src/slangpy_ext/utils/profiler.cpp`
   - `slangpy/__init__.pyi`
   - `slangpy/ui/__init__.pyi`
   - generated docs/stubs as required by the repo workflow

## Test Plan

- C++ tests:
  - `Profiler()` construction does not implicitly change the current profiler.
  - `ProfilerScope` activates only on the current thread and restores the previous profiler.
  - Destroying an inactive profiler leaves no stale current-profiler entry.
  - `stats_snapshot()` does not drain events, call `tick()`, or rebuild live worker state.
  - Worker-published stats eventually include CPU zones via a bounded wait helper.
  - `stop_trace()` preserves zones that completed before the stop timestamp.
  - Zones still open at `stop_trace()` are dropped from trace output after they close.
  - `clear_trace()` while zones are open does not corrupt later trace or stats.
  - Tiny GPU query pools force slot reuse after resolved and discarded work.
  - Stale GPU result events do not apply to a recycled slot with a newer ticket.
  - GPU query exhaustion skips GPU timing without blocking and still records CPU zones.
  - Same-command-encoder nested GPU zones preserve parent/child hierarchy when GPU timing is available.

- Python tests:
  - `with profiler:` activates only within the current thread/block.
  - `with profiler.frame("name"):` records a frame and activates the profiler for auto-zones in the block.
  - Same-thread nested Python frames fail clearly.
  - Functional auto-zones produce CPU trace/stats data without timestamp-query support.
  - Functional auto-zones still work when GPU timing is skipped due to query exhaustion.
  - `spy.ui.render_profiler_window` uses cached stats and does not force trace export.

- Verification commands, run outside the sandbox:
  - Build first with the active platform preset.
  - Run `pytest slangpy/tests/slangpy_tests/test_profiler.py -v`.
  - Run `python tools/ci.py unit-test-cpp` or the targeted C++ profiler test if available.
  - Run `python tools/ci.py typing-check-python`.
  - Run `pre-commit run --all-files` after implementation.

## Out Of Scope

- Pathtracer CUDA portability. Handle it as a separate task if needed.
- Trace clipping/synthetic close events.
- Cross-thread automatic profiler activation.
- Custom query reset batching or per-recording slot indexes without a measured need.
