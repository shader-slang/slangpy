# Publish CPU profiler history independently of GPU completion

This living ExecPlan follows `.agents/PLANS.md`.

## Purpose / Big Picture


The profiler headline and CPU graph should advance when CPU frames end even if GPU work has not been submitted or resolved. The zone table must retain aligned CPU/GPU statistics over completed frames. GPU graph samples use the same recent frame IDs as the CPU graph, with pending measurements shown as gaps. Diagnostics explain how far the completed table lags behind CPU history. Preserve the existing default snapshot API and all Level 1 UI refinements.

## Progress


- [x] (2026-10-30) Inspected frame sealing, GPU correlation, bounded pending retention, immutable snapshot publication, bindings, and UI selection.
- [x] (2026-10-30) Add recent snapshots and explicit pending GPU status without changing completed snapshot behavior.
- [x] (2026-10-30) Use recent snapshots for headline/history and completed snapshots for table; show both frame positions and lag.
- [x] (2026-10-30) Add native/Python regression coverage for unsubmitted work, later resolution, frame ordering, retention, clearing, and immutable arrays.
- [x] (2026-10-30) Build outside the sandbox, run tests outside the sandbox, inspect rendering, regenerate docstrings, and run pre-commit.

## Context and Orientation


`src/sgl/utils/profiler.cpp` owns the background collector. CPU frame seals arrive after prerequisite CPU events, so a sealed frame already has complete CPU data. `PendingFrameData` holds zones indexed by correlation ID; GPU results update those zones using both frame ID and correlation ID. `finalize_ready_frames()` currently moves frames into the completed deque in order, stopping at an unresolved GPU frame. Bounded pending retention eventually marks unresolved zones missing. `make_frame_stats()` builds immutable matrices from the completed deque. `src/sgl/utils/profiler.h` defines those matrices and GPU status values. Python bindings and read-only NumPy ownership are in `src/slangpy_ext/utils/profiler.cpp`.

`src/sgl/utils/profiler_ui.cpp` caches snapshots for pause display and preserves zone selection using hierarchical site IDs. Its table currently supplies the graph's indices; the refactor must map selection by path between different entry sets, never reuse a completed snapshot's row index in recent history. `tests/sgl/device/test_profiler.cpp` and `slangpy/tests/slangpy_tests/test_profiler.py` contain existing delayed-GPU and offscreen UI tests.

## Decision Log


Decision: Add `include_pending=false` to `Profiler::frame_stats_snapshot()`. The default remains the completed-frame view. With true, return the newest bounded window of CPU-ended frames, including pending GPU work. Both products are built in the same collector publication pass and swapped under the existing mutex. When no frames are pending, share the same immutable snapshot. Rationale: preserve callers' current semantics while reusing one statistics representation and avoiding another worker or collection pipeline. Date/Author: 2026-10-30, Codex.

Decision: Append `ProfilerFrameGpuStatus::pending`. Pending samples are distinct from permanently missing/incomplete GPU work, and excluded from GPU duration distributions. Existing enum values remain unchanged. Rationale: allow the graph and Python clients to distinguish temporary gaps from lost measurements. Date/Author: 2026-10-30, Codex.

Decision: Build recent records from retained completed records plus sealed pending records. Rebuilding a pending record for a snapshot must not increment hierarchy-loss diagnostics. Keep existing bounded retirement and late-result behavior. Rationale: no duplicate long-lived history or new mutation of published arrays. Date/Author: 2026-10-30, Codex.

## Plan of Work


First generalize statistics construction to select a bounded recent window while retaining completed defaults. Make frame-record construction read-only except for optional one-time hierarchy-loss accounting at finalization. Track unresolved GPU occurrences separately from missing ones. Publish and retire the additional snapshot using the existing lifetime mechanism that avoids releasing Python-owned objects on the worker thread.

Then update the UI to cache both snapshots. Headline and graph read recent history, table reads completed history. Match selection and colors by hierarchical site identity, including cases where a new zone appears only in recent history. GPU graphs reserve pending frame columns as gaps. Pause freezes both cached views. Diagnostics report latest CPU frame, completed table frame, age, and lag.

Finally exercise delayed command submission with explicit frame IDs and distinct zone names/durations. Verify CPU history advances, completed rows remain stable, late GPU results populate the correct frames, captured arrays remain immutable, clearing rejects old results, and retention stays bounded. Update `docs/src/profiling.rst` and generated Python docstrings.

## Concrete Steps


Run from `C:\src\slangpy`. All CMake builds and tests run outside the sandbox.

    cmake --build --preset windows-msvc-debug --target sgl_tests slangpy_ext pathtracer slangpy_pydoc
    .\build\windows-msvc\Debug\sgl_tests.exe -ts=profiler
    python -m pytest slangpy/tests/slangpy_tests/test_profiler.py -q --tb=short
    pre-commit run --all-files

Run focused offscreen UI probes to compare CPU and GPU graph modes while work is pending, then after submission. Save previews and logs under ignored `.temp/`. Re-run formatting hooks when they modify files.

## Validation and Acceptance


With an ended CPU frame containing unsubmitted GPU commands, the default snapshot contains no new completed row but the recent snapshot contains its CPU durations, cadence, and a pending GPU status. A later CPU-only frame must also appear in recent history despite the older pending frame. Resolving GPU work updates the original frame, not the newest one; previously returned snapshots and NumPy arrays remain unchanged. The table stays paired over completed frames. New zone paths and pause/resume must work across different snapshot entry sets. Exceeding the retention window and clearing history must not leak old frames or double-count diagnostics.

## Surprises and Discoveries


The existing collector already orders CPU prerequisites before frame seals and GPU results. No new producer synchronization is needed. The old `build_frame_record` increments hierarchy-loss counters, so using it to preview pending frames requires suppressing that accounting until finalization.

## Outcomes and Retrospective


Recent snapshots, pending GPU status, and the UI split are implemented. The default snapshot retains paired CPU/GPU comparisons; the opt-in recent snapshot drives headline and history without waiting for GPU completion. Final validation passed 19 native cases (237 assertions) and 27 Python cases, with three expected CUDA rasterization skips. Both D3D12 and Vulkan subprocess shutdown tests passed. The standalone UI probe verified GPU gaps, advancing CPU history, unchanged table values, selection after row reordering, and pause/resume, and now exits with status 0. Its previews were inspected. The Debug extension, tests, pathtracer, type stubs, and docstrings were built/generated successfully. Pre-commit and whitespace checks passed after formatting; an intermediate pre-commit run hit build-time source file locks and was rerun after the build finished. Worker replacement and lifecycle API redesign remain outside this change.

## Idempotence and Recovery


Preserve existing uncommitted Level 1 work. No dependency updates, commits, or external publication. Builds and tests are repeatable. Existing completed snapshot behavior remains the compatibility baseline.

## Interfaces and Dependencies


Use existing C++, ImGui, nanobind, NumPy, pytest, and doctest dependencies. Extend `Profiler::frame_stats_snapshot(bool include_pending = false)` and its Python argument, and append the pending GPU enum value. The native `ui::reset_profiler_window()` releases cached snapshots from Python's existing atexit hook. No new dependency or worker thread is required.

Revision note (2026-10-30): The UI reads the completed snapshot followed by the recent one. A publication may occur between reads; table lag is therefore computed from the actual displayed frame IDs, rather than the recent snapshot's pending count. Completed CPU/GPU table values always come from one snapshot.

Decision (2026-10-30, Codex): Reset the profiler window cache in the existing Python atexit hook while the interpreter and GIL are available. The native reset releases both cached snapshots and preserves the existing worker retirement mechanism. A subprocess regression covers Python ownership acquired after the UI has cached the native snapshot. This is necessary lifecycle cleanup for the new dual-view window, not a worker redesign.

Final evidence (2026-10-30): `.temp/profiler-level2-python.log` records 27 passed / 3 skipped; `.temp/profiler-level2-native.log` records 19 cases / 237 assertions. `.temp/profiler-level2-ui.log` records the interaction checks and clean exit. Previews `.temp/profiler-level2-cpu-pending.png`, `.temp/profiler-level2-gpu-pending.png`, and `.temp/profiler-level2-diagnostics.png` show both views and the diagnostics.
