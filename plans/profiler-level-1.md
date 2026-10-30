# Clarify profiler timings and fix capture membership

This ExecPlan follows `.agents/PLANS.md` and is maintained throughout implementation.

## Purpose / Big Picture


The profiler window should distinguish application cadence, CPU elapsed scope time, and measured GPU time. Selecting a GPU zone must show its history even when its parent has no GPU measurement. GPU capture membership must follow the CPU recording scope rather than approximate calibrated GPU timestamps. Existing trace and statistics APIs retain their meanings; additive fields describe frame intervals and sample timestamps.

## Progress


- [x] (2026-10-30) Reviewed collection, UI, bindings, examples, and existing tests. Working tree initially clean.
- [x] (2026-10-30) Reproduced the capture-boundary problem: pre-capture recording produced zero captured CPU zones and one captured GPU zone after delayed submission.
- [x] (2026-10-30) Added frame interval and timestamp data and regression coverage for capture membership, cadence, activation gaps, and array lifetime.
- [x] (2026-10-30) Simplified table and graph, exposed coverage and sample age, documented semantics, and regenerated Python docstrings.
- [x] (2026-10-30) Built Debug, passed 19 native cases and 20 Python cases (one CUDA rasterization skip), and inspected an offscreen rendering of the GPU-child case.
- [x] (2026-10-30) Ran pre-commit; accepted formatting updates and reran all hooks successfully.
- [x] (2026-10-30) Final Debug rebuild succeeded; 19 native cases (209 assertions) and 20 Python cases passed, with one expected CUDA rasterization skip. Final preview inspected and `git diff --check` passed.
- [x] (2026-10-30) Apply requested UI refinements: shared headline/value selector, compact labels, status in Diagnostics, coverage in tooltips, and selected subtree history. Rebuild, verify rendering, and run focused tests and pre-commit.

- [x] (2026-10-30) Replace separate history rows with one stacked subtree graph; verify inclusive timing partitioning, missing parents, and nested-band rendering.

## Surprises and Discoveries


The frame headline divides 1000 by CPU scope duration, while the pathtracer performs acquisition and presentation outside that scope. The graph clips GPU children to CPU-only parents' zero GPU time. Capture filtering compares calibrated GPU start timestamps to CPU capture boundaries; this can also admit work recorded before a capture but submitted during it.

The pre-change delayed-submission probe printed `pre-capture CPU zones: 0` and `pre-capture GPU zones: 1`. The new native regression records one zone before and one during capture in the same encoder and expects only the second in both capture timelines. This establishes a real boundary bug without claiming to reproduce issue #1223's intermittent missing-zone symptom.

## Decision Log


- Decision: Supersede separate history rows with one stacked column per frame. Compute each subtree's displayed size bottom-up as the larger of its measured inclusive duration and the sum of its child subtree sizes. The parent contributes only the positive remainder; missing measurements contribute no parent band, while measured descendants remain visible. Draw bands in hierarchy order, using existing table colors and per-band hover details.
  Rationale: The user wants a compact stack. This avoids adding inclusive parents on top of their children, never clips descendants to missing or smaller parent measurements, and keeps malformed or non-contained GPU relationships visible. The GPU stack is a grouped view, not a measured total GPU frame time or a measurement of exclusive time; explain this in its tooltip.
  Date/Author: 2026-10-30, Codex.


- Decision: Preserve `frame_time`, `latest_frame_ms`, and `sample_frame_time_ms` as CPU scope durations. Add explicitly named interval and start timestamp properties.
  Rationale: Avoid silently changing existing Python analysis. A frame interval is the difference between consecutive accepted frame starts; the first interval after activation is unavailable.
  Date/Author: 2026-10-30, Codex.
- Decision: Replace the stacked hierarchy graph with the selected zone's CPU or GPU history and remove share bars.
  Rationale: CPU nesting does not imply GPU containment, and inclusive times and independent percentiles cannot be treated as additive shares. This fixes visibility with less display logic.
  Date/Author: 2026-10-30, Codex.
- Decision: Keep GPU-dependent frame completion and the existing worker for this level.
  Rationale: Decoupling publication and changing collector ownership belong to the later structural refactor. Show frame identity, age, and pending count now.
  Date/Author: 2026-10-30, Codex.
- Decision: Clear retained frame statistics without resetting the cadence predecessor. Reset it only on enable/disable transitions.
  Rationale: Clearing history does not interrupt frame recording, whereas disabled periods must not count as a single slow frame.
  Date/Author: 2026-10-30, Codex.

## Outcomes and Retrospective


The UI separates cadence/FPS from CPU scope duration, keeps coverage in tooltips and sample age in Diagnostics, and graphs a selected subtree as one stack without clipping to CPU ancestors. Captures use recording-time membership. Python APIs are additive and existing scope-duration fields retain their meaning. Final validation passed 19 native cases (209 assertions) and 20 Python cases, with the CUDA rendering case skipped because it lacks rasterization. All pre-commit hooks and `git diff --check` passed. The final offscreen preview was inspected after rebuilding. Vulkan tests still emit the previously observed validation messages about unknown extension structures. The intermittent missing-zone symptom in #1223 has not been reproduced, so the capture-boundary correction is not claimed to close that issue.

## Context and Orientation


`src/sgl/utils/profiler.cpp` collects CPU events and GPU query results, retains traces, and publishes immutable statistics. `src/sgl/utils/profiler.h` defines those public native data structures. `src/slangpy_ext/utils/profiler.cpp` binds them to Python, including read-only arrays whose owner keeps their storage alive. `src/sgl/utils/profiler_ui.cpp` draws the ImGui profiler window. `tests/sgl/device/test_profiler.cpp` and `slangpy/tests/slangpy_tests/test_profiler.py` cover the native and Python contracts. `docs/src/profiling.rst` documents usage. GPU clock calibration converts GPU timestamps to approximate CPU clock positions; duration measurement itself uses timestamp differences.

## Plan of Work


First reproduce capture admission using a GPU zone recorded before capture and submitted after capture begins. Record CPU start time on GPU slots and use it for capture membership, retaining calibrated time solely for trace placement. Add regression tests that cover both delayed pre-capture work and work recorded during capture.

Next attach start-to-start intervals to accepted frames, excluding the first sample and resetting the predecessor on enable/disable transitions. Publish interval distributions and aligned read-only start/interval arrays while preserving CPU scope fields. Test exact relationships to trace timestamps, retained-window eviction, disabled gaps, empty snapshots, and Python array ownership.

Finally simplify the UI to a Mean/Latest/P95 selector for CPU frame time, zone times, and calls, GPU coverage in tooltips, explicit CPU and GPU labels, frame age and pending state in Diagnostics, and a single selected-subtree stacked history graph. Preserve selection by hierarchical site identity across snapshots. Document that timing sums are inclusive, coverage excludes unavailable/incomplete frames, FPS always uses the reciprocal of the mean measured frame interval, and all frame statistics still await GPU resolution.

## Concrete Steps


Run commands from `C:\src\slangpy`. All CMake builds and tests must run outside the sandbox.

    cmake --build --preset windows-msvc-debug --target sgl_tests slangpy_ext pathtracer
    .\build\windows-msvc\Debug\sgl_tests.exe -ts=profiler
    python -m pytest slangpy/tests/slangpy_tests/test_profiler.py -v
    pre-commit run --all-files

Build generated documentation with the existing `slangpy_pydoc` target when practical; inspect generated changes. Re-run pre-commit if it modifies files. Review the final diff and working tree.

## Validation and Acceptance


A deliberate gap outside a CPU frame scope increases frame interval without increasing that scope's elapsed time. FPS always uses the mean interval independently of the value selector and is unavailable without a valid interval. Python arrays remain immutable and usable after snapshot/profiler deletion. GPU children and grandchildren of a CPU-only parent remain visible together as contiguous colored bands when that parent is selected; each descendant is also independently selectable. A contained grandchild occupies its measured share of the parent total, rather than being added twice. A larger GPU grandchild on a separate encoder remains visible even when its measured parent is smaller. Missing GPU values show their status rather than a fabricated zero; an absent occurrence is a valid zero. Frame identity, age, and pending count explain delayed publication. Pre-capture GPU recordings stay outside capture even if execution occurs during capture; in-capture recordings remain included after query completion.

## Idempotence and Recovery


Builds and focused tests can be repeated. Preserve unrelated working-tree changes. Do not alter the slang-rhi dependency or publish changes to GitHub. Any issue-specific hypothesis not validated remains documented as such.

## Artifacts and Notes


Before implementation, the prior review built Debug and passed 18 native profiler cases, 16 Python cases, and all pre-commit hooks. Vulkan validation emitted messages about unknown extension structures during Python tests.

The offscreen GPU-child regression passed on D3D12 and Vulkan. A preview saved to `.temp/profiler-ui.png` was visually inspected: the CPU-only parent has GPU time n/a, its child has measured GPU time and a full independent history graph. Visual inspection prompted adaptive axis precision so sub-millisecond samples do not all label as 0.00 ms. Final build/test logs are saved under `.temp/`.

## Interfaces and Dependencies


Use existing C++ standard-library, ImGui, nanobind, NumPy, pytest, and doctest dependencies. Add `ProfilerFrame.interval_ns`, `ProfilerFrameStats.frame_interval`, `sample_frame_start_ns`, and `sample_frame_interval_ms`. Zero denotes an unavailable interval and is excluded from the interval distribution. Existing frame time properties remain CPU scope durations. No collector replacement or new external dependency is required.

Revision note (2026-10-30): Completed implementation and final validation. Recording-time membership is carried only through pending GPU results, avoiding permanent per-zone trace-storage overhead. Frame starts are sampled under the frame mutex so concurrent callers cannot invert successive accepted start timestamps.
- Decision: Following user feedback, graph the selected subtree in indented rows with a shared duration scale. Keep each entry independent, including GPU children of unmeasured CPU parents.
  Rationale: Retain visible nesting without implying GPU containment from CPU hierarchy. A scrollable graph keeps larger subtrees readable. Mean/Latest/P95 applies to the headline and table; P95 FPS is the reciprocal of P95 frame interval, describing slow-frame cadence.
  Date/Author: 2026-10-30, Codex.

Revision note (2026-10-30, UI follow-up): Applied the requested compact headline and shared selector, moved publication status into Diagnostics and GPU coverage into tooltips, and restored nested zones as indented independent history rows. Extended the offscreen regression to require visible GPU child and grandchild bands under a selected CPU-only root. Final build and 20 Python tests passed (one CUDA skip); native validation passed 19 cases / 209 assertions. Visually verified Mean, P95 with expanded Diagnostics, and Latest after selecting a child. Three rows fit without scrolling; larger subtrees scroll. Pre-commit and diff whitespace checks passed. Preview images are under `.temp/profiler-ui*.png`.

Revision note (2026-10-30, stacked graph): The user rejected separate lanes in favor of one stacked history plot. Replace the row renderer with bottom-up subtree sizing and contiguous bands. The expanded offscreen regression checks full plot height, contiguous bands, correct inclusive partitioning, and a grandchild recorded on a separate encoder whose duration exceeds its parent. The Debug build passed. Native tests passed 19 cases / 209 assertions; Python tests passed 22 cases with two CUDA rasterization skips. Both nested and non-contained GPU previews were inspected (`.temp/profiler-stacked-ui.png` and `.temp/profiler-stacked-expanded.png`). Pre-commit passed after formatting. The rendering assertion excludes the blue mode selector so it measures only graph bands.

Revision note (2026-10-30, FPS selector): The user agreed that plain FPS should consistently describe average cadence. Use the mean interval directly for FPS, keep the selector for CPU frame time and zone statistics, and update the tooltip and documentation. The prior decision to apply P95/Latest to FPS is superseded. The Debug extension build, pre-commit, and 22 Python profiler cases passed (two expected CUDA rendering skips). An offscreen interaction check verified identical FPS pixels across Mean/P95/Latest while CPU frame values changed; the P95 preview was inspected.
