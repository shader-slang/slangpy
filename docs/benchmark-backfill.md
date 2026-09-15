<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Historical benchmark backfill

How the benchmark backtest works, what it measures, why the current design is
being replaced with batched runs, and what the replacement looks like.

## What the backfill is for

BenchView holds benchmark timings keyed by commit. Only commits built after the
benchmark CI existed have data, so trends cannot be read back past that point.
The backfill fills the gap: it rebuilds historical commits, runs today's
benchmarks against them, and submits the results under the historical commit's
identity.

The supported range starts at the compatibility floor
`f3ad0fd91d8cf4eeb2be3b505765b43482aa952a` (2025-09-02, "System to allow tests to
be isolated to specific platform"), which is the oldest commit whose build the
current harness can drive. That is **402 commits** on `main` to today.

### Today's benchmarks against yesterday's library

Each run clones the target commit, builds it, then checks the *current*
benchmark harness over the top:

```
tools/ci.py  tools/gpu_clock.py  tools/backfill_benchmark_manifest.py
slangpy/testing/benchmark  slangpy/testing/helpers.py
slangpy/testing/plugin.py   slangpy/testing/crashpad.py
slangpy/benchmarks
```

Timings are only comparable across commits if the benchmark definitions are held
fixed, so the harness must be the same everywhere. The consequence is that the
harness constantly meets libraries that predate the APIs it uses.

Note that a backfill point varies **SlangPy, slang-rhi and the Slang compiler
together** — they are pinned as a unit per commit. A movement in the data
attributes to the combination, not to SlangPy alone. Recording the Slang and
slang-rhi versions as BenchView dimensions is still an open question.

## Current state

The first sweep is running and is roughly 15% through. Of the completed runs,
**57 succeeded and 5 failed** (~92%), with Windows and Linux succeeding at
similar rates.

### Failure classes found and fixed

Each was discovered on CI, one per cycle, and each is now handled:

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: crashpad` | overlay omitted `crashpad.py` | added to overlay; `is_supported()` probe |
| `get_device called when no device types are selected` | overlay omitted `slangpy/benchmarks` | added to overlay |
| `fatal: reference is not a tree` | `github.sha` resolved inside a fresh clone | resolve harness SHA with fetch + ref fallback |
| vcpkg msys2 404 | deleted upstream package | repin vcpkg, **after** `ci.py setup` (setup resets submodules) |
| new compiler warnings | `-Werror` on old code | `-DSGL_WARNINGS_AS_ERRORS=OFF` for historical builds |
| `enable_rhi_validation` rejected | postdates target | probe `Device.__init__.__doc__` |
| `set_cuda_context_current` missing | postdates target | `hasattr` guard |
| CUDA device creation fails outright | Windows CUDA broken before ~2026-02 | skip that device type, keep the others |
| `ImportError`/`BoundVariableException`/`ResolveException` in PPISP | **benchmark postdates the target** | benchmark manifest (below) |

The last row was the important one. Three different exceptions over three cycles
all had a single root cause: running a benchmark against a build that predates
the benchmark. Softening exceptions could not distinguish "did not exist yet"
from "regressed", so `tools/backfill_benchmark_manifest.py` now asks git which
benchmark modules exist in the target's tree and skips only the rest. Every
guard is gated on `BACKFILL_TARGET_SHA`, so ordinary CI still fails loudly.

### Data is a widening cone, not a flat panel

Benchmarks are skipped where they did not exist, so coverage grows over time:

| Target | cuda benchmarks measured |
|---|---|
| 2025-09 (floor) | 10 |
| 2026-03 | 44 |
| 2026-04 onward | 68 |

At the floor only `test_benchmark_interop.py` and `test_benchmark_tensor.py`
survive; `argcounts`, `autograd`, `bwd_diff` and `ppisp` all postdate it.
**Only `interop` and `tensor` span the full range.** Per-benchmark trends are
valid; aggregates across the whole benchmark set are not, because their
composition changes with time.

### A real regression, found by the sweep

The backtest bracketed a CUDA VRAM leak without being told to look for one:

```
2026-03-17  d49288ecdee2  success   <- last good
2026-03-20  ba04baab18d4  OOM
2026-03-30  6c4134e73d66  OOM
2026-04-02  8a763ee98552  clone 403 (benchmark outcome unknown)
2026-04-09  514bdf74393a  success   <- recovered
```

Failing runs show PyTorch holding 30.24 GiB of a 31.36 GiB card alongside
`tensor_bridge_create_zeros_like failed`. Successful runs from the same period
show no OOM at all, so this is not ambient GPU contention. The recovery window
contains `2a89baa6` — *"Support retain_graph=True and fix VRAM leak in autograd
backward (#914, #896)"* — which matches the symptom exactly. Probes either side
of that commit are queued to confirm.

### Known bad data

* **The old scheduler enumerated the wrong branch.** `backfill_benchmarks.py`
  used a single `--branch` flag for two unrelated things: which history to walk
  and which ref to dispatch the workflow from. Run against the development
  branch, it therefore treated that branch's own 35 commits as history. That is
  where the "435 commits" figure came from; `main` has 402. Four of those
  development commits were benchmarked and submitted under
  `BENCHVIEW_BENCHMARK_BRANCH: main`:

  ```
  7708f86d51ca  Re-enable more restricted gpu clock control on linux
  d7751febc4e5  Skip failed/cancelled runs when reconciling backfill state
  080dc9f304df  Skip benchmarks that did not exist in the backfill target
  759add3f9608  Skip benchmarks whose APIs postdate the target build
  ```

  They need deleting from BenchView. The replacement dispatcher separates the
  two concepts into `--branch` and `--workflow-ref`.
* **BenchView has not been wiped.** New rows are landing alongside pre-existing
  data.

### Windows nvrtc gap, late 2025

Reproduced at 2025-11-25 by the throughput probe, which gave the full message the
earlier truncated logs had hidden:

```
slang: (0): error 52002: could not find a suitable pass-through compiler for 'nvrtc'
```

Slang cannot locate nvrtc on the Windows runners for that era's CUDA targets, so
every `DeviceType.cuda` benchmark in the module fails. Linux is green on the same
commits. This is **not** Slang pass selection, which is what the truncated symptom
suggested, and it is not flaky: expect it across the surrounding window of commits.

Open question: this is a capability gap of the same kind as the missing CUDA
device, so it arguably belongs in the skip-and-report path rather than being
counted as a failure. Left as a failure for now, because unlike a device that
cannot be created, a missing pass-through compiler is also what a genuine
regression would look like.

## Why the current design is slow

One workflow run per commit. Measured on real runs:

| | Linux | Windows |
|---|---|---|
| queue wait | 29–45 min | 19–37 min |
| execution | **5 min** | **9–10 min** |

Execution breaks down as:

```
clone 21s · configure 10s · build 46s · torch bridge 22s · benchmark 214s
```

Two things follow, and they are not what one would guess:

1. **The build is not the problem.** 46 seconds — ccache is already effective.
   Per-commit setup is ~99 s against 214 s of actual measurement.
2. **~80% of wall time is queueing.** Only two runners carry the
   `nvrgfx-perf-kernelvm-bridge` label, and they are shared with Slang PR CI.

At the measured 6.71 runs/hour the remaining commits need ~50 h. The waste is
not compute, it is paying a ~30 minute queue admission **402 times**.

## Batched runs

Pay the queue cost once per 30 commits instead of once per commit.

### Shape

* `backfill-benchmark.yml` takes a **rev list** (`target_shas`) instead of a
  single `target_sha`, and `tools/backfill_batch.py` loops over it on both
  platforms.
* **45 commits per run**, giving 9 runs for the full 402.
* **All 9 dispatched at once** by `tools/dispatch_backfill_batches.py`. No
  scheduler process, no polling, no in-flight cap — GitHub queues them and
  drains as runners free up. This removes `tools/backfill_benchmarks.py` from
  the critical path entirely.
* **Generous `timeout-minutes`.** Set to 720. The worst 45-commit batch measures
  ~365 min on Windows, so this leaves 2x headroom.

### Measured, not estimated

Run 34920515414 benchmarked 15 commits spread evenly across the whole range, on
both platforms, behind a single 44 minute queue admission:

| | linux | windows |
|---|---|---|
| mean per commit | 3.3 min | 5.4 min |
| slowest commit | 5.4 min | 8.4 min |
| total execution | 50 min | 81 min |
| result | 15/15 | 14/15 |

Per-commit cost rises with commit age, from about 2 min at the floor to about 8
at head, because the manifest skips benchmarks that did not exist yet. That is
why the probe sampled the range evenly: a contiguous block of old commits would
have measured ~2 min/commit and badly understated the real cost.

Batch size follows from the Windows numbers. Per-commit cost is
`queue / n + 5.4`, so the 44 minute admission is already amortised to about
1 min/commit at n=45; going further buys almost nothing. Against the 720 minute
timeout the worst 45-commit batch is ~365 min, leaving 2x headroom, and with two
Windows runners more and smaller batches divide the fixed ~36 h of work more
evenly than fewer large ones. 45 is the chosen size; the timeout alone would
permit about 85.

Windows remains the bottleneck; batching removes the queue overhead, not the
execution time. Total wall time is bounded by Windows execution: 402 x 5.4 min =
~36 h of Windows runner time, divided by however many perf-labelled Windows
runners are online.

### Robustness

Results go straight to BenchView, so **no artifacts are needed**. The run only
has to report what happened:

* Each commit is isolated — a failure records and moves to the next, it does not
  abort the batch.
* Per-commit outcome accumulated and written to `$GITHUB_STEP_SUMMARY` as a
  table, plus echoed at the end of the job, so a finished run states plainly
  which commits succeeded and which failed.
* Because BenchView is written per commit as the loop proceeds, a job that dies
  or times out mid-batch keeps everything already submitted. Only the unreached
  commits are lost, and they are named in the log.
* Clean checkout between commits: `git reset --hard`, `git clean -xfd`, then
  `git checkout --detach <sha>`. The overlay writes untracked files and replaces
  tracked ones, so a plain checkout is not enough. The build is only 46 s, so a
  fully clean tree is affordable and removes the risk of stale artifacts
  contaminating the next commit.
* The harness is copied from the workflow's own checkout rather than checked out
  from a commit inside the historical clone. That deletes the whole
  "resolve the harness SHA in a fresh clone" problem along with its duplicated
  pwsh and bash implementations.

### Deliberate simplifications

* **No low-discrepancy ordering, no round-robin across batches.** The old
  scheduler samples history evenly so it can be stopped at any point; batched
  runs are fast enough that running to completion is the expectation. Contiguous
  chronological batches are therefore fine.

  The consequence, recorded so it is not a surprise: while the sweep is partially
  complete, coverage is a set of chronological chunks rather than an even spread.
  If that matters later, striding batch *k* to take every 9th commit rather than
  a contiguous block restores even coverage at no cost.
* **No dispatch pacing.** Submitting all batches immediately is simpler and the
  GitHub scheduler already queues correctly.

### Cutover

1. Land the batched workflow, `tools/backfill_batch.py` and
   `tools/dispatch_backfill_batches.py`.
2. Stop the running scheduler (`tools/backfill_benchmarks.py`).
3. Dispatch all 9 batches:

   ```
   python tools/dispatch_backfill_batches.py --workflow-ref <branch>
   ```

   `--exclude-file` takes a list of commits to leave out, for skipping ones that
   have already been benchmarked.
4. Read each run's summary for the succeeded/failed commit lists; re-dispatch
   any failures as a smaller rev list.
