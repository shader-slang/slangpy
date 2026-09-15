<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Historical benchmark backfill

BenchView holds benchmark timings keyed by commit, so trends cannot be read back
past the point where benchmark CI started running. The backfill fills that gap: it
rebuilds historical commits, runs today's benchmarks against them, and submits the
results under the historical commit's identity.

## Supported range

The inclusive compatibility floor is
`5c266df695fe69da052ef2495e55f9d571442e23` (2026-02-16), the oldest commit whose build
the current harness can drive. It is defined once, as `SUPPORTED_FLOOR_SHA` in
`tools/backfill_commit.py`, and commits below it are rejected before any setup or build
work starts.

It is set just past the last commit that needed an era-dependent accommodation, so the
harness carries none at all: see [Era compatibility](#era-compatibility). Older commits
are reachable only by reinstating those, which is a deliberate trade rather than a
configuration change.

## How a run works

One workflow run benchmarks exactly one commit, so the unit of work, the unit of
failure and the unit of retry are all the same thing. A runner that dies costs the one
commit it was holding, and re-running that commit needs no reasoning about which of its
neighbours had already been submitted.

`tools/backfill_commit.py` is what the run executes. It resets and cleans the historical
clone, checks the commit out, builds it untouched, and only then copies the current
harness over the top:

```
tools/ci.py  tools/gpu_clock.py  tools/backfill_benchmark_manifest.py
slangpy/testing/benchmark  slangpy/testing/helpers.py
slangpy/testing/plugin.py   slangpy/testing/crashpad.py
slangpy/benchmarks
```

Timings are only comparable across commits if the benchmark definitions are held
fixed, so the harness has to be identical everywhere. The consequence is that the
harness constantly meets libraries that predate the APIs it uses; see
[Era compatibility](#era-compatibility).

Results are submitted to BenchView per benchmark as they are produced, so a run that
dies partway still leaves everything it had already measured.

## Running a sweep

The history is hundreds of commits and there are only a couple of performance runners,
so `tools/backfill_benchmarks.py` is a supervising process rather than a bulk submit: it
keeps a bounded number of runs in flight, dispatching the next commit only as an earlier
one finishes. It is expected to run for a long time and to be interrupted.

Preview the commits without dispatching anything:

```
python tools/backfill_benchmarks.py --state-file sweep.json --dry-run
```

Dispatch the sweep, and leave it running:

```
python tools/backfill_benchmarks.py --state-file sweep.json
```

- `--state-file` is required. It records what each commit's run is doing and is
  rewritten atomically after every transition, so interrupting the dispatcher with
  Ctrl-C and starting it again on the same file resumes exactly where it left off.
  Commits that already have an outcome are not repeated, and runs still in flight are
  adopted rather than dispatched a second time. Without it a restart would benchmark
  the whole history again.
- `--branch` selects the history to benchmark and `--workflow-ref` selects the ref the
  workflow definition is read from. Keep these distinct: pointing `--branch` at a
  development branch benchmarks that branch's own commits and submits them to BenchView
  as if they were `main`.
- `--max-in-flight` caps concurrent runs. The default is sized to the performance
  runner pool; raising it past the number of runners only lengthens the GitHub queue.
- `--retry-failed` re-dispatches commits previously recorded as failed. Failures are
  otherwise terminal, so a resumed sweep does not retry them forever.

Per-commit cost falls the further back the target is, because the manifest skips more
benchmarks that did not exist yet. Windows is the bottleneck throughout.

### Recovering an interrupted dispatch

A commit is marked `dispatching` *before* the API call that creates its run, and
`in_flight` with the run id only after that call returns. Losing the dispatcher between
those two points would otherwise strand a run nobody was tracking, and re-dispatching
the commit would benchmark it twice.

The workflow's `run-name` carries the commit, which makes a run self-identifying. On
startup any commit left in `dispatching` is resolved by searching recent runs of the
workflow for that commit: if one exists it is adopted, and if none does the commit is
returned to the queue.

## Era compatibility

Every device and library API the harness uses exists throughout the supported range,
so it carries no probes or capability guards: a missing API is a regression everywhere,
including during a backfill. Only two things still vary with the target, and both are
about which benchmarks exist rather than what the build can do:

| Situation | Handling |
|---|---|
| Benchmark module postdates the target | `tools/backfill_benchmark_manifest.py` asks git which benchmark modules exist in the target's tree; the plugin skips the rest at collection |
| Individual benchmark's shader does not compile | Reported as skipped, not failed; a run where *every* benchmark is skipped fails, since that is a silently empty result |

This is what the floor buys. Below it the harness needed a vcpkg repin, a CUDA
device the runners could not create, and probes for APIs that did not exist yet —
each of which had to decide, with no evidence available, whether a failure was the
era or a regression. Lowering the floor means writing those back.

## Reading the data

A backfill point varies **SlangPy, slang-rhi and the Slang compiler together** — they
are pinned as a unit per commit. A movement in the data attributes to the combination,
not to SlangPy alone. Recording the Slang and slang-rhi versions as BenchView
dimensions is an open question.

Because benchmarks are skipped where they did not exist, coverage is a widening cone
rather than a flat panel: the number of benchmarks measured grows from the floor to
head. At the floor `interop`, `tensor` and `argcounts` survive; `autograd` (2026-02-17),
`ppisp` (2026-02-25) and `bwd_diff` (2026-04-02) all postdate it, so **only those three
span the full range.**

Per-benchmark trends are therefore valid, but aggregates across the whole benchmark set
are not, because their composition changes with time. To see which benchmarks a given
target supports, run `tools/backfill_benchmark_manifest.py` against it.
