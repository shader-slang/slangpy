---
description: These instructions should be loaded whenever the context refers to benchmarks or benchmarking
---

# SlangPy Benchmarks

## Overview

SlangPy has a custom benchmark framework built on pytest. Benchmarks live in `slangpy/benchmarks/` and the framework infrastructure is in `slangpy/testing/benchmark/`.

## Directory Structure

| Directory / File | Purpose |
|------------------|---------|
| `slangpy/benchmarks/` | Benchmark test files (`test_benchmark_*.py`) and Slang shader files |
| `slangpy/benchmarks/conftest.py` | Auto-imports benchmark fixtures and registers plugins |
| `slangpy/testing/benchmark/fixtures.py` | Pytest fixtures: `BenchmarkSlangFunction`, `BenchmarkPythonFunction`, `BenchmarkComputeKernel`, `ReportFixture` |
| `slangpy/testing/benchmark/plugin.py` | Pytest plugin adding local report and authenticated BenchView submission options |
| `slangpy/testing/benchmark/benchview.py` | Native BenchView payload construction, batching, and authenticated HTTP submission |
| `slangpy/testing/benchmark/report.py` | Legacy local report serialization and comparison data |
| `slangpy/testing/benchmark/table.py` | Terminal table display with color-coded deltas |
| `slangpy/testing/benchmark/utils.py` | Machine/GPU/commit info collection, JSON datetime helpers |

## Benchmark Fixtures

Three fixture types measure different things:

| Fixture | Measures | Timing Method |
|---------|----------|---------------|
| `benchmark_slang_function` | GPU execution time of a SlangPy `Function` | GPU timestamp queries (high precision) |
| `benchmark_compute_kernel` | GPU execution time of a raw `ComputeKernel` | GPU timestamp queries (high precision) |
| `benchmark_python_function` | Wall-clock time of a Python callable | `time.time()` with sub-iterations |

### Using `benchmark_slang_function`

Pass a `spy.Function` and its keyword arguments. The fixture handles warmup, GPU timestamp queries, and reporting.

```python
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_my_benchmark(device_type: spy.DeviceType, benchmark_slang_function: BenchmarkSlangFunction):
    device = helpers.get_device(device_type)
    # ... set up tensors and load module ...
    func = module.require_function("my_func")
    benchmark_slang_function(device, func, a=tensor_a, b=tensor_b, _result=tensor_c)
```

### Using `benchmark_python_function`

Wrap the code to benchmark in a callable. The fixture runs it `iterations × sub_iterations` times and reports per-call timing.

```python
def test_my_python_benchmark(device_type: spy.DeviceType, benchmark_python_function: BenchmarkPythonFunction):
    device = helpers.get_device(device_type)
    # ... set up resources ...
    def my_operation():
        func(a, b, _result=result)
    benchmark_python_function(device, my_operation, iterations=10, sub_iterations=20000)
```

### Using `benchmark_compute_kernel`

For raw compute kernel dispatch benchmarking:

```python
def test_my_kernel_benchmark(device_type: spy.DeviceType, benchmark_compute_kernel: BenchmarkComputeKernel):
    device = helpers.get_device(device_type)
    kernel = device.create_compute_kernel(program)
    benchmark_compute_kernel(device, kernel, spy.uint3(1024, 1, 1), a=buf_a, b=buf_b, res=buf_res)
```

## Running Benchmarks

```bash
# Run all benchmarks
pytest slangpy/benchmarks -v

# Run a specific benchmark file
pytest slangpy/benchmarks/test_benchmark_tensor.py -v

# Save results for later comparison
pytest slangpy/benchmarks -v --benchmark-save

# Save with a specific run ID
pytest slangpy/benchmarks -v --benchmark-save my_run

# Compare against a previous run
pytest slangpy/benchmarks -v --benchmark-compare

# Compare against a specific run ID
pytest slangpy/benchmarks -v --benchmark-compare my_run

# List saved runs
pytest slangpy/benchmarks --benchmark-list-runs

# Submit to a local BenchView server (PowerShell)
$env:BENCHVIEW_API_KEY = "<write key>"
pytest slangpy/benchmarks -v --benchmark-submit <request-id> --benchmark-api-url http://localhost:3000

# Run the CI wrapper against the nested hosted deployment
$env:BENCHVIEW_API_KEY = "<write key>"
python tools/ci.py benchmark-python --run-id <request-id> --api-url http://rtrci.nvidia.com/benchview
```

`BENCHVIEW_API_URL` may supply the base URL when direct pytest commands omit `--benchmark-api-url` or the CI wrapper omits `--api-url`. The URL is the root or nested BenchView application base, not the full submission endpoint. The write key is accepted only through `BENCHVIEW_API_KEY`; it is never a command-line option or printed by the benchmark plugin. `--benchmark-upload` remains an alias for `--benchmark-submit` for existing direct pytest scripts, but it now uses only the HTTP API and never connects to MongoDB.

The ordinary `.github/workflows/ci-benchmark.yml` workflow is manual and is also dispatched by the nightly scheduler. With no `revision` input it checks out and benchmarks the selected branch tip. With an exact 40-character `revision` it checks out that future-compatible commit while retaining the workflow branch name for BenchView. It always builds the selected source and invokes the same `tools/ci.py benchmark-python` command shown above on the Windows and Linux performance workers. Configure repository variable `BENCHVIEW_API_URL` with the root or nested BenchView base URL and repository secret `BENCHVIEW_API_KEY` with its write key before enabling the workflow.

## Benchmark action scheduling

The nightly `.github/workflows/schedule-benchmarks.yml` workflow runs once per day. It uses the authenticated GitHub API client supplied by `actions/github-script`, so it needs no checkout, Python environment, GitHub CLI, or extra secret. It examines one fixed 24-hour UTC interval and dispatches the ordinary workflow once for every `main` commit in that interval, oldest first, through the `revision` input. It deliberately does not inspect or suppress existing runs; manually triggering the scheduler repeats the complete 24-hour interval.

The historical backfill dispatcher uses the official GitHub CLI because it is an operator process rather than a GitHub-hosted workflow. It never accepts a token argument. Verify its prerequisite before any preview or dispatch:

```powershell
gh --version
gh auth status
```

Historical commits use the separate manual `.github/workflows/backfill-benchmark.yml` workflow. Its inclusive supported floor is `f3ad0fd91d8cf4eeb2be3b505765b43482aa952a` from 2 September 2025; older revisions are rejected before setup or build. A run takes a whole rev list rather than a single commit, because roughly 80% of a single-commit run's wall time is spent waiting for a performance runner. `tools/backfill_batch.py` clones once below the runner's temporary directory and then, for each commit, resets and cleans the tree, checks the commit out, builds it untouched, and only then copies the current harness over it. Each commit is isolated: a failure is recorded against that commit and the loop continues. Per-commit outcomes are written to the job summary as the loop proceeds, so a job that dies mid-batch still reports what it submitted and names what it never reached. Native PowerShell and Bash cleanup steps validate the resolved clone path before removing it. Submitted observations explicitly identify the historical SHA and branch `main`.

Preview the batches without dispatching anything:

```powershell
python tools/dispatch_backfill_batches.py --dry-run
```

Dispatch the whole sweep. Every batch is submitted immediately; GitHub queues them and drains as runners free up, so there is no scheduler process to supervise and nothing to resume:

```powershell
python tools/dispatch_backfill_batches.py
```

`--branch` selects the history to benchmark and `--workflow-ref` selects the ref the workflow definition is read from. Keep these distinct: pointing `--branch` at a development branch benchmarks that branch's own commits and submits them to BenchView as if they were `main`. `--batch-size` defaults to 30 commits per run and `--exclude-file` takes a list of commits to leave out, one per line, for skipping commits that have already been benchmarked.

Read each finished run's summary for its succeeded and failed commit lists, and re-dispatch any failures as a smaller rev list.

## Writing New Benchmarks

1. Create a `test_benchmark_*.py` file in `slangpy/benchmarks/`.
2. Import and use one of the three fixture types as a function argument.
3. Use `@pytest.mark.parametrize` for device types and other parameters.
4. Slang shader code can be inline strings or `.slang` files in the benchmarks directory.
5. For torch-dependent benchmarks, guard with `HAS_TORCH` and `pytest.skip()`.
6. All function arguments must have type annotations.

### Conventions

- Benchmark test functions are named `test_<description>`.
- Use `helpers.get_device(device_type)` or `helpers.get_torch_device(device_type)` for device creation.
- Use `helpers.DEFAULT_DEVICE_TYPES` for the standard device type parametrization.
- Inline Slang code uses uppercase module-level string constants (e.g., `ADD_FLOATS`).
- Pre-allocate GPU resources outside the benchmarked callable to measure dispatch overhead only, unless allocation is what you're benchmarking.

## Report Format

Reports are JSON files stored in `.benchmarks/`. Each benchmark entry includes:
- `name`, `function`, `params`, `meta` (device info)
- `min`, `max`, `mean` (trimmed), `median`, `stddev` — all in milliseconds
- `data` — raw timing samples (stripped when saving to disk)
- `cpu_time` — total wall-clock time including warmup

The terminal summary table shows color-coded deltas when comparing: green for >5% improvement, red for >5% regression.

Local report files retain this legacy shape. In parallel, fixtures accumulate native BenchView observations. Tests whose original pytest function name contains `_cpu` submit `cpu_time`; every other benchmark submits `gpu_time`. Both use milliseconds. This matches the existing imported history, including Python wrappers that synchronize GPU work. The stable test ID is the normalized source file plus the original pytest function, while pytest parameters keep their legacy string values as case dimensions and `DeviceType.cuda` is shortened to `cuda`. Project, commit, machine, OS, CPU, and GPU information use BenchView's dedicated run and environment fields.

One benchmark process submits observations in API-sized batches. Independent device or machine processes at the same Git revision and build configuration derive the same logical run key. A new benchmark execution uses fresh observation timestamps and execution identity, so it replaces matching test cases normally; retrying an unchanged request body is idempotent. Transient connection and gateway failures use five total attempts with exponential backoff while preserving the exact payload and idempotency key.
