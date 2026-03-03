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
| `slangpy/testing/benchmark/plugin.py` | Pytest plugin adding `--benchmark-save`, `--benchmark-compare`, `--benchmark-upload` CLI options |
| `slangpy/testing/benchmark/report.py` | `BenchmarkReport` / `Report` TypedDicts, serialization, MongoDB upload |
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
```

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
