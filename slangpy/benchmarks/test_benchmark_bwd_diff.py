# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Benchmark: DiffTensor vs DiffTensorView backward — atomic contention.

Two scenarios:

1. Uniform access: y[tid] = w
   All threads load the same weight w. In backward, gradients for w
   must be accumulated across all threads — atomic contention.

2. Per-element access: y[tid] = x[tid] * x[tid]
   Each thread reads/writes a unique index — no contention.
   loadOnce/storeOnce avoids unnecessary atomics.

GPU time is measured via timestamp query pools (no CPU overhead).
"""

import sys
import os
from typing import Optional
import pytest
import numpy as np

try:
    import torch
except ImportError:
    pytest.skip("PyTorch not installed", allow_module_level=True)

if sys.platform == "darwin":
    pytest.skip("PyTorch requires CUDA, not available on macOS", allow_module_level=True)

import slangpy as spy
from slangpy import diff_pair, Tensor

from slangpy.testing import helpers
from slangpy.testing.benchmark import BenchmarkSlangFunction

DEVICE_TYPES = [spy.DeviceType.cuda] if spy.DeviceType.cuda in helpers.DEFAULT_DEVICE_TYPES else []
if not DEVICE_TYPES:
    pytest.skip("Benchmarks require CUDA device type", allow_module_level=True)

TENSOR_SIZES = [65536, 1048576, 4194304]
CORRECTNESS_N = 64

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
EXTENSIONS_DIR = os.path.join(os.path.dirname(BENCH_DIR), "..").replace("\\", "/")

W_VAL = 2.0

# Module cache per device to avoid recompilation
_module_cache: dict[int, spy.Module] = {}


def _load_module(device: spy.Device) -> spy.Module:
    """Load the benchmark slang module with extensions.slang on the include path."""
    key = id(device)
    if key in _module_cache:
        return _module_cache[key]
    session = device.create_slang_session(
        {
            "include_paths": device.slang_session.desc.compiler_options.include_paths
            + [EXTENSIONS_DIR],
        }
    )
    slang_path = os.path.join(BENCH_DIR, "test_benchmark_bwd_diff.slang").replace("\\", "/")
    module = spy.Module(session.load_module(slang_path))
    _module_cache[key] = module
    return module


def _check_uniform_correctness(
    func: spy.Function, device: spy.Device, use_difftensor: bool = False
) -> None:
    """Verify y[tid]=w backward: w_grad = sum(output_grad) = CORRECTNESS_N."""
    n = CORRECTNESS_N
    if use_difftensor:
        w = Tensor.from_numpy(device, np.array([W_VAL], dtype=np.float32)).with_grads()
        result = Tensor.empty(device, shape=(n,), dtype=float).with_grads()
        assert result.grad_in is not None
        result.grad_in.storage.copy_from_numpy(np.ones(n, dtype=np.float32))
        func(count=n, w=w, output=result, _thread_count=n)
        func.bwds(count=n, w=w, output=result, _thread_count=n)
        assert w.grad_out is not None
        w_grad = w.grad_out.to_numpy()
        assert np.allclose(w_grad, [float(n)]), f"w_grad wrong: got {w_grad}, expected {n}"
    else:
        w = torch.tensor([W_VAL], device="cuda")
        w_grad = torch.zeros(1, device="cuda")
        output = torch.zeros(n, device="cuda")
        output_grad = torch.ones(n, device="cuda")
        func(count=n, w=w, output=output, _thread_count=n)
        func.bwds(
            count=n,
            w=diff_pair(w, w_grad),
            output=diff_pair(output, output_grad),
            _thread_count=n,
        )
        assert torch.allclose(
            w_grad, torch.tensor([float(n)], device="cuda")
        ), f"w_grad wrong: got {w_grad.item()}, expected {n}"


def _check_square_correctness(
    func: spy.Function, device: spy.Device, use_difftensor: bool = False
) -> None:
    """Verify f(x)=x*x backward: dx = 2*x with output_grad=1."""
    n = CORRECTNESS_N
    x_np = np.array([1.0, 2.0, 3.0] + [0.0] * (n - 3), dtype=np.float32)
    expected_np = 2.0 * x_np
    if use_difftensor:
        x = Tensor.from_numpy(device, x_np).with_grads()
        result = Tensor.empty(device, shape=(n,), dtype=float).with_grads()
        assert result.grad_in is not None
        result.grad_in.storage.copy_from_numpy(np.ones(n, dtype=np.float32))
        func(count=n, input=x, output=result, _thread_count=n)
        func.bwds(count=n, input=x, output=result, _thread_count=n)
        assert x.grad_out is not None
        x_grad = x.grad_out.to_numpy()
        assert np.allclose(
            x_grad, expected_np
        ), f"x_grad wrong: got {x_grad[:3].tolist()}, expected {expected_np[:3].tolist()}"
    else:
        x = torch.tensor(x_np, device="cuda")
        x_grad = torch.zeros(n, device="cuda")
        output = torch.zeros(n, device="cuda")
        output_grad = torch.ones(n, device="cuda")
        func(count=n, input=x, output=output, _thread_count=n)
        func.bwds(
            count=n,
            input=diff_pair(x, x_grad),
            output=diff_pair(output, output_grad),
            _thread_count=n,
        )
        expected = torch.tensor(expected_np, device="cuda")
        assert torch.allclose(
            x_grad, expected
        ), f"x_grad wrong: got {x_grad[:3].tolist()}, expected {expected[:3].tolist()}"


def _make_dt_uniform_kwargs(device: spy.Device, n: int) -> dict:
    """Create DiffTensor kwargs for uniform access benchmark."""
    w = Tensor.from_numpy(device, np.array([W_VAL], dtype=np.float32)).with_grads()
    result = Tensor.empty(device, shape=(n,), dtype=float).with_grads()
    assert result.grad_in is not None
    result.grad_in.storage.copy_from_numpy(np.ones(n, dtype=np.float32))
    return dict(count=n, w=w, output=result, _thread_count=n)


def _make_dtv_uniform_kwargs(n: int) -> dict:
    """Create DiffTensorView kwargs for uniform access benchmark."""
    return dict(
        count=n,
        w=diff_pair(
            torch.tensor([W_VAL], device="cuda"),
            torch.zeros(1, device="cuda"),
        ),
        output=diff_pair(
            torch.zeros(n, device="cuda"),
            torch.ones(n, device="cuda"),
        ),
        _thread_count=n,
    )


def _make_dt_square_kwargs(device: spy.Device, n: int) -> dict:
    """Create DiffTensor kwargs for per-element square benchmark."""
    x = Tensor.from_numpy(device, np.random.randn(n).astype(np.float32)).with_grads()
    result = Tensor.empty(device, shape=(n,), dtype=float).with_grads()
    assert result.grad_in is not None
    result.grad_in.storage.copy_from_numpy(np.ones(n, dtype=np.float32))
    return dict(count=n, input=x, output=result, _thread_count=n)


def _make_dtv_square_kwargs(n: int) -> dict:
    """Create DiffTensorView kwargs for per-element square benchmark."""
    return dict(
        count=n,
        input=diff_pair(
            torch.randn(n, device="cuda"),
            torch.zeros(n, device="cuda"),
        ),
        output=diff_pair(
            torch.zeros(n, device="cuda"),
            torch.ones(n, device="cuda"),
        ),
        _thread_count=n,
    )


# =============================================================================
# Uniform access: y[tid] = w (broadcast)
# =============================================================================


@pytest.mark.parametrize("n", TENSOR_SIZES)
@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_bwd_difftensor(
    device_type: spy.DeviceType,
    n: int,
    benchmark_slang_function: BenchmarkSlangFunction,
) -> None:
    """Uniform backward using DiffTensor (atomic)."""
    device = helpers.get_device(type=device_type)
    module = _load_module(device)
    func = module.require_function("broadcast_dt")
    _check_uniform_correctness(func, device, use_difftensor=True)
    kwargs = _make_dt_uniform_kwargs(device, n)
    benchmark_slang_function(device, func.bwds, **kwargs)


@pytest.mark.parametrize("n", TENSOR_SIZES)
@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_bwd_difftensor_uniform(
    device_type: spy.DeviceType,
    n: int,
    benchmark_slang_function: BenchmarkSlangFunction,
) -> None:
    """Uniform backward using DiffTensor loadUniform (wave reduction)."""
    device = helpers.get_device(type=device_type)
    module = _load_module(device)
    func = module.require_function("broadcast_dt_uniform")
    _check_uniform_correctness(func, device, use_difftensor=True)
    kwargs = _make_dt_uniform_kwargs(device, n)
    benchmark_slang_function(device, func.bwds, **kwargs)


@pytest.mark.parametrize("n", TENSOR_SIZES)
@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_bwd_dtv_load(
    device_type: spy.DeviceType,
    n: int,
    benchmark_slang_function: BenchmarkSlangFunction,
) -> None:
    """Uniform backward using DiffTensorView load() — atomic contention."""
    device = helpers.get_torch_device(device_type)
    module = _load_module(device)
    func = module.require_function("broadcast_dtv")
    _check_uniform_correctness(func, device)
    kwargs = _make_dtv_uniform_kwargs(n)
    benchmark_slang_function(device, func.bwds, **kwargs)


@pytest.mark.parametrize("n", TENSOR_SIZES)
@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_bwd_dtv_load_uniform(
    device_type: spy.DeviceType,
    n: int,
    benchmark_slang_function: BenchmarkSlangFunction,
) -> None:
    """Uniform backward using DiffTensorView loadUniform() — wave reduction."""
    device = helpers.get_torch_device(device_type)
    module = _load_module(device)
    func = module.require_function("broadcast_dtv_uniform")
    _check_uniform_correctness(func, device)
    kwargs = _make_dtv_uniform_kwargs(n)
    benchmark_slang_function(device, func.bwds, **kwargs)


# =============================================================================
# Per-element access: f(x) = x*x
# =============================================================================


@pytest.mark.parametrize("n", TENSOR_SIZES)
@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_bwd_perelement_difftensor(
    device_type: spy.DeviceType,
    n: int,
    benchmark_slang_function: BenchmarkSlangFunction,
) -> None:
    """Per-element backward using DiffTensor (atomic)."""
    device = helpers.get_device(type=device_type)
    module = _load_module(device)
    func = module.require_function("square_dt")
    _check_square_correctness(func, device, use_difftensor=True)
    kwargs = _make_dt_square_kwargs(device, n)
    benchmark_slang_function(device, func.bwds, **kwargs)


@pytest.mark.parametrize("n", TENSOR_SIZES)
@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_bwd_perelement_difftensor_once(
    device_type: spy.DeviceType,
    n: int,
    benchmark_slang_function: BenchmarkSlangFunction,
) -> None:
    """Per-element backward using DiffTensor loadOnce (no atomics)."""
    device = helpers.get_device(type=device_type)
    module = _load_module(device)
    func = module.require_function("square_dt_once")
    _check_square_correctness(func, device, use_difftensor=True)
    kwargs = _make_dt_square_kwargs(device, n)
    benchmark_slang_function(device, func.bwds, **kwargs)


@pytest.mark.parametrize("n", TENSOR_SIZES)
@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_bwd_perelement_load(
    device_type: spy.DeviceType,
    n: int,
    benchmark_slang_function: BenchmarkSlangFunction,
) -> None:
    """Per-element backward using DiffTensorView load/store (atomic)."""
    device = helpers.get_torch_device(device_type)
    module = _load_module(device)
    func = module.require_function("square_dtv")
    _check_square_correctness(func, device)
    kwargs = _make_dtv_square_kwargs(n)
    benchmark_slang_function(device, func.bwds, **kwargs)


@pytest.mark.parametrize("n", TENSOR_SIZES)
@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_bwd_perelement_load_once(
    device_type: spy.DeviceType,
    n: int,
    benchmark_slang_function: BenchmarkSlangFunction,
) -> None:
    """Per-element backward using DiffTensorView loadOnce/storeOnce (no atomics)."""
    device = helpers.get_torch_device(device_type)
    module = _load_module(device)
    func = module.require_function("square_dtv_once")
    _check_square_correctness(func, device)
    kwargs = _make_dtv_square_kwargs(n)
    benchmark_slang_function(device, func.bwds, **kwargs)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
