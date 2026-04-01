# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Unit tests for DiffTensor load_once and load_uniform methods.

Tests gradient correctness across all device types (D3D, Vulkan, CUDA).
"""

import pytest
import numpy as np
from pathlib import Path

from slangpy import DeviceType, Tensor
from slangpy.core.module import Module
from slangpy.testing import helpers


def load_module(device_type: DeviceType) -> Module:
    device = helpers.get_device(device_type)
    return Module.load_from_file(
        device,
        str(Path(__file__).parent / "test_difftensor_load_variants.slang"),
    )


# =============================================================================
# load_once: per-element f(x) = x*x, backward dx = 2*x
# Works on all device types (no wave intrinsics needed)
# =============================================================================


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_load_once_square_correctness(device_type: DeviceType) -> None:
    """load_once backward produces correct per-element gradients."""
    module = load_module(device_type)
    device = module.device

    x_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    x = Tensor.from_numpy(device, x_np).with_grads()
    result = Tensor.empty(device, shape=(5,), dtype=float).with_grads()
    assert result.grad_in is not None
    result.grad_in.storage.copy_from_numpy(np.ones(5, dtype=np.float32))

    func = module.square_load_once
    func(x, result, 0)
    func(x, result, 1)
    func(x, result, 2)
    func(x, result, 3)
    func(x, result, 4)

    # Verify forward
    result_np = result.to_numpy()
    assert np.allclose(result_np, x_np * x_np), f"Forward wrong: {result_np}"

    func.bwds(x, result, 0)
    func.bwds(x, result, 1)
    func.bwds(x, result, 2)
    func.bwds(x, result, 3)
    func.bwds(x, result, 4)

    # Verify backward: dx = 2*x
    assert x.grad_out is not None
    x_grad = x.grad_out.to_numpy()
    expected = 2.0 * x_np
    assert np.allclose(
        x_grad, expected, atol=1e-5
    ), f"Grad wrong: got {x_grad}, expected {expected}"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_load_once_matches_load(device_type: DeviceType) -> None:
    """load_once produces same gradients as load for unique-index access."""
    module = load_module(device_type)
    device = module.device

    x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    # Run with load_once
    x_once = Tensor.from_numpy(device, x_np).with_grads()
    result_once = Tensor.empty(device, shape=(3,), dtype=float).with_grads()
    assert result_once.grad_in is not None
    result_once.grad_in.storage.copy_from_numpy(np.ones(3, dtype=np.float32))
    for i in range(3):
        module.square_load_once(x_once, result_once, i)
    for i in range(3):
        module.square_load_once.bwds(x_once, result_once, i)

    # Run with load
    x_load = Tensor.from_numpy(device, x_np).with_grads()
    result_load = Tensor.empty(device, shape=(3,), dtype=float).with_grads()
    assert result_load.grad_in is not None
    result_load.grad_in.storage.copy_from_numpy(np.ones(3, dtype=np.float32))
    for i in range(3):
        module.square_load(x_load, result_load, i)
    for i in range(3):
        module.square_load.bwds(x_load, result_load, i)

    assert x_once.grad_out is not None
    assert x_load.grad_out is not None
    grad_once = x_once.grad_out.to_numpy()
    grad_load = x_load.grad_out.to_numpy()

    assert np.allclose(
        grad_once, grad_load, atol=1e-5
    ), f"Gradients differ: load_once={grad_once}, load={grad_load}"


# =============================================================================
# load_uniform: uniform y = w, backward w_grad = sum(output_grad) = n
# WaveActiveSum maps to subgroupAdd on Vulkan, works on all backends
# =============================================================================


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_load_uniform_broadcast_correctness(device_type: DeviceType) -> None:
    """load_uniform backward accumulates gradients via wave reduction."""
    module = load_module(device_type)
    device = module.device

    n = 5
    w = Tensor.from_numpy(device, np.array([3.0], dtype=np.float32)).with_grads()
    result = Tensor.empty(device, shape=(n,), dtype=float).with_grads()
    assert result.grad_in is not None
    result.grad_in.storage.copy_from_numpy(np.ones(n, dtype=np.float32))

    func = module.broadcast_load_uniform
    for i in range(n):
        func(w, result, i)

    # Verify forward: all outputs = w = 3.0
    result_np = result.to_numpy()
    assert np.allclose(result_np, 3.0), f"Forward wrong: {result_np}"

    for i in range(n):
        func.bwds(w, result, i)

    # Verify backward: w_grad = sum(output_grad) = n
    assert w.grad_out is not None
    w_grad = w.grad_out.to_numpy()
    assert np.allclose(w_grad, [float(n)], atol=1e-5), f"w_grad wrong: got {w_grad}, expected {n}"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_load_uniform_matches_load(device_type: DeviceType) -> None:
    """load_uniform produces same gradients as load for uniform access."""
    module = load_module(device_type)
    device = module.device

    n = 5

    # Run with load_uniform
    w_uniform = Tensor.from_numpy(device, np.array([3.0], dtype=np.float32)).with_grads()
    result_uniform = Tensor.empty(device, shape=(n,), dtype=float).with_grads()
    assert result_uniform.grad_in is not None
    result_uniform.grad_in.storage.copy_from_numpy(np.ones(n, dtype=np.float32))
    for i in range(n):
        module.broadcast_load_uniform(w_uniform, result_uniform, i)
    for i in range(n):
        module.broadcast_load_uniform.bwds(w_uniform, result_uniform, i)

    # Run with load
    w_load = Tensor.from_numpy(device, np.array([3.0], dtype=np.float32)).with_grads()
    result_load = Tensor.empty(device, shape=(n,), dtype=float).with_grads()
    assert result_load.grad_in is not None
    result_load.grad_in.storage.copy_from_numpy(np.ones(n, dtype=np.float32))
    for i in range(n):
        module.broadcast_load(w_load, result_load, i)
    for i in range(n):
        module.broadcast_load.bwds(w_load, result_load, i)

    assert w_uniform.grad_out is not None
    assert w_load.grad_out is not None
    grad_uniform = w_uniform.grad_out.to_numpy()
    grad_load = w_load.grad_out.to_numpy()

    assert np.allclose(
        grad_uniform, grad_load, atol=1e-5
    ), f"Gradients differ: load_uniform={grad_uniform}, load={grad_load}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
