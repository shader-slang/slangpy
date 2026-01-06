# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
from typing import Any
import pytest
import numpy as np

import slangpy as spy
from slangpy.testing import helpers
from slangpy.testing.benchmark import BenchmarkPythonFunction

HAS_TORCH = False
try:
    import torch

    HAS_TORCH = True
except ImportError:
    pass

HAS_SLANGTORCH = False
try:
    import slangtorch

    HAS_SLANGTORCH = True
except ImportError:
    pass

SLEEPS = True
ITERATIONS = 10
SUB_ITERATIONS = 2000
WARMUPS = 10
COUNTS = [1, 6]

RUN_SLANGTORCH_BENCHMARK = True
RUN_PURE_TORCH_BENCHMARK = True
RUN_TORCH_TENSOR_BENCHMARK = True
RUN_NATIVE_TENSOR_BENCHMARK = True

# ITERATIONS = 1
# SUB_ITERATIONS = 1
# WARMUPS = 1


@pytest.mark.parametrize("device_type", [spy.DeviceType.cuda])
@pytest.mark.parametrize("count", COUNTS)
def test_tensor_sum_torch(
    device_type: spy.DeviceType, count: int, benchmark_python_function: BenchmarkPythonFunction
):
    if not RUN_TORCH_TENSOR_BENCHMARK:
        pytest.skip("Torch tensor benchmark is not enabled")
    if not HAS_TORCH:
        pytest.skip("PyTorch is not installed")

    device = helpers.get_torch_device(device_type)
    inputs = [np.random.rand(1024, 1024).astype(np.float32) for _ in range(count)]
    result_tensor = torch.empty((1024, 1024), dtype=torch.float32, device="cuda")

    args: dict[str, Any] = {
        f"tensor_{i}": torch.from_numpy(input).cuda() for i, input in enumerate(inputs)
    }
    args["_result"] = result_tensor

    module = spy.Module(device.load_module("test_benchmark_tensor.slang"))
    func = module.require_function(f"sum")

    def tensor_addition():
        func(**args)

    benchmark_python_function(
        device,
        tensor_addition,
        iterations=ITERATIONS,
        warmup_iterations=WARMUPS,
        sub_iterations=SUB_ITERATIONS,
        sleeps=SLEEPS,
    )


@pytest.mark.parametrize("device_type", [spy.DeviceType.cuda])
@pytest.mark.parametrize("count", COUNTS)
def test_tensor_sum_slangtorch(
    device_type: spy.DeviceType, count: int, benchmark_python_function: BenchmarkPythonFunction
):
    if not RUN_SLANGTORCH_BENCHMARK:
        pytest.skip("slang-torch benchmark is not enabled")
    if not HAS_TORCH:
        pytest.skip("PyTorch is not installed")
    if not HAS_SLANGTORCH:
        pytest.skip("slang-torch is not installed")

    device = helpers.get_torch_device(device_type)
    module = slangtorch.loadModule(Path(__file__).parent / "test_benchmark_tensor_slangtorch.slang")
    inputs = [np.random.rand(1024, 1024).astype(np.float32) for _ in range(count)]
    result_tensor = torch.empty((1024, 1024), dtype=torch.float32, device="cuda")

    args: dict[str, Any] = {
        f"tensor_{i}": torch.from_numpy(input).cuda() for i, input in enumerate(inputs)
    }
    args["result"] = result_tensor

    func = getattr(module, f"sum_slangtorch_{count}")

    def tensor_addition():
        func(**args).launchRaw(blockSize=(32, 1, 1), gridSize=(64, 1, 1))

    benchmark_python_function(
        device,
        tensor_addition,
        iterations=ITERATIONS,
        warmup_iterations=WARMUPS,
        sub_iterations=SUB_ITERATIONS,
        sleeps=SLEEPS,
    )


@pytest.mark.parametrize("device_type", [spy.DeviceType.cuda])
@pytest.mark.parametrize("count", COUNTS)
def test_tensor_sum(
    device_type: spy.DeviceType, count: int, benchmark_python_function: BenchmarkPythonFunction
):
    if not RUN_NATIVE_TENSOR_BENCHMARK:
        pytest.skip("Native tensor benchmark is not enabled")

    device = helpers.get_torch_device(device_type)
    inputs = [np.random.rand(1024, 1024).astype(np.float32) for _ in range(count)]
    result_tensor = spy.Tensor.empty(device, shape=(1024, 1024), dtype=float)

    args: dict[str, Any] = {
        f"tensor_{i}": spy.Tensor.from_numpy(device, input) for i, input in enumerate(inputs)
    }
    args["_result"] = result_tensor

    module = spy.Module(device.load_module("test_benchmark_tensor.slang"))
    func = module.require_function(f"sum")

    def tensor_addition():
        func(**args)

    benchmark_python_function(
        device,
        tensor_addition,
        iterations=ITERATIONS,
        warmup_iterations=WARMUPS,
        sub_iterations=SUB_ITERATIONS,
        sleeps=SLEEPS,
    )


@pytest.mark.parametrize("device_type", [spy.DeviceType.cuda])
@pytest.mark.parametrize("count", COUNTS)
def test_tensor_sum_pure_torch(
    device_type: spy.DeviceType, count: int, benchmark_python_function: BenchmarkPythonFunction
):
    if not RUN_PURE_TORCH_BENCHMARK:
        pytest.skip("Pure Torch benchmark is not enabled")
    if not HAS_TORCH:
        pytest.skip("PyTorch is not installed")

    device = helpers.get_torch_device(device_type)
    inputs = [np.random.rand(1, 1).astype(np.float32) for _ in range(count)]
    result_tensor = torch.empty((1, 1), dtype=torch.float32, device="cuda")

    args: dict[str, Any] = {
        f"tensor_{i}": torch.from_numpy(input).cuda() for i, input in enumerate(inputs)
    }
    args["_result"] = result_tensor

    module = spy.Module(device.load_module("test_benchmark_tensor.slang"))
    func = module.require_function(f"sum")

    def tensor_addition():
        for i in range(count):
            args["_result"] += args[f"tensor_{i}"]

    benchmark_python_function(
        device,
        tensor_addition,
        iterations=ITERATIONS,
        warmup_iterations=WARMUPS,
        sub_iterations=SUB_ITERATIONS,
        sleeps=SLEEPS,
    )


if __name__ == "__main__":
    input("Press Enter to run the tests...")
    pytest.main([__file__, "-v", "-s"])
