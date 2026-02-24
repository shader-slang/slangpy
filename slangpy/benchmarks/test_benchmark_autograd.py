# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
from typing import Any, Optional
import pytest
import numpy as np

import slangpy as spy
from slangpy.testing import helpers
from slangpy.testing.benchmark import BenchmarkPythonFunction
from slangpy.core.native import NativeTorchTensorDiffPair

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
SUB_ITERATIONS = 20000
WARMUPS = 10

# ITERATIONS = 1
# SUB_ITERATIONS = 1
# WARMUPS = 1

# =============================================================================
# Autograd benchmarks
#
# All benchmarks evaluate the polynomial f(x) = a*x^2 + b*x + c and compute
# gradients w.r.t. x via backward(). The analytic derivative is 2*a*x + b.
#
# Four implementations are compared:
#   1. Pure PyTorch (torch ops + autograd)
#   2. slang-torch kernel + manual torch.autograd.Function
#   3. SlangPy function + manual torch.autograd.Function calling .bwds()
#   4. SlangPy automatic autograd (requires_grad=True torch tensors)
# =============================================================================

RUN_PURE_TORCH_BENCHMARK = False
RUN_SLANGTORCH_BENCHMARK = False
RUN_SLANGPY_MANUAL_HOOK_BENCHMARK = True
RUN_SLANGPY_AUTOMATIC_BENCHMARK = False

AUTOGRAD_TENSOR_SIZE = 32


@pytest.mark.parametrize("device_type", [spy.DeviceType.cuda])
def test_autograd_pure_torch(
    device_type: spy.DeviceType, benchmark_python_function: BenchmarkPythonFunction
) -> None:
    """Polynomial forward + backward using pure PyTorch ops."""
    if not RUN_PURE_TORCH_BENCHMARK:
        pytest.skip("Pure torch benchmark is not enabled")
    if not HAS_TORCH:
        pytest.skip("PyTorch is not installed")

    device = helpers.get_torch_device(device_type)
    a_val = 2.0
    b_val = 4.0
    c_val = 1.0

    x = torch.randn(AUTOGRAD_TENSOR_SIZE, dtype=torch.float32, device="cuda", requires_grad=True)

    def run() -> None:
        y = a_val * x * x + b_val * x + c_val
        y.backward(torch.ones_like(y))
        x.grad.zero_()  # type: ignore[union-attr]

    benchmark_python_function(
        device,
        run,
        iterations=ITERATIONS,
        warmup_iterations=WARMUPS,
        sub_iterations=SUB_ITERATIONS,
        sleeps=SLEEPS,
    )


@pytest.mark.parametrize("device_type", [spy.DeviceType.cuda])
def test_autograd_slangtorch(
    device_type: spy.DeviceType, benchmark_python_function: BenchmarkPythonFunction
) -> None:
    """Polynomial forward + backward using slang-torch kernels with a manual
    torch.autograd.Function, following the pattern from the slang-torch
    soft-rasterizer example."""
    if not RUN_SLANGTORCH_BENCHMARK:
        pytest.skip("slang-torch benchmark is not enabled")
    if not HAS_TORCH:
        pytest.skip("PyTorch is not installed")
    if not HAS_SLANGTORCH:
        pytest.skip("slang-torch is not installed")

    device = helpers.get_torch_device(device_type)
    st_module = slangtorch.loadModule(
        Path(__file__).parent / "test_benchmark_autograd_slangtorch.slang"
    )

    a_val = 2.0
    b_val = 4.0
    c_val = 1.0
    n = AUTOGRAD_TENSOR_SIZE

    x = torch.randn(n, dtype=torch.float32, device="cuda", requires_grad=True)
    ones = torch.ones_like(x)

    class PolynomialSlangTorch(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, a: float, b: float, c: float, x: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            st_module.polynomial_fwd(a=a, b=b, c=c, x=x, result=result).launchRaw(
                blockSize=(32, 1, 1), gridSize=((n + 31) // 32, 1, 1)
            )
            ctx.save_for_backward(x)
            ctx.a = a
            ctx.b = b
            ctx.c = c
            return result

        @staticmethod
        def backward(
            ctx: Any, grad_output: torch.Tensor
        ) -> tuple[None, None, None, Optional[torch.Tensor]]:
            (x,) = ctx.saved_tensors
            grad_x = torch.zeros_like(x)
            st_module.polynomial_bwd(
                a=ctx.a, b=ctx.b, c=ctx.c, x=x, grad_output=grad_output, grad_x=grad_x
            ).launchRaw(blockSize=(32, 1, 1), gridSize=((n + 31) // 32, 1, 1))
            return None, None, None, grad_x

    def run() -> None:
        y = PolynomialSlangTorch.apply(a_val, b_val, c_val, x)
        y.backward(ones)  # type: ignore[union-attr]

    benchmark_python_function(
        device,
        run,
        iterations=ITERATIONS,
        warmup_iterations=WARMUPS,
        sub_iterations=SUB_ITERATIONS,
        sleeps=SLEEPS,
    )


@pytest.mark.parametrize("device_type", [spy.DeviceType.cuda])
def test_autograd_slangpy_manual_hook(
    device_type: spy.DeviceType, benchmark_python_function: BenchmarkPythonFunction
) -> None:
    """Polynomial forward + backward using a SlangPy function with a manually
    written torch.autograd.Function that calls function.bwds()."""
    if not RUN_SLANGPY_MANUAL_HOOK_BENCHMARK:
        pytest.skip("SlangPy manual hook benchmark is not enabled")
    if not HAS_TORCH:
        pytest.skip("PyTorch is not installed")

    device = helpers.get_torch_device(device_type)
    module = spy.Module(device.load_module("test_benchmark_autograd.slang"))
    poly_func = module.require_function("polynomial")

    a_val = 2.0
    b_val = 4.0
    c_val = 1.0
    n = AUTOGRAD_TENSOR_SIZE

    x = torch.randn(n, dtype=torch.float32, device="cuda", requires_grad=True)
    ones = torch.ones_like(x)

    class PolynomialSlangPyManual(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, a: float, b: float, c: float, x: torch.Tensor) -> torch.Tensor:
            # Run the forward SlangPy kernel with plain (non-grad) tensors
            # to avoid triggering the automatic autograd path
            # x = x.detach()
            result = torch.empty_like(x)
            poly_func(a, b, c, x, _result=result)
            ctx.save_for_backward(x)
            ctx.a = a
            ctx.b = b
            ctx.c = c
            return result

        @staticmethod
        def backward(
            ctx: Any, grad_output: torch.Tensor
        ) -> tuple[None, None, None, Optional[torch.Tensor]]:
            (x,) = ctx.saved_tensors
            # Build diff-pair tensors for the backward call:
            #   - x is an input (read-only in forward) -> we want gradients for it
            #   - result is an output -> we provide upstream gradients

            x_pair = NativeTorchTensorDiffPair(x, torch.zeros_like(x), 0, True)
            result_pair = NativeTorchTensorDiffPair(None, grad_output, 1, False)
            poly_func.bwds(ctx.a, ctx.b, ctx.c, x_pair, _result=result_pair)
            grad: Optional[torch.Tensor] = x_pair.grad  # type: ignore[assignment]
            return None, None, None, grad

    def run() -> None:
        y = PolynomialSlangPyManual.apply(a_val, b_val, c_val, x)
        y.backward(ones)  # type: ignore[union-attr]

    benchmark_python_function(
        device,
        run,
        iterations=ITERATIONS,
        warmup_iterations=WARMUPS,
        sub_iterations=SUB_ITERATIONS,
        sleeps=SLEEPS,
    )


@pytest.mark.parametrize("device_type", [spy.DeviceType.cuda])
def test_autograd_slangpy_automatic(
    device_type: spy.DeviceType, benchmark_python_function: BenchmarkPythonFunction
) -> None:
    """Polynomial forward + backward using SlangPy's built-in automatic
    autograd integration (pass requires_grad=True torch tensors directly)."""
    if not RUN_SLANGPY_AUTOMATIC_BENCHMARK:
        pytest.skip("SlangPy automatic benchmark is not enabled")
    if not HAS_TORCH:
        pytest.skip("PyTorch is not installed")

    device = helpers.get_torch_device(device_type)
    module = spy.Module(device.load_module("test_benchmark_autograd.slang"))
    poly_func = module.require_function("polynomial")

    a_val = 2.0
    b_val = 4.0
    c_val = 1.0
    n = AUTOGRAD_TENSOR_SIZE

    x = torch.randn(n, dtype=torch.float32, device="cuda", requires_grad=True)

    def run() -> None:
        result = poly_func(a_val, b_val, c_val, x)
        result.backward(torch.ones_like(result))
        x.grad.zero_()  # type: ignore[union-attr]

    benchmark_python_function(
        device,
        run,
        iterations=ITERATIONS,
        warmup_iterations=WARMUPS,
        sub_iterations=SUB_ITERATIONS,
        sleeps=SLEEPS,
    )


if __name__ == "__main__":
    input("Press Enter to run the tests...")
    pytest.main([__file__, "-v", "-s"])
