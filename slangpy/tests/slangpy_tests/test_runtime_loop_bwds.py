# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Ensure .bwds() completes for runtime-bounded IDiffTensor stencils (#1167)."""

import functools
import os
import subprocess
import sys

import pytest
import numpy as np

import slangpy as spy
from slangpy import DeviceType
from slangpy.testing import helpers

MODULE = r"""
import slangpy;

[Differentiable]
float runtime_count(no_diff int2 pix, IDiffTensor<float, 2> src,
                    no_diff int radius, no_diff int width)
{
    float acc = 0.0;
    [MaxIters(17)]
    for (int i = 0; i < 2 * radius + 1; ++i)
    {
        int dx = i - radius;
        acc += src[uint2(clamp(pix.x + dx, 0, width - 1), pix.y)];
    }
    return acc;
}

[Differentiable]
float runtime_if_count(no_diff int2 pix, IDiffTensor<float, 2> src,
                       no_diff int radius, no_diff int width)
{
    float acc = 0.0;
    [MaxIters(17)]
    for (int i = 0; i < 17; ++i)
    {
        if (i < 2 * radius + 1)
        {
            int dx = i - radius;
            acc += src[uint2(clamp(pix.x + dx, 0, width - 1), pix.y)];
        }
    }
    return acc;
}

[Differentiable]
float runtime_break_count(no_diff int2 pix, IDiffTensor<float, 2> src,
                          no_diff int radius, no_diff int width)
{
    float acc = 0.0;
    [MaxIters(17)]
    for (int i = 0; i < 17; ++i)
    {
        if (i >= 2 * radius + 1)
            break;
        int dx = i - radius;
        acc += src[uint2(clamp(pix.x + dx, 0, width - 1), pix.y)];
    }
    return acc;
}
"""

HEIGHT = 8
WIDTH = 8
RADIUS = 2

# Generous for a Debug build on a loaded CI runner; the compile normally takes a few seconds.
COMPILE_TIMEOUT_S = 120

# Compiles the backward kernel on the CPU device, so the child never opens a GPU context.
# The CPU target cannot emit the float atomics, so bwds() raises once compilation is done;
# that specific error is the expected outcome, anything else fails the guard.
COMPILE_CHILD = r"""
import sys
import numpy as np
import slangpy as spy

device = spy.create_device(type=spy.DeviceType.cpu)
module = spy.Module.load_from_source(
    device,
    "runtime_loop_bwds_compile",
    sys.stdin.read(),
    options={"pipeline_compilation_policy": spy.PipelineCompilationPolicy.immediate},
)
src = spy.Tensor.from_numpy(device, np.zeros((8, 8), dtype=np.float32)).with_grads()
out = spy.Tensor.empty(device, shape=(8, 8), dtype=float).with_grads()
try:
    getattr(module, sys.argv[1]).bwds(spy.grid((8, 8)), src, 2, 8, _result=out)
except Exception as e:
    if "unexpected IR opcode during code emit" not in str(e):
        raise
"""


def stencil_reference(src: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Forward value and d(sum of outputs)/d(src) of the clamped horizontal box stencil."""
    height, width = src.shape
    out = np.zeros_like(src)
    grad = np.zeros_like(src)
    for y in range(height):
        for x in range(width):
            for dx in range(-radius, radius + 1):
                sx = min(max(x + dx, 0), width - 1)
                out[y, x] += src[y, sx]
                grad[y, sx] += 1.0
    return out, grad


@functools.cache
def bwds_compile_terminates(func_name: str) -> bool:
    """Compile the backward kernel of func_name in a child process; False if it times out."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", COMPILE_CHILD, func_name],
            input=MODULE,
            capture_output=True,
            text=True,
            timeout=COMPILE_TIMEOUT_S,
            env={**os.environ, "SLANGPY_DEVICE_TYPE_OVERRIDE": ""},
        )
    except subprocess.TimeoutExpired:
        return False
    assert result.returncode == 0, result.stderr
    return True


@pytest.mark.parametrize("func_name", ["runtime_count", "runtime_if_count", "runtime_break_count"])
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_runtime_loop_bwds(
    device_type: DeviceType, func_name: str, request: pytest.FixtureRequest
) -> None:
    if not bwds_compile_terminates(func_name):
        pytest.fail(f"{func_name} backward still compiling after {COMPILE_TIMEOUT_S}s (#1167)")

    device = helpers.get_device(device_type)
    module = helpers.create_module(device, MODULE)
    func = getattr(module, func_name)

    src_np = np.arange(HEIGHT * WIDTH, dtype=np.float32).reshape(HEIGHT, WIDTH)
    expected_out, expected_grad = stencil_reference(src_np, RADIUS)

    src = spy.Tensor.from_numpy(device, src_np).with_grads()
    out = func(spy.grid((HEIGHT, WIDTH)), src, RADIUS, WIDTH).with_grads()
    assert np.allclose(out.to_numpy(), expected_out)

    out.grad.copy_from_numpy(np.ones((HEIGHT, WIDTH), dtype=np.float32))
    func.bwds(spy.grid((HEIGHT, WIDTH)), src, RADIUS, WIDTH, _result=out)

    grad = src.grad.to_numpy()
    # Marked only here so a hang caught by the compile guard still fails on Vulkan.
    if device_type == DeviceType.vulkan:
        request.node.add_marker(
            pytest.mark.xfail(
                strict=False,
                reason=f"https://github.com/shader-slang/slangpy/issues/1227 ({device.info.adapter_name})",
            )
        )
    assert np.array_equal(
        grad, expected_grad
    ), f"{device.info.adapter_name}: grad=\n{grad}\nexpected=\n{expected_grad}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
