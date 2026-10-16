# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Ensure .bwds() completes for runtime-bounded IDiffTensor stencils (#1167)."""

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


@pytest.mark.parametrize("func_name", ["runtime_count", "runtime_if_count", "runtime_break_count"])
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_runtime_loop_bwds(device_type: DeviceType, func_name: str) -> None:
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
    assert grad.sum() == 320.0
    assert np.array_equal(grad, expected_grad), f"grad=\n{grad}\nexpected=\n{expected_grad}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
