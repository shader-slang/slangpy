# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Regression test for shader-slang/slangpy#1051.

Root cause: shader-slang/slang#12070 (Slang reverse-mode autodiff). Fixed upstream by
shader-slang/slang#12072.

Backward dispatch (``.bwds()``) of a ``[Differentiable]`` function crashes when the function
body contains a loop whose *start* value is a **runtime (non-constant) induction start**, e.g.
``for (int dx = -radius; dx <= radius; ++dx)`` where ``radius`` is a ``no_diff`` runtime param.
The forward pass is always correct, and a mathematically identical zero-based rewrite
(``for (int i = 0; i < 2*radius+1; ++i) { int dx = i - radius; ... }``) -- whose start is the
compile-time constant ``0`` -- produces correct gradients through the same call path.

The trigger is the loop start being non-constant, *not* its being negative: a constant negative
start such as ``for (int dx = -2; dx <= 2; ++dx)`` differentiates fine. The reporter's
``-radius`` was incidentally both runtime and negative. The root cause is upstream in the Slang
compiler's reverse-mode autodiff, not in SlangPy: SlangPy hands the user function verbatim to
``bwd_diff`` and does no loop/induction rewriting.

This test covers the two *constant-start* forms that must differentiate correctly -- a constant
negative start (``const_neg_start``) and the zero-based workaround (``zero_start``) -- which pin
the trigger to runtime-ness and guard the workaround. The crashing runtime-start form itself is
NOT exercised here: it is a hard native crash (SIGSEGV) that, on the CI GPU runners, lands in an
unkillable process (SlangPy's Crashpad handler / the GPU-driver fault path) and wedges the whole
"Unit Tests (Python)" step for hours rather than failing cleanly. The crash is documented in the
issues above and in ``docs/src/autodiff/autodiff.rst``; a live crash-reproducing test is not
worth hanging CI. Once the upstream fix (shader-slang/slang#12072) ships in a released wheel,
``runtime_start`` can be added here as an ordinary passing case with no subprocess scaffolding.
"""

import numpy as np
import pytest

import slangpy as spy
from slangpy import DeviceType
from slangpy.testing import helpers

# Two [Differentiable] functions with identical math and read footprint, both with a
# compile-time-constant loop start (so neither triggers #1051):
#   const_neg_start -- constant negative start (-2): shows negative is not the trigger
#   zero_start      -- zero-based workaround with an offset index
SHADER = r"""
[Differentiable]
float const_neg_start(no_diff int2 pix, IDiffTensor<float, 2> src,
                      no_diff int radius, no_diff int width)
{
    float acc = 0.0;
    [MaxIters(17)]
    for (int dx = -2; dx <= 2; ++dx)
        acc += src[uint2(clamp(pix.x + dx, 0, width - 1), pix.y)];
    return acc;
}

[Differentiable]
float zero_start(no_diff int2 pix, IDiffTensor<float, 2> src,
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
"""

# Use a non-square shape so the row/column axes cannot be confused by the oracle below.
H = 6  # rows       (pix.y, array axis 0)
W = 8  # columns    (pix.x, clamped against `width`, array axis 1)
R = 2  # radius; chosen so both functions cover the same [-2, 2] window


def _expected_grad() -> np.ndarray:
    """Independent numpy oracle for the input gradient of both loop variants.

    Each output element ``(y, x)`` sums ``src[y, clamp(x + dx, 0, W-1)]`` over ``dx`` in
    ``[-R, R]``; with the incoming result grad set to 1 everywhere, the gradient at a source
    element is the number of output elements that read it (a clamped box-filter footprint).
    """
    grad = np.zeros((H, W), dtype=np.float64)
    for y in range(H):
        for x in range(W):
            for dx in range(-R, R + 1):
                grad[y, min(max(x + dx, 0), W - 1)] += 1.0
    return grad


def _bwds_grad(device_type: DeviceType, func_name: str, src_np: np.ndarray) -> np.ndarray:
    """Run forward + ``.bwds()`` for ``func_name`` and return the input gradient as numpy."""
    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(device, func_name, SHADER)

    src = spy.Tensor.from_numpy(device, src_np).with_grads()
    out = function(spy.grid((H, W)), src, R, W)

    out = out.with_grads()
    out.grad.copy_from_numpy(np.ones((H, W), dtype=np.float32))
    function.bwds(spy.grid((H, W)), src, R, W, _result=out)

    return src.grad.to_numpy()


@pytest.mark.parametrize("func_name", ["const_neg_start", "zero_start"])
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_diff_loop_constant_start(device_type: DeviceType, func_name: str):
    """Constant loop starts differentiate correctly (#1051 does not fire on a constant start).

    ``const_neg_start`` (constant -2) shows a negative start is fine; ``zero_start`` is the
    documented workaround. Both isolate the #1051 trigger to a *runtime* (non-constant) start.
    """
    if helpers.should_skip_test_for_device(device_type):
        pytest.skip("device filtered out")
    grad = _bwds_grad(device_type, func_name, np.ones((H, W), dtype=np.float32))
    np.testing.assert_allclose(grad, _expected_grad(), rtol=1e-5, atol=1e-5)
