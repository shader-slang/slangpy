# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Regression test for slangpy#639.

A function with [BackwardDerivative] that takes an ITensor parameter would
crash the Slang compiler's type-flow specialization pass during the backward
pass compilation. The crash occurred in analyzeExtractExistentialWitnessTable
when the operand info was not an IRTaggedUnionType.
"""
import pytest
import numpy as np

import slangpy as spy
from slangpy import DeviceType
from slangpy.types import Tensor
from slangpy.testing import helpers

# pyright: reportOptionalMemberAccess=false, reportArgumentType=false

ITENSOR_CUSTOM_BACKWARD = r"""
[Differentiable]
float process_element(int coord, ITensor<float, 1> x) {
    return x[coord];
}

[BackwardDerivative(simple_bwd)]
float simple(int coord, ITensor<float, 1> x) {
    return process_element(coord, x);
}

void simple_bwd(int coord, ITensor<float, 1> x, float df) {
    bwd_diff(process_element)(coord, x, df);
}

[Differentiable]
float simple_auto(int coord, ITensor<float, 1> x) {
    return process_element(coord, x);
}
"""


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_forward_auto_diff(device_type: DeviceType):
    """Forward pass with [Differentiable] should work."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "simple_auto", ITENSOR_CUSTOM_BACKWARD)

    N = 8
    xt = Tensor.from_numpy(device, np.arange(N, dtype=np.float32))
    result = func(coord=spy.grid((N,)), x=xt)
    result_np = result.to_numpy().view(np.float32)
    np.testing.assert_allclose(result_np, np.arange(N, dtype=np.float32))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_forward_custom_backward(device_type: DeviceType):
    """Forward pass with [BackwardDerivative] should work (slangpy#639)."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "simple", ITENSOR_CUSTOM_BACKWARD)

    N = 8
    xt = Tensor.from_numpy(device, np.arange(N, dtype=np.float32))
    result = func(coord=spy.grid((N,)), x=xt)
    result_np = result.to_numpy().view(np.float32)
    np.testing.assert_allclose(result_np, np.arange(N, dtype=np.float32))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_backward_auto_diff(device_type: DeviceType):
    """Backward pass with [Differentiable] should work."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(
        device, "simple_auto", ITENSOR_CUSTOM_BACKWARD
    ).return_type(Tensor)

    N = 8
    xt = Tensor.from_numpy(device, np.arange(N, dtype=np.float32)).with_grads(zero=True)
    result = func(coord=spy.grid((N,)), x=xt)
    result = result.with_grads()
    result.grad.storage.copy_from_numpy(np.ones(N, dtype=np.float32))
    func.bwds(coord=spy.grid((N,)), x=xt, _result=result)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_backward_custom_backward(device_type: DeviceType):
    """Backward pass with [BackwardDerivative] should work (slangpy#639)."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(
        device, "simple", ITENSOR_CUSTOM_BACKWARD
    ).return_type(Tensor)

    N = 8
    xt = Tensor.from_numpy(device, np.arange(N, dtype=np.float32)).with_grads(zero=True)
    result = func(coord=spy.grid((N,)), x=xt)
    result = result.with_grads()
    result.grad.storage.copy_from_numpy(np.ones(N, dtype=np.float32))
    func.bwds(coord=spy.grid((N,)), x=xt, _result=result)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
