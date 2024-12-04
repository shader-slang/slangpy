# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

import slangpy.tests.helpers as helpers
from slangpy.backend import DeviceType, float3
from slangpy.types import NDDifferentiableBuffer, diffPair

# pyright: reportOptionalMemberAccess=false, reportArgumentType=false

POLYNOMIAL_OUT_PARAM = r"""
[Differentiable]
void polynomial(float a, float b, out float result) {
    result = a * a + b + 1;
}
"""

POLYNOMIAL_WITH_RETURN_VALUE_DOT_SLANG = r"""
[Differentiable]
float polynomial(float a, float b) {
    return a * a + b + 1;
}
"""

POLYNOMIAL_RETURN_VALUE_ND = r"""
float polynomial(float a, float b) {
    return a * a + b + 1;
}
"""

POLYNOMIAL_V3 = r"""
[Differentiable]
float3 polynomial(float3 a, float3 b) {
    return a * a + b + 1;
}
"""


def python_eval_polynomial(a: float, b: float) -> float:
    return a * a + b + 1


def python_eval_polynomial_a_deriv(a: float, b: float) -> float:
    return 2 * a


def python_eval_polynomial_b_deriv(a: float, b: float) -> float:
    return 1


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_call_none_differentiable(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device, "polynomial", POLYNOMIAL_RETURN_VALUE_ND
    )

    a = 5.0
    b = 10.0
    res = function(a, b)
    assert res == python_eval_polynomial(a, b)

    with pytest.raises(ValueError, match="Could not call function 'polynomial': Function is not differentiable"):
        function.bwds_diff(a, b, res)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_call_with_none_diff_scalars(device_type: DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device, "polynomial", POLYNOMIAL_WITH_RETURN_VALUE_DOT_SLANG
    )

    a = 5.0
    b = 10.0
    res = function(a, b)
    assert res == python_eval_polynomial(a, b)

    function.bwds_diff(a, b, res)


@pytest.mark.skip("Awaiting auto-diff changes")
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_call_with_diff_scalars(device_type: DeviceType):

    device = helpers.get_device(device_type)
    kernel_eval_polynomial = helpers.create_function_from_module(
        device, "polynomial", POLYNOMIAL_WITH_RETURN_VALUE_DOT_SLANG
    )

    a = 5.0
    b = 10.0
    res = kernel_eval_polynomial(a, b)
    expected = python_eval_polynomial(a, b)
    assert res == expected

    res_diff = diffPair(d=1.0)

    a_diff = diffPair(p=a)
    kernel_eval_polynomial.bwds_diff(a_diff, b, res_diff)
    exprected_grad = python_eval_polynomial_a_deriv(a, b)
    assert a_diff.grad == exprected_grad

    b_diff = diffPair(p=b)
    kernel_eval_polynomial.bwds_diff(a, b_diff, res_diff)
    exprected_grad = python_eval_polynomial_b_deriv(a, b)
    assert b_diff.grad == exprected_grad

    a_diff = diffPair(p=a)
    b_diff = diffPair(p=b)
    kernel_eval_polynomial.bwds_diff(a_diff, b_diff, res_diff)
    exprected_grad = python_eval_polynomial_a_deriv(a, b)
    assert a_diff.grad == exprected_grad
    exprected_grad = python_eval_polynomial_b_deriv(a, b)
    assert b_diff.grad == exprected_grad


@pytest.mark.skip("Awaiting auto-diff changes")
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_call_with_diff_pairs(device_type: DeviceType):

    device = helpers.get_device(device_type)
    kernel_eval_polynomial = helpers.create_function_from_module(
        device, "polynomial", POLYNOMIAL_WITH_RETURN_VALUE_DOT_SLANG
    )

    a = diffPair(5.0)
    b = diffPair(10.0)
    res = kernel_eval_polynomial(a, b)
    expected = python_eval_polynomial(a.primal, b.primal)
    assert res == expected

    kernel_eval_polynomial.bwds_diff(a, b, diffPair(d=1.0))
    exprected_grad = python_eval_polynomial_a_deriv(a.primal, b.primal)
    assert a.grad == exprected_grad
    exprected_grad = python_eval_polynomial_b_deriv(a.primal, b.primal)
    assert b.grad == exprected_grad


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_call_with_buffers(device_type: DeviceType):

    device = helpers.get_device(device_type)
    kernel_eval_polynomial = helpers.create_function_from_module(
        device, "polynomial", POLYNOMIAL_WITH_RETURN_VALUE_DOT_SLANG
    )

    a = NDDifferentiableBuffer(
        element_count=32,
        device=device,
        element_type=float,
        requires_grad=True,
    )
    a.buffer.from_numpy(np.random.rand(32).astype(np.float32))

    b = NDDifferentiableBuffer(
        element_count=32,
        device=device,
        element_type=float,
        requires_grad=True,
    )
    b.buffer.from_numpy(np.random.rand(32).astype(np.float32))

    res: NDDifferentiableBuffer = kernel_eval_polynomial(a, b)

    a_data = a.buffer.to_numpy().view(np.float32)
    b_data = b.buffer.to_numpy().view(np.float32)
    expected = python_eval_polynomial(a_data, b_data)
    res_data = res.buffer.to_numpy().view(np.float32)

    assert np.allclose(res_data, expected)

    res.grad.buffer.from_numpy(np.ones(res.shape.as_tuple(), dtype=np.float32))

    kernel_eval_polynomial.bwds_diff(a, b, res)
    a_grad_data = a.grad.buffer.to_numpy().view(np.float32)
    b_grad_data = b.grad.buffer.to_numpy().view(np.float32)

    exprected_grad = python_eval_polynomial_a_deriv(a_data, b_data)
    assert np.allclose(a_grad_data, exprected_grad)

    exprected_grad = python_eval_polynomial_b_deriv(a_data, b_data)
    assert np.allclose(b_grad_data, exprected_grad)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_vec3_call_with_buffers(device_type: DeviceType):

    device = helpers.get_device(device_type)
    kernel_eval_polynomial = helpers.create_function_from_module(
        device, "polynomial", POLYNOMIAL_V3
    )

    a = NDDifferentiableBuffer(
        element_count=32,
        device=device,
        element_type=float3,
        requires_grad=True,
    )
    a.buffer.from_numpy(np.random.rand(32*3).astype(np.float32))

    b = NDDifferentiableBuffer(
        element_count=32,
        device=device,
        element_type=float3,
        requires_grad=True,
    )
    b.buffer.from_numpy(np.random.rand(32*3).astype(np.float32))

    res: NDDifferentiableBuffer = kernel_eval_polynomial(a, b)

    a_data = a.buffer.to_numpy().view(np.float32).reshape(-1, 3)
    b_data = b.buffer.to_numpy().view(np.float32).reshape(-1, 3)
    expected = python_eval_polynomial(a_data, b_data)
    res_data = res.buffer.to_numpy().view(np.float32).reshape(-1, 3)

    assert np.allclose(res_data, expected)

    res.grad.buffer.from_numpy(np.ones(32*3, dtype=np.float32))

    kernel_eval_polynomial.bwds_diff(a, b, res)
    a_grad_data = a.grad.buffer.to_numpy().view(np.float32).reshape(-1, 3)
    b_grad_data = b.grad.buffer.to_numpy().view(np.float32).reshape(-1, 3)

    exprected_grad = python_eval_polynomial_a_deriv(a_data, b_data)
    assert np.allclose(a_grad_data, exprected_grad)

    exprected_grad = python_eval_polynomial_b_deriv(a_data, b_data)
    assert np.allclose(b_grad_data, exprected_grad)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_vec3_call_with_buffers_soa(device_type: DeviceType):

    device = helpers.get_device(device_type)
    kernel_eval_polynomial = helpers.create_function_from_module(
        device, "polynomial", POLYNOMIAL_V3
    )

    a_x = NDDifferentiableBuffer(
        element_count=32,
        device=device,
        element_type=float,
        requires_grad=True,
    )
    a_x.buffer.from_numpy(np.random.rand(32).astype(np.float32))

    a_y = NDDifferentiableBuffer(
        element_count=32,
        device=device,
        element_type=float,
        requires_grad=True,
    )
    a_y.buffer.from_numpy(np.random.rand(32).astype(np.float32))

    a_z = NDDifferentiableBuffer(
        element_count=32,
        device=device,
        element_type=float,
        requires_grad=True,
    )
    a_z.buffer.from_numpy(np.random.rand(32).astype(np.float32))

    b = NDDifferentiableBuffer(
        element_count=32,
        device=device,
        element_type=float3,
        requires_grad=True,
    )
    b.buffer.from_numpy(np.random.rand(32*3).astype(np.float32))

    res: NDDifferentiableBuffer = kernel_eval_polynomial({
        'x': a_x,
        'y': a_y,
        'z': a_z
    }, b)

    a_x_data = a_x.buffer.to_numpy().view(np.float32).reshape(-1, 1)
    a_y_data = a_y.buffer.to_numpy().view(np.float32).reshape(-1, 1)
    a_z_data = a_z.buffer.to_numpy().view(np.float32).reshape(-1, 1)
    a_data = np.column_stack((a_x_data, a_y_data, a_z_data))
    b_data = b.buffer.to_numpy().view(np.float32).reshape(-1, 3)
    expected = python_eval_polynomial(a_data, b_data)
    res_data = res.buffer.to_numpy().view(np.float32).reshape(-1, 3)

    assert np.allclose(res_data, expected)

    res.grad.buffer.from_numpy(np.ones(32*3, dtype=np.float32))

    kernel_eval_polynomial.bwds_diff({
        'x': a_x,
        'y': a_y,
        'z': a_z
    }, b, res)
    a_x_grad_data = a_x.grad.buffer.to_numpy().view(np.float32).reshape(-1, 1)
    a_y_grad_data = a_y.grad.buffer.to_numpy().view(np.float32).reshape(-1, 1)
    a_z_grad_data = a_z.grad.buffer.to_numpy().view(np.float32).reshape(-1, 1)
    a_grad_data = np.column_stack((a_x_grad_data, a_y_grad_data, a_z_grad_data))
    b_grad_data = b.grad.buffer.to_numpy().view(np.float32).reshape(-1, 3)

    exprected_grad = python_eval_polynomial_a_deriv(a_data, b_data)
    assert np.allclose(a_grad_data, exprected_grad)

    exprected_grad = python_eval_polynomial_b_deriv(a_data, b_data)
    assert np.allclose(b_grad_data, exprected_grad)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
