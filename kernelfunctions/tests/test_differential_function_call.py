import pytest
import sgl
import kernelfunctions.tests.helpers as helpers
from kernelfunctions.utils import diffPair

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


def python_eval_polynomial(a: float, b: float) -> float:
    return a * a + b + 1


def python_eval_polynomial_a_deriv(a: float, b: float) -> float:
    return 2 * a


def python_eval_polynomial_b_deriv(a: float, b: float) -> float:
    return 1


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_call_none_differentiable(device_type: sgl.DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device, "polynomial", POLYNOMIAL_RETURN_VALUE_ND
    )

    a = 5.0
    b = 10.0
    res = function(a, b)
    assert res == python_eval_polynomial(a, b)

    with pytest.raises(ValueError, match="No matching overload found"):
        function.backwards(a, b, res)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_call_with_none_diff_scalars(device_type: sgl.DeviceType):

    device = helpers.get_device(device_type)
    function = helpers.create_function_from_module(
        device, "polynomial", POLYNOMIAL_WITH_RETURN_VALUE_DOT_SLANG
    )

    a = 5.0
    b = 10.0
    res = function(a, b)
    assert res == python_eval_polynomial(a, b)

    function.backwards(a, b, res)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_call_with_diff_scalars(device_type: sgl.DeviceType):

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
    kernel_eval_polynomial.backwards(a_diff, b, res_diff)
    exprected_grad = python_eval_polynomial_a_deriv(a, b)
    assert a_diff.grad == exprected_grad

    b_diff = diffPair(p=b)
    kernel_eval_polynomial.backwards(a, b_diff, res_diff)
    exprected_grad = python_eval_polynomial_b_deriv(a, b)
    assert b_diff.grad == exprected_grad

    a_diff = diffPair(p=a)
    b_diff = diffPair(p=b)
    kernel_eval_polynomial.backwards(a_diff, b_diff, res_diff)
    exprected_grad = python_eval_polynomial_a_deriv(a, b)
    assert a_diff.grad == exprected_grad
    exprected_grad = python_eval_polynomial_b_deriv(a, b)
    assert b_diff.grad == exprected_grad


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_call_with_diff_pairs(device_type: sgl.DeviceType):

    device = helpers.get_device(device_type)
    kernel_eval_polynomial = helpers.create_function_from_module(
        device, "polynomial", POLYNOMIAL_WITH_RETURN_VALUE_DOT_SLANG
    )

    a = diffPair(5.0)
    b = diffPair(10.0)
    res = kernel_eval_polynomial(a, b)
    expected = python_eval_polynomial(a.primal, b.primal)
    assert res == expected

    kernel_eval_polynomial.backwards(a, b, diffPair(d=1.0))
    exprected_grad = python_eval_polynomial_a_deriv(a.primal, b.primal)
    assert a.grad == exprected_grad
    exprected_grad = python_eval_polynomial_b_deriv(a.primal, b.primal)
    assert b.grad == exprected_grad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
