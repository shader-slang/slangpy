# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Tests targeting coverage gaps in torchtensormarshall.py.

Exercises:
- NativeTorchTensorDiffPair factory path in create_torch_tensor_marshall (lines 269-314)
- TorchTensorMarshall properties: torch_dtype, slang_dtype, repr (lines 162, 166, 168)
- Error paths for unsupported types/dtypes (lines 114, 260, 287, 319, 330)
- hash_torch_tensor / hash_torch_diff_pair (lines 334, 338)
- build_shader_object with gradients raises NotImplementedError (line 240)
- Internal conversion helpers (lines 62, 70)
"""

import sys
import pytest

from slangpy import DeviceType
from slangpy.testing import helpers

if sys.platform == "darwin":
    pytest.skip("Torch tests require CUDA", allow_module_level=True)

try:
    import torch
except ImportError:
    pytest.skip("PyTorch not installed", allow_module_level=True)

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

from slangpy.torchintegration import diff_pair
import slangpy.torchintegration.torchtensormarshall as ttm

CUDA_TYPES = [t for t in helpers.DEFAULT_DEVICE_TYPES if t == DeviceType.cuda]
if not CUDA_TYPES:
    pytest.skip("No CUDA device available", allow_module_level=True)

SCALE_SHADER = r"""
void scale(float a, float factor, out float result) { result = a * factor; }
"""


def _get_layout(device_type: DeviceType):
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "scale", SCALE_SHADER)
    return func.module.layout


# ============================================================================
# DiffPair factory path (lines 269-314)
# ============================================================================


@pytest.mark.parametrize("device_type", CUDA_TYPES)
def test_diffpair_factory_primal_and_grad(device_type: DeviceType):
    """create_torch_tensor_marshall with diff_pair (primal+grad, default is_input=True)."""
    layout = _get_layout(device_type)

    primal = torch.tensor([1.0, 2.0, 3.0], device="cuda", dtype=torch.float32)
    grad = torch.zeros(3, device="cuda", dtype=torch.float32)
    pair = diff_pair(primal, grad)

    marshall = ttm.create_torch_tensor_marshall(layout, pair)
    assert marshall.has_derivative is True
    assert marshall.dims > 0


SQUARE_SHADER = r"""
[Differentiable]
float square(float x) { return x * x; }
"""


@pytest.mark.parametrize("device_type", CUDA_TYPES)
def test_diffpair_factory_output_via_backward(device_type: DeviceType):
    """Backward pass creates output diff pairs with is_input=False, exercising the d_in factory path."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "square", SQUARE_SHADER)

    x = torch.tensor([2.0, 3.0], device="cuda", dtype=torch.float32, requires_grad=True)
    result = func(x)
    result.backward(torch.ones_like(result))

    assert x.grad is not None
    torch.testing.assert_close(x.grad, torch.tensor([4.0, 6.0], device="cuda"))


@pytest.mark.parametrize("device_type", CUDA_TYPES)
def test_diffpair_factory_grad_only(device_type: DeviceType):
    """create_torch_tensor_marshall with primal=None falls back to grad for dtype/shape."""
    layout = _get_layout(device_type)

    grad = torch.tensor([1.0, 2.0], device="cuda", dtype=torch.float32)
    pair = diff_pair(None, grad)

    marshall = ttm.create_torch_tensor_marshall(layout, pair)
    assert marshall.has_derivative is True
    assert marshall.dims == 1


@pytest.mark.parametrize("device_type", CUDA_TYPES)
def test_diffpair_factory_no_grad(device_type: DeviceType):
    """create_torch_tensor_marshall with grad=None produces no derivative."""
    layout = _get_layout(device_type)

    primal = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    pair = diff_pair(primal, None)

    marshall = ttm.create_torch_tensor_marshall(layout, pair)
    assert marshall.has_derivative is False


# ============================================================================
# Properties (lines 162, 166, 168)
# ============================================================================


@pytest.mark.parametrize("device_type", CUDA_TYPES)
def test_marshall_properties(device_type: DeviceType):
    """Access torch_dtype, slang_dtype, repr, is_writable, has_derivative on TorchTensorMarshall."""
    layout = _get_layout(device_type)

    t = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    marshall = ttm.create_torch_tensor_marshall(layout, t)

    assert marshall.torch_dtype == torch.float32
    assert marshall.slang_dtype is not None
    assert "float" in marshall.slang_dtype.full_name
    assert marshall.is_writable is True
    assert marshall.has_derivative is False

    r = repr(marshall)
    assert "TorchTensor" in r
    assert "float" in r


# ============================================================================
# Error paths
# ============================================================================


def test_hash_torch_tensor_raises():
    """hash_torch_tensor always raises ValueError (line 334)."""
    with pytest.raises(ValueError, match="should not need a hash"):
        ttm.hash_torch_tensor(torch.tensor([1.0]))


def test_hash_torch_diff_pair_raises():
    """hash_torch_diff_pair always raises ValueError (line 338)."""
    pair = diff_pair(torch.tensor([1.0]), torch.tensor([0.0]))
    with pytest.raises(ValueError, match="should not need a hash"):
        ttm.hash_torch_diff_pair(pair)


@pytest.mark.parametrize("device_type", CUDA_TYPES)
def test_factory_unsupported_type_raises(device_type: DeviceType):
    """Passing a non-tensor to create_torch_tensor_marshall raises ValueError (line 330)."""
    layout = _get_layout(device_type)
    with pytest.raises(ValueError, match="unsupported"):
        ttm.create_torch_tensor_marshall(layout, "not a tensor")


@pytest.mark.parametrize("device_type", CUDA_TYPES)
def test_factory_unsupported_torch_dtype_raises(device_type: DeviceType):
    """Passing a tensor with unsupported dtype raises ValueError (line 319)."""
    layout = _get_layout(device_type)
    t = torch.tensor([1.0 + 2.0j], dtype=torch.complex64, device="cuda")
    with pytest.raises(ValueError, match="[Uu]nsupported"):
        ttm.create_torch_tensor_marshall(layout, t)


@pytest.mark.parametrize("device_type", CUDA_TYPES)
def test_diffpair_factory_unsupported_dtype_raises(device_type: DeviceType):
    """DiffPair factory raises for unsupported torch dtype (line 287)."""
    layout = _get_layout(device_type)
    primal = torch.tensor([1.0 + 2.0j], dtype=torch.complex64, device="cuda")
    grad = torch.tensor([0.0 + 0.0j], dtype=torch.complex64, device="cuda")
    pair = diff_pair(primal, grad)
    with pytest.raises(ValueError, match="[Uu]nsupported"):
        ttm.create_torch_tensor_marshall(layout, pair)


@pytest.mark.parametrize("device_type", CUDA_TYPES)
def test_build_shader_object_gradient_not_implemented(device_type: DeviceType):
    """build_shader_object with has_derivative raises NotImplementedError (line 240)."""
    layout = _get_layout(device_type)

    primal = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    grad = torch.tensor([0.0], device="cuda", dtype=torch.float32)
    pair = diff_pair(primal, grad)
    marshall = ttm.create_torch_tensor_marshall(layout, pair)

    assert marshall.has_derivative
    from slangpy.bindings.marshall import BindContext
    from slangpy.core.native import CallMode

    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "scale", SCALE_SHADER)
    ctx = BindContext(layout, CallMode.prim, func.module.device_module, {})

    with pytest.raises(NotImplementedError, match="[Gg]radient"):
        marshall.build_shader_object(ctx, primal)


# ============================================================================
# Internal conversion helpers (lines 62, 70)
# ============================================================================


@pytest.mark.parametrize("device_type", CUDA_TYPES)
def test_slang_dtype_to_torch_none_for_non_scalar(device_type: DeviceType):
    """_slang_dtype_to_torch returns None for non-scalar SlangType (line 62)."""
    layout = _get_layout(device_type)
    vec_type = layout.find_type_by_name("float2")
    assert ttm._slang_dtype_to_torch(vec_type) is None


@pytest.mark.parametrize("device_type", CUDA_TYPES)
def test_torch_dtype_to_slang_none_for_unsupported(device_type: DeviceType):
    """_torch_dtype_to_slang returns None for unsupported torch dtype (line 70)."""
    layout = _get_layout(device_type)
    assert ttm._torch_dtype_to_slang(torch.complex128, layout) is None


# ============================================================================
# C++ NativeTorchTensorDiffPair coverage (slangpytorchtensor.cpp)
# ============================================================================

DIFF_SRC = r"""
[Differentiable]
float square(float x) { return x * x; }
"""


@pytest.mark.parametrize("device_type", CUDA_TYPES)
def test_diffpair_read_signature(device_type: DeviceType):
    """Calling a function with a DiffPair triggers NativeTorchTensorDiffPair::read_signature."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "square", DIFF_SRC)

    primal = torch.tensor([2.0, 3.0, 4.0], device="cuda", dtype=torch.float32, requires_grad=True)
    grad = torch.ones(3, device="cuda", dtype=torch.float32)
    pair = diff_pair(primal, grad)

    result = func(pair)
    assert result is not None


@pytest.mark.parametrize("device_type", CUDA_TYPES)
def test_diffpair_get_shape_grad_only(device_type: DeviceType):
    """NativeTorchTensorMarshall::get_shape falls back to grad when primal=None."""
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "square", DIFF_SRC)

    grad = torch.ones(5, device="cuda", dtype=torch.float32)
    pair = diff_pair(None, grad)

    result = func(pair)
    assert result is not None
