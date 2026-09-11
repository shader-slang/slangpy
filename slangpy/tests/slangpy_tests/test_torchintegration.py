# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
from typing import Callable

import numpy as np
import pytest

from slangpy import DeviceType, Device, Module, diff_pair, grid
from slangpy.core.callsignature import ResolveException
from slangpy.core.native import (
    NativeCallDataCache,
    NativeTorchTensorDiffPair,
    SignatureBuilder,
)
from slangpy.testing import helpers

try:
    import torch
except ImportError:
    pytest.skip("Pytorch not installed", allow_module_level=True)

# Skip all tests in this file if running on MacOS
if sys.platform == "darwin":
    pytest.skip("PyTorch requires CUDA, that is not available on macOS", allow_module_level=True)

TEST_CODE = """
import tensor;
[Differentiable]
float square(float x) {
    return x * x;
}
"""

DEVICE_TYPES = helpers.DEFAULT_DEVICE_TYPES
# Metal does not support torch integration
if DeviceType.metal in DEVICE_TYPES:
    DEVICE_TYPES.remove(DeviceType.metal)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def get_test_tensors(device: Device, N: int = 4):
    weights = torch.randn(
        (5, 8), dtype=torch.float32, device=torch.device("cuda"), requires_grad=True
    )
    biases = torch.randn((5,), dtype=torch.float32, device=torch.device("cuda"), requires_grad=True)
    x = torch.randn((8,), dtype=torch.float32, device=torch.device("cuda"), requires_grad=False)

    return weights, biases, x


def load_test_module(device_type: DeviceType):
    device = helpers.get_torch_device(device_type)
    return Module.load_from_file(device, "test_torchintegration.slang")


def compare_tensors(a: torch.Tensor, b: torch.Tensor):
    assert a.shape == b.shape, f"Tensor shape {a.shape} does not match expected shape {b.shape}"
    err = torch.max(torch.abs(a - b)).item()
    assert err < 1e-4, f"Tensor deviates by {err} from reference"


@pytest.fixture(autouse=True)
def setup_bridge_mode(torch_bridge_mode: str):
    """Automatically use torch_bridge_mode fixture for all tests in this class."""
    pass


@pytest.mark.parametrize(
    "pair",
    [
        (torch.empty((1,), dtype=torch.float32).cuda(), "D1,S6,G0,V1"),
        (torch.empty((1,), dtype=torch.float32, requires_grad=True).cuda(), "D1,S6,G1,V1"),
        (torch.empty((1,), dtype=torch.float16).cuda(), "D1,S5,G0,V1"),
        (torch.empty((1,), dtype=torch.int32).cuda(), "D1,S3,G0,V1"),
        (torch.empty((1,), dtype=torch.uint8).cuda(), "D1,S0,G0,V1"),
        (torch.empty((1, 1, 1), dtype=torch.uint8).cuda(), "D3,S0,G0,V111"),
    ],
)
def test_torch_signature(pair: tuple[torch.Tensor, str]):
    cd = NativeCallDataCache()
    sig = SignatureBuilder()
    cd.get_value_signature(sig, pair[0])
    assert sig.str == f"torch\n[{pair[1]}]"


def _torch_signature(tensor: torch.Tensor) -> str:
    cache = NativeCallDataCache()
    signature = SignatureBuilder()
    cache.get_value_signature(signature, tensor)
    return signature.str


def test_torch_signature_shape_compatibility() -> None:
    rgba_small = torch.empty((720, 1280, 4), dtype=torch.float32, device="cuda")
    rgba_large = torch.empty((1080, 1920, 4), dtype=torch.float32, device="cuda")
    rgb = torch.empty((720, 1280, 3), dtype=torch.float32, device="cuda")
    mixed = torch.empty((2, 3, 4), dtype=torch.float32, device="cuda")

    rgba_small_signature = _torch_signature(rgba_small)
    assert rgba_small_signature == _torch_signature(rgba_large)
    assert rgba_small_signature != _torch_signature(rgb)
    assert _torch_signature(mixed) == "torch\n[D3,S6,G0,V234]"


def _diff_pair_signature(pair: NativeTorchTensorDiffPair) -> str:
    cache = NativeCallDataCache()
    signature = SignatureBuilder()
    cache.get_value_signature(signature, pair)
    return signature.str


def test_diff_pair_signature_includes_marshaller_configuration() -> None:
    float_value = torch.empty((8, 8), dtype=torch.float32, device="cuda")
    half_value = torch.empty((8, 8), dtype=torch.float16, device="cuda")

    signatures = {
        _diff_pair_signature(NativeTorchTensorDiffPair(float_value, float_value, is_input=True)),
        _diff_pair_signature(NativeTorchTensorDiffPair(float_value, float_value, is_input=False)),
        _diff_pair_signature(NativeTorchTensorDiffPair(float_value, None, is_input=True)),
        _diff_pair_signature(NativeTorchTensorDiffPair(None, float_value, is_input=False)),
    }
    assert len(signatures) == 4

    float_grad_signature = _diff_pair_signature(
        NativeTorchTensorDiffPair(None, float_value, is_input=False)
    )
    half_grad_signature = _diff_pair_signature(
        NativeTorchTensorDiffPair(None, half_value, is_input=False)
    )
    assert float_grad_signature != half_grad_signature


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize(
    "shape,dtype,expected",
    [
        ((2, 3, 3), torch.float32, 3.0),
        ((2, 3, 4), torch.float32, 4.0),
        ((2, 3, 4), torch.float16, 8.0),
    ],
)
def test_select_vector_overload(
    device_type: DeviceType,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    expected: float,
) -> None:
    module = load_test_module(device_type)
    value = torch.zeros(shape, dtype=dtype, device="cuda")

    result = module.select_vector_overload(value)

    assert isinstance(result, torch.Tensor)
    assert result.shape == value.shape
    assert result.dtype == value.dtype
    assert torch.all(result == expected)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_select_vector_overload_reuses_resolution_compatible_call_data(
    device_type: DeviceType,
) -> None:
    module = load_test_module(device_type)
    function = module.select_vector_overload.as_func()

    rgba_small = torch.empty((8, 16, 4), dtype=torch.float32, device="cuda")
    rgba_large = torch.empty((12, 20, 4), dtype=torch.float32, device="cuda")
    rgb = torch.empty((8, 16, 3), dtype=torch.float32, device="cuda")

    rgba_small_call_data = function.debug_build_call_data(rgba_small)
    rgba_large_call_data = function.debug_build_call_data(rgba_large)
    rgb_call_data = function.debug_build_call_data(rgb)

    assert rgba_small_call_data == rgba_large_call_data
    assert rgba_small_call_data != rgb_call_data


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_diff_pair_vector_overload_uses_shape_specific_call_data(
    device_type: DeviceType,
) -> None:
    module = load_test_module(device_type)
    function = module.select_differentiable_vector_overload.bwds

    float3 = torch.zeros((8, 16, 3), dtype=torch.float32, device="cuda")
    float4 = torch.zeros((8, 16, 4), dtype=torch.float32, device="cuda")

    float3_call_data = function.debug_build_call_data(
        diff_pair(float3, torch.zeros_like(float3)),
        _result=diff_pair(float3, torch.ones_like(float3)),
    )
    float4_call_data = function.debug_build_call_data(
        diff_pair(float4, torch.zeros_like(float4)),
        _result=diff_pair(float4, torch.ones_like(float4)),
    )

    assert float3_call_data != float4_call_data


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_diff_pair_scalar_call_data_is_rank_specific(
    device_type: DeviceType,
) -> None:
    module = load_test_module(device_type)
    function = module.differentiable_identity.bwds

    rank1 = torch.zeros((8,), dtype=torch.float32, device="cuda")
    rank2 = torch.zeros((8, 8), dtype=torch.float32, device="cuda")

    rank1_call_data = function.debug_build_call_data(
        diff_pair(rank1, torch.zeros_like(rank1)),
        _result=diff_pair(rank1, torch.ones_like(rank1)),
    )
    rank2_call_data = function.debug_build_call_data(
        diff_pair(rank2, torch.zeros_like(rank2)),
        _result=diff_pair(rank2, torch.ones_like(rank2)),
    )

    assert rank1_call_data != rank2_call_data


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize(
    "first_dtype,second_dtype",
    [
        (torch.float32, torch.float16),
        (torch.float16, torch.float32),
    ],
    ids=["float_then_half", "half_then_float"],
)
def test_diff_pair_scalar_overload_uses_dtype_specific_call_data(
    device_type: DeviceType,
    first_dtype: torch.dtype,
    second_dtype: torch.dtype,
) -> None:
    module = load_test_module(device_type)
    function = module.select_differentiable_scalar_overload.bwds

    first_value = torch.zeros((8, 8), dtype=first_dtype, device="cuda")
    second_value = torch.zeros((8, 8), dtype=second_dtype, device="cuda")

    first_call_data = function.debug_build_call_data(
        diff_pair(first_value, torch.zeros_like(first_value)),
        _result=diff_pair(first_value, torch.ones_like(first_value)),
    )
    second_call_data = function.debug_build_call_data(
        diff_pair(second_value, torch.zeros_like(second_value)),
        _result=diff_pair(second_value, torch.ones_like(second_value)),
    )

    assert first_call_data != second_call_data


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_select_vector_overload_with_explicit_mapping(device_type: DeviceType) -> None:
    module = load_test_module(device_type)
    value = torch.zeros((2, 3, 4), dtype=torch.float32, device="cuda")

    result = module.select_vector_overload.map((1, 0))(value)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 2, 4)
    assert torch.all(result == 4.0)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("matrix_shape", [(2, 3), (3, 2)])
def test_select_matrix_overload(
    device_type: DeviceType,
    matrix_shape: tuple[int, int],
) -> None:
    module = load_test_module(device_type)
    value = torch.randn((5,) + matrix_shape, dtype=torch.float32, device="cuda")

    result = module.select_matrix_overload(value)

    assert isinstance(result, torch.Tensor)
    assert result.shape == value.shape
    assert torch.all(result == value)


ADD_TESTS = [
    ("add", ()),
    ("add_vectors", (3,)),
    ("add_vectors_generic<4>", (4,)),
    ("add_arrays", (5,)),
]


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("extra_dims", [0, 1, 3])
@pytest.mark.parametrize(
    "func_and_shape", ADD_TESTS, ids=[f"{name}_{shape}" for name, shape in ADD_TESTS]
)
@pytest.mark.parametrize("result_mode", ["return", "pass", "out"])
def test_add_values(
    device_type: DeviceType,
    extra_dims: int,
    func_and_shape: tuple[str, tuple[int]],
    result_mode: str,
):

    module = load_test_module(device_type)

    func_name = func_and_shape[0]
    val_shape = func_and_shape[1]
    extra_shape = (5,) * extra_dims

    if len(extra_shape + val_shape) == 0:
        pytest.skip("No shape to test")

    a = torch.randn(
        extra_shape + val_shape,
        dtype=torch.float32,
        device=torch.device("cuda"),
        requires_grad=True,
    )
    b = torch.randn(
        extra_shape + val_shape,
        dtype=torch.float32,
        device=torch.device("cuda"),
        requires_grad=True,
    )

    if result_mode == "return":
        res = module[func_name](a, b)
    elif result_mode == "pass":
        res = torch.empty_like(a)
        module[func_name](a, b, _result=res)
    else:  # out
        res = torch.empty_like(a)
        if "<" in func_name:
            func_name = func_name.replace("<", "_out<")
        else:
            func_name += "_out"
        module[func_name](a, b, res)
    assert isinstance(res, torch.Tensor)

    test = a + b

    compare_tensors(a + b, res)

    # Not much to check for backwards pass of an 'add', but call it
    # so we at least catch any exceptions that fire.
    res.backward(torch.ones_like(res))


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("extra_dims", [0, 1, 3])
@pytest.mark.parametrize("func_and_shape", ADD_TESTS)
def test_add_values_fail(
    device_type: DeviceType, extra_dims: int, func_and_shape: tuple[str, tuple[int]]
):

    module = load_test_module(device_type)

    func_name = func_and_shape[0]
    val_shape = func_and_shape[1]
    if len(val_shape) == 0:
        pytest.skip("No shape to fail")

    extra_shape = (5,) * extra_dims

    val_shape = val_shape[0:-1] + (val_shape[-1] + 1,)

    a = torch.randn(extra_shape + val_shape, dtype=torch.float32, device=torch.device("cuda"))
    b = torch.randn(extra_shape + val_shape, dtype=torch.float32, device=torch.device("cuda"))

    with pytest.raises(ResolveException, match="does not match slang type"):
        res = module.add_vectors(a, b)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("extra_dims", [0, 1, 3])
def test_add_vectors_generic_explicit(device_type: DeviceType, extra_dims: int):
    pytest.skip("Crashes due to Slang compiler bug (#940)")

    module = load_test_module(device_type)

    extra_shape = (5,) * extra_dims

    a = torch.randn(extra_shape + (3,), dtype=torch.float32, device=torch.device("cuda"))
    b = torch.randn(extra_shape + (3,), dtype=torch.float32, device=torch.device("cuda"))

    # Can't currently infer generic vector from tensor shape, but explicit type map should work
    res = module.add_vectors_generic.map("float3", "float3")(a, b)
    assert isinstance(res, torch.Tensor)

    compare_tensors(a + b, res)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_polynomial(device_type: DeviceType):

    module = load_test_module(device_type)

    a = 2.0
    b = 4.0
    c = 1.0
    x = torch.randn((10,), dtype=torch.float32, device=torch.device("cuda"), requires_grad=True)

    res = module.polynomial(a, b, c, x)
    assert isinstance(res, torch.Tensor)

    compare_tensors(a * x * x + b * x + c, res)

    res.backward(torch.ones_like(res))

    compare_tensors(2 * a * x + b, x.grad)  # type: ignore


# This test ensures that the PyTorch integration doesn't fail if re-using the
# same cached call data.
@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_polynomial_multiple_calls(device_type: DeviceType):

    module = load_test_module(device_type)

    a = 2.0
    b = 4.0
    c = 1.0
    x = torch.randn((10,), dtype=torch.float32, device=torch.device("cuda"), requires_grad=True)

    res = module.polynomial(a, b, c, x)
    assert isinstance(res, torch.Tensor)

    compare_tensors(a * x * x + b * x + c, res)

    res.backward(torch.ones_like(res))
    compare_tensors(2 * a * x + b, x.grad)  # type: ignore

    res2 = module.polynomial(a, b, c, x)
    assert isinstance(res2, torch.Tensor)

    x.grad.zero_()  # Reset gradients before the second call
    res2.backward(torch.ones_like(res2))
    compare_tensors(2 * a * x + b, x.grad)  # type: ignore


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_polynomial_outparam(device_type: DeviceType):

    module = load_test_module(device_type)

    a = 2.0
    b = 4.0
    c = 1.0
    x = torch.randn((10,), dtype=torch.float32, device=torch.device("cuda"), requires_grad=True)
    res = torch.zeros_like(x)

    module.polynomial_out(a, b, c, x, res)

    compare_tensors(a * x * x + b * x + c, res)

    res.backward(torch.ones_like(res))

    compare_tensors(2 * a * x + b, x.grad)  # type: ignore


# Enable the vectors+arrays tests to reproduce compiler bugs
POLYNOMIAL_TESTS = [
    ("polynomial", ()),
    ("polynomial_vectors", (3,)),
    ("polynomial_arrays", (5,)),
]


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("extra_dims", [0, 1, 3])
@pytest.mark.parametrize(
    "func_and_shape",
    POLYNOMIAL_TESTS,
    ids=[f"{name}_{shape}" for name, shape in POLYNOMIAL_TESTS],
)
@pytest.mark.parametrize("result_mode", ["return", "pass", "out"])
def test_polynomials(
    device_type: DeviceType,
    extra_dims: int,
    func_and_shape: tuple[str, tuple[int]],
    result_mode: str,
):

    module = load_test_module(device_type)

    func_name = func_and_shape[0]
    val_shape = func_and_shape[1]
    extra_shape = (5,) * extra_dims

    if func_name == "polynomial_vectors":
        pytest.skip("Slang compiler bug: vector polynomial derivatives return 0 (#940)")

    if len(extra_shape + val_shape) == 0:
        pytest.skip("No shape to test")

    a = 2.0
    b = 4.0
    c = 1.0
    x = torch.randn(
        extra_shape + val_shape,
        dtype=torch.float32,
        device=torch.device("cuda"),
        requires_grad=True,
    )

    if result_mode == "return":
        res = module[func_name](a, b, c, x)
    elif result_mode == "pass":
        res = torch.empty_like(x)
        module[func_name](a, b, c, x, _result=res)
    else:  # out
        res = torch.empty_like(x)
        if "<" in func_name:
            func_name = func_name.replace("<", "_out<")
        else:
            func_name += "_out"
        module[func_name](a, b, c, x, res)
    assert isinstance(res, torch.Tensor)

    compare_tensors(a * x * x + b * x + c, res)

    res.backward(torch.ones_like(res))

    compare_tensors(2 * a * x + b, x.grad)  # type: ignore


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("extra_dims", [0, 1, 3])
@pytest.mark.parametrize("grads", [False, True])
def test_add_tensors(device_type: DeviceType, extra_dims: int, grads: bool):

    module = load_test_module(device_type)

    func_name = "add_tensors"
    val_shape = (8, 5)
    extra_shape = (5,) * extra_dims

    a = torch.randn(
        extra_shape + val_shape,
        dtype=torch.float32,
        device=torch.device("cuda"),
        requires_grad=grads,
    )
    b = torch.randn(
        extra_shape + val_shape,
        dtype=torch.float32,
        device=torch.device("cuda"),
        requires_grad=grads,
    )

    res = torch.empty_like(a)
    module[func_name](a, b, res)

    compare_tensors(a + b, res)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_struct_tensor_from_torch(device_type: DeviceType):
    """
    Test Tensor.from_torch() reinterprets a torch.Tensor as Tensor<PackedFloat2, 1>.
    """
    from slangpy import Tensor

    device = helpers.get_torch_device(device_type)
    module = load_test_module(device_type)

    input_torch = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        dtype=torch.float32,
        device=torch.device("cuda"),
    )
    output_torch = torch.zeros_like(input_torch)

    input_tensor = Tensor.from_torch(device, input_torch, dtype=module.PackedFloat2)
    output_tensor = Tensor.from_torch(device, output_torch, dtype=module.PackedFloat2)

    module.copy_struct_tensor(input_tensor, output_tensor)

    result = output_tensor.to_numpy()
    expected = input_torch.cpu().numpy().view(np.uint8).reshape(3, -1)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_struct_tensor_wrong_last_dim(device_type: DeviceType):
    """
    Test that Tensor.from_torch() raises when the last dimension doesn't match.
    """
    from slangpy import Tensor

    device = helpers.get_torch_device(device_type)
    module = load_test_module(device_type)

    bad_tensor = torch.zeros((3, 3), dtype=torch.float32, device=torch.device("cuda"))
    with pytest.raises(ValueError, match="does not match"):
        Tensor.from_torch(device, bad_tensor, dtype=module.PackedFloat2)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_struct_tensor_particle_update(device_type: DeviceType):
    """
    Test Tensor.from_torch() with a Particle struct (float3 + float3 = 6 floats).
    """
    from slangpy import Tensor

    device = helpers.get_torch_device(device_type)
    module = load_test_module(device_type)

    N = 5
    dt = 0.1
    input_torch = torch.randn((N, 6), dtype=torch.float32, device=torch.device("cuda"))
    output_torch = torch.zeros_like(input_torch)

    input_tensor = Tensor.from_torch(device, input_torch, dtype=module.Particle)
    output_tensor = Tensor.from_torch(device, output_torch, dtype=module.Particle)

    module.update_particle(input_tensor, dt, output_tensor)

    result_np = output_tensor.to_numpy()
    input_np = input_torch.cpu().numpy()
    expected = input_np.copy()
    expected[:, 0:3] = input_np[:, 0:3] + input_np[:, 3:6] * dt

    result_floats = np.frombuffer(result_np.tobytes(), dtype=np.float32).reshape(N, 6)
    np.testing.assert_allclose(result_floats, expected, atol=1e-4)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_empty_tensor_null_data_ptr(device_type: DeviceType):
    """
    Test that tensors with null data pointers (e.g., zero-element tensors) are accepted.
    """
    module = load_test_module(device_type)

    # Create empty tensors - these have null data pointers
    input_tensor = torch.empty((0,), dtype=torch.float32, device=torch.device("cuda"))
    output_tensor = torch.empty((0,), dtype=torch.float32, device=torch.device("cuda"))

    # This should not crash - empty tensors with null data_ptr should be accepted
    module.copy_tensor(input_tensor, output_tensor)

    # Verify tensors are still empty
    assert input_tensor.numel() == 0
    assert output_tensor.numel() == 0


SLICE_CASES = [
    pytest.param(4, lambda t: t[:3], id="prefix"),
    pytest.param(4, lambda t: t[1:], id="suffix_offset"),
    pytest.param(6, lambda t: t[::2], id="strided"),
    pytest.param(9, lambda t: t.reshape(3, 3).diagonal(), id="diagonal"),
]


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("source_size,slicer", SLICE_CASES)
def test_parameter_slice(
    device_type: DeviceType, source_size: int, slicer: Callable[[torch.Tensor], torch.Tensor]
):
    """
    Test that sliced PyTorch tensors can be passed as fixed-size array parameters.

    Covers prefix slices (zero offset, contiguous), suffix slices (non-zero
    offset, contiguous), and strided slices (non-contiguous).
    """
    module = load_test_module(device_type)

    scale = torch.rand(10, dtype=torch.float32, device=torch.device("cuda"))
    values = torch.rand(source_size, dtype=torch.float32, device=torch.device("cuda"))

    sliced = slicer(values)
    assert sliced.shape == (3,), f"Slice should produce 3 elements, got {sliced.shape}"

    res = module.scaled_sum(scale, sliced)
    assert isinstance(res, torch.Tensor)

    expected = scale * sliced.sum()
    compare_tensors(res, expected)


VECTOR_SLICE_CASES = [
    pytest.param(4, lambda t: t[:, :3], id="prefix"),
    pytest.param(4, lambda t: t[:, 1:], id="suffix_offset"),
    pytest.param(6, lambda t: t[:, ::2], id="strided"),
]


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("source_cols,slicer", VECTOR_SLICE_CASES)
def test_vector_parameter_slice(
    device_type: DeviceType,
    source_cols: int,
    slicer: Callable[[torch.Tensor], torch.Tensor],
):
    """
    Test that sliced PyTorch tensors can be passed as float3 vector parameters.

    The trailing dimension of the view maps to float3 components. Covers prefix
    (zero offset), suffix (non-zero offset), and strided (non-contiguous) slices.
    """
    module = load_test_module(device_type)

    batch = 5
    a = torch.rand(batch, source_cols, dtype=torch.float32, device=torch.device("cuda"))
    b = torch.rand(batch, source_cols, dtype=torch.float32, device=torch.device("cuda"))

    a_sliced = slicer(a)
    b_sliced = slicer(b)
    assert a_sliced.shape == (batch, 3)

    res = module.add_vectors(a_sliced, b_sliced)
    assert isinstance(res, torch.Tensor)

    compare_tensors(res, a_sliced + b_sliced)


RWTENSOR_SLICE_CASES = [
    pytest.param(6, lambda t: t[:3], id="prefix"),
    pytest.param(6, lambda t: t[1:4], id="offset"),
    pytest.param(6, lambda t: t[::2], id="strided"),
    pytest.param(9, lambda t: t.reshape(3, 3).diagonal(), id="diagonal"),
]


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("full_size,slicer", RWTENSOR_SLICE_CASES)
def test_rwtensor_slice_writeback(
    device_type: DeviceType,
    full_size: int,
    slicer: Callable[[torch.Tensor], torch.Tensor],
):
    """
    Test that write-back to a sliced RWTensor correctly updates only the
    sliced region of the underlying tensor.

    A sentinel-filled tensor is sliced, the slice is passed as RWTensor output,
    and we verify that only the sliced positions are overwritten.
    """
    module = load_test_module(device_type)

    input_data = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32, device=torch.device("cuda"))

    sentinel = -1.0
    output_full = torch.full(
        (full_size,), sentinel, dtype=torch.float32, device=torch.device("cuda")
    )
    output_slice = slicer(output_full)
    assert output_slice.shape == (3,)

    module.copy_tensor(input_data, output_slice)

    expected_full = torch.full_like(output_full, sentinel)
    slicer(expected_full)[:] = input_data
    compare_tensors(output_full, expected_full)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_vector_parameter_transpose(device_type: DeviceType):
    """
    Test that transposed (non-contiguous) tensors work as float3 vector params.

    Creates (3, batch) tensors and transposes to (batch, 3). The trailing
    dimension that maps to float3 has non-unit stride from the transpose.
    """
    module = load_test_module(device_type)

    batch = 5
    dev = torch.device("cuda")
    a = torch.rand(3, batch, dtype=torch.float32, device=dev).t()
    b = torch.rand(3, batch, dtype=torch.float32, device=dev).t()
    assert not a.is_contiguous()

    res = module.add_vectors(a, b)
    assert isinstance(res, torch.Tensor)

    compare_tensors(res, a + b)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_array_parameter_transpose(device_type: DeviceType):
    """
    Test that transposed tensors work as float[3] fixed-size array params.

    Creates (3, batch) tensors and transposes to (batch, 3). The trailing
    dimension that maps to float[3] has non-unit stride.
    """
    module = load_test_module(device_type)

    batch = 10
    dev = torch.device("cuda")
    scale = torch.rand(batch, dtype=torch.float32, device=dev)
    values = torch.rand(3, batch, dtype=torch.float32, device=dev).t()
    assert not values.is_contiguous()

    res = module.scaled_sum(scale, values)
    assert isinstance(res, torch.Tensor)

    expected = scale * values.sum(dim=-1)
    compare_tensors(res, expected)


TENSOR2D_VIEW_FACTORIES: list[tuple[str, Callable[..., torch.Tensor]]] = [
    ("transpose", lambda d: torch.randn(5, 8, dtype=torch.float32, device=d).t()),
    ("col_prefix", lambda d: torch.randn(5, 8, dtype=torch.float32, device=d)[:, :5]),
    ("col_offset", lambda d: torch.randn(5, 8, dtype=torch.float32, device=d)[:, 2:7]),
    ("col_strided", lambda d: torch.randn(5, 8, dtype=torch.float32, device=d)[:, ::2]),
    (
        "permute_3d_select",
        lambda d: torch.randn(5, 8, 3, dtype=torch.float32, device=d).permute(2, 0, 1)[0],
    ),
]


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize(
    "name,view_factory",
    TENSOR2D_VIEW_FACTORIES,
    ids=[name for name, _ in TENSOR2D_VIEW_FACTORIES],
)
def test_tensor_view(
    device_type: DeviceType,
    name: str,
    view_factory: Callable[[torch.device], torch.Tensor],
):
    """
    Test that non-contiguous 2D tensor views work correctly when bound to
    Tensor<float,2> / WTensor<float,2> parameters.

    Covers transposed, column-sliced (prefix and offset), and column-strided
    views, all of which produce non-contiguous memory layouts.
    """
    module = load_test_module(device_type)

    dev = torch.device("cuda")
    a = view_factory(dev)
    b = view_factory(dev)
    assert not a.is_contiguous()

    res = torch.empty(a.shape, dtype=torch.float32, device=dev)
    module.add_tensors(a, b, res)

    compare_tensors(res, a + b)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_wtensor_transpose_writeback(device_type: DeviceType):
    """
    Test that write-back to a transposed WTensor<float,2> output correctly
    places results in the non-contiguous view.
    """
    module = load_test_module(device_type)

    dev = torch.device("cuda")
    a = torch.randn(8, 5, dtype=torch.float32, device=dev)
    b = torch.randn(8, 5, dtype=torch.float32, device=dev)

    res_base = torch.zeros(5, 8, dtype=torch.float32, device=dev)
    res = res_base.t()  # (8, 5), non-contiguous
    assert not res.is_contiguous()

    module.add_tensors(a, b, res)

    compare_tensors(res, a + b)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_parameter_broadcast(device_type: DeviceType):
    """
    Test float[3] params with zero-stride (expand) broadcast input.

    A single row of 3 values is broadcast to (batch, 3) via expand, giving
    stride 0 in the batch dimension. Every batch invocation reads the same values.
    """
    module = load_test_module(device_type)

    dev = torch.device("cuda")
    batch = 10
    scale = torch.rand(batch, dtype=torch.float32, device=dev)
    values_single = torch.rand(3, dtype=torch.float32, device=dev)
    values = values_single.unsqueeze(0).expand(batch, -1)
    assert values.stride(0) == 0

    res = module.scaled_sum(scale, values)
    expected = scale * values_single.sum()
    compare_tensors(res, expected)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_vector_parameter_broadcast(device_type: DeviceType):
    """
    Test float3 params with zero-stride (expand) broadcast input.

    One input has stride 0 in the batch dim (same float3 for every row),
    while the other varies per row.
    """
    module = load_test_module(device_type)

    dev = torch.device("cuda")
    batch = 5
    a_single = torch.rand(1, 3, dtype=torch.float32, device=dev)
    a = a_single.expand(batch, -1)
    b = torch.rand(batch, 3, dtype=torch.float32, device=dev)
    assert a.stride(0) == 0

    res = module.add_vectors(a, b)
    compare_tensors(res, a + b)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_tensor_view_broadcast(device_type: DeviceType):
    """
    Test Tensor<float,2> with zero-stride (expand) broadcast on row dimension.

    One input has a single row broadcast to fill all rows (stride 0 on dim 0).
    """
    module = load_test_module(device_type)

    dev = torch.device("cuda")
    a_single = torch.randn(1, 5, dtype=torch.float32, device=dev)
    a = a_single.expand(8, -1)
    b = torch.randn(8, 5, dtype=torch.float32, device=dev)
    assert a.stride(0) == 0 and not a.is_contiguous()

    res = torch.empty(8, 5, dtype=torch.float32, device=dev)
    module.add_tensors(a, b, res)
    compare_tensors(res, a + b)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_copy_tensor_to_buffer(device_type: DeviceType):
    """
    Test that copy_torch_tensor_to_buffer correctly copies tensor data to a shared buffer.
    """
    from slangpy import BufferUsage, copy_torch_tensor_to_buffer

    # Get a device that shares the CUDA context with PyTorch
    device = helpers.get_torch_device(device_type)

    # Create a test tensor with known values
    tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32, device="cuda")

    # Create a shared buffer large enough for the tensor
    buffer = device.create_buffer(
        size=tensor.numel() * tensor.element_size(),
        struct_size=tensor.element_size(),
        usage=BufferUsage.unordered_access | BufferUsage.shader_resource | BufferUsage.shared,
    )

    # Copy tensor to buffer (on cuda device)
    copy_torch_tensor_to_buffer(tensor, buffer)

    # buffer.to_numpy is run on device, so if using interop need to make
    # sure we wait for the cuda work to complete
    device.sync_to_cuda()

    # Read back buffer contents via CPU and verify
    import numpy as np

    buffer_data = np.frombuffer(buffer.to_numpy().tobytes(), dtype=np.float32)
    expected = tensor.cpu().numpy()

    assert len(buffer_data) == len(
        expected
    ), f"Length mismatch: {len(buffer_data)} vs {len(expected)}"
    assert np.allclose(buffer_data, expected), f"Data mismatch: {buffer_data} vs {expected}"


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_copy_buffer_to_tensor(device_type: DeviceType):
    """
    Test that copy_buffer_to_torch_tensor correctly copies buffer data to a tensor.
    """
    from slangpy import BufferUsage, copy_buffer_to_torch_tensor

    device = helpers.get_torch_device(device_type)

    # Create a tensor to receive data
    tensor = torch.zeros(5, dtype=torch.float32, device="cuda")

    # Create a shared buffer and write known values
    import numpy as np

    test_values = np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float32)
    buffer = device.create_buffer(
        size=test_values.nbytes,
        struct_size=4,
        usage=BufferUsage.unordered_access | BufferUsage.shader_resource | BufferUsage.shared,
    )
    buffer.copy_from_numpy(test_values)

    # If using cuda interop, make sure cuda waits for device
    # to finish the copy_from_numpy
    device.sync_to_device()

    # Copy buffer to tensor (on cuda device)
    copy_buffer_to_torch_tensor(buffer, tensor)

    # Verify tensor contents
    result = tensor.cpu().numpy()
    assert np.allclose(result, test_values), f"Data mismatch: {result} vs {test_values}"


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_copy_noncontiguous_tensor_to_buffer(device_type: DeviceType):
    """
    Test that copy_torch_tensor_to_buffer works with non-contiguous tensors.
    """
    from slangpy import BufferUsage, copy_torch_tensor_to_buffer

    device = helpers.get_torch_device(device_type)

    # Create a non-contiguous tensor (transposed)
    base = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32, device="cuda")
    tensor = base.t()  # Transpose makes it non-contiguous
    assert not tensor.is_contiguous(), "Test tensor should be non-contiguous"

    # Create buffer for the contiguous data
    buffer = device.create_buffer(
        size=tensor.numel() * tensor.element_size(),
        struct_size=tensor.element_size(),
        usage=BufferUsage.unordered_access | BufferUsage.shader_resource | BufferUsage.shared,
    )

    # Copy tensor to buffer (on cuda device)
    copy_torch_tensor_to_buffer(tensor, buffer)

    # buffer.to_numpy is run on device, so if using interop need to make
    # sure we wait for the cuda work to complete
    device.sync_to_cuda()

    # Read back and verify - should match contiguous version of tensor
    import numpy as np

    buffer_data = np.frombuffer(buffer.to_numpy().tobytes(), dtype=np.float32)
    expected = tensor.contiguous().cpu().numpy().flatten()

    assert np.allclose(buffer_data, expected), f"Data mismatch: {buffer_data} vs {expected}"


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_tensor_buffer_roundtrip(device_type: DeviceType):
    """
    Test round-trip: tensor -> buffer -> tensor2.
    Verifies that data survives a complete copy cycle through the interop buffer.
    """
    from slangpy import BufferUsage, copy_torch_tensor_to_buffer, copy_buffer_to_torch_tensor

    device = helpers.get_torch_device(device_type)

    # Create source tensor with known values
    src_tensor = torch.tensor([1.5, 2.5, 3.5, 4.5, 5.5], dtype=torch.float32, device="cuda")

    # Create destination tensor (zeros)
    dst_tensor = torch.zeros_like(src_tensor)

    # Create shared buffer
    buffer = device.create_buffer(
        size=src_tensor.numel() * src_tensor.element_size(),
        struct_size=src_tensor.element_size(),
        usage=BufferUsage.unordered_access | BufferUsage.shader_resource | BufferUsage.shared,
    )

    # Copy: src_tensor -> buffer -> dst_tensor
    # There is no need for any device waits, as both operations happen
    # on the cuda device, even in the interop case.
    copy_torch_tensor_to_buffer(src_tensor, buffer)
    copy_buffer_to_torch_tensor(buffer, dst_tensor)

    # Verify dst_tensor matches src_tensor
    assert torch.allclose(
        src_tensor, dst_tensor
    ), f"Round-trip mismatch: {src_tensor} vs {dst_tensor}"


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_null_grad_difftensor(device_type: DeviceType):

    src = """
import slangpy;

[Differentiable]
void forward(uint index, DiffTensor<float, 1> x, WDiffTensor<float, 1> y)
{
    float x_i = x[index];
    y[index] = x_i * x_i * x_i;
}
"""
    import torch
    import torch.nn as nn

    device = helpers.get_torch_device(device_type)
    module = helpers.create_module(device, src)

    loss_fn = nn.MSELoss()
    targets = torch.ones(size=(4,), dtype=torch.float32, device="cuda")

    x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device="cuda", requires_grad=True)
    y = torch.zeros(size=(4,), dtype=torch.float32, device="cuda", requires_grad=True)

    module.forward(index=grid(shape=(4,)), x=x, y=y)
    loss = loss_fn(y, targets)
    loss.backward()

    assert x.grad is not None, "Gradients should flow back to x"
    expected_y = torch.tensor([1.0, 8.0, 27.0, 64.0], device="cuda")
    assert torch.allclose(y, expected_y), f"y = x^3 mismatch: {y} vs {expected_y}"


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_null_grad_idifftensor(device_type: DeviceType):

    src = """
import slangpy;

[Differentiable]
void forward(uint index, IDiffTensor<float, 1> x, IWDiffTensor<float, 1> y)
{
    float x_i = x[index];
    y[index] = x_i * x_i * x_i;
}
"""
    import torch
    import torch.nn as nn

    device = helpers.get_torch_device(device_type)
    module = helpers.create_module(device, src)

    loss_fn = nn.MSELoss()
    targets = torch.ones(size=(4,), dtype=torch.float32, device="cuda")

    x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device="cuda", requires_grad=True)
    y = torch.zeros(size=(4,), dtype=torch.float32, device="cuda", requires_grad=True)

    module.forward(index=grid(shape=(4,)), x=x, y=y)
    loss = loss_fn(y, targets)
    loss.backward()

    assert x.grad is not None, "Gradients should flow back to x"
    expected_y = torch.tensor([1.0, 8.0, 27.0, 64.0], device="cuda")
    assert torch.allclose(y, expected_y), f"y = x^3 mismatch: {y} vs {expected_y}"


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_nn_parameter_as_input(device_type: DeviceType):
    """
    Test that torch.nn.parameter.Parameter can be passed to a SlangPy function.
    nn.Parameter is a subclass of torch.Tensor and should be handled transparently.
    """
    import torch.nn as nn

    module = load_test_module(device_type)

    a = nn.Parameter(torch.randn((10,), dtype=torch.float32, device="cuda"))
    b = nn.Parameter(torch.randn((10,), dtype=torch.float32, device="cuda"))

    res = module.add(a, b)
    assert isinstance(res, torch.Tensor)
    compare_tensors(a + b, res)

    # Gradients should flow back through nn.Parameter
    res.backward(torch.ones_like(res))
    assert a.grad is not None
    assert b.grad is not None
    compare_tensors(a.grad, torch.ones_like(a))
    compare_tensors(b.grad, torch.ones_like(b))


def test_nn_parameter_signature():
    """
    Test that torch.nn.parameter.Parameter produces the same signature as torch.Tensor.
    """
    cd = NativeCallDataCache()

    # nn.Parameter defaults to requires_grad=True, so the tensor it is compared
    # against needs matching grad-ness for this to assert type-handling parity
    # rather than the grad bit added in #1052.
    param = torch.nn.parameter.Parameter(torch.empty((4, 4), dtype=torch.float32).cuda())
    tensor = torch.empty((4, 4), dtype=torch.float32, requires_grad=True).cuda()

    sig_param = SignatureBuilder()
    sig_tensor = SignatureBuilder()
    cd.get_value_signature(sig_param, param)
    cd.get_value_signature(sig_tensor, tensor)

    assert sig_param.str == sig_tensor.str

    tensor_nograd = torch.empty((4, 4), dtype=torch.float32).cuda()
    sig_nograd = SignatureBuilder()
    cd.get_value_signature(sig_nograd, tensor_nograd)
    assert sig_nograd.str != sig_param.str


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_nn_module_parameter_gradient(device_type: DeviceType):
    """
    Test that nn.Parameter from an nn.Module can be passed to slangpy functions
    with gradients flowing back into the module (typical training use-case).
    """
    import torch.nn as nn

    module = load_test_module(device_type)

    # Simulate a typical use-case: module parameters fed into a slangpy kernel
    linear = nn.Linear(10, 10, bias=True, device="cuda", dtype=torch.float32)
    bias = linear.bias  # nn.Parameter, shape (10,)

    x = torch.randn((10,), dtype=torch.float32, device="cuda", requires_grad=True)

    result = module.add(x, bias)
    assert isinstance(result, torch.Tensor)

    result.backward(torch.ones_like(result))
    assert x.grad is not None
    assert bias.grad is not None
    compare_tensors(x.grad, torch.ones_like(x))
    compare_tensors(bias.grad, torch.ones_like(bias))


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_zero_size_dispatch(device_type: DeviceType):
    """Dispatching with empty torch tensors should be a no-op, not crash."""
    module = load_test_module(device_type)
    a = torch.tensor([], dtype=torch.float32, device="cuda")
    b = torch.tensor([], dtype=torch.float32, device="cuda")
    result = module.add(a, b)
    assert result.numel() == 0


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_scalar_return_with_torch_input(device_type: DeviceType):
    """A scalar Slang return with a whole-tensor torch input comes back as a
    value-correct 0-D torch.Tensor. Distinct-per-index values guard against a
    silent-zero readback that a constant input would not catch."""
    module = load_test_module(device_type)
    tex = torch.arange(16, dtype=torch.float32, device="cuda") * 10.0
    result = module.read_element(tex, 3.0)
    assert isinstance(result, torch.Tensor)
    assert result.shape == ()
    assert result.item() == pytest.approx(30.0)


# ============================================================================
# DiffPair factory paths (torchtensormarshall.py coverage)
# ============================================================================

from slangpy.torchintegration import diff_pair
import slangpy.torchintegration.torchtensormarshall as ttm

SCALE_SHADER = r"""
void scale(float a, float factor, out float result) { result = a * factor; }
"""


def _get_layout(device_type: DeviceType):
    device = helpers.get_device(device_type)
    func = helpers.create_function_from_module(device, "scale", SCALE_SHADER)
    return func.module.layout


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_diffpair_factory_primal_and_grad(device_type: DeviceType):
    """create_torch_tensor_marshall with diff_pair (primal+grad, default is_input=True)."""
    layout = _get_layout(device_type)

    primal = torch.tensor([1.0, 2.0, 3.0], device="cuda", dtype=torch.float32)
    grad = torch.zeros(3, device="cuda", dtype=torch.float32)
    pair = diff_pair(primal, grad)

    marshall = ttm.create_torch_tensor_marshall(layout, pair)
    assert marshall.has_derivative is True
    assert marshall.dims > 0


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_diffpair_factory_grad_only(device_type: DeviceType):
    """create_torch_tensor_marshall with primal=None falls back to grad for dtype/shape."""
    layout = _get_layout(device_type)

    grad = torch.tensor([1.0, 2.0], device="cuda", dtype=torch.float32)
    pair = diff_pair(None, grad)

    marshall = ttm.create_torch_tensor_marshall(layout, pair)
    assert marshall.has_derivative is True
    assert marshall.dims == 1


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_diffpair_factory_no_grad(device_type: DeviceType):
    """create_torch_tensor_marshall with grad=None produces no derivative."""
    layout = _get_layout(device_type)

    primal = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    pair = diff_pair(primal, None)

    marshall = ttm.create_torch_tensor_marshall(layout, pair)
    assert marshall.has_derivative is False


DIFF_SRC = r"""
[Differentiable]
float square(float x) { return x * x; }
"""


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_diffpair_read_signature(device_type: DeviceType):
    """Calling a function with a DiffPair triggers NativeTorchTensorDiffPair::read_signature.

    NOTE: read_signature is currently unreachable due to a regression in PR #872
    where the SignatureBuffer optimization bypasses virtual dispatch. Filed as #923.
    This test exercises the forward pass but does NOT cover read_signature.
    """
    device = helpers.get_torch_device(device_type)
    func = helpers.create_function_from_module(device, "square", DIFF_SRC)

    primal = torch.tensor([2.0, 3.0, 4.0], device="cuda", dtype=torch.float32, requires_grad=True)
    grad = torch.ones(3, device="cuda", dtype=torch.float32)
    pair = diff_pair(primal, grad)

    result = func(pair)
    assert result is not None


@pytest.mark.skip(
    reason="diff_pair(None, grad) triggers interop buffer lifetime race (#929). "
    "Re-enable with DEVICE_TYPES parametrization once #929 is fixed."
)
def test_diffpair_get_shape_grad_only():
    """NativeTorchTensorMarshall::get_shape falls back to grad when primal=None.

    Skipped: dispatching diff_pair(None, grad) hits a cleanup race in
    create_zeroed_interop_buffer where an async CUDA memset outlives the
    interop buffer (#929). Crashes the worker and poisons the CUDA context
    for subsequent tests. Even CUDA-only + fallback bridge mode triggers it.
    """
    device = helpers.get_torch_device(DeviceType.cuda)
    func = helpers.create_function_from_module(device, "square", DIFF_SRC)

    grad = torch.ones(5, device="cuda", dtype=torch.float32)
    pair = diff_pair(None, grad)

    result = func(pair)
    assert result is not None


# ============================================================================
# NativeTorchTensorDiffPair nanobind binding coverage
# ============================================================================


@requires_cuda
def test_diffpair_repr():
    """NativeTorchTensorDiffPair.__repr__ formats primal/grad/index/is_input."""
    primal = torch.tensor([1.0], device="cuda")
    grad = torch.tensor([0.0], device="cuda")
    pair = diff_pair(primal, grad)

    r = repr(pair)
    assert "primal=Tensor" in r
    assert "grad=Tensor" in r
    assert "is_input=True" in r


def test_diffpair_repr_none():
    """__repr__ shows None for missing tensors."""
    pair = diff_pair(None, None)
    r = repr(pair)
    assert "primal=None" in r
    assert "grad=None" in r


@requires_cuda
def test_diffpair_property_setters():
    """Setting primal and grad properties exercises the nanobind setter lambdas."""
    pair = diff_pair(None, None)
    assert pair.primal is None
    assert pair.grad is None

    t = torch.tensor([1.0, 2.0], device="cuda")
    pair.primal = t
    assert pair.primal is not None
    assert torch.equal(pair.primal, t)

    g = torch.tensor([3.0, 4.0], device="cuda")
    pair.grad = g
    assert pair.grad is not None
    assert torch.equal(pair.grad, g)


@requires_cuda
def test_diffpair_clear_tensors():
    """clear_tensors() sets both primal and grad to None."""
    primal = torch.tensor([1.0], device="cuda")
    grad = torch.tensor([0.0], device="cuda")
    pair = diff_pair(primal, grad)
    assert pair.primal is not None
    assert pair.grad is not None

    pair.clear_tensors()
    assert pair.primal is None
    assert pair.grad is None


# ============================================================================
# TorchTensorMarshall properties and type conversion
# ============================================================================


@requires_cuda
@pytest.mark.parametrize("device_type", DEVICE_TYPES)
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


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_slang_dtype_to_torch_none_for_non_scalar(device_type: DeviceType):
    """_slang_dtype_to_torch returns None for non-scalar SlangType."""
    layout = _get_layout(device_type)
    vec_type = layout.find_type_by_name("float2")
    assert ttm._slang_dtype_to_torch(vec_type) is None


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_torch_dtype_to_slang_none_for_unsupported(device_type: DeviceType):
    """_torch_dtype_to_slang returns None for unsupported torch dtype."""
    layout = _get_layout(device_type)
    assert ttm._torch_dtype_to_slang(torch.complex128, layout) is None


# ============================================================================
# Error paths
# ============================================================================


def test_hash_torch_tensor_raises():
    """hash_torch_tensor always raises ValueError."""
    with pytest.raises(ValueError, match="should not need a hash"):
        ttm.hash_torch_tensor(torch.tensor([1.0]))


def test_hash_torch_diff_pair_raises():
    """hash_torch_diff_pair always raises ValueError."""
    pair = diff_pair(torch.tensor([1.0]), torch.tensor([0.0]))
    with pytest.raises(ValueError, match="should not need a hash"):
        ttm.hash_torch_diff_pair(pair)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_factory_unsupported_type_raises(device_type: DeviceType):
    """Passing a non-tensor to create_torch_tensor_marshall raises ValueError."""
    layout = _get_layout(device_type)
    with pytest.raises(ValueError, match="unsupported"):
        ttm.create_torch_tensor_marshall(layout, "not a tensor")


@requires_cuda
@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_factory_unsupported_torch_dtype_raises(device_type: DeviceType):
    """Passing a tensor with unsupported dtype raises ValueError."""
    layout = _get_layout(device_type)
    t = torch.tensor([1.0 + 2.0j], dtype=torch.complex64, device="cuda")
    with pytest.raises(ValueError, match=r"[Uu]nsupported"):
        ttm.create_torch_tensor_marshall(layout, t)


@requires_cuda
@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_diffpair_factory_unsupported_dtype_raises(device_type: DeviceType):
    """DiffPair factory raises for unsupported torch dtype."""
    layout = _get_layout(device_type)
    primal = torch.tensor([1.0 + 2.0j], dtype=torch.complex64, device="cuda")
    grad = torch.tensor([0.0 + 0.0j], dtype=torch.complex64, device="cuda")
    pair = diff_pair(primal, grad)
    with pytest.raises(ValueError, match=r"[Uu]nsupported"):
        ttm.create_torch_tensor_marshall(layout, pair)


IDENTITY_SRC = r"""
float identity(float x) { return x; }
"""

VEC_SRC = r"""
float2 scale_vec(float2 v) { return v * 2.0; }
"""


@requires_cuda
@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_torch_vector_dimension_mismatch_error(device_type: DeviceType):
    """Type resolution rejects a tensor whose trailing dim doesn't match the vector type."""
    device = helpers.get_torch_device(device_type)
    func = helpers.create_function_from_module(device, "scale_vec", VEC_SRC)

    bad = torch.tensor([[1.0, 2.0, 3.0]], device="cuda", dtype=torch.float32)
    with pytest.raises(ResolveException, match="does not match slang type"):
        func(bad)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_torch_cpu_tensor_rejected(device_type: DeviceType):
    """Non-CUDA torch tensors are rejected by write_shader_cursor_pre_dispatch."""
    device = helpers.get_torch_device(device_type)
    func = helpers.create_function_from_module(device, "identity", IDENTITY_SRC)

    cpu_tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)
    with pytest.raises(Exception, match=r"[Cc][Uu][Dd][Aa]"):
        func(cpu_tensor)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
