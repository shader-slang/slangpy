# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import sys

from typing import Callable

from slangpy import DeviceType, Device, Module
from slangpy.core.native import NativeCallDataCache, SignatureBuilder
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
        (torch.empty((1,), dtype=torch.float32).cuda(), "D1,S6"),
        (torch.empty((1,), dtype=torch.float32, requires_grad=True).cuda(), "D1,S6"),
        (torch.empty((1,), dtype=torch.float16).cuda(), "D1,S5"),
        (torch.empty((1,), dtype=torch.int32).cuda(), "D1,S3"),
        (torch.empty((1,), dtype=torch.uint8).cuda(), "D1,S0"),
        (torch.empty((1, 1, 1), dtype=torch.uint8).cuda(), "D3,S0"),
    ],
)
def test_torch_signature(pair: tuple[torch.Tensor, str]):
    cd = NativeCallDataCache()
    sig = SignatureBuilder()
    cd.get_value_signature(sig, pair[0])
    assert sig.str == f"torch\n[{pair[1]}]"


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

    with pytest.raises(ValueError, match="does not match expected shape"):
        res = module.add_vectors(a, b)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("extra_dims", [0, 1, 3])
def test_add_vectors_generic_explicit(device_type: DeviceType, extra_dims: int):
    pytest.skip("Crashes due to slang bug")

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
        pytest.skip("Slang bug currently causing derivatives to return 0")

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

    # Should this work??
    # res.backward(torch.ones_like(res))


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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
