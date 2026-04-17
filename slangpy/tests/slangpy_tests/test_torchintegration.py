# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import sys
import numpy as np

from slangpy import DeviceType, Device, Module, grid
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
        (torch.empty((1,), dtype=torch.uint16).cuda(), "D1,S27"),
        (torch.empty((1,), dtype=torch.uint32).cuda(), "D1,S28"),
        (torch.empty((1,), dtype=torch.uint64).cuda(), "D1,S29"),
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
def test_scalar_tensor_from_torch_accepts_singleton_last_dim(device_type: DeviceType):
    """
    Test Tensor.from_torch() accepts singleton trailing dimensions for scalar element types.
    """
    from slangpy import Tensor

    device = helpers.get_torch_device(device_type)
    module = load_test_module(device_type)

    input_torch = (
        torch.arange(12, dtype=torch.int32, device=torch.device("cuda"))
        .to(torch.uint32)
        .reshape(3, 4, 1)
    )
    output_torch = torch.zeros_like(input_torch)

    input_tensor = Tensor.from_torch(device, input_torch, dtype=module.uint)
    output_tensor = Tensor.from_torch(device, output_torch, dtype=module.uint)

    module.copy_uint_tensor(input_tensor, output_tensor)

    expected = input_torch.cpu().numpy().reshape(3, 4)
    np.testing.assert_array_equal(output_tensor.to_numpy(), expected)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_tensor_from_torch_syncs_cuda_to_device(device_type: DeviceType):
    """
    Test Tensor.from_torch() makes CUDA writes visible to subsequent device reads.
    """
    from slangpy import Tensor

    device = helpers.get_torch_device(device_type)
    module = load_test_module(device_type)

    input_torch = (
        torch.arange(12, dtype=torch.int32, device=torch.device("cuda"))
        .to(torch.uint32)
        .reshape(3, 4, 1)
    )

    tensor = Tensor.from_torch(device, input_torch, dtype=module.uint)

    expected = input_torch.cpu().numpy().reshape(3, 4)
    np.testing.assert_array_equal(tensor.to_numpy(), expected)


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
def test_copy_uint_tensor(device_type: DeviceType):
    """
    Test raw torch.Tensor marshalling for uint tensors.
    """
    module = load_test_module(device_type)

    input_torch = (
        torch.arange(12, dtype=torch.int32, device=torch.device("cuda"))
        .to(torch.uint32)
        .reshape(3, 4)
    )
    output_torch = torch.zeros_like(input_torch)

    module.copy_uint_tensor(input_torch, output_torch)

    assert torch.equal(output_torch, input_torch)


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

    param = torch.nn.parameter.Parameter(torch.empty((4, 4), dtype=torch.float32).cuda())
    tensor = torch.empty((4, 4), dtype=torch.float32).cuda()

    sig_param = SignatureBuilder()
    sig_tensor = SignatureBuilder()
    cd.get_value_signature(sig_param, param)
    cd.get_value_signature(sig_tensor, tensor)

    assert sig_param.str == sig_tensor.str


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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
